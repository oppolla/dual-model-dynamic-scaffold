import torch
import time
from collections import deque
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import contextlib
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_processor import LogitsProcessor
from sovl_utils import calculate_confidence, detect_repetitions, adjust_temperature

class GenerationManager:
    """Manages text generation, scaffold integration, and memory handling for the SOVL system."""
    
    def __init__(
        self,
        config_manager: Any,
        base_model: AutoModelForCausalLM,
        scaffolds: List[AutoModelForCausalLM],
        base_tokenizer: AutoTokenizer,
        scaffold_tokenizer: AutoTokenizer,
        state: SOVLState,
        logger: Logger,
        error_logger: Logger,
        cross_attention_injector: Any,
        scaffold_manager: Any,
        temperament: Any,
        curiosity_manager: Any = None,
    ):
        """Initialize GenerationManager with configuration and model components."""
        # Core components
        self.config_manager = config_manager
        self.base_model = base_model
        self.scaffolds = scaffolds
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.state = state
        self.logger = logger
        self.error_logger = error_logger
        self.cross_attention_injector = cross_attention_injector
        self.scaffold_manager = scaffold_manager
        self.temperament = temperament
        self.curiosity_manager = curiosity_manager

        # Configuration sections
        self._load_config_sections()

        # Device and memory settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaffold_unk_id = self.controls_config.get("scaffold_unk_id", scaffold_tokenizer.unk_token_id)
        self.use_token_map_memory = self.controls_config.get("use_token_map_memory", True)
        self.dynamic_cross_attn_mode = self.controls_config.get("dynamic_cross_attn_mode", None)

        # Generation settings
        self.max_retries = self.controls_config.get("max_generation_retries", 3)
        self.memory_threshold = self.controls_config.get("memory_threshold", 0.85)
        self.generation_callbacks: Dict[str, List[Callable]] = {
            "pre_generate": [],
            "post_generate": []
        }

    def _load_config_sections(self) -> None:
        """Load configuration sections from config manager."""
        self.controls_config = self.config_manager.get_section("controls_config")
        self.curiosity_config = self.config_manager.get_section("curiosity_config")
        self.training_config = self.config_manager.get_section("training_config")

    def register_callback(self, stage: str, callback: Callable) -> None:
        """Register a callback for generation stages."""
        if stage in self.generation_callbacks:
            self.generation_callbacks[stage].append(callback)
        else:
            self._log_error(
                error=f"Invalid callback stage: {stage}",
                context="register_callback"
            )

    def check_memory_health(self) -> bool:
        """Check GPU memory usage and clean up if necessary."""
        if not torch.cuda.is_available():
            return True

        try:
            memory_stats = torch.cuda.memory_stats()
            current_memory = memory_stats["allocated_bytes.all.current"]
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_ratio = current_memory / total_memory

            self._log_event(
                event="memory_health_check",
                data={
                    "memory_ratio": memory_ratio,
                    "current_memory": current_memory,
                    "total_memory": total_memory
                }
            )

            if memory_ratio > self.memory_threshold:
                torch.cuda.empty_cache()
                self._log_event(
                    event="memory_cleanup",
                    data={"reason": "threshold_exceeded"}
                )
                return False
            return True
        except Exception as e:
            self._log_error(
                error=f"Memory health check failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="check_memory_health"
            )
            return False

    def _handle_error_prompt(self, error_msg: str, temp_history: ConversationHistory) -> str:
        """Generate a response to a system error with retry mechanism."""
        original_history = self.state.history
        self.state.history = ConversationHistory(
            maxlen=self.controls_config.get("conversation_history_maxlen", 10)
        )

        try:
            for attempt in range(self.max_retries):
                try:
                    response = self.generate(
                        prompt=f"System error detected: {error_msg} What happened?",
                        max_new_tokens=self.curiosity_config.get("max_new_tokens", 60),
                        temperature=self.controls_config.get("base_temperature", 0.7) + 0.2 * (attempt + 1),
                        top_k=self.curiosity_config.get("top_k", 50),
                        do_sample=True,
                    )
                    self._log_event(
                        event="error_prompt_handled",
                        data={
                            "prompt": f"System error detected: {error_msg} What happened?",
                            "response": response,
                            "attempt": attempt + 1,
                            "confidence_score": 0.5
                        }
                    )
                    return response
                except torch.cuda.OutOfMemoryError:
                    if attempt < self.max_retries - 1:
                        torch.cuda.empty_cache()
                        continue
                    raise
                except Exception as e:
                    self._log_error(
                        error=f"Error handling prompt attempt {attempt + 1}: {str(e)}",
                        stack_trace=traceback.format_exc(),
                        context="handle_error_prompt"
                    )
                    if attempt == self.max_retries - 1:
                        raise
        except Exception as e:
            self._log_error(
                error=f"Failed to handle error prompt: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="handle_error_prompt"
            )
            return "Unable to process error prompt after retries."
        finally:
            self.state.history = original_history

    def has_repetition(self, output_ids: torch.Tensor, n: int = 3, threshold: float = 0.9) -> bool:
        """Check for repetition in generated output."""
        try:
            ids = output_ids.tolist()
            special_ids = {
                self.base_tokenizer.pad_token_id,
                self.base_tokenizer.eos_token_id,
                self.base_tokenizer.bos_token_id,
                self.base_tokenizer.unk_token_id,
            }
            filtered = [i for i in ids if i not in special_ids]

            for i in range(len(filtered) - 2 * n):
                seq1 = filtered[i: i + n]
                seq2 = filtered[i + n: i + 2 * n]
                if len(seq1) == len(seq2) and seq1 == seq2:
                    return True
                if len(seq1) == len(seq2):
                    embeddings1 = self.base_model.get_input_embeddings()(
                        torch.tensor(seq1, device=self.device)
                    )
                    embeddings2 = self.base_model.get_input_embeddings()(
                        torch.tensor(seq2, device=self.device)
                    )
                    similarity = torch.cosine_similarity(
                        embeddings1.mean(dim=0), embeddings2.mean(dim=0), dim=0
                    )
                    if similarity > threshold:
                        return True
            return False
        except Exception as e:
            self._log_error(
                error=f"Repetition check failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="has_repetition"
            )
            return False

    def tokenize_and_map(
        self,
        prompts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = 'max_length'
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and map to scaffold tokens."""
        try:
            max_length = max_length or self.training_config.get("max_seq_length", 128)
            prompts = [prompts] if isinstance(prompts, str) else prompts

            batch_size = self.training_config.get("batch_size", 1)
            input_batches = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]
            all_input_ids = []
            all_attention_masks = []

            for batch in input_batches:
                inputs = self.base_tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=padding,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                scaffold_input_ids = self.scaffold_manager.map_sequence(inputs.input_ids)
                scaffold_attention_mask = (
                    scaffold_input_ids != self.scaffold_tokenizer.pad_token_id
                ).int()
                all_input_ids.append(scaffold_input_ids)
                all_attention_masks.append(scaffold_attention_mask)

            return {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
            }
        except Exception as e:
            self._log_error(
                error=f"Tokenization failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="tokenize_and_map"
            )
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get hidden states from scaffold model."""
        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16
            ):
                scaffold_outputs = self.scaffolds[0](
                    **{k: v for k, v in scaffold_inputs.items() if k in ['input_ids', 'attention_mask']},
                    output_hidden_states=True,
                )
                hidden_states = (
                    scaffold_outputs.hidden_states[-1]
                    if hasattr(scaffold_outputs, 'hidden_states')
                    else scaffold_outputs.base_model_output.hidden_states[-1]
                )
                return hidden_states.detach()
        except Exception as e:
            self._log_error(
                error=f"Scaffold hidden states failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="get_scaffold_hidden_states"
            )
            raise

    @contextlib.contextmanager
    def _scaffold_context(self, scaffold_hidden_states: torch.Tensor):
        """Manage scaffold context with memory monitoring."""
        try:
            with self.state.memory_lock:
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    if mem_before > self.memory_threshold:
                        torch.cuda.empty_cache()
                self._temp_scaffold_context = scaffold_hidden_states.detach()
            yield
        finally:
            self._clear_scaffold_cache()

    def _clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches with memory optimization."""
        with self.state.memory_lock:
            try:
                if hasattr(self, '_temp_scaffold_context') and self._temp_scaffold_context is not None:
                    if isinstance(self._temp_scaffold_context, torch.Tensor):
                        self._temp_scaffold_context = self._temp_scaffold_context.to('cpu')
                    del self._temp_scaffold_context
                self._temp_scaffold_context = None

                if self.state.last_prompt_embedding is not None:
                    if isinstance(self.state.last_prompt_embedding, torch.Tensor):
                        self.state.last_prompt_embedding = self.state.last_prompt_embedding.to('cpu')
                    del self.state.last_prompt_embedding
                    self.state.last_prompt_embedding = None

                if self.state.dream_memory:
                    new_memory = deque(maxlen=self.state.dream_memory_maxlen)
                    for tensor, weight in self.state.dream_memory:
                        new_memory.append((tensor.to('cpu'), weight))
                    self.state.dream_memory = new_memory

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._log_event(
                    event="scaffold_cache_cleared",
                    data={}
                )
            except Exception as e:
                self._log_error(
                    error=f"Failed to clear scaffold cache: {str(e)}",
                    stack_trace=traceback.format_exc(),
                    context="clear_scaffold_cache"
                )

    def _update_token_map_memory(self, prompt: str, confidence: float) -> None:
        """Update token map weights."""
        if not self.use_token_map_memory:
            return
        try:
            self.scaffold_manager.update_token_map_memory(
                prompt=prompt,
                confidence=confidence,
                tokenizer=self.base_tokenizer,
                memory_decay_rate=self.controls_config.get("memory_decay_rate", 0.95),
            )
        except Exception as e:
            self._log_error(
                error=f"Token map update failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="update_token_map_memory"
            )

    def prepare_for_training(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare data for training."""
        try:
            prompts = [item['prompt'] for item in batch]
            scaffold_inputs = self.tokenize_and_map(prompts)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
            return {
                "scaffold_hidden_states": scaffold_hidden_states,
                "prompts": prompts
            }
        except Exception as e:
            self._log_error(
                error=f"Training preparation failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="prepare_for_training"
            )
            raise

    def _log_event(self, event: str, data: Dict[str, Any]) -> None:
        """Log an event with common metadata."""
        self.logger.record({
            "event": event,
            **data,
            "timestamp": time.time(),
            "conversation_id": self.state.history.conversation_id,
            "state_hash": self.state.state_hash(),
        })

    def _log_error(self, error: str, context: str, stack_trace: Optional[str] = None) -> None:
        """Log an error with common metadata."""
        self.error_logger.record({
            "error": error,
            "context": context,
            "timestamp": time.time(),
            "stack_trace": stack_trace or traceback.format_exc(),
            "conversation_id": self.state.history.conversation_id,
            "state_hash": self.state.state_hash(),
        })

    def _prepare_generation_params(self, max_new_tokens: int, scaffold_weight: Optional[float], **kwargs) -> Dict[str, Any]:
        """Prepare and validate generation parameters."""
        return {
            "max_new_tokens": max_new_tokens or self.curiosity_config.get("max_new_tokens", 50),
            "scaffold_weight": scaffold_weight,
            "temperature": kwargs.get("temperature", self.controls_config.get("base_temperature", 0.7)),
            "top_k": kwargs.get("top_k", self.curiosity_config.get("top_k", 30)),
            "do_sample": kwargs.get("do_sample", False),
            "prompt_count": 1 if isinstance(kwargs.get("prompts"), str) else len(kwargs.get("prompts", [])),
        }

    def _compute_dynamic_factor(self) -> Optional[torch.Tensor]:
        """Compute dynamic cross-attention factor based on configuration."""
        if not self.controls_config.get("enable_dynamic_cross_attention", False) or not self.dynamic_cross_attn_mode:
            return None

        try:
            last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
            if self.dynamic_cross_attn_mode == 'confidence':
                return torch.tensor(last_conf, device=self.device, dtype=torch.float)
            elif self.dynamic_cross_attn_mode == 'temperament':
                return torch.tensor(self.temperament.score, device=self.device, dtype=torch.float)
            return None
        except Exception as e:
            self._log_error(
                error=f"Failed to compute dynamic factor: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="compute_dynamic_factor"
            )
            return None

    def _prepare_dream_memory(self) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Prepare dream memory tensors if available."""
        dream_memory_info = {"used": False, "tensor_count": 0, "shapes": []}
        memory_tensors = None
        dream_memory_weight = self.controls_config.get("dream_memory_weight", 0.1)

        if self.state.dream_memory and dream_memory_weight > 0:
            try:
                with self.state.memory_lock:
                    dream_tensors, dream_weights = zip(*self.state.dream_memory)
                    dream_memory_info["tensor_count"] = len(dream_tensors)
                    dream_memory_info["shapes"] = [list(t.shape) for t in dream_tensors]
                    for tensor in dream_tensors:
                        if tensor.shape[-1] != self.state.hidden_size:
                            raise ValueError(
                                f"Dream tensor shape {tensor.shape} mismatches hidden_size {self.state.hidden_size}"
                            )
                    dream_tensors = torch.stack([t.detach().to(self.device) for t in dream_tensors])
                    dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=self.device)
                    memory_tensors = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / dream_weights.sum()
                    dream_memory_info["used"] = True
            except Exception as e:
                dream_memory_info["error"] = str(e)
                self._log_error(
                    error=f"Dream memory preparation failed: {str(e)}",
                    stack_trace=traceback.format_exc(),
                    context="prepare_dream_memory"
                )

        return memory_tensors, dream_memory_info

    def _handle_repetition(self, seq: torch.Tensor, seq_ids: List[int], outputs: Any) -> List[int]:
        """Handle detected repetition in generated sequence."""
        if self.controls_config.get("enable_repetition_check", True) and self.has_repetition(seq):
            original_text = self.base_tokenizer.decode(seq_ids, skip_special_tokens=True)
            for j in range(len(seq_ids) - 6):
                if all(seq_ids[j + k] == seq_ids[j + k + 3] for k in range(3)):
                    seq_ids = seq_ids[:j + 3]
                    break
            self._log_event(
                event="repetition_detected",
                data={
                    "original_text": original_text,
                    "truncated_at": j + 3
                }
            )
        return seq_ids

    def _update_curiosity(self, prompt: str, response: str, confidence_score: float) -> str:
        """Update curiosity system and append questions if needed."""
        if not self.curiosity_config.get("enable_curiosity", True) or not self.curiosity_manager:
            return response

        try:
            self.state.curiosity.prune_old_questions(self.curiosity_config.get("question_timeout", 3600.0))
            self.curiosity_manager.update_pressure(
                self.temperament.score,
                confidence_score,
                0.0,
                self.state.curiosity.context_vector,
            )
            if self.curiosity_manager.should_erupt():
                question = self.curiosity_manager.generate_question(
                    state=self.state,
                    tokenizer=self.base_tokenizer,
                    model=self.scaffolds[0],
                    prompt=prompt,
                )
                if question and isinstance(question, str) and question.strip():
                    response += f" {question}"
                    self.state.curiosity.update_question_history(question, time.time())
                    self._log_event(
                        event="curiosity_question",
                        data={
                            "prompt": question,
                            "response": "",
                            "confidence_score": 0.0,
                            "is_system_question": True
                        }
                    )
        except Exception as e:
            self._log_error(
                error=f"Curiosity eruption failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="update_curiosity"
            )
        return response

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 50,
        scaffold_weight: Optional[float] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate response(s) for the given prompt(s)."""
        single_prompt = isinstance(prompts, str)
        prompts = [prompts] if single_prompt else prompts
        responses = []

        try:
            # Check gestation state
            if self.state.gestation_state == "gestating":
                self._log_event(
                    event="gestation_skip",
                    data={"reason": "System is in gestation state"}
                )
                return "System is currently in gestation state. Please try again later."

            generation_params = self._prepare_generation_params(max_new_tokens, scaffold_weight, prompts=prompts, **kwargs)
            for callback in self.generation_callbacks["pre_generate"]:
                callback(generation_params)

            self._log_event(event="generation_initiated", data={"params": generation_params})

            if not self.check_memory_health():
                self._log_event(
                    event="memory_warning",
                    data={"warning": "Memory health check failed, attempting cleanup"}
                )

            start_time = time.time()
            base_inputs = self.base_tokenizer(
                prompts, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)
            input_ids = base_inputs['input_ids']
            input_lengths = base_inputs['attention_mask'].sum(dim=1)

            scaffold_inputs = self.tokenize_and_map(prompts)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

            temperature = self.temperament.adjust_parameter(
                base_value=kwargs.get('temperature', self.controls_config.get("base_temperature", 0.7)),
                parameter_type="temperature",
            )
            kwargs['temperature'] = temperature

            dynamic_factor = self._compute_dynamic_factor()
            memory_tensors, dream_memory_info = self._prepare_dream_memory()

            with self._scaffold_context(scaffold_hidden_states):
                self.cross_attention_injector.set_influence(model=self.base_model, weight=scaffold_weight)
                for attempt in range(self.max_retries):
                    try:
                        outputs = self.base_model.generate(
                            input_ids,
                            max_new_tokens=generation_params["max_new_tokens"],
                            pad_token_id=self.base_tokenizer.pad_token_id,
                            eos_token_id=self.base_tokenizer.eos_token_id,
                            temperature=temperature * (1 + 0.1 * attempt),
                            return_dict_in_generate=True,
                            output_scores=True,
                            scaffold_context=scaffold_hidden_states,
                            memory_tensors=memory_tensors,
                            memory_weight=self.controls_config.get("dream_memory_weight", 0.1),
                            dynamic_factor=dynamic_factor,
                            **kwargs,
                        )
                        generated_ids = outputs.sequences
                        break
                    except torch.cuda.OutOfMemoryError:
                        if attempt < self.max_retries - 1:
                            torch.cuda.empty_cache()
                            continue
                        raise
                    except Exception as e:
                        self._log_error(
                            error=f"Generation attempt {attempt + 1} failed: {str(e)}",
                            stack_trace=traceback.format_exc(),
                            context="generate"
                        )
                        if attempt == self.max_retries - 1:
                            raise

                for i, (seq, input_length) in enumerate(zip(generated_ids, input_lengths)):
                    prompt = prompts[i]
                    seq_ids = seq[input_length:].tolist()
                    confidence_score = 0.5

                    if self.controls_config.get("enable_confidence_tracking", True):
                        try:
                            confidence_score = calculate_confidence(outputs.scores, seq_ids)
                            with self.state.memory_lock:
                                self.state.sleep_confidence_sum += confidence_score
                                self.state.sleep_confidence_count += 1
                                self.state.confidence_history.append(confidence_score)
                                
                                # Update gestation state based on confidence
                                if len(self.state.confidence_history) >= self.controls_config.get("gestation_history_threshold", 100):
                                    avg_confidence = sum(self.state.confidence_history[-100:]) / 100
                                    if avg_confidence < self.controls_config.get("gestation_confidence_threshold", 0.3):
                                        self.state.gestation_state = "gestating"
                                        self._log_event(
                                            event="gestation_triggered",
                                            data={"avg_confidence": avg_confidence}
                                        )
                        except Exception as e:
                            self._log_error(
                                error=f"Confidence calculation failed: {str(e)}",
                                stack_trace=traceback.format_exc(),
                                context="generate_confidence"
                            )

                    seq_ids = self._handle_repetition(seq, seq_ids, outputs)
                    response = self.base_tokenizer.decode(seq_ids, skip_special_tokens=True)
                    
                    # Update curiosity and generate questions if needed
                    if self.curiosity_manager and self.controls_config.get("enable_curiosity", True):
                        response = self._update_curiosity(prompt, response, confidence_score)
                        if self.curiosity_manager.should_generate_question():
                            question = self.curiosity_manager.generate_question(prompt, response)
                            if question:
                                self._log_event(
                                    event="curiosity_question_generated",
                                    data={"question": question}
                                )
                                response += f"\n\nQuestion: {question}"

                    self._log_event(
                        event="generation_completed",
                        data={
                            "prompt": prompt,
                            "response": response,
                            "confidence_score": confidence_score,
                            "generation_params": generation_params,
                            "dream_memory_info": dream_memory_info
                        }
                    )
                    self.state.history.add_message(prompt, response)

                    if self.use_token_map_memory:
                        self._update_token_map_memory(prompt, confidence_score)

                    responses.append(response)

            for callback in self.generation_callbacks["post_generate"]:
                callback(responses, generation_params)

            print(f"Generation completed in {time.time() - start_time:.2f}s")
            return responses[0] if single_prompt else responses

        except torch.cuda.OutOfMemoryError as oom:
            self._log_error(
                error="CUDA out of memory",
                context="generate_oom",
                stack_trace=traceback.format_exc(),
                additional_data={
                    "type": "OOM",
                    "prompt": prompts[0][:200] if prompts else "",
                    "memory_stats": {
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "max_allocated": torch.cuda.max_memory_allocated(),
                    } if torch.cuda.is_available() else None
                }
            )
            torch.cuda.empty_cache()

            if self.controls_config.get("enable_error_listening", True):
                return self._handle_error_prompt("GPU memory error occurred", self.state.history)
            return "System is low on memory - please try a shorter prompt"

        except Exception as e:
            self._log_error(
                error=f"Generation error: {str(e)}",
                context="generate",
                stack_trace=traceback.format_exc(),
                additional_data={
                    "type": type(e).__name__,
                    "prompt": prompts[0][:200] if prompts else ""
                }
            )

            if self.controls_config.get("enable_error_listening", True):
                try:
                    return self._handle_error_prompt(f"Generation error: {str(e)}", self.state.history)
                except Exception as inner_e:
                    self._log_error(
                        error=f"Failed to handle generation error: {str(inner_e)}",
                        context="generate_error_handling",
                        stack_trace=traceback.format_exc(),
                        additional_data={"original_error": str(e)}
                    )
            return "An error occurred during generation"

        finally:
            if 'generated_ids' in locals():
                del generated_ids
            if 'outputs' in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._log_event(event="generate_cleanup", data={})
