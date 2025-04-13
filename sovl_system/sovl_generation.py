import torch
import time
import random
from collections import deque
from typing import Optional, Dict, Any, List
import contextlib
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_processor import LogitsProcessor
from sovl_utils import calculate_confidence, detect_repetitions

class GenerationManager:
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

        # Cache config sections
        self.controls_config = config_manager.get_section("controls_config")
        self.curiosity_config = config_manager.get_section("curiosity_config")
        self.training_config = config_manager.get_section("training_config")

        # Initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaffold_unk_id = self.controls_config.get("scaffold_unk_id", scaffold_tokenizer.unk_token_id)
        self.use_token_map_memory = self.controls_config.get("use_token_map_memory", True)
        self.dynamic_cross_attn_mode = self.controls_config.get("dynamic_cross_attn_mode", None)

        # Initialize generation parameters
        self.max_retries = self.controls_config.get("max_generation_retries", 3)
        self.memory_threshold = self.controls_config.get("memory_threshold", 0.85)

    def _handle_error_prompt(self, error_msg: str, temp_history: ConversationHistory) -> str:
        """Generate a response to a system error with retry mechanism."""
        try:
            self.history = ConversationHistory(maxlen=self.controls_config.get("conversation_history_maxlen", 10))
            for attempt in range(self.max_retries):
                try:
                    response = self.generate(
                        f"System error detected: {error_msg} What happened?",
                        max_new_tokens=self.curiosity_config.get("max_new_tokens", 60),
                        temperature=self.controls_config.get("base_temperature", 0.7) + 0.2 * (attempt + 1),
                        top_k=self.curiosity_config.get("top_k", 50),
                        do_sample=True,
                    )
                    self.logger.record({
                        "prompt": f"System error detected: {error_msg} What happened?",
                        "response": response,
                        "timestamp": time.time(),
                        "conversation_id": self.history.conversation_id,
                        "is_error_prompt": True,
                        "confidence_score": 0.5,
                        "attempt": attempt + 1,
                        "state_hash": self.state.state_hash(),
                    })
                    return response
                except Exception as e:
                    self.error_logger.record({
                        "error": f"Error handling prompt attempt {attempt + 1}: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })
                    if attempt == self.max_retries - 1:
                        raise
        finally:
            self.history = temp_history
        return "Unable to process error prompt after retries."

    def has_repetition(self, output_ids: torch.Tensor, n: int = 3, threshold: float = 0.9) -> bool:
        """Check for repetition in generated output with similarity threshold."""
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
                seq1 = filtered[i : i + n]
                seq2 = filtered[i + n : i + 2 * n]
                if len(seq1) == len(seq2) and seq1 == seq2:
                    return True
                # Add similarity check for near-repetitions
                if len(seq1) == len(seq2):
                    embeddings1 = self.base_model.get_input_embeddings()(torch.tensor(seq1, device=self.device))
                    embeddings2 = self.base_model.get_input_embeddings()(torch.tensor(seq2, device=self.device))
                    similarity = torch.cosine_similarity(embeddings1.mean(dim=0), embeddings2.mean(dim=0), dim=0)
                    if similarity > threshold:
                        return True
            return False
        except Exception as e:
            self.error_logger.record({
                "error": f"Repetition check failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })
            return False

    def tokenize_and_map(self, prompts: str or List[str], max_length: Optional[int] = None, padding: str = 'max_length') -> Dict[str, torch.Tensor]:
        """Tokenize prompts and map to scaffold tokens with batch processing."""
        try:
            max_length = max_length or self.training_config.get("max_seq_length", 128)
            if isinstance(prompts, str):
                prompts = [prompts]

            # Batch processing for efficiency
            batch_size = self.training_config.get("batch_size", 1)
            input_batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
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
                scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()
                all_input_ids.append(scaffold_input_ids)
                all_attention_masks.append(scaffold_attention_mask)

            return {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
            }
        except Exception as e:
            self.error_logger.record({
                "error": f"Tokenization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get hidden states from scaffold model with optimized memory usage."""
        try:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
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
            self.error_logger.record({
                "error": f"Scaffold hidden states failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })
            raise

    @contextlib.contextmanager
    def _scaffold_context(self, scaffold_hidden_states: torch.Tensor):
        """Manage scaffold context with safe tensor handling and memory monitoring."""
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

    def _clear_scaffold_cache(self):
        """Clear scaffold-related caches safely with memory optimization."""
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

                self.logger.record({
                    "event": "scaffold_cache_cleared",
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash(),
                })
            except Exception as e:
                self.error_logger.record({
                    "error": f"Failed to clear scaffold cache: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash(),
                })

    def _update_token_map_memory(self, prompt: str, confidence: float):
        """Update token map weights with validation."""
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
            self.error_logger.record({
                "error": f"Token map update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, scaffold_weight: Optional[float] = None, **kwargs) -> str:
        """Generate a response for the given prompt with enhanced error handling and efficiency."""
        generated_ids = []
        try:
            max_new_tokens = max_new_tokens or self.curiosity_config.get("max_new_tokens", 50)
            generation_params = {
                "prompt_length": len(prompt),
                "max_new_tokens": max_new_tokens,
                "scaffold_weight": scaffold_weight,
                "temperature": kwargs.get("temperature", self.controls_config.get("base_temperature", 0.7)),
                "top_k": kwargs.get("top_k", self.curiosity_config.get("top_k", 30)),
                "do_sample": kwargs.get("do_sample", False),
            }
            self.logger.record({
                "event": "generation_initiated",
                "params": generation_params,
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })

            start_time = time.time()
            base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = base_inputs['input_ids']
            input_length = input_ids.shape[1]

            scaffold_inputs = self.tokenize_and_map(prompt)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

            # Adjust temperature based on temperament
            temperature = self.temperament.adjust_parameter(
                base_value=kwargs.get('temperature', self.controls_config.get("base_temperature", 0.7)),
                parameter_type="temperature",
            )
            kwargs['temperature'] = temperature

            # Handle dynamic cross-attention factors
            dynamic_factor = None
            if self.controls_config.get("enable_dynamic_cross_attention", False) and self.dynamic_cross_attn_mode:
                try:
                    last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
                    if self.dynamic_cross_attn_mode == 'confidence':
                        dynamic_factor = torch.tensor(last_conf, device=self.device, dtype=torch.float)
                    elif self.dynamic_cross_attn_mode == 'temperament':
                        dynamic_factor = torch.tensor(self.temperament.score, device=self.device, dtype=torch.float)
                except Exception as e:
                    self.logger.record({
                        "warning": f"Failed to compute dynamic factor: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })

            # Prepare dream memory with validation
            memory_tensors = None
            dream_memory_info = {"used": False, "tensor_count": 0, "shapes": []}
            dream_memory_weight = self.controls_config.get("dream_memory_weight", 0.1)
            if self.state.dream_memory and dream_memory_weight > 0:
                try:
                    with self.state.memory_lock:
                        dream_tensors, dream_weights = zip(*self.state.dream_memory)
                        dream_memory_info["tensor_count"] = len(dream_tensors)
                        dream_memory_info["shapes"] = [list(t.shape) for t in dream_tensors]
                        for tensor in dream_tensors:
                            if tensor.shape[-1] != self.state.hidden_size:
                                raise ValueError(f"Dream tensor shape {tensor.shape} mismatches hidden_size {self.state.hidden_size}")
                        dream_tensors = torch.stack([t.detach().to(self.device) for t in dream_tensors])
                        dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=self.device)
                        memory_tensors = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / dream_weights.sum()
                        dream_memory_info["used"] = True
                except Exception as e:
                    self.logger.record({
                        "warning": f"Dream memory preparation failed: {str(e)}",
                        "timestamp": time.time(),
                        "dream_memory_len": len(self.state.dream_memory),
                        "dream_tensor_shapes": [tuple(t.shape) for t, _ in self.state.dream_memory] if self.state.dream_memory else [],
                        "state_hash": self.state.state_hash(),
                    })
                    dream_memory_info["error"] = str(e)

            # Perform generation with scaffold context and retry logic
            with self._scaffold_context(scaffold_hidden_states):
                self.cross_attention_injector.set_influence(model=self.base_model, weight=scaffold_weight)
                chunk_size = self.training_config.get("generation_chunk_size", 512)
                for attempt in range(self.max_retries):
                    try:
                        for chunk_start in range(0, input_ids.size(1), chunk_size):
                            chunk_end = chunk_start + chunk_size
                            input_chunk = input_ids[:, chunk_start:chunk_end]
                            outputs = self.base_model.generate(
                                input_chunk,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=self.base_tokenizer.pad_token_id,
                                eos_token_id=self.base_tokenizer.eos_token_id,
                                temperature=temperature * (1 + 0.1 * attempt),
                                return_dict_in_generate=True,
                                output_scores=True,
                                scaffold_context=scaffold_hidden_states,
                                memory_tensors=memory_tensors,
                                memory_weight=dream_memory_weight,
                                dynamic_factor=dynamic_factor,
                                **kwargs,
                            )
                            generated_ids.extend(outputs.sequences[0][input_length:].tolist())
                        break
                    except Exception as e:
                        self.error_logger.record({
                            "error": f"Generation attempt {attempt + 1} failed: {str(e)}",
                            "timestamp": time.time(),
                            "stack_trace": traceback.format_exc(),
                            "conversation_id": self.state.history.conversation_id,
                            "state_hash": self.state.state_hash(),
                        })
                        if attempt == self.max_retries - 1:
                            raise

            # Calculate confidence and handle repetition
            confidence_score = 0.5
            if self.controls_config.get("enable_confidence_tracking", True):
                try:
                    confidence_score = calculate_confidence(outputs.scores, generated_ids)
                    with self.state.memory_lock:
                        self.state.sleep_confidence_sum += confidence_score
                        self.state.sleep_confidence_count += 1
                        self.state.confidence_history.append(confidence_score)
                except Exception as e:
                    self.logger.record({
                        "warning": f"Confidence calculation failed: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })

            if self.controls_config.get("enable_repetition_check", True) and self.has_repetition(torch.tensor(generated_ids)):
                original_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                for i in range(len(generated_ids) - 6):
                    if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                        generated_ids = generated_ids[:i + 3]
                        break
                self.logger.record({
                    "warning": "Repetition detected",
                    "original_text": original_text,
                    "truncated_at": i + 3,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash(),
                })

            response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Handle curiosity eruptions
            if self.curiosity_config.get("enable_curiosity", True) and self.curiosity_manager:
                try:
                    self.state.curiosity.prune_old_questions(self.curiosity_config.get("question_timeout", 3600.0))
                    self.curiosity_manager.update_pressure(
                        self.temperament.score,
                        confidence_score,
                        0.0,
                        self.state.curiosity.context_vector,
                    )
                    if self.curiosity_manager.should_erupt():
                        q = self.curiosity_manager.generate_question(
                            state=self.state,
                            tokenizer=self.base_tokenizer,
                            model=self.scaffolds[0],
                            prompt=prompt,
                        )
                        if q and isinstance(q, str) and q.strip():
                            response += f" {q}"
                            self.state.curiosity.update_question_history(q, time.time())
                            self.logger.record({
                                "prompt": q,
                                "response": "",
                                "timestamp": time.time(),
                                "conversation_id": self.state.history.conversation_id,
                                "confidence_score": 0.0,
                                "is_system_question": True,
                                "state_hash": self.state.state_hash(),
                            })
                except Exception as e:
                    self.logger.record({
                        "warning": f"Curiosity eruption failed: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })

            # Log and update state
            log_entry = {
                "prompt": prompt,
                "response": response,
                "timestamp": start_time,
                "conversation_id": self.state.history.conversation_id,
                "confidence_score": confidence_score,
                "generation_params": generation_params,
                "dream_memory_info": dream_memory_info,
                "state_hash": self.state.state_hash(),
            }
            self.logger.record(log_entry)
            self.state.history.add_message(prompt, response)

            if self.use_token_map_memory:
                self._update_token_map_memory(prompt, confidence_score)

            print(f"Generation completed in {time.time() - start_time:.2f}s")
            return response

        except torch.cuda.OutOfMemoryError as oom:
            error_details = {
                "error": "CUDA out of memory",
                "type": "OOM",
                "prompt": prompt[:200],
                "timestamp": time.time(),
                "memory_stats": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated(),
                } if torch.cuda.is_available() else None,
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            }
            self.error_logger.record(error_details)
            torch.cuda.empty_cache()

            if self.controls_config.get("enable_error_listening", True):
                try:
                    return self._handle_error_prompt("GPU memory error occurred", self.state.history)
                except Exception as e:
                    self.error_logger.record({
                        "error": f"Failed to handle OOM error: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })
                    return "System is low on memory - please try a shorter prompt"
            return "System is low on memory - please try a shorter prompt"

        except Exception as e:
            error_details = {
                "error": str(e),
                "type": type(e).__name__,
                "prompt": prompt[:200],
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            }
            self.error_logger.record(error_details)

            if self.controls_config.get("enable_error_listening", True):
                try:
                    return self._handle_error_prompt(f"Generation error: {str(e)}", self.state.history)
                except Exception as inner_e:
                    self.error_logger.record({
                        "error": f"Failed to handle generation error: {str(inner_e)}",
                        "original_error": str(e),
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id,
                        "state_hash": self.state.state_hash(),
                    })
            return "An error occurred during generation"

        finally:
            if generated_ids:
                del generated_ids
            if 'outputs' in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.record({
                "event": "generate_cleanup",
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash(),
            })
