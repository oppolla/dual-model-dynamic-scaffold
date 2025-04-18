import torch
import time
from collections import deque
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Set
import contextlib
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_processor import LogitsProcessor
from sovl_utils import calculate_confidence, detect_repetitions, adjust_temperature
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_curiosity import CuriosityManager

class GenerationManager:
    """Manages text generation, scaffold integration, and memory handling for the SOVL system."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        base_model: AutoModelForCausalLM,
        scaffolds: List[AutoModelForCausalLM],
        base_tokenizer: AutoTokenizer,
        scaffold_tokenizer: AutoTokenizer,
        state: SOVLState,
        logger: Logger,
        error_manager: ErrorManager,
        cross_attention_injector: Any,
        scaffold_manager: Any,
        temperament: Any,
        device: torch.device,
        curiosity_manager: Any = None,
    ):
        """Initialize GenerationManager with configuration and model components."""
        # Core components
        self._config_manager = config_manager
        self.base_model = base_model.to(device)
        self.scaffolds = [scaffold.to(device) for scaffold in scaffolds]
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.state = state
        self.logger = logger
        self.error_manager = error_manager
        self.cross_attention_injector = cross_attention_injector
        self.scaffold_manager = scaffold_manager
        self.temperament = temperament
        self.curiosity_manager = curiosity_manager
        self.device = device

        # Initialize configuration
        self._initialize_config()
        
        # Log initialization with config values
        self._log_initialization()

        # Memory settings
        self.scaffold_unk_id = self._get_config_value("controls_config.scaffold_unk_id", scaffold_tokenizer.unk_token_id)
        self.use_token_map_memory = self._get_config_value("controls_config.use_token_map_memory", True)
        self.dynamic_cross_attn_mode = self._get_config_value("controls_config.dynamic_cross_attn_mode", None)

        # Generation settings
        self.max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
        self.memory_threshold = self._get_config_value("controls_config.memory_threshold", 0.85)
        self.generation_callbacks: Dict[str, List[Callable]] = {
            "pre_generate": [],
            "post_generate": []
        }

        # Validate and initialize curiosity state
        self._validate_curiosity_state()

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Validate required configuration sections
            required_sections = ["controls_config", "curiosity_config", "training_config"]
            for section in required_sections:
                if not self._config_manager.validate_section(section):
                    raise ValueError(f"Missing required configuration section: {section}")

            # Validate specific configuration values
            self._validate_config_values()

        except Exception as e:
            self._log_error(
                Exception(f"Configuration initialization failed: {str(e)}"),
                "config_initialization",
                traceback.format_exc()
            )
            raise

    def _validate_config_values(self) -> None:
        """Validate specific configuration values."""
        try:
            # Memory threshold validation
            memory_threshold = self._get_config_value("controls_config.memory_threshold", 0.85)
            if not 0.0 <= memory_threshold <= 1.0:
                raise ValueError(f"Invalid memory_threshold: {memory_threshold}")

            # Generation retries validation
            max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
            if not isinstance(max_retries, int) or max_retries < 1:
                raise ValueError(f"Invalid max_generation_retries: {max_retries}")

            # Temperature validation
            base_temperature = self._get_config_value("controls_config.base_temperature", 0.7)
            if not 0.0 <= base_temperature <= 2.0:
                raise ValueError(f"Invalid base_temperature: {base_temperature}")

            # Top-k validation
            top_k = self._get_config_value("curiosity_config.top_k", 30)
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"Invalid top_k: {top_k}")

        except Exception as e:
            self._log_error(
                Exception(f"Configuration validation failed: {str(e)}"),
                "config_validation",
                traceback.format_exc()
            )
            raise

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get a configuration value with validation."""
        try:
            return self._config_manager.get(key, default)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get config value for {key}: {str(e)}"),
                "config_access",
                traceback.format_exc()
            )
            return default

    def _update_config(self, key: str, value: Any) -> bool:
        """Update a configuration value with validation."""
        try:
            return self._config_manager.update(key, value)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to update config value for {key}: {str(e)}"),
                "config_update",
                traceback.format_exc()
            )
            return False

    def _load_config_sections(self) -> None:
        """Load configuration sections from config manager."""
        self.controls_config = self._config_manager.get_section("controls_config")
        self.curiosity_config = self._config_manager.get_section("curiosity_config")
        self.training_config = self._config_manager.get_section("training_config")
        
    def _log_initialization(self) -> None:
        """Log GenerationManager initialization with config values."""
        self._log_event(
            event_type="generation_manager_init",
            message="GenerationManager initialized successfully",
            level="info",
            additional_info={
                "controls_config": {k: self.controls_config.get(k) for k in [
                    "memory_threshold", "max_generation_retries", "scaffold_unk_id", 
                    "use_token_map_memory", "dynamic_cross_attn_mode", "conversation_history_maxlen",
                    "memory_decay_rate", "enable_repetition_check", "enable_confidence_tracking",
                    "enable_error_listening", "dream_memory_weight", "base_temperature"
                ]},
                "curiosity_config": {k: self.curiosity_config.get(k) for k in [
                    "max_new_tokens", "top_k", "weight_ignorance", "weight_novelty"
                ]}
            }
        )

    def register_callback(self, stage: str, callback: Callable) -> None:
        """Register a callback for generation stages."""
        if stage in self.generation_callbacks:
            self.generation_callbacks[stage].append(callback)
        else:
            self._log_error(
                Exception(f"Invalid callback stage: {stage}"),
                "register_callback"
            )

    def _log_event(self, event_type: str, message: str, level: str = "info", additional_info: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info={
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "device": str(self.device),
                **(additional_info or {})
            }
        )

    def _log_error(self, error: Exception, context: str, stack_trace: Optional[str] = None) -> None:
        """Log an error with context and stack trace."""
        self.logger.log_error(
            error_msg=str(error),
            error_type=f"generation_{context}_error",
            stack_trace=stack_trace or traceback.format_exc(),
            additional_info={
                "context": context,
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "device": str(self.device),
                "timestamp": time.time()
            }
        )

    def _log_memory_health(self, memory_ratio: float, current_memory: int, total_memory: int) -> None:
        """Log memory health status with standardized metadata."""
        self.logger.log_memory_health(
            model_size=current_memory,
            health_status="healthy" if memory_ratio <= self.memory_threshold else "unhealthy",
            device=self.device,
            additional_info={
                "memory_ratio": memory_ratio,
                "current_memory": current_memory,
                "total_memory": total_memory,
                "threshold": self.memory_threshold,
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "timestamp": time.time()
            }
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

            # Log memory health using standardized method
            self._log_memory_health(memory_ratio, current_memory, total_memory)

            if memory_ratio > self.memory_threshold:
                torch.cuda.empty_cache()
                self._log_event(
                    "memory_cleanup",
                    {
                        "reason": "threshold_exceeded",
                        "memory_ratio": memory_ratio,
                        "threshold": self.memory_threshold
                    }
                )
                return False
            return True
        except Exception as e:
            self._log_error(
                Exception(f"Memory health check failed: {str(e)}"),
                "check_memory_health",
                traceback.format_exc()
            )
            return False

    def _handle_error_prompt(self, error_msg: str) -> str:
        """Generate a response to a system error."""
        try:
            temp_history = self.state.history
            self.state.history = ConversationHistory(
                maxlen=self.controls_config.get("conversation_history_maxlen", 10)
            )
            response = self.generate(
                f"System error detected: {error_msg} What happened?",
                max_new_tokens=self.curiosity_config.get("max_new_tokens", 60),
                temperature=self.controls_config.get("base_temperature", 0.7) + 0.2,
                top_k=self.curiosity_config.get("top_k", 50),
                do_sample=True
            )
            self._log_event(
                "error_prompt_handled",
                {
                    "prompt": f"System error detected: {error_msg} What happened?",
                    "response": response,
                    "is_error_prompt": True,
                    "confidence_score": 0.5
                }
            )
            self.state.history = temp_history
            return response
        except Exception as e:
            self._log_error(
                Exception(f"Error prompt handling failed: {str(e)}"),
                "handle_error_prompt",
                traceback.format_exc()
            )
            return "An error occurred while handling the error prompt"

    def has_repetition(self, output_ids: torch.Tensor, n: int = 3) -> bool:
        """Check for repetition in generated output."""
        try:
            ids = output_ids.tolist()
            special_ids = {
                self.base_tokenizer.pad_token_id,
                self.base_tokenizer.eos_token_id,
                self.base_tokenizer.bos_token_id,
                self.base_tokenizer.unk_token_id
            }
            filtered = [i for i in ids if i not in special_ids]
            for i in range(len(filtered) - 2 * n):
                if filtered[i:i + n] == filtered[i + n:i + 2 * n]:
                    return True
            return False
        except Exception as e:
            self._log_error(
                Exception(f"Repetition check failed: {str(e)}"),
                "has_repetition",
                traceback.format_exc()
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
                Exception(f"Tokenization failed: {str(e)}"),
                "tokenize_and_map",
                traceback.format_exc()
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
                Exception(f"Scaffold hidden states failed: {str(e)}"),
                "get_scaffold_hidden_states",
                traceback.format_exc()
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
                    "scaffold_cache_cleared",
                    {}
                )
            except Exception as e:
                self._log_error(
                    Exception(f"Failed to clear scaffold cache: {str(e)}"),
                    "clear_scaffold_cache",
                    traceback.format_exc()
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
                Exception(f"Token map update failed: {str(e)}"),
                "update_token_map_memory",
                traceback.format_exc()
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
                Exception(f"Training preparation failed: {str(e)}"),
                "prepare_for_training",
                traceback.format_exc()
            )
            raise

    def _prepare_generation_params(self, max_new_tokens: int, scaffold_weight: Optional[float], **kwargs) -> Dict[str, Any]:
        """Prepare and validate generation parameters."""
        try:
            return {
                "max_new_tokens": max_new_tokens or self._get_config_value("curiosity_config.max_new_tokens", 8),
                "scaffold_weight": scaffold_weight,
                "temperature": kwargs.get("temperature", self._get_config_value("controls_config.base_temperature", 0.7)),
                "top_k": kwargs.get("top_k", self._get_config_value("curiosity_config.top_k", 30)),
                "do_sample": kwargs.get("do_sample", False),
                "prompt_count": 1 if isinstance(kwargs.get("prompts"), str) else len(kwargs.get("prompts", [])),
            }
        except Exception as e:
            self._log_error(
                Exception(f"Failed to prepare generation parameters: {str(e)}"),
                "prepare_generation_params",
                traceback.format_exc()
            )
            raise

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
                Exception(f"Failed to compute dynamic factor: {str(e)}"),
                "compute_dynamic_factor",
                traceback.format_exc()
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
                    Exception(f"Dream memory preparation failed: {str(e)}"),
                    "prepare_dream_memory",
                    traceback.format_exc()
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
                "repetition_detected",
                {
                    "original_text": original_text,
                    "truncated_at": j + 3
                }
            )
        return seq_ids

    def _update_curiosity(self, text: str, confidence: float) -> None:
        """Update curiosity state with generated text and confidence."""
        try:
            if not self.curiosity_manager:
                return
                
            # Tokenize and get embeddings for the generated text
            inputs = self.base_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                text_embedding = outputs.hidden_states[-1].mean(dim=1)
            
            # Update curiosity with the new text and confidence
            self.curiosity_manager.update_curiosity(
                text=text,
                confidence=confidence,
                embedding=text_embedding
            )
            
            # Log curiosity update
            self._log_event(
                "curiosity_update",
                {
                    "text": text[:200],  # Log first 200 chars
                    "confidence": confidence,
                    "embedding_shape": text_embedding.shape,
                    "curiosity_pressure": self.curiosity_manager.get_pressure()
                }
            )
            
        except Exception as e:
            self._log_error(
                Exception(f"curiosity_update_error: {str(e)}"),
                "curiosity_update_error",
                {
                    "text": text[:200],
                    "confidence": confidence
                }
            )

    @synchronized()
    def calculate_confidence_score(self, logits: torch.Tensor, generated_ids: List[int]) -> float:
        """Calculate confidence score for generated output."""
        try:
            processor = LogitsProcessor(logits)
            return processor.calculate_confidence(generated_ids)
        except Exception as e:
            self._log_error(
                Exception(f"Confidence score calculation failed: {str(e)}"),
                "calculate_confidence_score",
                traceback.format_exc()
            )
            return 0.5

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        conversation_id: Optional[str] = None,
        state_hash: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Generate text using the base model and scaffolds with proper device handling."""
        try:
            # Validate device state
            self._validate_device_state()
            
            # Store state metadata for logging
            self._store_state_metadata(conversation_id, state_hash)
            
            # Update temperament before generation
            if hasattr(self, 'temperament_adjuster'):
                self.temperament_adjuster.update_temperament(self.curiosity_manager)
            
            # Sync temperament score with state
            if hasattr(self.state, 'temperament_score'):
                self.temperament.score = self.state.temperament_score
            
            # Get curiosity pressure for temperature adjustment
            curiosity_pressure = None
            if self.curiosity_manager:
                curiosity_pressure = self.curiosity_manager.get_pressure()
            
            # Get query embedding for curiosity computation
            query_embedding = None
            if self.curiosity_manager:
                inputs = self.base_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.base_model(**inputs, output_hidden_states=True)
                    query_embedding = outputs.hidden_states[-1].mean(dim=1)
            
            # Compute curiosity score
            base_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
            scaf_conf = self.state.scaffold_confidence_history[-1] if self.state.scaffold_confidence_history else 0.5
            curiosity_score = self.compute_curiosity(base_conf, scaf_conf, query_embedding)
            
            # Adjust temperature based on current temperament, curiosity pressure, and curiosity score
            adjusted_temperature = self.temperament.adjust_parameter(
                base_value=temperature,
                parameter_type="temperature",
                curiosity_pressure=curiosity_pressure
            )
            
            # Further adjust temperature based on curiosity score
            if curiosity_score > 0.7:  # High curiosity
                adjusted_temperature = min(1.5, adjusted_temperature * 1.2)
            elif curiosity_score < 0.3:  # Low curiosity
                adjusted_temperature = max(0.5, adjusted_temperature * 0.8)
            
            # Log generation start with temperament and curiosity info
            self._log_event(
                "generation_start",
                {
                    "prompt": prompt,
                    "max_length": max_length,
                    "base_temperature": temperature,
                    "adjusted_temperature": adjusted_temperature,
                    "top_p": top_p,
                    "num_return_sequences": num_return_sequences,
                    "device": str(self.device),
                    "temperament_score": self.temperament.score,
                    "curiosity_pressure": curiosity_pressure,
                    "curiosity_score": curiosity_score,
                    "mood_label": self.temperament.mood_label
                }
            )
            
            # Check memory health before generation
            if not self.check_memory_health():
                self._log_warning("Memory health check failed before generation")
                return self.error_manager.handle_generation_error(
                    Exception("Memory health check failed"),
                    prompt
                )

            # Tokenize input
            inputs = self.base_tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with base model using adjusted temperature
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=adjusted_temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decode and return generated sequences
            generated_sequences = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update curiosity state with generated text
            if self.curiosity_manager:
                for sequence in generated_sequences:
                    self._update_curiosity(sequence, base_conf)
            
            # Log successful generation with final parameters
            self._log_event(
                "generation_complete",
                {
                    "generated_sequences": generated_sequences,
                    "device": str(self.device),
                    "final_temperature": adjusted_temperature,
                    "final_temperament_score": self.temperament.score,
                    "final_curiosity_pressure": curiosity_pressure,
                    "final_curiosity_score": curiosity_score
                }
            )
            
            return generated_sequences
            
        except torch.cuda.OutOfMemoryError as oom:
            self.error_manager.context.logger.log_error(
                error_msg="CUDA out of memory",
                error_type="generation_oom",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "prompt": prompt[:200],
                    "memory_stats": {
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "max_allocated": torch.cuda.max_memory_allocated(),
                    } if torch.cuda.is_available() else None,
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash
                }
            )
            return self.error_manager.handle_generation_error(oom, prompt)
            
        except Exception as e:
            self._log_error(
                Exception(f"generation_error: {str(e)}"),
                "generation_error",
                {
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "adjusted_temperature": adjusted_temperature if 'adjusted_temperature' in locals() else None,
                    "top_p": top_p,
                    "device": str(self.device),
                    "temperament_score": self.temperament.score if hasattr(self, 'temperament') else None,
                    "curiosity_pressure": curiosity_pressure if 'curiosity_pressure' in locals() else None,
                    "curiosity_score": curiosity_score if 'curiosity_score' in locals() else None
                }
            )
            return self.error_manager.handle_generation_error(e, prompt)

    def _validate_device_state(self) -> None:
        """Validate that all models are on the correct device."""
        try:
            # Check base model
            if self.base_model.device != self.device:
                self._log_warning(f"Base model device mismatch: {self.base_model.device} != {self.device}")
                self.base_model = self.base_model.to(self.device)
            
            # Check scaffolds
            for i, scaffold in enumerate(self.scaffolds):
                if scaffold.device != self.device:
                    self._log_warning(f"Scaffold {i} device mismatch: {scaffold.device} != {self.device}")
                    self.scaffolds[i] = scaffold.to(self.device)
                    
        except Exception as e:
            self._log_error(
                Exception(f"device_validation_error: {str(e)}"),
                "device_validation",
                traceback.format_exc()
            )
            raise

    def _store_state_metadata(self, conversation_id: Optional[str], state_hash: Optional[str]) -> None:
        """Store state metadata for logging."""
        self.conversation_id = conversation_id
        self.state_hash = state_hash

    def _validate_curiosity_state(self) -> None:
        """Validate and initialize curiosity state if needed."""
        try:
            if not hasattr(self.state, 'curiosity'):
                self.state.curiosity = CuriosityState(
                    pressure=0.0,
                    unanswered_questions=deque(maxlen=self.curiosity_config.get("queue_maxlen", 10)),
                    novelty_scores=deque(maxlen=self.curiosity_config.get("metrics_maxlen", 1000))
                )
                self._log_event(
                    "curiosity_state_init",
                    {
                        "reason": "missing_curiosity_state",
                        "queue_maxlen": self.curiosity_config.get("queue_maxlen", 10),
                        "metrics_maxlen": self.curiosity_config.get("metrics_maxlen", 1000)
                    }
                )
            
            # Validate curiosity manager
            if self.curiosity_manager is None:
                self._log_warning(
                    "curiosity_manager_missing",
                    "Curiosity manager not provided, curiosity features will be disabled"
                )
            else:
                # Ensure curiosity manager is using the same state
                self.curiosity_manager.set_state(self.state)
                self._log_event(
                    "curiosity_manager_sync",
                    {
                        "state_hash": self.state.state_hash,
                        "conversation_id": self.state.history.conversation_id
                    }
                )
                
        except Exception as e:
            self._log_error(
                Exception(f"Failed to validate curiosity state: {str(e)}"),
                "curiosity_validation",
                traceback.format_exc()
            )
            raise

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        query_embedding: torch.Tensor,
    ) -> float:
        """Compute curiosity score based on confidence and embeddings."""
        try:
            if not self.curiosity_manager:
                return 0.5

            # Get memory embeddings with memory limits
            memory_embeddings = self._get_valid_memory_embeddings(self.state)
            
            # Compute curiosity score
            score = self.curiosity_manager.compute_curiosity(
                base_conf=base_conf,
                scaf_conf=scaf_conf,
                state=self.state,
                query_embedding=query_embedding,
                device=self.device
            )
            
            return score
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to compute curiosity: {str(e)}"),
                "compute_curiosity",
                traceback.format_exc()
            )
            return 0.5

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """Get valid memory embeddings while respecting memory limits."""
        try:
            memory_embeddings = []
            hidden_size = self._config_manager.get("hidden_size")
            total_size = 0.0
            
            if not hasattr(state, 'dream_memory') or not state.dream_memory:
                return memory_embeddings
                
            for entry in state.dream_memory:
                tensor = entry["tensor"]
                # Calculate memory size in MB
                memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                
                # Check memory limit
                if total_size + memory_size > self._config_manager.get("max_memory_mb", 512.0):
                    self._log_event("memory_limit_reached", {
                        "total_size_mb": total_size,
                        "skipped_size_mb": memory_size,
                        "max_memory_mb": self._config_manager.get("max_memory_mb", 512.0)
                    })
                    break
                    
                # Validate tensor shape
                if tensor.shape[-1] == hidden_size:
                    try:
                        # Validate and move tensor to correct device
                        tensor = self._validate_device(tensor, f"dream_memory_tensor_{len(memory_embeddings)}")
                        memory_embeddings.append(tensor)
                        total_size += memory_size
                    except Exception as e:
                        self._log_warning("tensor_validation_failed", {
                            "error": str(e),
                            "tensor_shape": list(tensor.shape)
                        })
                        continue
                else:
                    self._log_warning("invalid_tensor_shape", {
                        "expected_size": hidden_size,
                        "actual_size": tensor.shape[-1]
                    })
                    
            return memory_embeddings
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get valid memory embeddings: {str(e)}"),
                "get_valid_memory_embeddings",
                traceback.format_exc()
            )
            return []

def calculate_confidence(logits: torch.Tensor, generated_ids: torch.Tensor) -> float:
    """Calculate confidence score for generated tokens."""
    try:
        # Get probabilities for generated tokens
        probs = torch.softmax(logits, dim=-1)
        token_probs = torch.gather(probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1)
        
        # Calculate average confidence
        confidence = token_probs.mean().item()
        return max(0.0, min(1.0, confidence))
    except Exception as e:
        logger.record({
            "error": f"Confidence calculation failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return 0.5

def detect_repetitions(token_ids: List[int], special_ids: Set[int], min_rep_length: int = 3) -> Optional[Tuple[int, int]]:
    """Detect repeating token sequences."""
    try:
        filtered = [i for i in token_ids if i not in special_ids]
        for i in range(len(filtered) - 2 * min_rep_length + 1):
            window = filtered[i:i + min_rep_length]
            next_window = filtered[i + min_rep_length:i + 2 * min_rep_length]
            if window == next_window:
                return (i, i + min_rep_length)
        return None
    except Exception as e:
        logger.record({
            "error": f"Repetition detection failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return None

def adjust_temperature(
    base_temp: float,
    temperament_score: float,
    mood_influence: float = 0.3,
    min_temp: float = 0.5,
    max_temp: float = 1.5,
    curiosity_pressure: Optional[float] = None
) -> float:
    """Adjust temperature based on temperament and curiosity."""
    try:
        # Clamp input values
        base_temp = max(min_temp, min(max_temp, base_temp))
        temperament_score = max(-1.0, min(1.0, temperament_score))
        mood_influence = max(0.0, min(1.0, mood_influence))
        
        # Calculate temperature adjustment
        temp_adjustment = mood_influence * 0.3 * temperament_score
        if curiosity_pressure is not None:
            curiosity_pressure = max(0.0, min(1.0, curiosity_pressure))
            temp_adjustment += curiosity_pressure * 0.1
        
        # Apply adjustment
        adjusted_temp = max(min_temp, min(max_temp, base_temp + temp_adjustment))
        return adjusted_temp
    except Exception as e:
        logger.record({
            "error": f"Temperature adjustment failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return base_temp
