import torch
import json
import os
from collections import deque, defaultdict
from threading import Lock
import time
import traceback
from typing import Optional, Dict, List, Tuple, Any, Union
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_utils import memory_usage, safe_divide
from sovl_config import ConfigManager
from sovl_hardware import HardwareManager  # New import
import gc

"""
Manages memory-related operations, using HardwareManager for hardware access.
"""

class MemoryManager:
    """
    Manages memory for the SOVL system, handling token maps, dream memory,
    conversation history, scaffold context, and GPU memory health.
    """
    _CONFIG_RANGES = {
        "memory_threshold": (0.5, 0.95),
        "memory_decay_rate": (0.8, 1.0),
        "dream_memory_decay": (0.8, 1.0),
        "dream_prune_threshold": (0.0, 0.5),
        "token_map_weight_cap": (1.0, 5.0),
        "dream_memory_weight": (0.0, 0.5)
    }
    _MAXLEN_RANGES = {
        "dream_memory_maxlen": (5, 50),
        "conversation_history_maxlen": (5, 50),
        "confidence_history_maxlen": (3, 1000),
        "temperament_history_maxlen": (3, 1000)
    }
    _CLEANUP_COOLDOWN = 60.0
    _MEM_USAGE_HISTORY_MAXLEN = 10

    def __init__(self, config_manager: ConfigManager, device: torch.device, logger: Logger):
        """Initialize MemoryManager with configuration and dependencies."""
        self._config_manager = config_manager
        self._device = device
        self._logger = logger
        self.hardware = HardwareManager(config_manager, logger)  # Initialize HardwareManager
        self._memory_lock = Lock()
        self._last_cleanup_time = 0.0
        self._mem_usage_history = deque(maxlen=self._MEM_USAGE_HISTORY_MAXLEN)
        self._state = None
        self._conversation_history = None
        self._token_map = {}
        self._scaffold_context = {}
        self._dream_memory = deque(maxlen=10)
        self._memory_health = {
            "gpu_usage": 0.0,
            "cpu_usage": 0.0,
            "last_check": 0.0
        }
        self._cleanup_cooldown = self._CLEANUP_COOLDOWN

        # Initialize configuration
        self._initialize_config()

        # Initialize memory thresholds and batch sizes
        self.dynamic_threshold_base = self._config_manager.get("memory_config.memory_threshold", 0.85)
        self.memory_decay_rate = self._config_manager.get("memory_config.memory_decay_rate", 0.95)
        self.max_batch_size = self._config_manager.get("memory_config.max_batch_size", 32)
        self.initial_batch_size = self._config_manager.get("memory_config.initial_batch_size", 8)
        self.batch_size = self.initial_batch_size
        self.memory_threshold = self.dynamic_threshold_base

        # Log initialization
        self._logger.record_event(
            event_type="memory_manager_initialized",
            message="Memory manager initialized",
            level="info",
            device=str(self._device),
            config_device=str(config_manager.device)
        )

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        with self._memory_lock:
            try:
                # Get memory config section
                memory_config = self._config_manager.get_section("memory_config", {})
                
                # Validate and set memory-specific config values
                self.memory_threshold = self._validate_config_value(
                    "memory_threshold",
                    memory_config.get("memory_threshold", 0.85),
                    (0.5, 0.95)
                )
                
                self.memory_decay_rate = self._validate_config_value(
                    "memory_decay_rate",
                    memory_config.get("memory_decay_rate", 0.95),
                    (0.8, 1.0)
                )
                
                self.use_scaffold_memory = self._config_manager.get("memory_config.use_scaffold_memory", True)
                self.use_token_map_memory = self._config_manager.get("memory_config.use_token_map_memory", True)
                self.scaffold_weight = self._validate_config_value(
                    "scaffold_weight",
                    memory_config.get("scaffold_weight", 1.0),
                    (0.0, 1.0)
                )
                
                # Validate and set dream memory config values
                self.dream_memory_maxlen = self._validate_config_value(
                    "dream_memory_maxlen",
                    memory_config.get("dream_memory_maxlen", 10),
                    (5, 50),
                    is_int=True
                )
                
                self.dream_memory_decay = self._validate_config_value(
                    "dream_memory_decay",
                    memory_config.get("dream_memory_decay", 0.95),
                    (0.8, 1.0)
                )
                
                self.dream_prune_threshold = self._validate_config_value(
                    "dream_prune_threshold",
                    memory_config.get("dream_prune_threshold", 0.1),
                    (0.0, 0.5)
                )
                
                self.dream_memory_weight = self._validate_config_value(
                    "dream_memory_weight",
                    memory_config.get("dream_memory_weight", 0.1),
                    (0.0, 0.5)
                )
                
                # Update config with validated values
                memory_config.update({
                    "memory_threshold": self.memory_threshold,
                    "memory_decay_rate": self.memory_decay_rate,
                    "dream_memory_maxlen": self.dream_memory_maxlen,
                    "dream_memory_decay": self.dream_memory_decay,
                    "dream_prune_threshold": self.dream_prune_threshold,
                    "dream_memory_weight": self.dream_memory_weight,
                    "scaffold_weight": self.scaffold_weight
                })
                
                self._config_manager.update_section("memory_config", memory_config)
                
                # Log successful initialization
                self._log_event(
                    "memory_config_initialized",
                    "Memory configuration initialized successfully",
                    level="info"
                )
                
            except Exception as e:
                self._log_error(
                    f"Failed to initialize memory config: {str(e)}",
                    error_type="config_error",
                    stack_trace=traceback.format_exc(),
                    context="config_initialization"
                )
                raise

    def _validate_config_value(self, key: str, value: Any, valid_range: Tuple[float, float], is_int: bool = False) -> Union[float, int]:
        """Validate a configuration value against a range."""
        try:
            if is_int:
                if not isinstance(value, int):
                    raise ValueError(f"Config {key} must be an integer")
                min_val, max_val = valid_range
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Config {key}={value} outside valid range [{min_val}, {max_val}]")
                return value
            else:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Config {key} must be a number")
                min_val, max_val = valid_range
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Config {key}={value} outside valid range [{min_val}, {max_val}]")
                return float(value)
        except Exception as e:
            self._log_error(
                f"Config validation failed for {key}: {str(e)}",
                error_type="config_validation_error",
                context="config_validation"
            )
            raise

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log an event with standardized format."""
        # Prepare data before acquiring any locks
        additional_info = {
            "conversation_id": self._conversation_history.conversation_id if self._conversation_history else None,
            "state_hash": self._state.state_hash() if self._state else None,
            **kwargs
        }
        
        # Log without holding any locks
        try:
            self._logger.record_event(
                event_type=event_type,
                message=message,
                level=level,
                additional_info=additional_info
            )
        except Exception as e:
            print(f"Failed to log event: {str(e)}")

    def _log_error(self, error_msg: str, error_type: str, stack_trace: Optional[str] = None, context: Optional[Dict] = None):
        """Log an error with consistent formatting and context."""
        try:
            # Ensure context is a dictionary
            if context is None:
                context = {}
            
            # Add memory-specific context
            memory_stats = self.hardware.get_detailed_memory_stats()
            context.update(memory_stats)
            
            # Add system state context
            context.update({
                "memory_threshold": self.memory_threshold,
                "batch_size": self.batch_size,
                "max_history_size": self.max_history_size,
                "memory_usage_history": len(self._mem_usage_history)
            })
            
            # Log error with consistent format
            self._logger.log_error(
                error_msg=error_msg,
                error_type=error_type,
                stack_trace=stack_trace,
                additional_info=context
            )
        except Exception as e:
            # If logging fails, at least print the error
            print(f"Failed to log error: {str(e)}")
            print(f"Original error: {error_msg}")

    def _log_warning(self, message: str, context: str = "memory", **kwargs) -> None:
        """Log a warning with standardized format."""
        try:
            self._logger.record_event(
                event_type=f"{context}_warning",
                message=message,
                level="warning",
                additional_info={
                    "conversation_id": self._conversation_history.conversation_id if self._conversation_history else None,
                    "state_hash": self._state.state_hash() if self._state else None,
                    **kwargs
                }
            )
        except Exception as e:
            print(f"Failed to log warning: {str(e)}")

    def initialize_token_map(self, base_tokenizer, scaffold_tokenizer) -> None:
        """Initialize token map for mapping base model tokens to scaffold tokens."""
        with self._state.lock:
            try:
                # Create initial token map
                token_map = defaultdict(lambda: {'ids': [scaffold_tokenizer.unk_token_id], 'weight': 1.0})
                
                # Populate token map from tokenizer vocabularies
                for base_token, base_id in base_tokenizer.get_vocab().items():
                    normalized = base_token.replace("Ġ", "").replace("##", "")
                    scaffold_ids = scaffold_tokenizer.encode(
                        normalized,
                        add_special_tokens=False,
                        max_length=3,
                        truncation=True
                    ) or [scaffold_tokenizer.unk_token_id]
                    token_map[base_id] = {'ids': scaffold_ids, 'weight': 1.0}

                # Update state with new token map
                self._state.update_token_map(dict(token_map))
                
                # Log initialization
                self._log_event(
                    "token_map_initialized",
                    message="Token map initialized with base and scaffold tokenizers",
                    level="info",
                    base_vocab_size=len(base_tokenizer.get_vocab()),
                    state_hash=self._state.state_hash()
                )
                
            except Exception as e:
                self._log_error(
                    f"Failed to initialize token map: {str(e)}",
                    error_type="token_map_error",
                    stack_trace=traceback.format_exc(),
                    context={
                        "base_vocab_size": len(base_tokenizer.get_vocab()),
                        "state_hash": self._state.state_hash()
                    }
                )
                raise

    def sync_token_map(self) -> None:
        """Synchronize token map from state to local memory."""
        if self._state is None:
            raise ValueError("State not set. Call set_state first.")
        
        with self._memory_lock:
            try:
                # Get current token map from state
                with self._state.lock:
                    token_map = self._state.get_token_map()
                
                # Validate token map structure
                if not isinstance(token_map, dict):
                    raise ValueError("Token map must be a dictionary")
                    
                for token_id, mapping in token_map.items():
                    if not isinstance(mapping, dict):
                        raise ValueError(f"Invalid mapping for token {token_id}")
                    if 'ids' not in mapping or 'weight' not in mapping:
                        raise ValueError(f"Missing required fields in mapping for token {token_id}")
                    if not isinstance(mapping['ids'], list):
                        raise ValueError(f"Invalid ids type for token {token_id}")
                    if not isinstance(mapping['weight'], (int, float)):
                        raise ValueError(f"Invalid weight type for token {token_id}")
                
                # Update local token map
                self._token_map = defaultdict(lambda: None, token_map)
                
                # Log synchronization
                self._log_event(
                    "token_map_synced",
                    message="Token map synchronized from state",
                    level="info",
                    token_map_size=len(token_map),
                    state_hash=self._state.state_hash()
                )
                
            except Exception as e:
                self._log_error(
                    error_msg=f"Failed to sync token map: {str(e)}",
                    error_type="token_map_error",
                    stack_trace=traceback.format_exc(),
                    context={
                        "state_hash": self._state.state_hash() if self._state else None
                    }
                )
                raise

    def set_state(self, state: SOVLState) -> None:
        """Set the state object for memory synchronization."""
        with self._memory_lock:
            self._state = state
            self._dream_memory = state.dream_memory
            self._conversation_history = state.conversation_history
            self._log_event(
                "memory_state_set",
                message="State set for memory synchronization",
                level="info",
                dream_memory_len=len(self._dream_memory)
            )

    def set_hidden_size(self, hidden_size: int) -> None:
        """Set the hidden size for validating dream memory tensors."""
        self._hidden_size = hidden_size
        self._log_event(
            "hidden_size_set",
            message="Hidden size set for dream memory validation",
            level="info",
            hidden_size=hidden_size
        )

    def update_token_map_memory(self, prompt: str, confidence: float, tokenizer) -> None:
        """Update token map weights based on prompt and confidence."""
        if not self.use_token_map_memory:
            return

        try:
            with self._state.lock:
                # Get current token map from state
                token_map = self._state.get_token_map()
                
                # Update weights based on prompt
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                for token_id in input_ids:
                    if token_id in token_map:
                        current_weight = token_map[token_id]['weight']
                        new_weight = current_weight * self.memory_decay_rate + confidence * (1 - self.memory_decay_rate)
                        token_map[token_id]['weight'] = min(max(new_weight, 0.1), self.token_map_weight_cap)

                # Update state with modified token map
                self._state.update_token_map(token_map)
                
                self._log_event(
                    "token_map_updated",
                    message="Token map weights updated based on prompt",
                    level="info",
                    prompt_length=len(prompt),
                    confidence=confidence,
                    state_hash=self._state.state_hash()
                )
        except Exception as e:
            self._log_error(
                f"Failed to update token map: {str(e)}",
                error_type="token_map_error",
                stack_trace=traceback.format_exc(),
                context={
                    "prompt_length": len(prompt),
                    "confidence": confidence,
                    "state_hash": self._state.state_hash()
                }
            )

    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Retrieve GPU memory statistics."""
        try:
            stats = self.hardware.get_memory_stats()
            return {
                "allocated": stats["allocated_mb"] / 1024,  # Convert MB to GB
                "reserved": stats["reserved_mb"] / 1024,   # Convert MB to GB
                "total": stats["total_memory_mb"] / 1024,  # Convert MB to GB
                "available": stats["available_mb"] / 1024  # Convert MB to GB
            }
        except Exception as e:
            self._log_event(
                "memory_stats_failed",
                message="Failed to retrieve memory stats",
                level="warning"
            )
            return None

    def check_memory_health(self, model_size: int, trainer: Optional[Any] = None):
        """Autonomically reduce GPU memory usage if approaching capacity."""
        try:
            stats = self.hardware.get_memory_stats()
            current_memory = stats["allocated_mb"] * 1024 * 1024  # Convert MB to bytes
            total_memory = stats["total_memory_mb"] * 1024 * 1024  # Convert MB to bytes
            memory_ratio = current_memory / total_memory if total_memory > 0 else 0.0
            
            # Log memory usage
            self._log_event(
                event_type="memory_health_check",
                message="Memory health check performed",
                level="info",
                memory_ratio=memory_ratio,
                model_size=model_size
            )
            
            # Update memory usage history
            with self._memory_lock:
                self._mem_usage_history.append(memory_ratio)
                if len(self._mem_usage_history) > self.max_history_size:
                    self._mem_usage_history.pop(0)
                
                # Calculate dynamic threshold
                if len(self._mem_usage_history) > 0:
                    avg_usage = sum(self._mem_usage_history) / len(self._mem_usage_history)
                    self.memory_threshold = min(
                        self.dynamic_threshold_base,
                        avg_usage * self.memory_decay_rate
                    )
            
            # Check if memory usage exceeds threshold
            if memory_ratio > self.memory_threshold:
                # Reduce batch size if trainer is provided
                if trainer is not None:
                    new_batch_size = max(
                        self.min_batch_size,
                        int(self.batch_size * 0.5)
                    )
                    if new_batch_size != self.batch_size:
                        self.batch_size = new_batch_size
                        trainer.train_batch_size = new_batch_size
                        self._log_event(
                            event_type="memory_health_action",
                            message="Batch size reduced due to high memory usage",
                            level="warning",
                            new_batch_size=new_batch_size,
                            memory_ratio=memory_ratio,
                            threshold=self.memory_threshold
                        )
                
                # Clear cache if memory usage is very high
                if memory_ratio > 0.9:
                    self.hardware.clear_memory_cache()
                    self._log_event(
                        event_type="memory_health_action",
                        message="Memory cache cleared due to very high memory usage",
                        level="warning",
                        memory_ratio=memory_ratio
                    )
                
                return False
            
            return True
        except Exception as e:
            # Log detailed error information
            self._log_error(
                error_msg=f"Memory health check failed: {str(e)}",
                error_type="memory_error",
                stack_trace=traceback.format_exc(),
                context={
                    "model_size": model_size,
                    "trainer_present": trainer is not None,
                    "device": str(self._device)
                }
            )
            return False

    def get_batch_size(self):
        """Get current batch size."""
        return self.batch_size

    def reset_batch_size(self):
        """Reset batch size to initial value."""
        self.batch_size = self.initial_batch_size

    def clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches safely."""
        with self._memory_lock:
            try:
                if self._scaffold_context is not None:
                    if isinstance(self._scaffold_context, torch.Tensor):
                        self._scaffold_context = self._scaffold_context.detach().cpu()
                    self._scaffold_context = None

                if self._state and self._state.last_prompt_embedding is not None:
                    if isinstance(self._state.last_prompt_embedding, torch.Tensor):
                        self._state.last_prompt_embedding = self._state.last_prompt_embedding.detach().cpu()
                    self._state.last_prompt_embedding = None

                if self._dream_memory:
                    self._dream_memory = deque(
                        [(entry["tensor"].detach().cpu(), entry["weight"], entry["metadata"])
                         for entry in self._dream_memory],
                        maxlen=self.dream_memory_maxlen
                    )
                    self._state.dream_memory = self._dream_memory

                self.hardware.clear_memory_cache()

                self._log_event(
                    "scaffold_cache_cleared",
                    message="Scaffold cache cleared",
                    level="info",
                    device=str(self._device)
                )
            except Exception as e:
                self._log_error(
                    error_msg=f"Failed to clear scaffold cache: {str(e)}",
                    error_type="cache_error",
                    stack_trace=traceback.format_exc()
                )

    def set_scaffold_context(self, scaffold_hidden_states: torch.Tensor) -> None:
        """Set temporary scaffold context for generation."""
        with self._memory_lock:
            # Ensure tensor is on correct device
            scaffold_hidden_states = scaffold_hidden_states.to(self._device)
            self._scaffold_context = scaffold_hidden_states.detach() if isinstance(scaffold_hidden_states, torch.Tensor) else scaffold_hidden_states
            self._log_event(
                "scaffold_context_set",
                message="Scaffold context set for generation",
                level="info",
                context_shape=list(scaffold_hidden_states.shape) if isinstance(scaffold_hidden_states, torch.Tensor) else None,
                device=str(scaffold_hidden_states.device)
            )

    def get_scaffold_context(self) -> Optional[torch.Tensor]:
        """Retrieve the current scaffold context."""
        with self._memory_lock:
            if self._scaffold_context is None:
                return None
            # Ensure tensor is on correct device
            return self._scaffold_context.to(self._device)

    def append_dream_memory(self, tensor: torch.Tensor, weight: float, metadata: Optional[Dict] = None) -> None:
        """Append a tensor to dream memory with associated weight and metadata."""
        if not self._hidden_size:
            self._log_error(
                error_msg="Hidden size not set for dream memory",
                error_type="config_error",
                stack_trace=None
            )
            return

        try:
            with self._memory_lock:
                if tensor.shape[-1] != self._hidden_size:
                    raise ValueError(f"Dream tensor shape {tensor.shape} mismatches hidden_size {self._hidden_size}")
                
                # Ensure tensor is on correct device
                tensor = tensor.to(self._device)
                
                entry = {
                    "tensor": tensor,
                    "weight": min(max(weight, 0.0), 1.0),
                    "metadata": metadata or {"timestamp": time.time()}
                }
                self._dream_memory.append(entry)
                self._state.dream_memory = self._dream_memory
                
                self._log_event(
                    "dream_memory_appended",
                    message="Tensor appended to dream memory",
                    level="info",
                    tensor_shape=list(tensor.shape),
                    weight=weight,
                    device=str(tensor.device),
                    dream_memory_len=len(self._dream_memory)
                )
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to append dream memory: {str(e)}",
                error_type="dream_memory_error",
                stack_trace=traceback.format_exc()
            )

    def prune_dream_memory(self) -> None:
        """Prune dream memory based on weight threshold and decay."""
        with self._memory_lock:
            try:
                if not self._dream_memory:
                    return

                new_memory = deque(maxlen=self.dream_memory_maxlen)
                for entry in self._dream_memory:
                    new_weight = entry["weight"] * self.dream_memory_decay
                    if new_weight >= self.dream_prune_threshold:
                        entry["weight"] = new_weight
                        new_memory.append(entry)
                original_len = len(self._dream_memory)
                self._dream_memory = new_memory
                self._state.dream_memory = self._dream_memory

                self._log_event(
                    "dream_memory_pruned",
                    message="Dream memory pruned based on weight threshold",
                    level="info",
                    original_length=original_len,
                    new_length=len(self._dream_memory)
                )
            except Exception as e:
                self._log_error(
                    error_msg=f"Failed to prune dream memory: {str(e)}",
                    error_type="dream_memory_error",
                    stack_trace=traceback.format_exc()
                )

    def get_dream_memory_tensors(self) -> Optional[torch.Tensor]:
        """Aggregate dream memory tensors for generation, weighted by importance."""
        if not self._dream_memory:
            return None

        try:
            with self._memory_lock:
                # Ensure all tensors are on correct device
                tensors = [entry["tensor"].to(self._device) for entry in self._dream_memory]
                weights = [entry["weight"] for entry in self._dream_memory]
                
                if not weights:
                    return None
                    
                dream_tensors = torch.stack(tensors)
                dream_weights = torch.tensor(weights, dtype=torch.float32, device=self._device)
                weight_sum = dream_weights.sum()
                
                if weight_sum == 0:
                    return None
                    
                aggregated = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / weight_sum
                
                self._log_event(
                    "dream_memory_aggregated",
                    message="Dream memory tensors aggregated",
                    level="info",
                    tensor_count=len(dream_tensors),
                    device=str(aggregated.device),
                    shapes=[list(t.shape) for t in tensors]
                )
                
                return aggregated
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to aggregate dream memory: {str(e)}",
                error_type="dream_memory_error",
                stack_trace=traceback.format_exc()
            )
            return None

    def save_state(self, path_prefix: str) -> None:
        """Save memory-related state to disk."""
        try:
            with self._memory_lock:
                with open(f"{path_prefix}_token_map.json", "w") as f:
                    json.dump({str(k): v for k, v in self._token_map.items()}, f)

                dream_state = [
                    {
                        "tensor": entry["tensor"].cpu().numpy().tolist(),
                        "weight": entry["weight"],
                        "metadata": entry["metadata"]
                    }
                    for entry in self._dream_memory
                ]
                with open(f"{path_prefix}_dream_memory.json", "w") as f:
                    json.dump(dream_state, f)

                self._log_event(
                    "memory_state_saved",
                    message="Memory state saved to disk",
                    level="info",
                    path_prefix=path_prefix,
                    token_map_size=len(self._token_map),
                    dream_memory_len=len(self._dream_memory)
                )
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to save memory state: {str(e)}",
                error_type="state_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def load_state(self, path_prefix: str) -> None:
        """Load memory-related state from disk."""
        try:
            with self._memory_lock:
                if os.path.exists(f"{path_prefix}_token_map.json"):
                    with open(f"{path_prefix}_token_map.json", "r") as f:
                        loaded_map = json.load(f)
                    self._token_map = defaultdict(lambda: [{'ids': [self._scaffold_unk_id], 'weight': 1.0}],
                                                 {int(k): v for k, v in loaded_map.items()})
                    self._log_event(
                        "token_map_loaded",
                        message="Token map loaded from disk",
                        level="info",
                        size=len(self._token_map)
                    )

                if os.path.exists(f"{path_prefix}_dream_memory.json"):
                    with open(f"{path_prefix}_dream_memory.json", "r") as f:
                        dream_state = json.load(f)
                    self._dream_memory = deque(maxlen=self.dream_memory_maxlen)
                    for item in dream_state:
                        tensor = torch.tensor(item["tensor"], device=self._device)
                        entry = {
                            "tensor": tensor,
                            "weight": item["weight"],
                            "metadata": item.get("metadata", {"timestamp": time.time()})
                        }
                        self._dream_memory.append(entry)
                    self._state.dream_memory = self._dream_memory
                    self._log_event(
                        "dream_memory_loaded",
                        message="Dream memory loaded from disk",
                        level="info",
                        length=len(self._dream_memory)
                    )

        except Exception as e:
            self._log_error(
                error_msg=f"Failed to load memory state: {str(e)}",
                error_type="state_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def toggle_memory_modes(self, mode: str) -> None:
        """Toggle scaffold and token map memory usage."""
        modes = {
            'scaffold_mem': (True, False),
            'token_mem': (False, True),
            'both_mem': (True, True),
            'no_mem': (False, False)
        }
        if mode not in modes:
            self._log_error(
                error_msg=f"Invalid memory mode: {mode}",
                error_type="config_error",
                stack_trace=None
            )
            raise ValueError(f"Invalid memory mode. Use: {', '.join(modes.keys())}")

        scaffold_mem, token_mem = modes[mode]
        updates = {
            "controls_config.use_scaffold_memory": scaffold_mem,
            "controls_config.use_token_map_memory": token_mem
        }
        self._config_manager.update_batch(updates)
        self.use_scaffold_memory = scaffold_mem
        self.use_token_map_memory = token_mem
        self._controls_config.update({"use_scaffold_memory": scaffold_mem, "use_token_map_memory": token_mem})
        self._log_event(
            "memory_modes_toggled",
            message=f"Memory modes toggled to {mode}",
            level="info",
            mode=mode,
            scaffold_memory=scaffold_mem,
            token_map_memory=token_mem
        )

    def new_conversation(self) -> None:
        """Start a new conversation with proper synchronization."""
        try:
            with self._state.lock:
                # Store old conversation ID for logging
                old_id = self._conversation_history.conversation_id if self._conversation_history else None
                
                # Create new conversation history
                self._conversation_history = ConversationHistory(
                    maxlen=self.conversation_history_maxlen
                )
                
                # Update state with new conversation history
                self._state.conversation_history = self._conversation_history
                
                # Clear scaffold cache
                self.clear_scaffold_cache()
                
                # Reset curiosity state for new conversation if available
                if self._state.curiosity:
                    self._state.curiosity.reset_for_conversation(self._conversation_history.conversation_id)
                
                # Log the conversation change
                self._log_event(
                    "new_conversation",
                    message="New conversation started",
                    level="info",
                    new_id=self._conversation_history.conversation_id,
                    old_id=old_id,
                    state_hash=self._state.state_hash()
                )
                
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to start new conversation: {str(e)}",
                error_type="conversation_error",
                stack_trace=traceback.format_exc(),
                context={
                    "old_id": old_id if 'old_id' in locals() else None,
                    "state_hash": self._state.state_hash() if self._state else None
                }
            )
            raise

    def log_memory_stats(self, label: str = "", verbose: bool = False) -> None:
        """Log detailed memory statistics."""
        stats = {
            "timestamp": time.time(),
            "event": "memory_stats",
            "label": label,
            "gpu_allocated": None,
            "gpu_reserved": None,
            "gpu_memory_percent": None,
            "dream_memory_len": len(self._dream_memory),
            "token_map_size": len(self._token_map),
            "conversation_history_len": len(self._conversation_history),
            "conversation_id": self._conversation_history.conversation_id,
            "state_hash": self._state.state_hash() if self._state else None
        }

        mem_stats = self.get_memory_stats()
        if mem_stats:
            try:
                stats["gpu_allocated"] = mem_stats['allocated']
                stats["gpu_reserved"] = mem_stats['reserved']
                stats["gpu_memory_percent"] = (mem_stats['allocated'] / mem_stats['total']) * 100 if mem_stats['total'] > 0 else None
            except Exception as e:
                self._log_event(
                    "memory_stats_calculation_failed",
                    message=f"Memory stats calculation failed: {str(e)}",
                    level="warning"
                )

        self._logger.record(stats)
        if verbose and stats["gpu_allocated"] is not None:
            print(f"\n--- Memory Stats ({label}) ---")
            print(f"Allocated: {stats['gpu_allocated']:.2f} GB")
            print(f"Reserved:  {stats['gpu_reserved']:.2f} GB")
            print(f"Dream Memory: {stats['dream_memory_len']} items")
            print(f"Token Map: {stats['token_map_size']} entries")
            print(f"Conversation History: {stats['conversation_history_len']} messages")

    def cleanup(self, model_size: float, trainer) -> None:
        """Perform memory cleanup operations to free up GPU memory."""
        mem_stats = self.get_memory_stats()
        if not mem_stats:
            return

        with self._memory_lock:
            current_mem = mem_stats['allocated'] * (1024 ** 3)  # Convert GB to bytes
            total_mem = mem_stats['total'] * (1024 ** 3)        # Convert GB to bytes
            mem_ratio = current_mem / total_mem if total_mem > 0 else 0.0

            if mem_ratio > self.memory_threshold and time.time() - self._last_cleanup_time > self._cleanup_cooldown:
                self.hardware.clear_memory_cache()
                self._last_cleanup_time = time.time()
                self._log_event(
                    "memory_cleanup",
                    message="Memory cleanup performed",
                    level="info",
                    details={
                        "current_memory": current_mem,
                        "total_memory": total_mem,
                        "memory_ratio": mem_ratio,
                        "threshold": self.memory_threshold
                    }
                )
                print(f"Memory cleaned up (GPU: {mem_ratio:.0%}, Threshold: {self.memory_threshold:.2f})")

    def tune_memory_config(self, **kwargs) -> None:
        """Dynamically adjust memory configuration with validation."""
        with self._memory_lock:
            try:
                old_config = {
                    "memory_threshold": self.memory_threshold,
                    "memory_decay_rate": self.memory_decay_rate,
                    "dream_memory_maxlen": self.dream_memory_maxlen,
                    "dream_memory_decay": self.dream_memory_decay,
                    "dream_prune_threshold": self.dream_prune_threshold,
                    "dream_memory_weight": self.dream_memory_weight,
                    "scaffold_weight": self.scaffold_weight
                }
                
                updates = {}
                for key, value in kwargs.items():
                    if key in old_config:
                        if key in ["dream_memory_maxlen"]:
                            updates[key] = self._validate_config_value(key, value, (5, 50), is_int=True)
                        elif key in ["memory_threshold", "memory_decay_rate", "dream_memory_decay", "dream_prune_threshold", "dream_memory_weight", "scaffold_weight"]:
                            updates[key] = self._validate_config_value(key, value, (0.0, 1.0))
                
                # Apply updates
                for key, value in updates.items():
                    setattr(self, key, value)
                    self._config_manager.set(f"memory_config.{key}", value)
                
                # Log configuration changes
                self._log_event(
                    "memory_config_updated",
                    "Memory configuration updated",
                    level="info",
                    old_config=old_config,
                    new_config=updates
                )
                
            except Exception as e:
                self._log_error(
                    f"Failed to tune memory config: {str(e)}",
                    error_type="config_tuning_error",
                    stack_trace=traceback.format_exc(),
                    context="config_tuning"
                )
                raise

class MemoryMonitor:
    """Monitors memory usage using HardwareManager."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger
    ):
        """Initialize memory monitor with configuration."""
        self.config_manager = config_manager
        self.logger = logger
        self.hardware = HardwareManager(config_manager, logger)  # Initialize HardwareManager
        self._initialized = False
        
        # Initialize configuration
        self._initialize_config()
        
        # Mark as initialized
        self._initialized = True
        
    def _initialize_config(self) -> None:
        """Initialize memory monitoring configuration."""
        try:
            # Get memory configuration
            self.max_memory_mb = self.config_manager.get("memory_config.max_memory_mb", 1024)
            self.memory_threshold = self.config_manager.get("memory_config.memory_threshold", 0.8)
            self.check_interval = self.config_manager.get("memory_config.check_interval", 1000)
            
            self.logger.record_event(
                event_type="memory_monitor_initialized",
                message="Memory monitor configuration initialized",
                level="info",
                max_memory_mb=self.max_memory_mb,
                memory_threshold=self.memory_threshold,
                check_interval=self.check_interval
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to initialize memory configuration: {str(e)}",
                error_type="memory_config_error",
                stack_trace=traceback.format_exc()
            )
            raise
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            # Get memory stats from HardwareManager
            gc.collect()  # Force garbage collection
            stats = self.hardware.get_memory_stats()
            
            # Calculate percentage used
            percentage_used = (stats["allocated_mb"] / self.max_memory_mb) if self.max_memory_mb > 0 else 0
            
            return {
                "allocated_mb": stats["allocated_mb"],
                "reserved_mb": stats["reserved_mb"],
                "percentage_used": percentage_used,
                "max_memory_mb": self.max_memory_mb,
                "available_mb": stats["available_mb"]
            }
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to get memory usage: {str(e)}",
                error_type="memory_stats_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "percentage_used": 0,
                "max_memory_mb": self.max_memory_mb,
                "available_mb": self.max_memory_mb
            }
            
    def check_memory_usage(self, threshold: Optional[float] = None) -> bool:
        """Check if memory usage is below threshold."""
        try:
            # Use provided threshold or default
            threshold = threshold or self.memory_threshold
            
            # Get current memory usage
            memory_stats = self.get_memory_usage()
            percentage_used = memory_stats["percentage_used"]
            
            # Check if below threshold
            is_below_threshold = percentage_used < threshold
            
            if not is_below_threshold:
                self.logger.record_event(
                    event_type="memory_threshold_exceeded",
                    message="Memory usage above threshold",
                    level="warning",
                    percentage_used=percentage_used,
                    threshold=threshold,
                    allocated_mb=memory_stats["allocated_mb"],
                    max_memory_mb=memory_stats["max_memory_mb"]
                )
                
            return is_below_threshold
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to check memory usage: {str(e)}",
                error_type="memory_check_error",
                stack_trace=traceback.format_exc()
            )
            return True  # Default to safe state
            
    def log_memory_usage(self) -> None:
        """Log current memory usage statistics."""
        try:
            memory_stats = self.get_memory_usage()
            self.logger.record_event(
                event_type="memory_usage_stats",
                message="Memory usage statistics",
                level="info",
                additional_info=memory_stats
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to log memory usage: {str(e)}",
                error_type="memory_logging_error",
                stack_trace=traceback.format_exc()
            )
            
    def is_memory_available(self, required_mb: float) -> bool:
        """Check if required memory is available."""
        try:
            memory_stats = self.get_memory_usage()
            available_mb = memory_stats["available_mb"]
            
            # Check if enough memory is available
            is_available = available_mb >= required_mb
            
            if not is_available:
                self.logger.record_event(
                    event_type="insufficient_memory",
                    message="Insufficient memory available",
                    level="warning",
                    required_mb=required_mb,
                    available_mb=available_mb,
                    allocated_mb=memory_stats["allocated_mb"],
                    max_memory_mb=memory_stats["max_memory_mb"]
                )
                
            return is_available
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to check memory availability: {str(e)}",
                error_type="memory_availability_error",
                stack_trace=traceback.format_exc()
            )
            return False  # Default to safe state
