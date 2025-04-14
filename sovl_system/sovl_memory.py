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

class MemoryManager:
    """
    Manages memory for the SOVL system, handling token maps, dream memory,
    conversation history, scaffold context, and GPU memory health.
    """
    _CONFIG_RANGES = {
        "memory_threshold": (0.7, 0.95),
        "memory_decay_rate": (0.8, 0.99),
        "dream_memory_decay": (0.8, 0.99),
        "dream_prune_threshold": (0.0, 0.5),
        "token_map_weight_cap": (1.0, 5.0),
    }
    _MAXLEN_RANGES = {
        "dream_memory_maxlen": (5, 20),
        "conversation_history_maxlen": (5, 20),
        "confidence_history_maxlen": (3, 10),
        "temperament_history_maxlen": (3, 10),
    }
    _CLEANUP_COOLDOWN = 60.0
    _MEM_USAGE_HISTORY_MAXLEN = 10

    def __init__(self, config_manager, device: torch.device, logger: Logger):
        self._config_manager = config_manager
        self._device = device
        self._logger = logger
        self._error_logger = Logger(
            log_file="sovl_memory_errors.jsonl",
            max_size_mb=10,
            compress_old=True
        )
        self._memory_lock = Lock()

        # Initialize configuration
        self._controls_config = config_manager.get_section("controls_config")
        self._core_config = config_manager.get_section("core_config")
        self._training_config = config_manager.get_section("training_config")
        self._initialize_config()

        # Memory structures
        self._token_map = defaultdict(lambda: None)
        self._special_token_map = {}
        self._scaffold_unk_id = None
        self._dream_memory = deque(maxlen=self.dream_memory_maxlen)
        self._conversation_history = ConversationHistory(maxlen=self.conversation_history_maxlen)
        self._mem_usage_history = deque(maxlen=self._MEM_USAGE_HISTORY_MAXLEN)
        self._temp_scaffold_context = None
        self._last_cleanup = 0.0

        # State integration
        self._state = None
        self._hidden_size = None

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        self.memory_threshold = self._validate_config(
            "memory_threshold", self._controls_config.get("memory_threshold", 0.85)
        )
        self.memory_decay_rate = self._validate_config(
            "memory_decay_rate", self._controls_config.get("memory_decay_rate", 0.95)
        )
        self.dream_memory_maxlen = self._validate_maxlen(
            "dream_memory_maxlen", self._controls_config.get("dream_memory_maxlen", 10)
        )
        self.dream_memory_decay = self._validate_config(
            "dream_memory_decay", self._controls_config.get("dream_memory_decay", 0.95)
        )
        self.dream_prune_threshold = self._validate_config(
            "dream_prune_threshold", self._controls_config.get("dream_prune_threshold", 0.1)
        )
        self.conversation_history_maxlen = self._validate_maxlen(
            "conversation_history_maxlen", self._controls_config.get("conversation_history_maxlen", 10)
        )
        self.confidence_history_maxlen = self._validate_maxlen(
            "confidence_history_maxlen", self._controls_config.get("confidence_history_maxlen", 5)
        )
        self.temperament_history_maxlen = self._validate_maxlen(
            "temperament_history_maxlen", self._controls_config.get("temperament_history_maxlen", 5)
        )
        self.use_scaffold_memory = self._controls_config.get("use_scaffold_memory", True)
        self.use_token_map_memory = self._controls_config.get("use_token_map_memory", True)
        self.token_map_weight_cap = self._validate_config(
            "token_map_weight_cap", self._controls_config.get("token_map_weight_cap", 2.0)
        )
        self._dynamic_threshold_base = self.memory_threshold

    def _validate_config(self, key: str, value: float) -> float:
        """Validate configuration parameter within defined range."""
        min_val, max_val = self._CONFIG_RANGES[key]
        if not (min_val <= value <= max_val):
            self._log_error(
                f"Invalid {key}: {value} not in range [{min_val}, {max_val}]",
                conversation_id=self._conversation_history.conversation_id
            )
            raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
        return value

    def _validate_maxlen(self, key: str, value: int) -> int:
        """Validate maxlen parameter within defined range."""
        min_val, max_val = self._MAXLEN_RANGES[key]
        if not (min_val <= value <= max_val):
            self._log_error(
                f"Invalid {key}: {value} not in range [{min_val}, {max_val}]",
                conversation_id=self._conversation_history.conversation_id
            )
            raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
        return value

    def _log_error(self, error: str, **kwargs) -> None:
        """Log an error with context."""
        self._error_logger.record({
            "error": error,
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc() if "stack_trace" in kwargs else None,
            "conversation_id": kwargs.get("conversation_id", None),
            "state_hash": self._state.state_hash() if self._state else None,
            **{k: v for k, v in kwargs.items() if k not in ("conversation_id",)}
        })

    def _log_event(self, event: str, **kwargs) -> None:
        """Log an event with context."""
        self._logger.record({
            "event": event,
            "timestamp": time.time(),
            "conversation_id": self._conversation_history.conversation_id,
            "state_hash": self._state.state_hash() if self._state else None,
            **kwargs
        })

    def initialize_token_map(self, base_tokenizer, scaffold_tokenizer) -> None:
        """Initialize token map for mapping base model tokens to scaffold tokens."""
        with self._memory_lock:
            self._token_map = defaultdict(lambda: [{'ids': [scaffold_tokenizer.unk_token_id], 'weight': 1.0}])
            for base_token, base_id in base_tokenizer.get_vocab().items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(
                    normalized,
                    add_special_tokens=False,
                    max_length=3,
                    truncation=True
                ) or [scaffold_tokenizer.unk_token_id]
                self._token_map[base_id] = {'ids': scaffold_ids, 'weight': 1.0}

            self._special_token_map = {
                base_tokenizer.pad_token_id: scaffold_tokenizer.pad_token_id,
                base_tokenizer.eos_token_id: scaffold_tokenizer.eos_token_id or scaffold_tokenizer.sep_token_id,
                base_tokenizer.unk_token_id: scaffold_tokenizer.unk_token_id,
            }
            self._scaffold_unk_id = self._controls_config.get("scaffold_unk_id", scaffold_tokenizer.unk_token_id)

            self._log_event(
                "token_map_initialized",
                base_vocab_size=len(base_tokenizer.get_vocab())
            )

    def set_state(self, state: SOVLState) -> None:
        """Set the state object for memory synchronization."""
        with self._memory_lock:
            self._state = state
            self._dream_memory = state.dream_memory
            self._conversation_history = state.conversation_history
            self._log_event(
                "memory_state_set",
                dream_memory_len=len(self._dream_memory)
            )

    def set_hidden_size(self, hidden_size: int) -> None:
        """Set the hidden size for validating dream memory tensors."""
        self._hidden_size = hidden_size
        self._log_event("hidden_size_set", hidden_size=hidden_size)

    def update_token_map_memory(self, prompt: str, confidence: float, tokenizer) -> None:
        """Update token map weights based on prompt and confidence."""
        if not self.use_token_map_memory:
            return

        try:
            with self._memory_lock:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                for token_id in input_ids:
                    if token_id in self._token_map:
                        current_weight = self._token_map[token_id]['weight']
                        new_weight = current_weight * self.memory_decay_rate + confidence * (1 - self.memory_decay_rate)
                        self._token_map[token_id]['weight'] = min(max(new_weight, 0.1), self.token_map_weight_cap)

                self._log_event(
                    "token_map_updated",
                    prompt_length=len(prompt),
                    confidence=confidence
                )
        except Exception as e:
            self._log_error(f"Token map update failed: {str(e)}", stack_trace=True)

    def _get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Retrieve GPU memory statistics."""
        if not torch.cuda.is_available():
            return None
        mem_stats = memory_usage(self._device)
        if not mem_stats:
            self._log_event("memory_stats_failed")
            return None
        return mem_stats

    def check_memory_health(self, model_size: float, trainer) -> None:
        """Reduce GPU memory usage if nearing capacity."""
        mem_stats = self._get_memory_stats()
        if not mem_stats:
            return

        with self._memory_lock:
            current_mem = mem_stats['allocated'] * (1024 ** 3)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            mem_ratio = current_mem / total_mem
            self._mem_usage_history.append(mem_ratio)

            avg_mem_usage = sum(self._mem_usage_history) / len(self._mem_usage_history) if self._mem_usage_history else mem_ratio
            dynamic_threshold = min(
                0.95,
                max(
                    0.7,
                    self._dynamic_threshold_base * (1 + (model_size / total_mem) * 0.1 - avg_mem_usage * 0.2)
                )
            )
            lifecycle_stage = safe_divide(
                getattr(trainer, 'data_exposure', 0),
                getattr(trainer, 'lora_capacity', 1),
                default=0.0
            )

            if lifecycle_stage < 0.25 and mem_ratio > dynamic_threshold:
                actions = {
                    "memory_pruned": False,
                    "quantization_changed": False,
                    "cache_cleared": False,
                    "batch_size_reduced": False,
                    "new_batch_size": None
                }

                torch.cuda.empty_cache()
                actions["cache_cleared"] = True

                with self._state.memory_lock:
                    if len(self._state.dream_memory) > 0:
                        original_len = len(self._state.dream_memory)
                        sorted_mem = sorted(self._state.dream_memory, key=lambda x: x[1], reverse=True)
                        keep_len = max(1, original_len // 2)
                        self._state.dream_memory = deque(maxlen=self._state.dream_memory_maxlen)
                        for tensor, weight in sorted_mem[:keep_len]:
                            if weight > 0.5:
                                self._state.append_dream_memory(tensor.detach().cpu(), weight)
                        if len(self._state.dream_memory) < original_len:
                            actions["memory_pruned"] = True

                current_batch_size = self._training_config.get("batch_size", 1)
                if not hasattr(self, '_original_batch_size'):
                    self._original_batch_size = current_batch_size
                if current_batch_size > 1:
                    new_batch_size = max(1, current_batch_size // 2)
                    self._config_manager.update("training_config.batch_size", new_batch_size)
                    self._training_config["batch_size"] = new_batch_size
                    actions["batch_size_reduced"] = True
                    actions["new_batch_size"] = new_batch_size

                quantization_mode = self._core_config.get("quantization", "fp16")
                if quantization_mode != "int8":
                    self._config_manager.update("core_config.quantization", "int8")
                    self._core_config["quantization"] = "int8"
                    actions["quantization_changed"] = True

                self._log_event(
                    "memory_threshold_exceeded",
                    details={
                        "current_memory": current_mem,
                        "total_memory": total_mem,
                        "memory_pruned": actions["memory_pruned"],
                        "quantization_changed": actions["quantization_changed"],
                        "cache_cleared": actions["cache_cleared"],
                        "batch_size_reduced": actions["batch_size_reduced"],
                        "new_batch_size": actions["new_batch_size"],
                        "dynamic_threshold": dynamic_threshold,
                        "threshold": self.memory_threshold,
                        "dream_memory_len": len(self._state.dream_memory)
                    }
                )
                print(f"Memory adjusted (GPU: {mem_ratio:.0%}, Threshold: {dynamic_threshold:.2f}) - "
                      f"Cache Cleared: {actions['cache_cleared']}, Pruned: {actions['memory_pruned']}, "
                      f"Batch Reduced: {actions['batch_size_reduced']}, Quantized: {actions['quantization_changed']}")

            elif mem_ratio < dynamic_threshold * 0.8 and hasattr(self, '_original_batch_size'):
                new_batch_size = self._original_batch_size
                self._config_manager.update("training_config.batch_size", new_batch_size)
                self._training_config["batch_size"] = new_batch_size
                print(f"Restored batch size to {new_batch_size}")
                delattr(self, '_original_batch_size')

    def clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches safely."""
        with self._memory_lock:
            try:
                if self._temp_scaffold_context is not None:
                    if isinstance(self._temp_scaffold_context, torch.Tensor):
                        self._temp_scaffold_context = self._temp_scaffold_context.detach().cpu()
                    self._temp_scaffold_context = None

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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._log_event("scaffold_cache_cleared")
            except Exception as e:
                self._log_error(f"Failed to clear scaffold cache: {str(e)}", stack_trace=True)

    def set_scaffold_context(self, scaffold_hidden_states: torch.Tensor) -> None:
        """Set temporary scaffold context for generation."""
        with self._memory_lock:
            self._temp_scaffold_context = scaffold_hidden_states.detach() if isinstance(scaffold_hidden_states, torch.Tensor) else scaffold_hidden_states
            self._log_event(
                "scaffold_context_set",
                context_shape=list(scaffold_hidden_states.shape) if isinstance(scaffold_hidden_states, torch.Tensor) else None
            )

    def get_scaffold_context(self) -> Optional[torch.Tensor]:
        """Retrieve the current scaffold context."""
        with self._memory_lock:
            return self._temp_scaffold_context

    def append_dream_memory(self, tensor: torch.Tensor, weight: float, metadata: Optional[Dict] = None) -> None:
        """Append a tensor to dream memory with associated weight and metadata."""
        if not self._hidden_size:
            self._log_error("Hidden size not set for dream memory")
            return

        try:
            with self._memory_lock:
                if tensor.shape[-1] != self._hidden_size:
                    raise ValueError(f"Dream tensor shape {tensor.shape} mismatches hidden_size {self._hidden_size}")
                entry = {
                    "tensor": tensor.detach().to(self._device),
                    "weight": min(max(weight, 0.0), 1.0),
                    "metadata": metadata or {"timestamp": time.time()}
                }
                self._dream_memory.append(entry)
                self._state.dream_memory = self._dream_memory
                self._log_event(
                    "dream_memory_appended",
                    tensor_shape=list(tensor.shape),
                    weight=weight,
                    metadata=metadata,
                    dream_memory_len=len(self._dream_memory)
                )
        except Exception as e:
            self._log_error(f"Failed to append dream memory: {str(e)}", stack_trace=True)

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
                    original_length=original_len,
                    new_length=len(self._dream_memory)
                )
            except Exception as e:
                self._log_error(f"Failed to prune dream memory: {str(e)}", stack_trace=True)

    def get_dream_memory_tensors(self) -> Optional[torch.Tensor]:
        """Aggregate dream memory tensors for generation, weighted by importance."""
        if not self._dream_memory:
            return None

        try:
            with self._memory_lock:
                tensors = [entry["tensor"] for entry in self._dream_memory]
                weights = [entry["weight"] for entry in self._dream_memory]
                if not weights:
                    return None
                dream_tensors = torch.stack([t.detach().to(self._device) for t in tensors])
                dream_weights = torch.tensor(weights, dtype=torch.float32, device=self._device)
                weight_sum = dream_weights.sum()
                if weight_sum == 0:
                    return None
                aggregated = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / weight_sum
                self._log_event(
                    "dream_memory_aggregated",
                    tensor_count=len(dream_tensors),
                    shapes=[list(t.shape) for t in tensors]
                )
                return aggregated
        except Exception as e:
            self._log_error(f"Dream memory aggregation failed: {str(e)}", stack_trace=True)
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
                    path_prefix=path_prefix,
                    token_map_size=len(self._token_map),
                    dream_memory_len=len(self._dream_memory)
                )
        except Exception as e:
            self._log_error(f"Memory state save failed: {str(e)}", stack_trace=True)
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
                    self._log_event("token_map_loaded", size=len(self._token_map))

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
                    self._log_event("dream_memory_loaded", length=len(self._dream_memory))

        except Exception as e:
            self._log_error(f"Memory state load failed: {str(e)}", stack_trace=True)
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
            self._log_error(f"Invalid memory mode: {mode}")
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
            mode=mode,
            scaffold_memory=scaffold_mem,
            token_map_memory=token_mem
        )

    def new_conversation(self) -> None:
        """Start a new conversation, resetting relevant memory."""
        with self._memory_lock:
            old_id = self._conversation_history.conversation_id
            self._conversation_history = ConversationHistory(maxlen=self.conversation_history_maxlen)
            self._state.conversation_history = self._conversation_history
            self.clear_scaffold_cache()
            if self._state.curiosity:
                self._state.curiosity.reset_for_conversation(self._conversation_history.conversation_id)
            self._log_event(
                "new_conversation",
                new_id=self._conversation_history.conversation_id,
                old_id=old_id
            )

    def log_memory_stats(self, label: str = "", verbose: bool = False) -> None:
        """Log detailed memory statistics, including CPU memory if available."""
        stats = {
            "timestamp": time.time(),
            "event": "memory_stats",
            "label": label,
            "gpu_allocated": None,
            "gpu_reserved": None,
            "gpu_memory_percent": None,
            "cpu_available": None,
            "dream_memory_len": len(self._dream_memory),
            "token_map_size": len(self._token_map),
            "conversation_history_len": len(self._conversation_history),
            "conversation_id": self._conversation_history.conversation_id,
            "state_hash": self._state.state_hash() if self._state else None
        }

        mem_stats = self._get_memory_stats()
        if mem_stats:
            try:
                stats["gpu_allocated"] = mem_stats['allocated']
                stats["gpu_reserved"] = mem_stats['reserved']
                total_memory = torch.cuda.get_device_properties(0).total_memory
                stats["gpu_memory_percent"] = (stats["gpu_allocated"] * (1024 ** 3) / total_memory * 100) if total_memory > 0 else None
            except Exception as e:
                self._log_event("memory_stats_calculation_failed", warning=str(e))

        try:
            import psutil
            stats["cpu_available"] = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            pass

        self._logger.record(stats)
        if verbose and stats["gpu_allocated"] is not None:
            print(f"\n--- Memory Stats ({label}) ---")
            print(f"Allocated: {stats['gpu_allocated']:.2f} GB")
            print(f"Reserved:  {stats['gpu_reserved']:.2f} GB")
            print(f"CPU Available: {stats['cpu_available']:.2f} GB" if stats['cpu_available'] else "CPU stats unavailable")
            print(f"Dream Memory: {stats['dream_memory_len']} items")
            print(f"Token Map: {stats['token_map_size']} entries")
            print(f"Conversation History: {stats['conversation_history_len']} messages")

    def cleanup(self, model_size: float, trainer) -> None:
        """Perform memory cleanup operations to free up GPU memory."""
        mem_stats = self._get_memory_stats()
        if not mem_stats:
            return

        with self._memory_lock:
            current_mem = mem_stats['allocated'] * (1024 ** 3)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            mem_ratio = current_mem / total_mem

            if mem_ratio > self.memory_threshold and time.time() - self._last_cleanup > self._CLEANUP_COOLDOWN:
                torch.cuda.empty_cache()
                self._last_cleanup = time.time()
                self._log_event(
                    "memory_cleanup",
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
            old_config = {
                "memory_threshold": self.memory_threshold,
                "memory_decay_rate": self.memory_decay_rate,
                "dream_memory_maxlen": self.dream_memory_maxlen,
                "dream_memory_decay": self.dream_memory_decay,
                "dream_prune_threshold": self.dream_prune_threshold,
                "conversation_history_maxlen": self.conversation_history_maxlen,
                "confidence_history_maxlen": self.confidence_history_maxlen,
                "temperament_history_maxlen": self.temperament_history_maxlen,
                "token_map_weight_cap": self.token_map_weight_cap,
            }
            updates = {}
            try:
                for key, value in kwargs.items():
                    if key in self._CONFIG_RANGES:
                        validated_value = self._validate_config(key, value)
                        setattr(self, key, validated_value)
                        updates[f"controls_config.{key}"] = validated_value
                    elif key in self._MAXLEN_RANGES:
                        validated_value = self._validate_maxlen(key, value)
                        setattr(self, key, validated_value)
                        updates[f"controls_config.{key}"] = validated_value
                        if key == "dream_memory_maxlen":
                            self._dream_memory = deque(self._dream_memory, maxlen=validated_value)
                            self._state.dream_memory = self._dream_memory
                        elif key == "conversation_history_maxlen":
                            self._conversation_history = ConversationHistory(maxlen=validated_value)
                            self._state.conversation_history = self._conversation_history
                    else:
                        raise ValueError(f"Unknown configuration parameter: {key}")

                self._config_manager.update_batch(updates)
                self._controls_config.update({k.split('.')[-1]: v for k, v in updates.items()})
                self._log_event(
                    "memory_config_tuned",
                    old_config=old_config,
                    new_config=kwargs
                )
            except Exception as e:
                self._log_error(f"Memory config tuning failed: {str(e)}", stack_trace=True)
                raise
