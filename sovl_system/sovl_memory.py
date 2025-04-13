from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple, Union
import torch
import time
import os
from threading import Lock
from sovl_logger import Logger
from sovl_utils import safe_divide, float_lt, log_memory_usage
import traceback

@dataclass
class MemoryConfig:
    """Configuration for memory management system."""
    # Thresholds and limits
    memory_threshold: float = 0.85  # GPU memory usage threshold for cleanup
    dynamic_threshold_base: float = 0.8  # Base for dynamic threshold calculation
    memory_decay_rate: float = 0.95  # Rate at which token map weights decay
    dream_memory_maxlen: int = 10  # Maximum dream memory entries
    dream_memory_weight: float = 0.1  # Weight for dream memory in generation
    dream_prune_threshold: float = 0.1  # Minimum weight to keep dream memory
    token_map_weight_cap: float = 2.0  # Maximum weight for token map entries
    # Cleanup strategies
    enable_quantization_fallback: bool = True  # Allow switching to int8 on OOM
    enable_batch_reduction: bool = True  # Allow reducing batch size on OOM
    enable_memory_pruning: bool = True  # Allow pruning low-weight memories
    # Validation ranges
    _ranges: Dict[str, Tuple[float, float]] = None
    _maxlen_ranges: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        self._ranges = {
            "memory_threshold": (0.7, 0.95),
            "dynamic_threshold_base": (0.7, 0.9),
            "memory_decay_rate": (0.8, 0.99),
            "dream_memory_weight": (0.0, 0.5),
            "dream_prune_threshold": (0.0, 0.5),
            "token_map_weight_cap": (1.0, 5.0)
        }
        self._maxlen_ranges = {
            "dream_memory_maxlen": (5, 20)
        }
        
        for key, (min_val, max_val) in self._ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
                
        for key, (min_val, max_val) in self._maxlen_ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key in self._ranges:
                min_val, max_val = self._ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
                setattr(self, key, value)
            elif key in self._maxlen_ranges:
                min_val, max_val = self._maxlen_ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

class MemoryManager:
    """Comprehensive memory management system with monitoring, optimization, and logging."""
    
    def __init__(self, 
                 config: MemoryConfig,
                 logger: Logger,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize memory management system.
        
        Args:
            config: Memory configuration
            logger: Logger instance for recording events
            device: Device for memory operations
        """
        self.config = config
        self.logger = logger
        self.device = device
        self.memory_lock = Lock()
        self.mem_usage_history: Deque[float] = deque(maxlen=10)
        self._original_batch_size: Optional[int] = None
        self._last_cleanup: float = 0.0
        self._cleanup_cooldown: float = 60.0  # Seconds between cleanups
        
        # Initialize logging
        self.logger.record({
            "event": "memory_manager_init",
            "config": vars(config),
            "device": str(device),
            "timestamp": time.time()
        })

    def check_memory_health(self) -> bool:
        """
        Monitor memory usage and trigger cleanup if needed.
        
        Returns:
            bool: True if cleanup was performed, False otherwise
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            with self.memory_lock:
                # Get current memory stats
                mem_stats = self._get_memory_stats()
                if not mem_stats:
                    return False
                    
                current_mem = mem_stats['allocated']
                total_mem = mem_stats['total']
                mem_ratio = current_mem / total_mem
                self.mem_usage_history.append(mem_ratio)
                
                # Calculate dynamic threshold
                dynamic_threshold = self._calculate_dynamic_threshold(total_mem)
                
                # Check if we need cleanup
                if mem_ratio > dynamic_threshold:
                    self._perform_memory_cleanup(mem_stats, dynamic_threshold)
                    return True
                    
                # Restore original batch size if memory is low
                elif (mem_ratio < dynamic_threshold * 0.8 and 
                      hasattr(self, '_original_batch_size')):
                    self._restore_original_batch_size()
                    
                return False
                
        except Exception as e:
            self.logger.record({
                "error": f"Memory health check failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get current memory statistics in GB."""
        try:
            if not torch.cuda.is_available():
                return None
                
            stats = {
                'allocated': torch.cuda.memory_allocated() / (1024 ** 3),
                'reserved': torch.cuda.memory_reserved() / (1024 ** 3),
                'total': torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3)
            }
            return stats
            
        except Exception as e:
            self.logger.record({
                "warning": f"Failed to get memory stats: {str(e)}",
                "timestamp": time.time()
            })
            return None

    def _calculate_dynamic_threshold(self, total_mem: float) -> float:
        """
        Calculate dynamic memory threshold based on usage patterns.
        
        Args:
            total_mem: Total available GPU memory in GB
            
        Returns:
            float: Dynamic threshold (0.0-1.0)
        """
        avg_mem_usage = (sum(self.mem_usage_history) / len(self.mem_usage_history) 
                         if self.mem_usage_history else 0.0)
        
        # Adjust based on recent memory usage patterns
        return min(
            0.95,  # Absolute maximum
            max(
                0.7,  # Absolute minimum
                self.config.dynamic_threshold_base * (1 + (avg_mem_usage * 0.1))
        )

    def _perform_memory_cleanup(self, 
                              mem_stats: Dict[str, float],
                              threshold: float) -> None:
        """
        Execute memory cleanup procedures.
        
        Args:
            mem_stats: Current memory statistics
            threshold: Current memory threshold
        """
        if time.time() - self._last_cleanup < self._cleanup_cooldown:
            return
            
        self._last_cleanup = time.time()
        actions = {
            'cache_cleared': False,
            'memory_pruned': False,
            'quantization_changed': False,
            'batch_size_reduced': False
        }
        
        # 1. Clear PyTorch cache
        torch.cuda.empty_cache()
        actions['cache_cleared'] = True
        
        # 2. Prune dream memory
        if self.config.enable_memory_pruning:
            actions['memory_pruned'] = self._prune_dream_memory()
            
        # 3. Reduce batch size if enabled
        if self.config.enable_batch_reduction:
            actions['batch_size_reduced'] = self._reduce_batch_size()
            
        # 4. Fallback to quantization if enabled
        if (self.config.enable_quantization_fallback and 
            torch.cuda.get_device_properties(0).total_memory < 16 * (1024 ** 3)):  # <16GB GPU
            actions['quantization_changed'] = self._switch_to_quantization()
            
        # Log cleanup actions
        self.logger.record({
            "event": "memory_cleanup",
            "actions": actions,
            "memory_before": mem_stats,
            "memory_after": self._get_memory_stats(),
            "threshold": threshold,
            "timestamp": time.time()
        })

    def _prune_dream_memory(self) -> bool:
        """Prune low-weight dream memory entries."""
        try:
            if not hasattr(self, '_dream_memory'):
                return False
                
            original_len = len(self._dream_memory)
            if original_len == 0:
                return False
                
            # Sort by weight and keep top entries
            sorted_mem = sorted(self._dream_memory.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
            keep_len = max(1, original_len // 2)
            self._dream_memory = {
                k: v for k, v in sorted_mem[:keep_len] 
                if v > self.config.dream_prune_threshold
            }
            
            return len(self._dream_memory) < original_len
            
        except Exception as e:
            self.logger.record({
                "warning": f"Dream memory pruning failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _reduce_batch_size(self) -> bool:
        """Reduce training batch size to conserve memory."""
        try:
            # This would interact with the trainer's config
            # Implementation depends on how batch size is managed
            return False
            
        except Exception as e:
            self.logger.record({
                "warning": f"Batch reduction failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _switch_to_quantization(self) -> bool:
        """Switch model to quantized version to save memory."""
        try:
            # This would trigger model reload with quantization
            # Actual implementation depends on model management
            return False
            
        except Exception as e:
            self.logger.record({
                "warning": f"Quantization switch failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _restore_original_batch_size(self) -> None:
        """Restore original batch size when memory is available."""
        try:
            # Implementation depends on batch size management
            delattr(self, '_original_batch_size')
            self.logger.record({
                "event": "batch_size_restored",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.record({
                "warning": f"Batch size restoration failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def clear_caches(self, full: bool = False) -> None:
        """
        Clear various memory caches.
        
        Args:
            full: If True, perform more aggressive cleanup
        """
        with self.memory_lock:
            try:
                # Clear PyTorch CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Additional cleanup if requested
                if full:
                    self._clear_scaffold_cache()
                    self._prune_dream_memory()
                    
                self.logger.record({
                    "event": "cache_cleared",
                    "full_cleanup": full,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                self.logger.record({
                    "error": f"Cache clearance failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })

    def _clear_scaffold_cache(self) -> None:
        """Clear scaffold-related cached tensors."""
        try:
            if hasattr(self, '_temp_scaffold_context'):
                if isinstance(self._temp_scaffold_context, torch.Tensor):
                    self._temp_scaffold_context = self._temp_scaffold_context.detach().cpu()
                del self._temp_scaffold_context
                
            self.logger.record({
                "event": "scaffold_cache_cleared",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.record({
                "error": f"Scaffold cache clearance failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def log_memory_stats(self, label: str = "") -> None:
        """
        Log detailed memory statistics.
        
        Args:
            label: Context label for the log entry
        """
        stats = {
            "timestamp": time.time(),
            "event": "memory_stats",
            "label": label,
            "gpu_allocated": None,
            "gpu_reserved": None,
            "gpu_total": None,
            "cpu_available": None
        }
        
        try:
            # GPU stats
            if torch.cuda.is_available():
                mem_stats = self._get_memory_stats()
                if mem_stats:
                    stats.update({
                        "gpu_allocated": mem_stats['allocated'],
                        "gpu_reserved": mem_stats['reserved'],
                        "gpu_total": mem_stats['total']
                    })
            
            # CPU stats
            try:
                import psutil
                stats['cpu_available'] = psutil.virtual_memory().available / (1024 ** 3)
            except ImportError:
                pass
                
            self.logger.record(stats)
            
        except Exception as e:
            self.logger.record({
                "error": f"Memory stats logging failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def manage_token_map_memory(self, 
                              token_map: Dict[int, Dict],
                              prompt: str,
                              confidence: float) -> None:
        """
        Update token map weights based on usage.
        
        Args:
            token_map: Token mapping dictionary
            prompt: Input prompt text
            confidence: Generation confidence score
        """
        if not token_map or confidence <= 0:
            return
            
        with self.memory_lock:
            try:
                tokens = self._tokenize_prompt(prompt)
                for token_id in tokens:
                    if token_id in token_map:
                        token_map[token_id]['weight'] = min(
                            token_map[token_id]['weight'] + confidence * 0.1,
                            self.config.token_map_weight_cap
                        )
                
                # Apply decay to all tokens
                for token_id in token_map:
                    token_map[token_id]['weight'] *= self.config.memory_decay_rate
                    
                self.logger.record({
                    "event": "token_map_updated",
                    "prompt_length": len(prompt),
                    "confidence": confidence,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                self.logger.record({
                    "error": f"Token map update failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })

    def _tokenize_prompt(self, prompt: str) -> list:
        """Tokenize prompt (implementation depends on tokenizer availability)."""
        # This would need access to the tokenizer - would be provided by main system
        # Placeholder implementation
        return []

    def get_dream_memory_context(self) -> Optional[torch.Tensor]:
        """
        Get weighted dream memory context for generation.
        
        Returns:
            Optional[torch.Tensor]: Combined memory context or None
        """
        if not hasattr(self, '_dream_memory') or not self._dream_memory:
            return None
            
        with self.memory_lock:
            try:
                dream_tensors = []
                dream_weights = []
                
                for tensor, weight in self._dream_memory.items():
                    if weight > self.config.dream_prune_threshold:
                        dream_tensors.append(tensor.to(self.device))
                        dream_weights.append(weight)
                        
                if not dream_tensors:
                    return None
                    
                dream_tensors = torch.stack(dream_tensors)
                dream_weights = torch.tensor(dream_weights, 
                                           dtype=torch.float32, 
                                           device=self.device)
                
                # Weighted average
                return torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / dream_weights.sum()
                
            except Exception as e:
                self.logger.record({
                    "error": f"Dream memory context failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                return None

    def update_dream_memory(self, 
                          new_memory: torch.Tensor,
                          weight: float = 1.0) -> None:
        """
        Update dream memory with new entry.
        
        Args:
            new_memory: Memory tensor to add
            weight: Initial weight for the memory
        """
        if not hasattr(self, '_dream_memory'):
            self._dream_memory = deque(maxlen=self.config.dream_memory_maxlen)
            
        with self.memory_lock:
            try:
                # Detach and clone to prevent memory leaks
                mem_tensor = new_memory.detach().clone().cpu()
                self._dream_memory.append((mem_tensor, weight))
                
                self.logger.record({
                    "event": "dream_memory_updated",
                    "new_weight": weight,
                    "current_size": len(self._dream_memory),
                    "timestamp": time.time()
                })
                
            except Exception as e:
                self.logger.record({
                    "error": f"Dream memory update failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })

    def get_state(self) -> Dict:
        """Get current state for serialization."""
        state = {
            "memory_usage_history": list(self.mem_usage_history),
            "last_cleanup": self._last_cleanup,
            "original_batch_size": self._original_batch_size if hasattr(self, '_original_batch_size') else None
        }
        
        if hasattr(self, '_dream_memory'):
            state["dream_memory"] = [
                (tensor.cpu().numpy().tolist(), weight) 
                for tensor, weight in self._dream_memory
            ]
            
        return state

    def load_state(self, state: Dict) -> None:
        """Load state from serialized data."""
        with self.memory_lock:
            self.mem_usage_history = deque(state.get("memory_usage_history", []), maxlen=10)
            self._last_cleanup = state.get("last_cleanup", 0.0)
            
            if "original_batch_size" in state and state["original_batch_size"] is not None:
                self._original_batch_size = state["original_batch_size"]
                
            if "dream_memory" in state:
                self._dream_memory = deque(maxlen=self.config.dream_memory_maxlen)
                for tensor_data, weight in state["dream_memory"]:
                    tensor = torch.tensor(tensor_data)
                    self._dream_memory.append((tensor, weight))
                    
            self.logger.record({
                "event": "memory_state_loaded",
                "state_keys": list(state.keys()),
                "timestamp": time.time()
            })

    def tune(self, **kwargs) -> None:
        """
        Dynamically adjust memory configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        old_config = vars(self.config).copy()
        self.config.update(**kwargs)
        
        # Adjust structures if their sizes changed
        if "dream_memory_maxlen" in kwargs:
            if hasattr(self, '_dream_memory'):
                self._dream_memory = deque(self._dream_memory, maxlen=self.config.dream_memory_maxlen)
                
        self.logger.record({
            "event": "memory_config_updated",
            "old_config": old_config,
            "new_config": vars(self.config),
            "timestamp": time.time()
        })
