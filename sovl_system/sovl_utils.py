import torch
import math
import time
from typing import Union, Tuple, Optional, List, Dict, Deque, Set
from collections import deque
import numpy as np
import random
from threading import Lock
import traceback
from functools import wraps

from sovl_logger import Logger

class NumericalGuard:
    """Context manager for precision-sensitive blocks with mixed precision support."""
    
    def __init__(self, dtype: torch.dtype = torch.float32, no_grad: bool = False, mixed_precision: bool = False):
        self.dtype = dtype
        self.no_grad = no_grad
        self.mixed_precision = mixed_precision
        self.contexts: List = []
        self.orig_dtype: Optional[torch.dtype] = None
        
    def __enter__(self):
        self.orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        
        if self.no_grad:
            self.contexts.append(torch.no_grad().__enter__())
        if self.mixed_precision and torch.cuda.is_available():
            self.contexts.append(torch.cuda.amp.autocast(enabled=True, dtype=self.dtype).__enter__())
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_dtype(self.orig_dtype)
        for context in reversed(self.contexts):
            context.__exit__(exc_type, exc_val, exc_tb)

def _check_bitsandbytes_availability(quantization_mode: str, logger: Logger) -> bool:
    """Check if bitsandbytes is available for quantization."""
    try:
        import bitsandbytes as bnb
        if quantization_mode == "int8":
            from bitsandbytes.nn import Linear8bitLt
        elif quantization_mode == "int4":
            from bitsandbytes.nn import Linear4bit
        return True
    except ImportError:
        logger.record({
            "warning": f"bitsandbytes not available for {quantization_mode}. Falling back to fp16.",
            "timestamp": time.time(),
            "conversation_id": "init"
        })
        return False

def validate_quantization_mode(quantization_mode: str, logger: Logger) -> str:
    """
    Validate and handle quantization mode configuration.
    
    Args:
        quantization_mode: The requested quantization mode ('fp16', 'int8', 'int4')
        logger: Logger instance for recording events
        
    Returns:
        Validated quantization mode
    """
    valid_modes = {"fp16", "int8", "int4"}
    if quantization_mode not in valid_modes:
        logger.record({
            "warning": f"Invalid quantization mode '{quantization_mode}'. Defaulting to 'fp16'.",
            "timestamp": time.time(),
            "conversation_id": "init"
        })
        return "fp16"

    if quantization_mode in {"int8", "int4"} and not _check_bitsandbytes_availability(quantization_mode, logger):
        return "fp16"
    
    return quantization_mode

def validate_layer_indices(indices: List[int], total_layers: int, context: str = "", logger: Optional[Logger] = None) -> List[int]:
    """
    Validate and filter layer indices.
    
    Args:
        indices: List of layer indices
        total_layers: Total number of layers in model
        context: Context for error messages
        logger: Optional logger for warnings
    
    Returns:
        Valid layer indices
    """
    if not indices:
        return []
    
    valid_indices = [i for i in indices if 0 <= i < total_layers]
    if len(valid_indices) != len(indices) and logger:
        invalid = set(indices) - set(valid_indices)
        logger.record({
            "warning": f"Invalid layer indices {invalid} for model with {total_layers} layers",
            "context": context,
            "timestamp": time.time()
        })
    
    return valid_indices

def safe_compare(
    a: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    mode: str = 'gt',
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8,
    logger: Optional[Logger] = None
) -> Union[bool, torch.Tensor]:
    """
    Perform tolerant comparison for floats/tensors with device awareness.
    
    Args:
        a: First value/tensor
        b: Second value/tensor
        mode: Comparison mode ('gt', 'lt', 'eq', 'ne')
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        logger: Optional logger for error reporting
    
    Returns:
        Boolean or tensor result
    """
    comparison_modes = {
        'gt': lambda diff, tol: diff > tol,
        'lt': lambda diff, tol: diff < -tol,
        'eq': lambda diff, tol: torch.abs(diff) <= tol,
        'ne': lambda diff, tol: torch.abs(diff) > tol
    }
    
    if mode not in comparison_modes:
        raise ValueError(f"Invalid mode '{mode}'. Use: gt/lt/eq/ne")
    
    try:
        # Convert scalars to tensors if needed
        if isinstance(a, torch.Tensor) and isinstance(b, (int, float)):
            b = torch.tensor(b, device=a.device, dtype=a.dtype)
        elif isinstance(b, torch.Tensor) and isinstance(a, (int, float)):
            a = torch.tensor(a, device=b.device, dtype=b.dtype)
        elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.device != b.device:
                b = b.to(a.device)
            if a.dtype != b.dtype:
                b = b.to(a.dtype)
        
        diff = a - b
        tol = rel_tol * torch.maximum(torch.abs(a), torch.abs(b)) + abs_tol if isinstance(a, torch.Tensor) \
              else rel_tol * max(abs(a), abs(b)) + abs_tol
        
        return comparison_modes[mode](diff, tol)
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Safe compare failed: {str(e)}",
                "mode": mode,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        raise

def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    default: float = 0.0,
    epsilon: float = 1e-10,
    logger: Optional[Logger] = None
) -> torch.Tensor:
    """Perform batch-safe division with auto device handling and type promotion."""
    try:
        if not isinstance(numerator, torch.Tensor) or not isinstance(denominator, torch.Tensor):
            raise ValueError("Both numerator and denominator must be tensors")
        
        result_dtype = torch.promote_types(numerator.dtype, denominator.dtype)
        numerator = numerator.to(result_dtype, device=numerator.device)
        denominator = denominator.to(result_dtype, device=numerator.device)
        
        mask = torch.abs(denominator) > epsilon
        return torch.where(
            mask,
            numerator / (denominator + epsilon),
            torch.tensor(default, device=numerator.device, dtype=result_dtype)
        )
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Safe divide failed: {str(e)}",
                "numerator_shape": str(numerator.shape) if isinstance(numerator, torch.Tensor) else "N/A",
                "denominator_shape": str(denominator.shape) if isinstance(denominator, torch.Tensor) else "N/A",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        raise

def float_lt(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float comparison with epsilon tolerance."""
    return a < b - eps

def float_gt(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float comparison with epsilon tolerance."""
    return a > b + eps

def float_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float comparison with epsilon tolerance."""
    return abs(a - b) <= eps

def memory_usage(device: torch.device = None) -> Dict[str, float]:
    """Get memory usage statistics in GB."""
    if device is None or device.type != 'cuda':
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated(device) / (1024 ** 3),
        'reserved': torch.cuda.memory_reserved(device) / (1024 ** 3),
        'max_allocated': torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    }

def log_memory_usage(label: str = "", device: torch.device = None, logger: Optional[Logger] = None):
    """Log memory usage statistics."""
    if logger:
        stats = memory_usage(device)
        if stats:
            logger.record({
                'event': 'memory_usage',
                'memory_stats': stats,
                'label': label,
                'timestamp': time.time()
            })

def dynamic_batch_size(
    base_size: int,
    memory_threshold: float = 0.8,
    safety_factor: float = 0.9,
    logger: Optional[Logger] = None
) -> int:
    """
    Adjust batch size based on available GPU memory.
    
    Args:
        base_size: Base batch size
        memory_threshold: Memory usage threshold (0 to 1)
        safety_factor: Safety margin for memory allocation
        logger: Optional logger for debugging
    
    Returns:
        Adjusted batch size
    """
    if not torch.cuda.is_available():
        return base_size
    
    try:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated()
        available = (total_mem * memory_threshold * safety_factor) - allocated
        
        if available <= 0:
            adjusted = max(1, base_size // 4)
        else:
            sample_mem = allocated / base_size if base_size > 0 else 1e6
            adjusted = min(base_size, int(available / sample_mem))
            adjusted = max(1, adjusted)
        
        if logger:
            logger.record({
                'event': 'dynamic_batch_size',
                'base_size': base_size,
                'adjusted_size': adjusted,
                'available_memory': available / (1024 ** 3),
                'timestamp': time.time()
            })
        return adjusted
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Dynamic batch size failed: {str(e)}",
                "base_size": base_size,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        return max(1, base_size // 4)

def cosine_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
    batch_size: int = 1000,
    logger: Optional[Logger] = None
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of vectors.
    
    Args:
        a: (n, d) tensor
        b: (m, d) tensor
        eps: Small value to avoid division by zero
        batch_size: Batch size for large tensors
        logger: Optional logger for debugging
    
    Returns:
        (n, m) similarity matrix
    """
    try:
        if a.dim() != 2 or b.dim() != 2 or a.size(1) != b.size(1):
            raise ValueError(f"Invalid tensor shapes: a={a.shape}, b={b.shape}")
        
        with NumericalGuard(dtype=a.dtype):
            n, m = a.size(0), b.size(0)
            result = torch.zeros(n, m, device=a.device, dtype=a.dtype)
            
            b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                a_batch = a[i:end_i]
                a_norm = a_batch / (a_batch.norm(dim=1, keepdim=True) + eps)
                result[i:end_i] = torch.mm(a_norm, b_norm.transpose(0, 1))
            
            if logger:
                logger.record({
                    'event': 'cosine_similarity',
                    'a_shape': a.shape,
                    'b_shape': b.shape,
                    'result_shape': result.shape,
                    'timestamp': time.time()
                })
            return result
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Cosine similarity failed: {str(e)}",
                "a_shape": str(a.shape) if isinstance(a, torch.Tensor) else "N/A",
                "b_shape": str(b.shape) if isinstance(b, torch.Tensor) else "N/A",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        raise

def normalize_scores(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling and softmax normalization."""
    with NumericalGuard(dtype=scores.dtype):
        scores = scores / max(1e-10, temperature)
        return torch.softmax(scores, dim=-1)

def weighted_sample(scores: torch.Tensor, top_k: Optional[int] = None) -> int:
    """Sample from scores with optional top-k filtering."""
    with NumericalGuard(dtype=scores.dtype):
        if top_k is not None and top_k > 0 and top_k < scores.shape[-1]:
            values, indices = torch.topk(scores, top_k)
            scores = torch.zeros_like(scores).scatter_(-1, indices, values)
        return torch.multinomial(scores, num_samples=1).item()

def decay_weights(weights: Deque[float], decay_rate: float) -> Deque[float]:
    """Apply exponential decay to a deque of weights."""
    return deque([w * decay_rate for w in weights], maxlen=weights.maxlen)

def float_compare(a: float, b: float, mode: str = 'eq', eps: float = 1e-6) -> bool:
    """Float comparison with epsilon tolerance."""
    comparisons = {
        'lt': lambda a, b: a < b - eps,
        'gt': lambda a, b: a > b + eps,
        'eq': lambda a, b: abs(a - b) <= eps
    }
    
    if mode not in comparisons:
        raise ValueError(f"Invalid mode '{mode}'. Use: lt/gt/eq")
    
    return comparisons[mode](a, b)

def get_parameter_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

def set_seed(seed: int, lock: Optional[Lock] = None):
    """Set random seeds for reproducibility with thread safety."""
    def _set_seeds():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    if lock:
        with lock:
            _set_seeds()
    else:
        _set_seeds()

def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate memory usage of a tensor in bytes."""
    return tensor.numel() * tensor.element_size()

def detect_repetitions(
    token_ids: List[int],
    special_ids: Set[int],
    min_rep_length: int = 3,
    max_scan: int = 100,
    logger: Optional[Logger] = None
) -> Optional[Tuple[int, int]]:
    """
    Detect repeating token sequences.
    
    Args:
        token_ids: List of token IDs
        special_ids: Set of special token IDs to ignore
        min_rep_length: Minimum sequence length to check
        max_scan: Maximum number of tokens to scan
        logger: Optional logger for debugging
    
    Returns:
        (start_idx, end_idx) of first repetition found or None
    """
    try:
        filtered = [i for i in token_ids if i not in special_ids]
        scan_range = min(len(filtered), max_scan)
        
        for i in range(scan_range - 2 * min_rep_length + 1):
            window = filtered[i:i + min_rep_length]
            next_window = filtered[i + min_rep_length:i + 2 * min_rep_length]
            if window == next_window:
                if logger:
                    logger.record({
                        'event': 'repetition_detected',
                        'start_idx': i,
                        'end_idx': i + min_rep_length,
                        'length': min_rep_length,
                        'timestamp': time.time()
                    })
                return (i, i + min_rep_length)
        return None
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Repetition detection failed: {str(e)}",
                "token_ids_length": len(token_ids),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        raise

def log_gradient_norms(model: torch.nn.Module, logger: Optional[Logger] = None) -> Dict[str, float]:
    """Log L2 norms of gradients by parameter name."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    
    if logger and norms:
        logger.record({
            'event': 'gradient_norms',
            'gradient_norms': norms,
            'timestamp': time.time(),
            'total_params': len(norms),
            'mean_norm': sum(norms.values()) / len(norms) if norms else 0
        })
    
    return norms

def adjust_temperature(
    base_temp: float,
    temperament_score: float,
    mood_influence: float = 0.3,
    min_temp: float = 0.5,
    max_temp: float = 1.5,
    curiosity_pressure: Optional[float] = None,
    logger: Optional[Logger] = None
) -> float:
    """
    Adjust temperature based on temperament and curiosity.
    
    Args:
        base_temp: Base temperature from config
        temperament_score: Current temperament (-1 to 1)
        mood_influence: Mood effect strength
        min_temp: Minimum temperature allowed
        max_temp: Maximum temperature allowed
        curiosity_pressure: Optional curiosity pressure (0 to 1)
        logger: Optional logger for debugging
    
    Returns:
        Adjusted temperature value
    """
    try:
        with NumericalGuard():
            # Clamp input values
            base_temp = max(min_temp, min(max_temp, base_temp))
            temperament_score = max(-1.0, min(1.0, temperament_score))
            mood_influence = max(0.0, min(1.0, mood_influence))
            
            temp_adjustment = mood_influence * 0.3 * temperament_score
            if curiosity_pressure is not None:
                curiosity_pressure = max(0.0, min(1.0, curiosity_pressure))
                temp_adjustment += curiosity_pressure * 0.1
            
            adjusted_temp = max(min_temp, min(max_temp, base_temp + temp_adjustment))
            
            if logger:
                logger.record({
                    'event': 'temperature_adjustment',
                    'base_temp': base_temp,
                    'temperament_score': temperament_score,
                    'mood_influence': mood_influence,
                    'curiosity_pressure': curiosity_pressure,
                    'adjusted_temp': adjusted_temp,
                    'timestamp': time.time()
                })
            return adjusted_temp
    
    except Exception as e:
        if logger:
            logger.record({
                "error": f"Temperature adjustment failed: {str(e)}",
                "base_temp": base_temp,
                "temperament_score": temperament_score,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
        return base_temp

def synchronized(lock_name: str):
    """Thread synchronization decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_name)
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator
