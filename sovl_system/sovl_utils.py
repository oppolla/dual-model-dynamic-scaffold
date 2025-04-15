import torch
import math
import time
from typing import Union, Tuple, Optional, List, Dict, Deque, Set, Callable, Any
from collections import deque
import numpy as np
import random
from threading import Lock
import traceback
from functools import wraps

from sovl_logger import Logger

class NumericalGuard:
    """Context manager for numerical stability."""
    def __enter__(self):
        torch.set_grad_enabled(False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(True)

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers with a default fallback."""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default

def safe_compare(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Safely compare two floating point numbers."""
    try:
        return abs(a - b) < tolerance
    except Exception:
        return False

def float_compare(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Compare two floating point numbers with tolerance."""
    try:
        return abs(a - b) < tolerance
    except Exception:
        return False

def float_gt(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Check if a is greater than b with tolerance."""
    try:
        return a > b + tolerance
    except Exception:
        return False

def validate_quantization_mode(mode: str, logger: Optional[Logger] = None) -> str:
    """Validate and handle quantization mode."""
    valid_modes = {'fp16', 'int8', 'int4'}
    if mode not in valid_modes:
        if logger:
            logger.record_event(
                event_type="invalid_quantization_mode",
                message=f"Invalid quantization mode: {mode}",
                level="warning",
                additional_info={"defaulting_to": "fp16"}
            )
        return 'fp16'
    return mode

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
            logger.record_event(
                event_type="memory_usage",
                message="Memory usage statistics",
                level="info",
                additional_info={
                    "memory_stats": stats,
                    "label": label
                }
            )

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
            logger.record_event(
                event_type="batch_size_adjustment",
                message="Batch size adjusted based on memory",
                level="info",
                additional_info={
                    "base_size": base_size,
                    "adjusted_size": adjusted,
                    "available_memory": available / (1024 ** 3)
                }
            )
        return adjusted
    
    except Exception as e:
        if logger:
            logger.record_event(
                event_type="batch_size_error",
                message=f"Dynamic batch size failed: {str(e)}",
                level="error",
                additional_info={
                    "base_size": base_size,
                    "error": str(e)
                }
            )
        return max(1, base_size // 4)

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
                    logger.record_event(
                        event_type="repetition_detected",
                        message="Token repetition detected",
                        level="warning",
                        additional_info={
                            "start_idx": i,
                            "end_idx": i + min_rep_length,
                            "length": min_rep_length
                        }
                    )
                return (i, i + min_rep_length)
        return None
    
    except Exception as e:
        if logger:
            logger.record_event(
                event_type="repetition_detection_error",
                message=f"Repetition detection failed: {str(e)}",
                level="error",
                additional_info={
                    "token_ids_length": len(token_ids),
                    "error": str(e)
                }
            )
        raise

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

def synchronized(lock: Optional[Lock] = None) -> Callable:
    """
    Thread synchronization decorator.
    
    Args:
        lock: Optional Lock instance. If not provided, will use the instance's lock attribute.
        
    Returns:
        Decorated function with thread synchronization.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Use provided lock or instance lock
            lock_to_use = lock if lock is not None else getattr(self, 'lock')
            if not isinstance(lock_to_use, Lock):
                raise AttributeError(f"Lock attribute not found or invalid: {lock_to_use}")
                
            with lock_to_use:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator
