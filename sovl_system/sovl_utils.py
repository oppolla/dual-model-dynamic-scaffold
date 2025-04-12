import torch
import math
from typing import Union, Tuple

class NumericalGuard:
    """Context manager for precision-sensitive blocks"""
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        
    def __enter__(self):
        self.orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        return self
        
    def __exit__(self, *args):
        torch.set_default_dtype(self.orig_dtype)

def safe_compare(
    a: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    mode: str = 'gt',
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8
) -> Union[bool, torch.Tensor]:
    """
    Unified tolerant comparison for floats/tensors
    Modes: 'gt' (greater than), 'lt' (less than), 'eq' (equal), 'ne' (not equal)
    """
    diff = a - b
    tol = (rel_tol * torch.maximum(torch.abs(a), torch.abs(b)) + abs_tol if isinstance(a, torch.Tensor) \
          else rel_tol * max(abs(a), abs(b)) + abs_tol
    
    if mode == 'gt': return diff > tol
    elif mode == 'lt': return diff < -tol
    elif mode == 'eq': return abs(diff) <= tol
    elif mode == 'ne': return abs(diff) > tol
    raise ValueError(f"Invalid mode '{mode}'. Use: gt/lt/eq/ne")

def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    default: float = 0.0,
    epsilon: float = 1e-10
) -> torch.Tensor:
    """Batch-safe division with auto device handling"""
    mask = torch.abs(denominator) > epsilon
    return torch.where(mask, numerator / (denominator + epsilon), 
                     torch.tensor(default, device=denominator.device))
