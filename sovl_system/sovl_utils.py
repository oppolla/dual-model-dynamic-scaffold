import torch
import math
import time
from typing import Union, Tuple, Optional, List, Dict, Deque
from collections import deque
import numpy as np
import random

class NumericalGuard:
    """Context manager for precision-sensitive blocks with optional gradient control"""
    def __init__(self, dtype: torch.dtype = torch.float32, no_grad: bool = False):
        self.dtype = dtype
        self.no_grad = no_grad
        
    def __enter__(self):
        self.orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        if self.no_grad:
            self.grad_context = torch.no_grad()
            self.grad_context.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_default_dtype(self.orig_dtype)
        if hasattr(self, 'grad_context'):
            self.grad_context.__exit__(exc_type, exc_val, exc_tb)

def safe_compare(
    a: Union[float, torch.Tensor],
    b: Union[float, torch.Tensor],
    mode: str = 'gt',
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8
) -> Union[bool, torch.Tensor]:
    """
    Unified tolerant comparison for floats/tensors with device awareness
    Modes: 'gt' (greater than), 'lt' (less than), 'eq' (equal), 'ne' (not equal)
    """
    if isinstance(a, torch.Tensor) and isinstance(b, (int, float)):
        b = torch.tensor(b, device=a.device)
    elif isinstance(b, torch.Tensor) and isinstance(a, (int, float)):
        a = torch.tensor(a, device=b.device)
    
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
    """Batch-safe division with auto device handling and type promotion"""
    if numerator.device != denominator.device:
        denominator = denominator.to(numerator.device)
    
    # Type promotion for mixed precision
    result_dtype = torch.promote_types(numerator.dtype, denominator.dtype)
    numerator = numerator.to(result_dtype)
    denominator = denominator.to(result_dtype)
    
    mask = torch.abs(denominator) > epsilon
    return torch.where(mask, numerator / (denominator + epsilon), 
                     torch.tensor(default, device=denominator.device, dtype=result_dtype))

def memory_usage(device: torch.device = None) -> Dict[str, float]:
    """Get memory usage statistics in GB"""
    stats = {}
    if device and device.type == 'cuda':
        stats = {
            'allocated': torch.cuda.memory_allocated(device) / (1024 ** 3),
            'reserved': torch.cuda.memory_reserved(device) / (1024 ** 3),
            'max_allocated': torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        }
    return stats

def log_memory_usage(label: str = "", device: torch.device = None, logger=None):
    """Log memory usage with optional logger"""
    stats = memory_usage(device)
    if stats:
        msg = f"{label} - Memory: Alloc={stats['allocated']:.2f}GB, Resv={stats['reserved']:.2f}GB, Max={stats['max_allocated']:.2f}GB"
        if logger:
            logger({'memory_stats': stats, 'label': label, 'timestamp': time.time()})
        else:
            print(msg)

def dynamic_batch_size(
    base_size: int,
    memory_threshold: float = 0.8,
    safety_factor: float = 0.9
) -> int:
    """
    Dynamically adjust batch size based on available GPU memory
    Returns adjusted batch size that should fit within threshold
    """
    if not torch.cuda.is_available():
        return base_size
    
    total_mem = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    available = (total_mem * memory_threshold * safety_factor) - allocated
    
    if available <= 0:
        return max(1, base_size // 4)
    
    # Estimate memory per sample (crude approximation)
    sample_mem = allocated / base_size if base_size > 0 else 1e6
    adjusted_size = min(base_size, int(available / sample_mem))
    return max(1, adjusted_size)

def cosine_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of vectors
    a: (n, d) tensor
    b: (m, d) tensor
    Returns: (n, m) similarity matrix
    """
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def normalize_scores(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling and softmax normalization"""
    if temperature != 1.0:
        scores = scores / temperature
    return torch.softmax(scores, dim=-1)

def weighted_sample(scores: torch.Tensor, top_k: int = None) -> int:
    """Sample from scores with optional top-k filtering"""
    if top_k is not None and top_k > 0 and top_k < scores.shape[-1]:
        values, indices = torch.topk(scores, top_k)
        scores = torch.zeros_like(scores).scatter(-1, indices, values)
    return torch.multinomial(scores, num_samples=1).item()

def decay_weights(weights: Deque[float], decay_rate: float) -> Deque[float]:
    """Apply exponential decay to a deque of weights"""
    return deque([w * decay_rate for w in weights], maxlen=weights.maxlen)

def float_lt(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float less-than with epsilon tolerance"""
    return a < b - eps

def float_gt(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float greater-than with epsilon tolerance"""
    return a > b + eps

def float_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    """Float equality with epsilon tolerance"""
    return abs(a - b) <= eps

def get_parameter_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate memory usage of a tensor in bytes"""
    return tensor.numel() * tensor.element_size()

def validate_layer_indices(
    indices: List[int],
    total_layers: int,
    context: str = ""
) -> List[int]:
    """Validate and filter layer indices with context for error messages"""
    if not indices:
        return []
    
    valid = [i for i in indices if 0 <= i < total_layers]
    if len(valid) != len(indices):
        invalid = set(indices) - set(valid)
        print(f"Warning in {context}: Invalid layer indices {invalid} for model with {total_layers} layers")
    return valid

def log_gradient_norms(model: torch.nn.Module, logger=None) -> Dict[str, float]:
    """Log L2 norms of gradients by parameter name"""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    
    if logger and norms:
        logger({
            'gradient_norms': norms,
            'timestamp': time.time(),
            'total_params': len(norms),
            'mean_norm': sum(norms.values()) / len(norms) if norms else 0
        })
    
    return norms
