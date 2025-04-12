from typing import Union, List, Tuple
import torch
from math_utils import safe_divide

"""
Robust logits processing abstraction
Place this in your project root or modules/ directory
"""

class LogitsProcessor:
    def __init__(self, logits: Union[List[torch.Tensor], torch.Tensor]):
        self.logits = self._validate(logits)
        
    def _validate(self, logits) -> List[torch.Tensor]:
        """Convert and validate logits into consistent format"""
        if isinstance(logits, torch.Tensor):
            if logits.dim() != 3:
                raise ValueError(f"Logits tensor must be 3D (got shape {logits.shape})")
            return [logits[i] for i in range(logits.size(0))]
            
        if not isinstance(logits, (list, tuple)):
            raise TypeError(f"Logits must be tensor, list or tuple (got {type(logits)})")
            
        return logits

    def calculate_confidence(self, generated_ids: List[int]) -> float:
        """Calculate generation confidence score"""
        if len(self.logits) != len(generated_ids):
            raise ValueError(f"Logits length ({len(self.logits)}) ≠ generated_ids length ({len(generated_ids)})")
            
        try:
            stacked = torch.stack(self.logits)
            probs = torch.softmax(stacked, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            
            # Low confidence if probabilities are flat
            if max_probs.var().item() < 1e-5:
                return 0.2
                
            return max_probs.mean().item()
        except Exception as e:
            raise RuntimeError(f"Confidence calculation failed: {str(e)}")
