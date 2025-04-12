import torch
from typing import Union, List
from sovl_utils import NumericalGuard, safe_divide

class LogitsError(Exception):
    """Custom exceptions for processing failures"""
    pass

class SOVLProcessor:
    def __init__(self, device: str = None):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _validate_logits(self, logits: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Convert input to 3D tensor (batch, seq_len, vocab_size) with validation"""
        if isinstance(logits, list):
            logits = torch.stack(logits)
            
        if not isinstance(logits, torch.Tensor):
            raise LogitsError(f"Expected Tensor/list, got {type(logits)}")
            
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        elif logits.dim() != 3:
            raise LogitsError(f"Logits must be 2D/3D (got {logits.dim()}D)")
            
        if not torch.isfinite(logits).all():
            raise LogitsError("Logits contain NaN/inf values")
            
        return logits.to(self.device)

    def calculate_confidence(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Batched confidence calculation
        Args:
            logits: Input logits (batch_size, seq_len, vocab_size)
            generated_ids: Optional mask for valid positions
        Returns:
            confidence scores (batch_size,)
        """
        with NumericalGuard():
            logits = self._validate_logits(logits)
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            
            if generated_ids is not None:
                mask = (generated_ids != -100).float().to(self.device)
                conf = safe_divide(
                    (max_probs * mask).sum(dim=1),
                    mask.sum(dim=1),
                    default=0.2
                )
            else:
                conf = max_probs.mean(dim=1)
                
            # Detect flat distributions
            low_conf = (max_probs.var(dim=-1) < 1e-5)
            conf[low_conf] = 0.2
            
            return conf.squeeze()
