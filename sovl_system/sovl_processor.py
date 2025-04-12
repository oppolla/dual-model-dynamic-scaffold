import torch
from typing import Union, List, Optional
from sovl_utils import NumericalGuard, safe_divide

class LogitsError(Exception):
    """Custom exceptions for processing failures"""
    pass

class SOVLProcessor:
    FLAT_DISTRIBUTION_CONFIDENCE = 0.2

    def __init__(self, device: Optional[str] = None):
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'

    def _validate_logits(self, logits: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Convert input to 3D tensor (batch, seq_len, vocab_size) with validation.

        Args:
            logits: Input logits, can be a single Tensor or a list of Tensors.

        Returns:
            A 3D Tensor (batch_size, seq_len, vocab_size) on the specified device.

        Raises:
            LogitsError: If the input is not a Tensor or list of Tensors, or if the
                         dimensions are incorrect, or if the Tensor contains NaN/inf values,
                         or if the Tensor is not a floating-point type.
        """
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

        if logits.dtype not in [torch.float16, torch.float32, torch.float64]:
            raise LogitsError(f"Logits must be float type, got {logits.dtype}")

        return logits.to(self.device)

    def _validate_generated_ids(self, generated_ids: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Validates generated_ids against logits for shape and type.

        Args:
            generated_ids: Optional mask for valid positions.
            logits: Input logits Tensor.

        Returns:
            A Tensor of generated_ids on the specified device or None.

        Raises:
            LogitsError: If generated_ids is not a LongTensor or if its shape
                         mismatches with logits.
        """
        if generated_ids is None:
            return None

        if not isinstance(generated_ids, torch.Tensor) or generated_ids.dtype != torch.long:
            raise LogitsError("generated_ids must be a LongTensor")

        if generated_ids.dim() != 2 or generated_ids.shape[:2] != logits.shape[:2]:
            raise LogitsError("generated_ids shape mismatch with logits")

        return generated_ids.to(self.device)

    def calculate_confidence(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batched confidence calculation.

        Args:
            logits: Input logits (batch_size, seq_len, vocab_size).
            generated_ids: Optional mask for valid positions (batch_size, seq_len).

        Returns:
            confidence scores (batch_size,).
        """
        with NumericalGuard():
            logits = self._validate_logits(logits)
            generated_ids = self._validate_generated_ids(generated_ids, logits)

            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values

            if generated_ids is not None:
                mask = (generated_ids != -100).float().to(self.device)
                conf = safe_divide(
                    (max_probs * mask).sum(dim=1),
                    mask.sum(dim=1),
                    default=SOVLProcessor.FLAT_DISTRIBUTION_CONFIDENCE
                )
            else:
                conf = max_probs.mean(dim=1)

            # Detect flat distributions and assign a default low confidence.
            low_conf = (max_probs.var(dim=-1) < 1e-5)
            conf[low_conf] = SOVLProcessor.FLAT_DISTRIBUTION_CONFIDENCE

            return conf.squeeze()
