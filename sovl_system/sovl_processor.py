import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Union, List, Optional, Dict, Any, Tuple
import torch
import traceback
from dataclasses import dataclass

# Assuming these are external utilities with the same functionality
from sovl_utils import NumericalGuard, safe_divide
from sovl_logger import Logger


class LogitsError(Exception):
    """Custom exception for logits processing failures."""
    pass


class EventType(Enum):
    """Enum for logging event types."""
    PROCESSOR_INIT = "processor_init"
    CONFIDENCE_CALC = "confidence_calculation"
    PROCESSOR_TUNE = "processor_tune"
    PROCESSOR_LOAD_STATE = "processor_load_state"
    PROCESSOR_RESET = "processor_reset"
    TOKEN_MAP_UPDATE = "token_map_updated"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ProcessorConfig:
    """Configuration for SOVLProcessor with validation."""
    flat_distribution_confidence: float = 0.2
    confidence_var_threshold: float = 1e-5
    confidence_smoothing_factor: float = 0.0
    max_confidence_history: int = 10

    _RANGES: Dict[str, Tuple[float, float]] = None
    _HISTORY_RANGES: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Initialize and validate configuration parameters."""
        self._RANGES = {
            "flat_distribution_confidence": (0.0, 0.5),
            "confidence_var_threshold": (1e-6, 1e-4),
            "confidence_smoothing_factor": (0.0, 1.0),
        }
        self._HISTORY_RANGES = {
            "max_confidence_history": (5, 20),
        }
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        for key, (min_val, max_val) in self._RANGES.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be in [{min_val}, {max_val}], got {value}")
        for key, (min_val, max_val) in self._HISTORY_RANGES.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be in [{min_val}, {max_val}], got {value}")

    def update(self, **kwargs) -> None:
        """Update configuration parameters with validation."""
        for key, value in kwargs.items():
            if key in self._RANGES:
                min_val, max_val = self._RANGES[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be in [{min_val}, {max_val}], got {value}")
            elif key in self._HISTORY_RANGES:
                min_val, max_val = self._HISTORY_RANGES[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be in [{min_val}, {max_val}], got {value}")
            else:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        self._validate_config()


class TensorValidator:
    """Handles tensor validation and conversion for logits and generated IDs."""
    
    def __init__(self, device: torch.device, logger: Logger):
        self.device = device
        self.logger = logger

    def validate_logits(self, logits: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Convert and validate logits to a 3D tensor (batch, seq_len, vocab_size).

        Args:
            logits: Input logits, single tensor or list of tensors.

        Returns:
            Validated 3D tensor on the specified device.

        Raises:
            LogitsError: If validation fails.
        """
        try:
            if isinstance(logits, list):
                logits = torch.stack(logits)
            if not isinstance(logits, torch.Tensor):
                raise LogitsError(f"Expected tensor/list, got {type(logits)}")
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            elif logits.dim() != 3:
                raise LogitsError(f"Logits must be 2D or 3D, got {logits.dim()}D")
            if not torch.isfinite(logits).all():
                raise LogitsError("Logits contain NaN or inf values")
            if logits.dtype not in (torch.float16, torch.float32, torch.float64):
                raise LogitsError(f"Logits must be float type, got {logits.dtype}")
            return logits.to(self.device)
        except Exception as e:
            self._log_error("Logits validation failed", str(e), logits=logits)
            raise LogitsError(f"Logits validation failed: {str(e)}")

    def validate_generated_ids(self, generated_ids: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Validate generated IDs against logits.

        Args:
            generated_ids: Optional tensor of generated IDs (batch, seq_len).
            logits: Reference logits tensor.

        Returns:
            Validated generated IDs tensor or None.

        Raises:
            LogitsError: If validation fails.
        """
        if generated_ids is None:
            return None
        try:
            if not isinstance(generated_ids, torch.Tensor) or generated_ids.dtype != torch.long:
                raise LogitsError("Generated IDs must be LongTensor")
            if generated_ids.dim() != 2 or generated_ids.shape[:2] != logits.shape[:2]:
                raise LogitsError("Generated IDs shape mismatch with logits")
            return generated_ids.to(self.device)
        except Exception as e:
            self._log_error(
                "Generated IDs validation failed", str(e),
                generated_ids=generated_ids, logits=logits
            )
            raise LogitsError(f"Generated IDs validation failed: {str(e)}")

    def _log_error(self, message: str, error: str, **kwargs) -> None:
        """Log validation errors with context."""
        log_data = {
            "event": EventType.ERROR.value,
            "error": f"{message}: {error}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        }
        for key, value in kwargs.items():
            log_data[f"{key}_shape"] = str(getattr(value, 'shape', 'N/A'))
        self.logger.record(log_data)


class SOVLProcessor:
    """Processes logits to calculate confidence with temperament and curiosity integration."""
    
    # Constants for adjustments
    TEMPERAMENT_SCALE: float = 0.1
    CURIOSITY_BOOST: float = 0.05
    MAX_CONFIDENCE: float = 1.0
    MIN_CONFIDENCE: float = 0.0
    PADDING_ID: int = -100

    def __init__(self, config: ProcessorConfig, logger: Logger, device: torch.device):
        """
        Initialize the processor.

        Args:
            config: Configuration instance.
            logger: Logger for event recording.
            device: Device for tensor operations.
        """
        self.config = config
        self.logger = logger
        self.device = device
        self._confidence_history: deque = deque(maxlen=config.max_confidence_history)
        self._lock = Lock()
        self._validator = TensorValidator(device, logger)
        self.scaffold_unk_id: Optional[int] = None
        self.token_map: Dict = {}
        self._log_init()

    def _log_init(self) -> None:
        """Log initialization event."""
        self.logger.record({
            "event": EventType.PROCESSOR_INIT.value,
            "config": vars(self.config),
            "device": str(self.device),
            "timestamp": time.time()
        })

    # Confidence Calculation Methods
    def calculate_confidence(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor] = None,
        temperament_influence: Optional[float] = None,
        curiosity_pressure: Optional[float] = None
    ) -> torch.Tensor:
        """
        Calculate batched confidence scores.

        Args:
            logits: Input logits (batch, seq_len, vocab_size).
            generated_ids: Optional mask for valid positions (batch, seq_len).
            temperament_influence: Temperament score (-1.0 to 1.0).
            curiosity_pressure: Curiosity pressure (0.0 to 1.0).

        Returns:
            Confidence scores (batch,).

        Raises:
            LogitsError: If processing fails.
        """
        try:
            with self._lock, NumericalGuard():
                return self._compute_confidence(
                    logits, generated_ids, temperament_influence, curiosity_pressure
                )
        except Exception as e:
            self._log_confidence_error(e, logits, generated_ids, temperament_influence, curiosity_pressure)
            raise LogitsError(f"Confidence calculation failed: {str(e)}")

    def _compute_confidence(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor],
        temperament_influence: Optional[float],
        curiosity_pressure: Optional[float]
    ) -> torch.Tensor:
        """Core confidence computation logic."""
        logits = self._validator.validate_logits(logits)
        generated_ids = self._validator.validate_generated_ids(generated_ids, logits)

        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        conf = self._aggregate_confidence(max_probs, generated_ids)
        conf = self._adjust_for_distribution(conf, max_probs)
        conf = self._apply_temperament(conf, temperament_influence)
        conf = self._apply_curiosity(conf, curiosity_pressure)
        conf = self._smooth_confidence(conf)
        self._update_history(conf)

        self._log_confidence(conf, logits, temperament_influence, curiosity_pressure)
        return conf.squeeze()

    def _aggregate_confidence(self, max_probs: torch.Tensor, generated_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Aggregate confidence scores."""
        if generated_ids is not None:
            mask = (generated_ids != self.PADDING_ID).float().to(self.device)
            return safe_divide(
                (max_probs * mask).sum(dim=1),
                mask.sum(dim=1),
                default=self.config.flat_distribution_confidence
            )
        return max_probs.mean(dim=1)

    def _adjust_for_distribution(self, conf: torch.Tensor, max_probs: torch.Tensor) -> torch.Tensor:
        """Adjust confidence for flat distributions."""
        low_conf = max_probs.var(dim=-1) < self.config.confidence_var_threshold
        conf = conf.clone()
        conf[low_conf] = self.config.flat_distribution_confidence
        return conf

    def _apply_temperament(self, conf: torch.Tensor, temperament: Optional[float]) -> torch.Tensor:
        """Apply temperament influence."""
        if temperament is not None:
            if not (-1.0 <= temperament <= 1.0):
                raise ValueError(f"Temperament must be in [-1.0, 1.0], got {temperament}")
            conf = conf * (1.0 + temperament * self.TEMPERAMENT_SCALE)
        return conf

    def _apply_curiosity(self, conf: torch.Tensor, curiosity: Optional[float]) -> torch.Tensor:
        """Apply curiosity pressure."""
        if curiosity is not None:
            if not (0.0 <= curiosity <= 1.0):
                raise ValueError(f"Curiosity must be in [0.0, 1.0], got {curiosity}")
            conf = conf + curiosity * self.CURIOSITY_BOOST
            conf = torch.clamp(conf, self.MIN_CONFIDENCE, self.MAX_CONFIDENCE)
        return conf

    def _smooth_confidence(self, conf: torch.Tensor) -> torch.Tensor:
        """Apply confidence smoothing based on history."""
        if self._confidence_history and self.config.confidence_smoothing_factor > 0:
            avg_hist_conf = torch.tensor(list(self._confidence_history), device=self.device).mean()
            conf = (1 - self.config.confidence_smoothing_factor) * conf + \
                   self.config.confidence_smoothing_factor * avg_hist_conf
        return conf

    def _update_history(self, conf: torch.Tensor) -> None:
        """Update confidence history."""
        if conf.dim() == 0:
            self._confidence_history.append(float(conf.item()))
        else:
            self._confidence_history.extend(conf.tolist())

    def _log_confidence(
        self,
        conf: torch.Tensor,
        logits: torch.Tensor,
        temperament: Optional[float],
        curiosity: Optional[float]
    ) -> None:
        """Log confidence calculation event."""
        self.logger.record({
            "event": EventType.CONFIDENCE_CALC.value,
            "confidence": conf.tolist(),
            "logits_shape": logits.shape,
            "temperament_influence": temperament,
            "curiosity_pressure": curiosity,
            "timestamp": time.time()
        })

    def _log_confidence_error(
        self,
        error: Exception,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor],
        temperament: Optional[float],
        curiosity: Optional[float]
    ) -> None:
        """Log confidence calculation errors."""
        self.logger.record({
            "event": EventType.ERROR.value,
            "error": f"Confidence calculation failed: {str(error)}",
            "logits_shape": str(getattr(logits, 'shape', 'N/A')),
            "generated_ids_shape": str(getattr(generated_ids, 'shape', 'N/A')),
            "temperament_influence": temperament,
            "curiosity_pressure": curiosity,
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })

    # Configuration and State Management
    def tune(self, **kwargs) -> None:
        """
        Update processor configuration.

        Args:
            **kwargs: Configuration parameters to update.
        """
        try:
            with self._lock:
                old_config = vars(self.config).copy()
                self.config.update(**kwargs)
                if "max_confidence_history" in kwargs:
                    self._confidence_history = deque(
                        self._confidence_history, maxlen=self.config.max_confidence_history
                    )
                self.logger.record({
                    "event": EventType.PROCESSOR_TUNE.value,
                    "old_config": old_config,
                    "new_config": vars(self.config),
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "event": EventType.ERROR.value,
                "error": f"Processor tuning failed: {str(e)}",
                "kwargs": kwargs,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def get_state(self) -> Dict[str, Any]:
        """Export processor state."""
        with self._lock:
            return {"confidence_history": list(self._confidence_history)}

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load processor state.

        Args:
            state: State dictionary.
        """
        try:
            with self._lock:
                self._confidence_history = deque(
                    state.get("confidence_history", []),
                    maxlen=self.config.max_confidence_history
                )
                self.logger.record({
                    "event": EventType.PROCESSOR_LOAD_STATE.value,
                    "state": state,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "event": EventType.ERROR.value,
                "error": f"Failed to load state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def reset(self) -> None:
        """Reset processor state."""
        with self._lock:
            self._confidence_history.clear()
            self.logger.record({
                "event": EventType.PROCESSOR_RESET.value,
                "timestamp": time.time()
            })

    # Token Mapping Methods
    def validate_and_map_tokens(self, base_ids: torch.Tensor, max_expanded_len: int, max_seq_length: int) -> torch.Tensor:
        """
        Map and validate token IDs.

        Args:
            base_ids: Input token IDs (batch, seq_len).
            max_expanded_len: Maximum expanded length.
            max_seq_length: Maximum sequence length.

        Returns:
            Mapped token IDs tensor.
        """
        batch_size, seq_len = base_ids.shape
        mapped_ids = torch.full(
            (batch_size, max_expanded_len),
            self.scaffold_unk_id,
            dtype=torch.long,
            device=self.device
        )

        for batch_idx in range(batch_size):
            position = 0
            truncated = False

            for base_id in base_ids[batch_idx]:
                if base_id == self.PADDING_ID:
                    continue

                mapped_tokens = self._get_mapped_tokens(base_id)
                if position + len(mapped_tokens) > max_expanded_len:
                    truncated = True
                    break

                for token in mapped_tokens:
                    if position >= max_expanded_len:
                        truncated = True
                        break
                    mapped_ids[batch_idx, position] = token
                    position += 1

                if truncated:
                    break

            if truncated:
                self.logger.record({
                    "event": EventType.WARNING.value,
                    "warning": f"Token mapping truncated to {max_expanded_len}",
                    "original_length": seq_len,
                    "allowed_length": max_expanded_len,
                    "timestamp": time.time()
                })

        return mapped_ids[:, :min(max_expanded_len, max_seq_length)]

    def _get_mapped_tokens(self, base_id: torch.Tensor) -> List[int]:
        """Get mapped tokens for a base ID with error handling."""
        try:
            mapped_entry = self.token_map.get(int(base_id.item()), [self.scaffold_unk_id])
            return mapped_entry['ids'] if isinstance(mapped_entry, dict) else mapped_entry
        except Exception as e:
            self.logger.record({
                "event": EventType.WARNING.value,
                "warning": f"Token mapping error for ID {base_id}: {str(e)}",
                "timestamp": time.time()
            })
            return [self.scaffold_unk_id]

    def set_token_map(self, token_map: Dict, scaffold_unk_id: int) -> None:
        """
        Set token mapping and unknown token ID.

        Args:
            token_map: Mapping of base to scaffold tokens.
            scaffold_unk_id: Unknown token ID.
        """
        with self._lock:
            self.token_map = token_map
            self.scaffold_unk_id = scaffold_unk_id

    def update_token_map_memory(self, prompt: str, confidence: float, tokenizer: Any, memory_decay_rate: float = 0.95) -> None:
        """
        Update token map weights.

        Args:
            prompt: Input prompt text.
            confidence: Confidence score.
            tokenizer: Tokenizer instance.
            memory_decay_rate: Memory decay rate.
        """
        with self._lock:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            for token_id in tokens:
                if token_id in self.token_map:
                    self.token_map[token_id]['weight'] = min(
                        self.token_map[token_id]['weight'] + confidence * 0.1,
                        2.0
                    )
            for token_id in self.token_map:
                self.token_map[token_id]['weight'] *= memory_decay_rate

            self.logger.record({
                "event": EventType.TOKEN_MAP_UPDATE.value,
                "prompt_length": len(prompt),
                "confidence": confidence,
                "timestamp": time.time()
            })
