import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Union, List, Optional, Dict, Any, Tuple
import torch
import traceback
from dataclasses import dataclass
from sovl_utils import NumericalGuard, safe_divide
from sovl_logger import Logger
from sovl_config import ConfigManager


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
        self._valid_dtypes = (torch.float16, torch.float32, torch.float64)
        self._valid_dims = (2, 3)
        self._clamp_range = (-100.0, 100.0)  # Reasonable range for logits

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
            # Fast path for common case: single tensor on correct device
            if isinstance(logits, torch.Tensor) and logits.device == self.device:
                if logits.dim() in self._valid_dims and logits.dtype in self._valid_dtypes:
                    return self._handle_nan_inf(logits)
                
            # Handle list of tensors
            if isinstance(logits, list):
                logits = torch.stack(logits)
                
            # Type check
            if not isinstance(logits, torch.Tensor):
                raise LogitsError(f"Expected tensor/list, got {type(logits)}")
                
            # Dimension check
            if logits.dim() not in self._valid_dims:
                raise LogitsError(f"Logits must be 2D or 3D, got {logits.dim()}D")
                
            # Dtype check
            if logits.dtype not in self._valid_dtypes:
                raise LogitsError(f"Logits must be float type, got {logits.dtype}")
                
            # Move to device and handle NaN/Inf
            logits = logits.to(self.device)
            return self._handle_nan_inf(logits)
            
        except Exception as e:
            self._log_error("Logits validation failed", str(e), logits=logits)
            raise LogitsError(f"Logits validation failed: {str(e)}")

    def _handle_nan_inf(self, logits: torch.Tensor) -> torch.Tensor:
        """Handle NaN and Inf values in logits with logging and fallback strategy."""
        if not torch.isfinite(logits).all():
            # Count problematic values
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            total_elements = logits.numel()
            
            # Log the issue
            self.logger.record({
                "event": EventType.WARNING.value,
                "message": "Logits contain NaN/Inf values",
                "timestamp": time.time(),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "total_elements": total_elements,
                "nan_percentage": (nan_count / total_elements) * 100,
                "inf_percentage": (inf_count / total_elements) * 100
            })
            
            # Apply fallback strategy: clamp values and replace NaN with 0
            logits = torch.nan_to_num(logits, nan=0.0, posinf=self._clamp_range[1], neginf=self._clamp_range[0])
            logits = torch.clamp(logits, *self._clamp_range)
            
        return logits

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
            # Fast path for common case: tensor on correct device with matching shape
            if (isinstance(generated_ids, torch.Tensor) and 
                generated_ids.device == self.device and 
                generated_ids.dtype == torch.long and
                generated_ids.dim() == 2 and 
                generated_ids.shape[:2] == logits.shape[:2]):
                return generated_ids
                
            # Basic validation
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

    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """
        Initialize the processor.

        Args:
            config_manager: Configuration manager instance.
            logger: Logger for event recording.
            device: Device for tensor operations.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        
        # Load processor configuration
        processor_config = self.config_manager.get_section("processor_config", {})
        self.config = ProcessorConfig(
            flat_distribution_confidence=processor_config.get("flat_distribution_confidence", 0.2),
            confidence_var_threshold=processor_config.get("confidence_var_threshold", 1e-5),
            confidence_smoothing_factor=processor_config.get("confidence_smoothing_factor", 0.0),
            max_confidence_history=processor_config.get("max_confidence_history", 10)
        )
        
        # Initialize token mapping components
        self._initialize_token_mapping()
        
        self._lock = Lock()
        self._validator = TensorValidator(device, logger)
        self._log_init()

    def _initialize_token_mapping(self) -> None:
        """Initialize token mapping components with validation."""
        try:
            # Get token mapping configuration
            token_config = self.config_manager.get_section("token_config", {})
            
            # Initialize scaffold_unk_id with validation
            self.scaffold_unk_id = token_config.get("scaffold_unk_id", 0)
            if not isinstance(self.scaffold_unk_id, int) or self.scaffold_unk_id < 0:
                self.logger.record({
                    "event": EventType.WARNING.value,
                    "message": f"Invalid scaffold_unk_id: {self.scaffold_unk_id}, using default 0",
                    "timestamp": time.time()
                })
                self.scaffold_unk_id = 0
            
            # Initialize token map with validation
            self.token_map = token_config.get("token_map", {})
            if not isinstance(self.token_map, dict):
                self.logger.record({
                    "event": EventType.WARNING.value,
                    "message": "Invalid token_map type, using empty dict",
                    "timestamp": time.time()
                })
                self.token_map = {}
            
            # Validate token map entries
            valid_entries = {}
            for base_id, scaffold_ids in self.token_map.items():
                try:
                    # Ensure base_id is a valid integer
                    base_id = int(base_id)
                    if base_id < 0:
                        continue
                        
                    # Ensure scaffold_ids is a list of valid integers
                    if not isinstance(scaffold_ids, (list, tuple)):
                        continue
                        
                    valid_scaffold_ids = []
                    for sid in scaffold_ids:
                        try:
                            sid = int(sid)
                            if sid >= 0:
                                valid_scaffold_ids.append(sid)
                        except (ValueError, TypeError):
                            continue
                            
                    if valid_scaffold_ids:
                        valid_entries[base_id] = valid_scaffold_ids
                        
                except (ValueError, TypeError):
                    continue
                    
            self.token_map = valid_entries
            
            # Log initialization
            self.logger.record({
                "event": EventType.PROCESSOR_INIT.value,
                "message": "Token mapping initialized",
                "timestamp": time.time(),
                "scaffold_unk_id": self.scaffold_unk_id,
                "token_map_size": len(self.token_map)
            })
            
        except Exception as e:
            self.logger.record({
                "event": EventType.ERROR.value,
                "message": f"Failed to initialize token mapping: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            # Use safe defaults
            self.scaffold_unk_id = 0
            self.token_map = {}

    def _log_init(self) -> None:
        """Log initialization event."""
        self.logger.record({
            "event": EventType.PROCESSOR_INIT.value,
            "config": vars(self.config),
            "device": str(self.device),
            "token_mapping": {
                "scaffold_unk_id": self.scaffold_unk_id,
                "token_map_size": len(self.token_map)
            },
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
        # Note: Confidence history is now managed by SOVLState
        return conf

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
                
                # Update config in config_manager
                processor_config = self.config_manager.get_section("processor_config", {})
                processor_config.update(kwargs)
                self.config_manager.update_section("processor_config", processor_config)
                
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
            return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load processor state.

        Args:
            state: State dictionary.
        """
        try:
            with self._lock:
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
            self.logger.record({
                "event": EventType.PROCESSOR_RESET.value,
                "timestamp": time.time()
            })
