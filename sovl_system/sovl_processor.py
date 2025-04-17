import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Union, List, Optional, Dict, Any, Tuple, Set
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
    # Repetition detection configuration
    min_rep_length: int = 3
    max_rep_scan: int = 100
    rep_confidence_penalty: float = 0.3
    enable_rep_detection: bool = True

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ProcessorConfig':
        """Create a ProcessorConfig instance from ConfigManager."""
        processor_config = config_manager.get_section("processor_config", {})
        return cls(
            flat_distribution_confidence=processor_config.get("flat_distribution_confidence", 0.2),
            confidence_var_threshold=processor_config.get("confidence_var_threshold", 1e-5),
            confidence_smoothing_factor=processor_config.get("confidence_smoothing_factor", 0.0),
            max_confidence_history=processor_config.get("max_confidence_history", 10),
            min_rep_length=processor_config.get("min_rep_length", 3),
            max_rep_scan=processor_config.get("max_rep_scan", 100),
            rep_confidence_penalty=processor_config.get("rep_confidence_penalty", 0.3),
            enable_rep_detection=processor_config.get("enable_rep_detection", True)
        )

    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        try:
            # Validate confidence parameters
            if not (0.0 <= self.flat_distribution_confidence <= 0.5):
                raise ValueError(f"flat_distribution_confidence must be in [0.0, 0.5], got {self.flat_distribution_confidence}")
            if not (1e-6 <= self.confidence_var_threshold <= 1e-4):
                raise ValueError(f"confidence_var_threshold must be in [1e-6, 1e-4], got {self.confidence_var_threshold}")
            if not (0.0 <= self.confidence_smoothing_factor <= 1.0):
                raise ValueError(f"confidence_smoothing_factor must be in [0.0, 1.0], got {self.confidence_smoothing_factor}")
            if not (5 <= self.max_confidence_history <= 20):
                raise ValueError(f"max_confidence_history must be in [5, 20], got {self.max_confidence_history}")

            # Validate repetition detection parameters
            if not (2 <= self.min_rep_length <= 10):
                raise ValueError(f"min_rep_length must be in [2, 10], got {self.min_rep_length}")
            if not (50 <= self.max_rep_scan <= 200):
                raise ValueError(f"max_rep_scan must be in [50, 200], got {self.max_rep_scan}")
            if not (0.0 <= self.rep_confidence_penalty <= 1.0):
                raise ValueError(f"rep_confidence_penalty must be in [0.0, 1.0], got {self.rep_confidence_penalty}")
            if not isinstance(self.enable_rep_detection, bool):
                raise ValueError(f"enable_rep_detection must be boolean, got {type(self.enable_rep_detection)}")

        except ValueError as e:
            raise ConfigurationError(f"Invalid processor configuration: {str(e)}")

    def update(self, config_manager: ConfigManager, **kwargs) -> None:
        """Update configuration parameters with validation."""
        try:
            # Validate new values before updating
            for key, value in kwargs.items():
                if key == "flat_distribution_confidence" and not (0.0 <= value <= 0.5):
                    raise ValueError(f"flat_distribution_confidence must be in [0.0, 0.5], got {value}")
                elif key == "confidence_var_threshold" and not (1e-6 <= value <= 1e-4):
                    raise ValueError(f"confidence_var_threshold must be in [1e-6, 1e-4], got {value}")
                elif key == "confidence_smoothing_factor" and not (0.0 <= value <= 1.0):
                    raise ValueError(f"confidence_smoothing_factor must be in [0.0, 1.0], got {value}")
                elif key == "max_confidence_history" and not (5 <= value <= 20):
                    raise ValueError(f"max_confidence_history must be in [5, 20], got {value}")
                elif key == "min_rep_length" and not (2 <= value <= 10):
                    raise ValueError(f"min_rep_length must be in [2, 10], got {value}")
                elif key == "max_rep_scan" and not (50 <= value <= 200):
                    raise ValueError(f"max_rep_scan must be in [50, 200], got {value}")
                elif key == "rep_confidence_penalty" and not (0.0 <= value <= 1.0):
                    raise ValueError(f"rep_confidence_penalty must be in [0.0, 1.0], got {value}")
                elif key == "enable_rep_detection" and not isinstance(value, bool):
                    raise ValueError(f"enable_rep_detection must be boolean, got {type(value)}")

            # Update local config
            for key, value in kwargs.items():
                setattr(self, key, value)

            # Update config in ConfigManager
            processor_config = config_manager.get_section("processor_config", {})
            processor_config.update(kwargs)
            config_manager.update_section("processor_config", processor_config)

        except ValueError as e:
            raise ConfigurationError(f"Invalid processor configuration update: {str(e)}")


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
        self.logger.log_error(
            error_msg=f"{message}: {error}",
            error_type="validation_error",
            stack_trace=traceback.format_exc(),
            additional_info={f"{key}_shape": str(getattr(value, 'shape', 'N/A')) for key, value in kwargs.items()}
        )


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
        self.config = ProcessorConfig.from_config_manager(config_manager)
        
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
                self.logger.log_training_event(
                    event_type="token_mapping_warning",
                    message=f"Invalid scaffold_unk_id: {self.scaffold_unk_id}, using default 0",
                    level="warning"
                )
                self.scaffold_unk_id = 0
            
            # Initialize token map with validation
            self.token_map = token_config.get("token_map", {})
            if not isinstance(self.token_map, dict):
                self.logger.log_training_event(
                    event_type="token_mapping_warning",
                    message="Invalid token_map type, using empty dict",
                    level="warning"
                )
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
            self.logger.log_training_event(
                event_type="token_mapping_initialized",
                message="Token mapping initialized successfully",
                level="info",
                additional_info={
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map_size": len(self.token_map)
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_type="token_mapping_error",
                message="Failed to initialize token mapping",
                error=str(e),
                stack_trace=traceback.format_exc(),
                additional_info={
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map_size": len(self.token_map)
                }
            )
            # Use safe defaults
            self.scaffold_unk_id = 0
            self.token_map = {}

    def _log_init(self) -> None:
        """Log initialization event."""
        self.logger.log_training_event(
            event_type="processor_initialized",
            message="Processor initialized successfully",
            level="info",
            additional_info={
                "config": vars(self.config),
                "device": str(self.device),
                "token_mapping": {
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map_size": len(self.token_map)
                }
            }
        )

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
        
        # Apply repetition penalty if enabled
        if self.config.enable_rep_detection and generated_ids is not None:
            rep_indices = self.detect_repetitions(generated_ids)
            if rep_indices is not None:
                conf = conf * (1.0 - self.config.rep_confidence_penalty)
        
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
        self.logger.record_event(
            event_type="confidence_calculated",
            message="Confidence calculation completed",
            level="info",
            additional_info={
                "confidence": conf.tolist(),
                "logits_shape": logits.shape,
                "temperament_influence": temperament,
                "curiosity_pressure": curiosity
            }
        )

    def _log_confidence_error(
        self,
        error: Exception,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor],
        temperament: Optional[float],
        curiosity: Optional[float]
    ) -> None:
        """Log confidence calculation errors."""
        self.logger.log_error(
            error_msg="Confidence calculation failed",
            error_type="confidence_error",
            stack_trace=traceback.format_exc(),
            additional_info={
                "logits_shape": str(getattr(logits, 'shape', 'N/A')),
                "generated_ids_shape": str(getattr(generated_ids, 'shape', 'N/A')),
                "temperament_influence": temperament,
                "curiosity_pressure": curiosity
            }
        )

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
                self.config.update(self.config_manager, **kwargs)
                
                self.logger.log_training_event(
                    event_type="processor_tuned",
                    message="Processor configuration updated",
                    level="info",
                    additional_info={
                        "old_config": old_config,
                        "new_config": vars(self.config)
                    }
                )
        except Exception as e:
            self.logger.log_error(
                error_type="tuning_error",
                message="Processor tuning failed",
                error=str(e),
                stack_trace=traceback.format_exc(),
                additional_info={"kwargs": kwargs}
            )
            raise

    def get_state(self) -> Dict[str, Any]:
        """Export processor state."""
        with self._lock:
            return {
                "config": vars(self.config),
                "token_mapping": {
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map": self.token_map
                }
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load processor state.

        Args:
            state: State dictionary.
        """
        try:
            with self._lock:
                if "config" in state:
                    self.config.update(self.config_manager, **state["config"])
                if "token_mapping" in state:
                    self.scaffold_unk_id = state["token_mapping"].get("scaffold_unk_id", 0)
                    self.token_map = state["token_mapping"].get("token_map", {})
                
                self.logger.log_training_event(
                    event_type="state_loaded",
                    message="Processor state loaded",
                    level="info",
                    additional_info={"state": state}
                )
        except Exception as e:
            self.logger.log_error(
                error_type="state_error",
                message="Failed to load processor state",
                error=str(e),
                stack_trace=traceback.format_exc()
            )
            raise

    def reset(self) -> None:
        """Reset processor state."""
        with self._lock:
            # Reset to default configuration
            self.config = ProcessorConfig.from_config_manager(self.config_manager)
            self.scaffold_unk_id = 0
            self.token_map = {}
            
            self.logger.log_training_event(
                event_type="processor_reset",
                message="Processor state reset",
                level="info"
            )

    def detect_repetitions(
        self,
        token_ids: Union[List[int], torch.Tensor],
        special_ids: Optional[Set[int]] = None,
        min_rep_length: Optional[int] = None,
        max_scan: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Detect repeating token sequences with optimized batch processing.
        
        Args:
            token_ids: List of token IDs or tensor of shape (batch_size, seq_len)
            special_ids: Set of special token IDs to ignore
            min_rep_length: Minimum sequence length to check
            max_scan: Maximum number of tokens to scan
            batch_size: Optional batch size for processing
            
        Returns:
            (start_idx, end_idx) of first repetition found or None
        """
        try:
            with self._lock:
                # Use config values if not specified
                if min_rep_length is None:
                    min_rep_length = self.config.min_rep_length
                if max_scan is None:
                    max_scan = self.config.max_rep_scan
                if special_ids is None:
                    special_ids = {
                        self.base_tokenizer.pad_token_id,
                        self.base_tokenizer.eos_token_id,
                        self.base_tokenizer.bos_token_id,
                        self.base_tokenizer.unk_token_id
                    }
                
                # Convert to tensor if needed
                if isinstance(token_ids, list):
                    token_ids = torch.tensor(token_ids, device=self.device)
                
                # Handle batch processing
                if token_ids.dim() == 2:
                    if batch_size is None:
                        batch_size = token_ids.size(0)
                    
                    for i in range(0, token_ids.size(0), batch_size):
                        batch = token_ids[i:i + batch_size]
                        result = self._detect_repetitions_batch(
                            batch, special_ids, min_rep_length, max_scan
                        )
                        if result is not None:
                            return result
                    return None
                
                # Single sequence processing
                return self._detect_repetitions_single(
                    token_ids, special_ids, min_rep_length, max_scan
                )
                
        except Exception as e:
            self.logger.log_error(
                error_msg="Repetition detection failed",
                error_type="detection_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "token_ids_shape": str(getattr(token_ids, 'shape', 'N/A')),
                    "min_rep_length": min_rep_length,
                    "max_scan": max_scan,
                    "batch_size": batch_size
                }
            )
            return None

    def _detect_repetitions_batch(
        self,
        token_ids: torch.Tensor,
        special_ids: Set[int],
        min_rep_length: int,
        max_scan: int
    ) -> Optional[Tuple[int, int]]:
        """Detect repetitions in a batch of sequences."""
        # Create mask for special tokens
        special_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for sid in special_ids:
            special_mask |= (token_ids == sid)
        
        # Filter out special tokens
        filtered = token_ids[~special_mask]
        
        # Process each sequence in the batch
        for i in range(filtered.size(0)):
            result = self._detect_repetitions_single(
                filtered[i], special_ids, min_rep_length, max_scan
            )
            if result is not None:
                return result
        
        return None

    def _detect_repetitions_single(
        self,
        token_ids: torch.Tensor,
        special_ids: Set[int],
        min_rep_length: int,
        max_scan: int
    ) -> Optional[Tuple[int, int]]:
        """Detect repetitions in a single sequence."""
        # Convert to list for processing
        ids = token_ids.tolist()
        filtered = [i for i in ids if i not in special_ids]
        scan_range = min(len(filtered), max_scan)
        
        # Use sliding window with early stopping
        for i in range(scan_range - 2 * min_rep_length + 1):
            window = filtered[i:i + min_rep_length]
            next_window = filtered[i + min_rep_length:i + 2 * min_rep_length]
            
            if window == next_window:
                self.logger.record({
                    "event": EventType.WARNING.value,
                    "message": "Repetition detected",
                    "start_idx": i,
                    "end_idx": i + min_rep_length,
                    "length": min_rep_length,
                    "timestamp": time.time()
                })
                return (i, i + min_rep_length)
        
        return None
    
class SoulLogitsProcessor(LogitsProcessor):
    """Boosts token probabilities for .soul file keywords during generation."""
    
    def __init__(self, soul_keywords: Dict[str, float], tokenizer: PreTrainedTokenizer):
        self.soul_keywords = soul_keywords
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        try:
            for keyword, weight in self.soul_keywords.items():
                token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
                for token_id in token_ids:
                    scores[:, token_id] += weight * 2.0  # Hypersensitive boost
            return scores
        except Exception as e:
            # Log error but don't interrupt generation
            self.logger.log_error(
                error_msg=f"Failed to apply soul logits processing: {str(e)}",
                error_type="soul_logits_error",
                stack_trace=traceback.format_exc(),
                additional_info={"keywords": self.soul_keywords}
            )
            return scores
