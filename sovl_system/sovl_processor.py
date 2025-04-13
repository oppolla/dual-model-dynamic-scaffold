import torch
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
from threading import Lock
import traceback
from sovl_utils import NumericalGuard, safe_divide
from sovl_logger import Logger

class LogitsError(Exception):
    """Custom exception for logits processing failures."""
    pass

@dataclass
class ProcessorConfig:
    """Configuration for SOVLProcessor, aligned with ConfigManager."""
    flat_distribution_confidence: float = 0.2
    confidence_var_threshold: float = 1e-5
    confidence_smoothing_factor: float = 0.0
    max_confidence_history: int = 10
    _ranges: Dict[str, Tuple[float, float]] = None
    _history_ranges: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        self._ranges = {
            "flat_distribution_confidence": (0.0, 0.5),
            "confidence_var_threshold": (1e-6, 1e-4),
            "confidence_smoothing_factor": (0.0, 1.0),
        }
        self._history_ranges = {
            "max_confidence_history": (5, 20),
        }
        for key, (min_val, max_val) in self._ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
        for key, (min_val, max_val) in self._history_ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")

    def update(self, **kwargs) -> None:
        """Dynamically update configuration parameters with validation."""
        for key, value in kwargs.items():
            if key in self._ranges:
                min_val, max_val = self._ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            elif key in self._history_ranges:
                min_val, max_val = self._history_ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)

class SOVLProcessor:
    """Processes logits to calculate confidence, integrated with SOVLSystem architecture."""
    def __init__(self, config: ProcessorConfig, logger: Logger, device: torch.device):
        """
        Initialize SOVLProcessor.

        Args:
            config: Processor configuration
            logger: Logger instance for recording events
            device: Device for tensor operations
        """
        self.config = config
        self.logger = logger
        self.device = device
        self._confidence_history: Deque[float] = deque(maxlen=config.max_confidence_history)
        self._lock = Lock()
        self._initialize_logging()
        self.scaffold_unk_id = None  # Should be set after initialization
        self.token_map = {}  # Should be populated with token mapping data

    def _initialize_logging(self) -> None:
        """Initialize logging with system startup event."""
        self.logger.record({
            "event": "processor_init",
            "config": vars(self.config),
            "device": str(self.device),
            "timestamp": time.time()
        })

    def _validate_logits(self, logits: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Convert input to 3D tensor (batch, seq_len, vocab_size) with validation.

        Args:
            logits: Input logits, can be a single Tensor or a list of Tensors.

        Returns:
            A 3D Tensor (batch_size, seq_len, vocab_size) on the specified device.

        Raises:
            LogitsError: If input validation fails.
        """
        try:
            with self._lock:
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
        except Exception as e:
            self.logger.record({
                "error": f"Logits validation failed: {str(e)}",
                "logits_shape": str(getattr(logits, 'shape', 'N/A')),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise LogitsError(f"Logits validation failed: {str(e)}")

    def _validate_generated_ids(self, generated_ids: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Validates generated_ids against logits for shape and type.

        Args:
            generated_ids: Optional mask for valid positions.
            logits: Input logits Tensor.

        Returns:
            A Tensor of generated_ids on the specified device or None.

        Raises:
            LogitsError: If generated_ids validation fails.
        """
        try:
            with self._lock:
                if generated_ids is None:
                    return None

                if not isinstance(generated_ids, torch.Tensor) or generated_ids.dtype != torch.long:
                    raise LogitsError("generated_ids must be a LongTensor")

                if generated_ids.dim() != 2 or generated_ids.shape[:2] != logits.shape[:2]:
                    raise LogitsError("generated_ids shape mismatch with logits")

                return generated_ids.to(self.device)
        except Exception as e:
            self.logger.record({
                "error": f"Generated IDs validation failed: {str(e)}",
                "generated_ids_shape": str(getattr(generated_ids, 'shape', 'N/A')),
                "logits_shape": str(logits.shape),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise LogitsError(f"Generated IDs validation failed: {str(e)}")

    def calculate_confidence(
        self,
        logits: Union[torch.Tensor, List[torch.Tensor]],
        generated_ids: Optional[torch.Tensor] = None,
        temperament_influence: Optional[float] = None,
        curiosity_pressure: Optional[float] = None
    ) -> torch.Tensor:
        """
        Batched confidence calculation with temperament and curiosity integration.

        Args:
            logits: Input logits (batch_size, seq_len, vocab_size).
            generated_ids: Optional mask for valid positions (batch_size, seq_len).
            temperament_influence: Optional temperament score to adjust confidence (-1.0 to 1.0).
            curiosity_pressure: Optional curiosity pressure to boost confidence (0.0 to 1.0).

        Returns:
            Confidence scores (batch_size,).
        """
        try:
            with self._lock:
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
                            default=self.config.flat_distribution_confidence
                        )
                    else:
                        conf = max_probs.mean(dim=1)

                    # Detect flat distributions
                    low_conf = (max_probs.var(dim=-1) < self.config.confidence_var_threshold)
                    conf[low_conf] = self.config.flat_distribution_confidence

                    # Apply temperament influence
                    if temperament_influence is not None:
                        if not (-1.0 <= temperament_influence <= 1.0):
                            raise ValueError(f"Temperament influence must be between -1.0 and 1.0, got {temperament_influence}")
                        conf = conf * (1.0 + temperament_influence * 0.1)  # Scale confidence slightly

                    # Apply curiosity pressure
                    if curiosity_pressure is not None:
                        if not (0.0 <= curiosity_pressure <= 1.0):
                            raise ValueError(f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}")
                        conf = conf + curiosity_pressure * 0.05  # Slight boost for curiosity
                        conf = torch.clamp(conf, 0.0, 1.0)

                    # Apply smoothing
                    if self._confidence_history and self.config.confidence_smoothing_factor > 0:
                        avg_hist_conf = torch.tensor(list(self._confidence_history), device=self.device).mean()
                        conf = (1 - self.config.confidence_smoothing_factor) * conf + \
                               self.config.confidence_smoothing_factor * avg_hist_conf

                    # Update history
                    if conf.dim() == 0:
                        self._confidence_history.append(float(conf.item()))
                    else:
                        self._confidence_history.extend(conf.tolist())

                    # Log confidence calculation
                    self.logger.record({
                        "event": "confidence_calculation",
                        "confidence": conf.tolist(),
                        "logits_shape": logits.shape,
                        "temperament_influence": temperament_influence,
                        "curiosity_pressure": curiosity_pressure,
                        "timestamp": time.time()
                    })

                    return conf.squeeze()

        except Exception as e:
            self.logger.record({
                "error": f"Confidence calculation failed: {str(e)}",
                "logits_shape": str(getattr(logits, 'shape', 'N/A')),
                "generated_ids_shape": str(getattr(generated_ids, 'shape', 'N/A')),
                "temperament_influence": temperament_influence,
                "curiosity_pressure": curiosity_pressure,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise LogitsError(f"Confidence calculation failed: {str(e)}")

    def tune(self, **kwargs) -> None:
        """
        Dynamically tune processor configuration parameters.

        Args:
            **kwargs: Parameters to update (e.g., flat_distribution_confidence)
        """
        try:
            with self._lock:
                old_config = vars(self.config).copy()
                self.config.update(**kwargs)
                if "max_confidence_history" in kwargs:
                    self._confidence_history = deque(self._confidence_history, maxlen=self.config.max_confidence_history)
                self.logger.record({
                    "event": "processor_tune",
                    "old_config": old_config,
                    "new_config": vars(self.config),
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Processor tuning failed: {str(e)}",
                "kwargs": kwargs,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def get_state(self) -> Dict[str, Any]:
        """
        Export current state for serialization.

        Returns:
            Dictionary containing processor state
        """
        with self._lock:
            return {
                "confidence_history": list(self._confidence_history)
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from serialized data.

        Args:
            state: Dictionary containing processor state
        """
        try:
            with self._lock:
                self._confidence_history = deque(
                    state.get("confidence_history", []),
                    maxlen=self.config.max_confidence_history
                )
                self.logger.record({
                    "event": "processor_load_state",
                    "state": state,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load processor state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def reset(self) -> None:
        """
        Reset processor state.
        """
        with self._lock:
            self._confidence_history.clear()
            self.logger.record({
                "event": "processor_reset",
                "timestamp": time.time()
            })

    def validate_and_map_tokens(self, base_ids: torch.Tensor, max_expanded_len: int, max_seq_length: int) -> torch.Tensor:
        """
        Validates and maps token IDs with proper error handling and logging.
        
        Args:
            base_ids: Input tensor of token IDs (batch_size, seq_len)
            max_expanded_len: Maximum length after expansion
            max_seq_length: Maximum allowed sequence length
            
        Returns:
            Mapped token IDs tensor
        """
        batch_size, seq_len = base_ids.shape
        mapped_ids = torch.full((batch_size, max_expanded_len), 
                              self.scaffold_unk_id, 
                              dtype=torch.long,
                              device=self.device)
        
        for batch_idx in range(batch_size):
            position = 0
            truncated = False
            
            for base_id_item in base_ids[batch_idx]:
                if base_id_item == -100:  # Skip padding
                    continue
                    
                try:
                    mapped_entry = self.token_map.get(base_id_item, [self.scaffold_unk_id])
                    mapped_tokens = mapped_entry['ids'] if isinstance(mapped_entry, dict) else mapped_entry
                except Exception as e:
                    self.logger.record({
                        "warning": f"Token mapping error for ID {base_id_item}: {str(e)}",
                        "timestamp": time.time()
                    })
                    mapped_tokens = [self.scaffold_unk_id]

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
                    "warning": f"Token mapping truncated to {max_expanded_len}",
                    "original_length": seq_len,
                    "allowed_length": max_expanded_len,
                    "timestamp": time.time()
                })

        return mapped_ids[:, :min(max_expanded_len, max_seq_length)]

    def set_token_map(self, token_map: Dict, scaffold_unk_id: int) -> None:
        """
        Sets the token mapping dictionary and UNK token ID.
        
        Args:
            token_map: Dictionary mapping base tokens to scaffold tokens
            scaffold_unk_id: Unknown token ID for scaffold model
        """
        with self._lock:
            self.token_map = token_map
            self.scaffold_unk_id = scaffold_unk_id

    def update_token_map_memory(self, prompt: str, confidence: float, tokenizer, memory_decay_rate: float = 0.95) -> None:
        """
        Update token map weights based on prompt and confidence.
        
        Args:
            prompt: Input prompt text
            confidence: Confidence score
            tokenizer: Tokenizer to use for encoding
            memory_decay_rate: Rate at which memory decays
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
                "event": "token_map_updated",
                "prompt_length": len(prompt),
                "confidence": confidence,
                "timestamp": time.time()
            })
