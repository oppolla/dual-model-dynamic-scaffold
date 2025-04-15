import time
from typing import Deque, Optional, Dict, Any
from collections import deque
import torch
from dataclasses import dataclass
import traceback
import uuid
from threading import RLock
from sovl_utils import synchronized, safe_divide, float_lt
from sovl_config import ConfigManager
from sovl_state import SOVLState
from sovl_logger import Logger, LoggerConfig

@dataclass
class TemperamentConfig:
    """Configuration for the temperament system."""
    eager_threshold: float = 0.7
    sluggish_threshold: float = 0.3
    mood_influence: float = 0.5
    curiosity_boost: float = 0.2
    restless_drop: float = 0.1
    melancholy_noise: float = 0.1
    confidence_feedback_strength: float = 0.3
    temp_smoothing_factor: float = 0.1
    decay_rate: float = 0.1
    history_maxlen: int = 5
    confidence_history_maxlen: int = 5
    early_lifecycle: float = 0.25
    mid_lifecycle: float = 0.75
    lifecycle_params: Optional[Dict[str, Dict[str, float]]] = None

    def __post_init__(self):
        """Initialize and validate configuration."""
        self._ranges = {
            "eager_threshold": (0.7, 0.9),
            "sluggish_threshold": (0.3, 0.6),
            "mood_influence": (0.0, 1.0),
            "curiosity_boost": (0.0, 0.5),
            "restless_drop": (0.0, 0.5),
            "melancholy_noise": (0.0, 0.1),
            "confidence_feedback_strength": (0.0, 1.0),
            "temp_smoothing_factor": (0.0, 1.0),
            "decay_rate": (0.0, 1.0),
            "early_lifecycle": (0.1, 0.3),
            "mid_lifecycle": (0.6, 0.8),
        }
        self._history_ranges = {
            "history_maxlen": (3, 10),
            "confidence_history_maxlen": (3, 10),
        }
        self.validate()

    def validate(self) -> None:
        """Validate all configuration parameters."""
        for key, (min_val, max_val) in self._ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{key} must be between {min_val} and {max_val}, got {value}"
                )
        for key, (min_val, max_val) in self._history_ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{key} must be between {min_val} and {max_val}, got {value}"
                )
        if self.lifecycle_params is not None:
            for stage, params in self.lifecycle_params.items():
                if not isinstance(params, dict) or "bias" not in params or "decay" not in params:
                    raise ValueError(
                        f"lifecycle_params for {stage} must contain 'bias' and 'decay'"
                    )

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key in self._ranges:
                min_val, max_val = self._ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{key} must be between {min_val} and {max_val}, got {value}"
                    )
                setattr(self, key, value)
            elif key in self._history_ranges:
                min_val, max_val = self._history_ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{key} must be between {min_val} and {max_val}, got {value}"
                    )
                setattr(self, key, value)
            elif key == "lifecycle_params":
                self.lifecycle_params = value
                self.validate()
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

class TemperamentSystem:
    """Manages the temperament state and dynamics of the SOVL system."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        state: SOVLState,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize temperament system.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance from LoggingManager
            state: SOVLState instance for state management
            device: Device for tensor operations
        """
        self.config = config
        self.logger = logger
        self.state = state
        self._device = device
        self._mood_cache = None
        self._mood_cache_time = 0.0
        self._last_update = time.time()
        self._lock = RLock()
        
        # Log device initialization
        self._log_event(
            event_type="temperament_init",
            message="TemperamentSystem initialized",
            level="info",
            additional_info={
                "device": str(device),
                "conversation_id": state.conversation_id,
                "state_hash": state.state_hash
            }
        )
        
        self._load_config()

    def _load_config(self) -> None:
        """Load and validate configuration."""
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.stability_threshold = self.config.get('stability_threshold', 0.1)
        self.max_volatility = self.config.get('max_volatility', 0.5)
        
        # Validate configuration
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")
            
        self.logger.record_event(
            event_type="temperament_config_loaded",
            message="Temperament configuration loaded",
            level="info",
            additional_info={
                "learning_rate": self.learning_rate,
                "stability_threshold": self.stability_threshold,
                "max_volatility": self.max_volatility,
                "conversation_id": self.state.conversation_id,
                "state_hash": self.state.state_hash
            }
        )

    def _log_error(self, error_msg: str, error_type: str, stack_trace: str, **kwargs) -> None:
        """Log an error with standardized format."""
        self.logger.record_event(
            event_type="temperament_error",
            message=error_msg,
            level="error",
            additional_info={
                "error_type": error_type,
                "stack_trace": stack_trace,
                "conversation_id": self.state.conversation_id,
                "state_hash": self.state.state_hash,
                **kwargs
            }
        )

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log an event with standardized format."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info={
                "conversation_id": self.state.conversation_id,
                "state_hash": self.state.state_hash,
                **kwargs
            }
        )

    def update_temperament(
        self,
        performance_metrics: Dict[str, float],
        external_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Update temperament based on performance metrics and external factors.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            external_factors: Optional dictionary of external factors
            
        Returns:
            Updated temperament dictionary
        """
        try:
            with self._lock:
                # Calculate temperament adjustments
                adjustments = self._calculate_adjustments(performance_metrics, external_factors)
                
                # Get current state
                current_temperament = self.state.temperament
                
                # Apply adjustments with stability check
                new_temperament = {}
                for trait, value in current_temperament.items():
                    adjustment = adjustments.get(trait, 0.0)
                    new_value = value + adjustment
                    new_temperament[trait] = max(0.0, min(1.0, new_value))
                    
                # Update state through SOVLState
                self.state.update_temperament(new_temperament)
                
                # Log update
                self._log_event(
                    event_type="temperament_updated",
                    message="Temperament updated with new adjustments",
                    level="info",
                    additional_info={
                        "old_temperament": current_temperament,
                        "new_temperament": new_temperament,
                        "adjustments": adjustments,
                        "performance_metrics": performance_metrics,
                        "external_factors": external_factors,
                        "conversation_id": self.state.conversation_id,
                        "state_hash": self.state.state_hash
                    }
                )
                
                return new_temperament
                
        except Exception as e:
            self._log_error(
                error_msg=f"Temperament update failed: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                performance_metrics=performance_metrics,
                external_factors=external_factors,
                conversation_id=self.state.conversation_id,
                state_hash=self.state.state_hash
            )
            return self.state.temperament

    def _calculate_adjustments(
        self,
        performance_metrics: Dict[str, float],
        external_factors: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate temperament adjustments based on metrics and factors."""
        adjustments = {}
        
        # Base adjustments from performance metrics
        if 'loss' in performance_metrics:
            loss = performance_metrics['loss']
            adjustments['curiosity'] = -loss * self.learning_rate
            adjustments['cautiousness'] = loss * self.learning_rate
            
        if 'accuracy' in performance_metrics:
            accuracy = performance_metrics['accuracy']
            adjustments['creativity'] = accuracy * self.learning_rate
            adjustments['focus'] = (1 - accuracy) * self.learning_rate
            
        # Apply external factors if provided
        if external_factors:
            for factor, value in external_factors.items():
                if factor in self.state.temperament:
                    adjustments[factor] = adjustments.get(factor, 0.0) + value * self.learning_rate
                    
        return adjustments
        
    def get_temperament_score(self, trait: str) -> float:
        """
        Get current score for a specific temperament trait.
        
        Args:
            trait: Name of the temperament trait
            
        Returns:
            Current score for the trait
        """
        return self.state.temperament.get(trait, 0.5)
        
    def get_stability(self) -> float:
        """
        Calculate current temperament stability.
        
        Returns:
            Stability score between 0 and 1
        """
        if len(self.state.history) < 2:
            return 1.0
            
        current = self.state.temperament
        previous = self.state.history[-2]
        
        differences = [
            abs(current[trait] - previous[trait])
            for trait in current.keys()
        ]
        
        avg_difference = sum(differences) / len(differences)
        stability = 1.0 - min(1.0, avg_difference / self.stability_threshold)
        
        return stability

    @property
    def score(self) -> float:
        """Current temperament score (-1.0 to 1.0)."""
        return self.state.temperament_score

    @property
    def mood_label(self) -> str:
        """Get current mood label based on score, cached for 1 second."""
        if self._mood_cache is not None and (time.time() - self._mood_cache_time) < 1.0:
            return self._mood_cache
        
        score = self.score
        if float_lt(score, -0.5):
            label = "melancholic"
        elif float_lt(score, 0.0):
            label = "restless"
        elif float_lt(score, self.config.sluggish_threshold):
            label = "calm"
        else:
            label = "curious"
        
        self._mood_cache = label
        self._mood_cache_time = time.time()
        return label

    @synchronized("_lock")
    def adjust_temperament(self, eager_threshold: Optional[float] = None, 
                          sluggish_threshold: Optional[float] = None,
                          mood_influence: Optional[float] = None,
                          curiosity_boost: Optional[float] = None,
                          restless_drop: Optional[float] = None,
                          melancholy_noise: Optional[float] = None,
                          conf_feedback_strength: Optional[float] = None,
                          temp_smoothing_factor: Optional[float] = None,
                          decay_rate: Optional[float] = None) -> None:
        """
        Adjust temperament parameters with validation.

        Args:
            eager_threshold: Threshold for eager/curious temperament (0.7-0.9)
            sluggish_threshold: Threshold for sluggish/calm temperament (0.3-0.6)
            mood_influence: Mood influence factor (0.0-1.0)
            curiosity_boost: Curiosity boost factor (0.0-0.5)
            restless_drop: Restless drop factor (0.0-0.5)
            melancholy_noise: Melancholy noise factor (0.0-0.1)
            conf_feedback_strength: Confidence feedback strength (0.0-1.0)
            temp_smoothing_factor: Temperature smoothing factor (0.0-1.0)
            decay_rate: Temperament decay rate (0.0-1.0)
        """
        try:
            updates = self._validate_and_collect_updates(
                eager_threshold, sluggish_threshold, mood_influence,
                curiosity_boost, restless_drop, melancholy_noise,
                conf_feedback_strength, temp_smoothing_factor, decay_rate
            )
            
            if updates:
                self._apply_config_updates(updates)
                self._log_adjustments(updates)
                self._log_event(
                    event_type="temperament_adjusted",
                    message=f"Temperament params updated: {updates}",
                    level="info",
                    additional_info={"updates": updates}
                )

        except Exception as e:
            self._log_error(
                error_msg=f"Temperament adjustment failed: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc()
            )
            raise

    def _validate_and_collect_updates(self, *params: Optional[float]) -> Dict[str, float]:
        """Validate parameters and collect valid updates."""
        param_names = [
            "eager_threshold", "sluggish_threshold", "mood_influence",
            "curiosity_boost", "restless_drop", "melancholy_noise",
            "conf_feedback_strength", "temp_smoothing_factor", "decay_rate"
        ]
        ranges = [
            (0.7, 0.9), (0.3, 0.6), (0.0, 1.0),
            (0.0, 0.5), (0.0, 0.5), (0.0, 0.1),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        ]
        
        updates = {}
        for name, value, (min_val, max_val) in zip(param_names, params, ranges):
            if value is not None and min_val <= value <= max_val:
                updates[name] = value
        return updates

    def _apply_config_updates(self, updates: Dict[str, float]) -> None:
        """Apply validated updates to configuration."""
        for key, value in updates.items():
            setattr(self.config, key, value)

    def _log_adjustments(self, updates: Dict[str, float]) -> None:
        """Log temperament adjustments."""
        self._log_event(
            event_type="temperament_adjusted",
            message="Temperament parameters adjusted",
            level="info",
            additional_info={"updates": updates}
        )

    @synchronized("_lock")
    def update_temperament(self, confidence: float, lifecycle_stage: float, 
                          time_since_last: Optional[float] = None,
                          curiosity_pressure: Optional[float] = None) -> None:
        """
        Update temperament based on confidence and lifecycle stage.

        Args:
            confidence: Current confidence score (0.0-1.0)
            lifecycle_stage: Current lifecycle stage (0.0-1.0)
            time_since_last: Time since last update in seconds
            curiosity_pressure: Optional curiosity pressure (0.0-1.0)
        """
        try:
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            if not (0.0 <= lifecycle_stage <= 1.0):
                raise ValueError(f"Lifecycle stage must be between 0.0 and 1.0, got {lifecycle_stage}")
            if curiosity_pressure is not None and not (0.0 <= curiosity_pressure <= 1.0):
                raise ValueError(f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}")

            self._update_confidence(confidence)
            avg_confidence = self._calculate_avg_confidence()
            target_score = self._compute_target_score(avg_confidence, lifecycle_stage, curiosity_pressure)
            self._apply_smoothing(target_score, time_since_last)
            self._update_state()
            self._log_update(confidence, avg_confidence, lifecycle_stage)
            self._log_event(
                event_type="temperament_updated",
                message="Temperament state updated",
                level="info",
                additional_info={
                    "score": self._score,
                    "mood": self.mood_label,
                    "confidence": confidence,
                    "avg_confidence": avg_confidence,
                    "lifecycle_stage": lifecycle_stage
                }
            )

        except Exception as e:
            self._log_error(
                error_msg=f"Temperament update failed: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                conversation_id=self.state.conversation_id,
                state_hash=self.state.state_hash
            )
            raise

    def _update_confidence(self, confidence: float) -> None:
        """Update confidence tracking."""
        self.state.confidence_history.append(confidence)
        self.state.sleep_confidence_sum += confidence
        self.state.sleep_confidence_count += 1

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence with safe division."""
        return safe_divide(
            self.state.sleep_confidence_sum,
            self.state.sleep_confidence_count,
            default=0.5
        )

    def _compute_target_score(self, avg_confidence: float, lifecycle_stage: float, 
                             curiosity_pressure: Optional[float]) -> float:
        """Compute target temperament score."""
        base_score = 2.0 * (avg_confidence - 0.5)
        bias = self._calculate_lifecycle_bias(lifecycle_stage)
        
        target_score = base_score + bias + (
            self.config.confidence_feedback_strength * (avg_confidence - 0.5)
        )

        if curiosity_pressure is not None:
            target_score += self.config.curiosity_boost * curiosity_pressure

        target_score += self._calculate_noise()
        return max(-1.0, min(1.0, target_score))

    def _calculate_lifecycle_bias(self, lifecycle_stage: float) -> float:
        """Calculate lifecycle bias based on stage."""
        try:
            if float_lt(lifecycle_stage, self.config.early_lifecycle):
                return self.config.curiosity_boost * (1 - lifecycle_stage / self.config.early_lifecycle)
            
            if float_lt(lifecycle_stage, self.config.mid_lifecycle):
                if len(self.state.history) >= self.config.history_maxlen:
                    # Ensure tensor is created on correct device
                    history_tensor = torch.tensor(
                        list(self.state.history),
                        device=self._device
                    )
                    variance = torch.var(history_tensor).item()
                    return -0.2 * variance
                return 0.0
            
            return -self.config.curiosity_boost * (
                (lifecycle_stage - self.config.mid_lifecycle) / 
                (1.0 - self.config.mid_lifecycle)
            )
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to calculate lifecycle bias: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                lifecycle_stage=lifecycle_stage
            )
            return 0.0

    def _calculate_noise(self) -> float:
        """Calculate random noise for temperament."""
        try:
            # Ensure tensors are created on correct device
            noise = torch.randn(1, device=self._device).item() * self.config.melancholy_noise
            if self.mood_label == "melancholic":
                noise += torch.randn(1, device=self._device).item() * self.config.melancholy_noise
            return noise
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to calculate noise: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc()
            )
            return 0.0

    def _validate_tensor_device(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """
        Validate and move tensor to correct device if needed.
        
        Args:
            tensor: Input tensor
            name: Name of tensor for logging
            
        Returns:
            Tensor on correct device
        """
        if tensor.device != self._device:
            self._log_event(
                event_type="temperament_device_mismatch",
                message=f"Moving {name} from {tensor.device} to {self._device}",
                level="warning",
                additional_info={
                    "tensor_name": name,
                    "source_device": str(tensor.device),
                    "target_device": str(self._device),
                    "conversation_id": self.state.conversation_id,
                    "state_hash": self.state.state_hash
                }
            )
            return tensor.to(self._device)
        return tensor

    def _apply_smoothing(self, target_score: float, time_since_last: Optional[float]) -> None:
        """Apply smoothing to temperament score."""
        if time_since_last is None:
            time_since_last = time.time() - self._last_update
        
        decay = self.config.decay_rate * time_since_last
        smoothing = self.config.temp_smoothing_factor * (1.0 - decay)
        self._score = (1.0 - smoothing) * target_score + smoothing * self._score
        self._score = max(-1.0, min(1.0, self._score))

    def _update_state(self) -> None:
        """Update internal state after temperament change."""
        self.state.history.append(self._score)
        self._last_update = time.time()
        self._mood_cache = None

    def _log_update(self, confidence: float, avg_confidence: float, lifecycle_stage: float) -> None:
        """Log temperament update details."""
        self._log_event(
            event_type="temperament_updated",
            message="Temperament state updated",
            level="info",
            additional_info={
                "score": self._score,
                "mood": self.mood_label,
                "confidence": confidence,
                "avg_confidence": avg_confidence,
                "lifecycle_stage": lifecycle_stage
            }
        )

    @synchronized("_lock")
    def compute_and_update(self, sleep_confidence_sum: float, sleep_confidence_count: int,
                         data_exposure: float, lora_capacity: float,
                         curiosity_pressure: Optional[float] = None) -> None:
        """
        Compute averages and update temperament state.

        Args:
            sleep_confidence_sum: Sum of sleep confidence scores
            sleep_confidence_count: Number of sleep confidence scores
            data_exposure: Current data exposure
            lora_capacity: LoRA capacity
            curiosity_pressure: Optional curiosity pressure (0.0-1.0)
        """
        try:
            # Calculate average confidence
            avg_confidence = safe_divide(
                sleep_confidence_sum,
                sleep_confidence_count,
                default=0.5
            )
            
            # Calculate lifecycle stage
            lifecycle_stage = safe_divide(
                data_exposure,
                lora_capacity,
                default=0.0
            )
            
            # Update temperament
            self.update_temperament(
                confidence=avg_confidence,
                lifecycle_stage=lifecycle_stage,
                time_since_last=None,  # TODO: Add time tracking
                curiosity_pressure=curiosity_pressure
            )
            
            # Log the update
            self._log_event(
                event_type="temperament_computed",
                message="Temperament computation completed",
                level="info",
                additional_info={
                    "avg_confidence": avg_confidence,
                    "lifecycle_stage": lifecycle_stage,
                    "curiosity_pressure": curiosity_pressure,
                    "score": self._score,
                    "mood": self.mood_label
                }
            )
            
        except Exception as e:
            self._log_error(
                error_msg=f"Temperament computation failed: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc()
            )
            raise

    @classmethod
    def create_from_config(
        cls,
        config_manager: ConfigManager,
        logger: Any,
        device: torch.device = torch.device("cpu")
    ) -> 'TemperamentSystem':
        """
        Create a TemperamentSystem instance from a ConfigManager.
        
        Args:
            config_manager: ConfigManager instance containing temperament settings
            logger: Logger instance from LoggingManager
            device: Device for tensor operations
            
        Returns:
            Initialized TemperamentSystem instance
        """
        controls_config = config_manager.get_section("controls_config")
        temperament_config = TemperamentConfig(
            eager_threshold=controls_config.get("temp_eager_threshold", 0.7),
            sluggish_threshold=controls_config.get("temp_sluggish_threshold", 0.3),
            mood_influence=controls_config.get("temp_mood_influence", 0.5),
            curiosity_boost=controls_config.get("temp_curiosity_boost", 0.2),
            restless_drop=controls_config.get("temp_restless_drop", 0.1),
            melancholy_noise=controls_config.get("temp_melancholy_noise", 0.1),
            confidence_feedback_strength=controls_config.get("temp_conf_feedback_strength", 0.3),
            temp_smoothing_factor=controls_config.get("temp_smoothing_factor", 0.1),
            history_maxlen=controls_config.get("temperament_history_maxlen", 5),
            lifecycle_params={
                "gestation": controls_config.get("temp_gestation_params", {"bias": 0.2, "decay": 0.1}),
                "awakening": controls_config.get("temp_awakening_params", {"bias": 0.1, "decay": 0.05}),
                "maturity": controls_config.get("temp_maturity_params", {"bias": 0.0, "decay": 0.0}),
                "decline": controls_config.get("temp_decline_params", {"bias": -0.1, "decay": 0.05})
            }
        )
        return cls(config=temperament_config, logger=logger, device=device)

    def update_from_state(self, state: SOVLState, curiosity_pressure: Optional[float] = None) -> None:
        """
        Update temperament based on system state and curiosity pressure.
        
        Args:
            state: Current system state
            curiosity_pressure: Optional curiosity pressure value (0.0-1.0)
        """
        try:
            # Log warning if no curiosity pressure provided
            if curiosity_pressure is None:
                self._log_event(
                    event_type="temperament_warning",
                    message="No curiosity pressure provided",
                    level="warning",
                    additional_info={
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
            else:
                # Validate curiosity pressure
                if not 0.0 <= curiosity_pressure <= 1.0:
                    raise ValueError(f"Invalid curiosity pressure: {curiosity_pressure}")
                    
                self._log_event(
                    event_type="temperament_curiosity_pressure",
                    message="Using curiosity pressure for temperament update",
                    level="info",
                    additional_info={
                        "pressure": curiosity_pressure,
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
            
            # Calculate averages and update temperament
            self.compute_and_update(
                sleep_confidence_sum=state.sleep_confidence_sum,
                sleep_confidence_count=state.sleep_confidence_count,
                data_exposure=state.data_exposure,
                lora_capacity=state.lora_capacity,
                curiosity_pressure=curiosity_pressure
            )
            
            # Update state with new temperament
            state.temperament_score = self.score
            state.mood_label = self.mood_label
            
            # Log the update
            self._log_event(
                event_type="temperament_updated",
                message=f"Temperament updated: score={self.score:.3f}, mood={self.mood_label}",
                level="info",
                additional_info={
                    "score": self.score,
                    "mood": self.mood_label,
                    "curiosity_pressure": curiosity_pressure,
                    "data_exposure": state.data_exposure,
                    "lora_capacity": state.lora_capacity,
                    "conversation_id": state.conversation_id,
                    "state_hash": state.state_hash
                }
            )
            
        except Exception as e:
            self._log_error(
                error_msg=f"Failed to update temperament from state: {str(e)}",
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
                curiosity_pressure=curiosity_pressure,
                data_exposure=state.data_exposure if state else None,
                lora_capacity=state.lora_capacity if state else None
            )
            raise
