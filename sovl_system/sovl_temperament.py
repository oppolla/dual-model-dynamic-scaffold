import time
from typing import Deque, Optional
from collections import deque
import torch
import traceback
from threading import RLock
from sovl_utils import synchronized, safe_divide, float_lt

class TemperamentSystem:
    """Manages model temperament with thread-safe operations."""
    
    STATE_VERSION = "1.0"
    
    def __init__(self, config: 'TemperamentConfig', logger: 'Logger', 
                 device: torch.device = torch.device("cpu")):
        """
        Initialize temperament system.

        Args:
            config: Temperament configuration
            logger: Logger instance
            device: Device for tensor operations
        """
        self._config = config
        self._logger = logger
        self._device = device
        self._score: float = 0.0
        self._history: Deque[float] = deque(maxlen=config.history_maxlen)
        self._confidence_history: Deque[float] = deque(maxlen=config.confidence_history_maxlen)
        self._sleep_confidence_sum: float = 0.0
        self._sleep_confidence_count: int = 0
        self._last_update: float = time.time()
        self._lock = RLock()
        self._mood_cache: Optional[str] = None
        self._mood_cache_time: float = 0
        self._initialize_logging()

    def _initialize_logging(self) -> None:
        """Initialize logging configuration."""
        self._logger.record({
            "event": "temperament_system_initialized",
            "state_version": self.STATE_VERSION,
            "timestamp": time.time()
        })

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
            eager_threshold: Threshold for eager temperament (0.7-0.9)
            sluggish_threshold: Threshold for sluggish temperament (0.4-0.6)
            mood_influence: Mood influence factor (0-1)
            curiosity_boost: Curiosity boost factor (0-0.5)
            restless_drop: Restless drop factor (0-0.5)
            melancholy_noise: Melancholy noise factor (0-0.05)
            conf_feedback_strength: Confidence feedback strength (0-1)
            temp_smoothing_factor: Temperature smoothing factor (0-1)
            decay_rate: Temperament decay rate (0-1)
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
                print(f"Temperament params updated: {updates}")

        except Exception as e:
            self._logger.record({
                "error": f"Temperament adjustment failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _validate_and_collect_updates(self, *params: Optional[float]) -> dict:
        """Validate parameters and collect valid updates."""
        param_names = [
            "eager_threshold", "sluggish_threshold", "mood_influence",
            "curiosity_boost", "restless_drop", "melancholy_noise",
            "conf_feedback_strength", "temp_smoothing_factor", "decay_rate"
        ]
        ranges = [
            (0.7, 0.9), (0.4, 0.6), (0.0, 1.0),
            (0.0, 0.5), (0.0, 0.5), (0.0, 0.05),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        ]
        
        updates = {}
        for name, value, (min_val, max_val) in zip(param_names, params, ranges):
            if value is not None and min_val <= value <= max_val:
                updates[name] = value
        return updates

    def _apply_config_updates(self, updates: dict) -> None:
        """Apply validated updates to configuration."""
        for key, value in updates.items():
            setattr(self._config, key, value)

    def _log_adjustments(self, updates: dict) -> None:
        """Log temperament Ricciardi adjustments."""
        self._logger.record({
            "event": "temperament_adjusted",
            "updates": updates,
            "timestamp": time.time()
        })

    @synchronized("_lock")
    def update_temperament(self, confidence: float, lifecycle_stage: float, 
                          time_since_last: Optional[float] = None,
                          curiosity_pressure: Optional[float] = None) -> None:
        """
        Update temperament based on confidence and lifecycle stage.

        Args:
            confidence: Current confidence score (0-1)
            lifecycle_stage: Current lifecycle stage (0-1)
            time_since_last: Time since last update in seconds
            curiosity_pressure: Optional curiosity pressure (0-1)
        """
        try:
            self._update_confidence(confidence)
            avg_confidence = self._calculate_avg_confidence()
            target_score = self._compute_target_score(avg_confidence, lifecycle_stage, curiosity_pressure)
            self._apply_smoothing(target_score, time_since_last)
            self._update_state()
            self._log_update(confidence, avg_confidence, lifecycle_stage)
            print(f"Temperament score: {self._score:.3f} ({self.mood_label}, "
                  f"lifecycle: {lifecycle_stage:.2f}), confidence feedback: {avg_confidence:.2f}")

        except Exception as e:
            self._logger.record({
                "error": f"Temperament update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "confidence": confidence,
                "lifecycle_stage": lifecycle_stage,
                "curiosity_pressure": curiosity_pressure
            })
            raise

    def _update_confidence(self, confidence: float) -> None:
        """Update confidence tracking."""
        self._confidence_history.append(confidence)
        self._sleep_confidence_sum += confidence
        self._sleep_confidence_count += 1

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence with safe division."""
        return safe_divide(
            self._sleep_confidence_sum,
            self._sleep_confidence_count,
            default=0.5
        )

    def _compute_target_score(self, avg_confidence: float, lifecycle_stage: float, 
                            curiosity_pressure: Optional[float]) -> float:
        """Compute target temperament score."""
        base_score = 2.0 * (avg_confidence - 0.5)
        bias = self._calculate_lifecycle_bias(lifecycle_stage)
        
        target_score = base_score + bias + (
            self._config.confidence_feedback_strength * (avg_confidence - 0.5)
        )

        if curiosity_pressure is not None:
            target_score += self._config.curiosity_boost * curiosity_pressure

        target_score += self._calculate_noise()
        return max(-1.0, min(1.0, target_score))

    def _calculate_lifecycle_bias(self, lifecycle_stage: float) -> float:
        """Calculate lifecycle bias based on stage."""
        if float_lt(lifecycle_stage, self._config.early_lifecycle):
            return self._config.curiosity_boost * (1 - lifecycle_stage / self._config.early_lifecycle)
        
        if float_lt(lifecycle_stage, self._config.mid_lifecycle):
            if len(self._history) >= self._config.history_maxlen:
                variance = torch.var(torch.tensor(list(self._history), device=self._device)).item()
                return -0.2 * variance
            return 0.0
        
        return -self._config.curiosity_boost * (
            (lifecycle_stage - self._config.mid_lifecycle) / 
            (1.0 - self._config.mid_lifecycle)
        )

    def _calculate_noise(self) -> float:
        """Calculate random noise for temperament."""
        return torch.randn(1, device=self._device).item() * self._config.melancholy_noise

    def _apply_smoothing(self, target_score: float, time_since_last: Optional[float]) -> None:
        """Apply smoothing to temperament score."""
        if time_since_last is None:
            time_since_last = time.time() - self._last_update
        
        decay = self._config.decay_rate * time_since_last
        smoothing = self._config.temp_smoothing_factor * (1.0 - decay)
        self._score = (1.0 - smoothing) * target_score + smoothing * self._score

    def _update_state(self) -> None:
        """Update internal state after temperament change."""
        self._history.append(self._score)
        self._last_update = time.time()
        self._mood_cache = None

    def _log_update(self, confidence: float, avg_confidence: float, lifecycle_stage: float) -> None:
        """Log temperament update details."""
        self._logger.record({
            "event": "temperament_updated",
            "score": self._score,
            "mood_label": self.mood_label,
            "confidence": confidence,
            "avg_confidence": avg_confidence,
            "lifecycle_stage": lifecycle_stage,
            "timestamp": time.time()
        })

    @property
    def mood_label(self) -> str:
        """Get current mood label based on score."""
        if self._mood_cache is not None and (time.time() - self._mood_cache_time) < 1.0:
            return self._mood_cache
        
        if self._score > self._config.eager_threshold:
            label = "eager"
        elif self._score < self._config.sluggish_threshold:
            label = "sluggish"
        else:
            label = "balanced"
        
        self._mood_cache = label
        self._mood_cache_time = time.time()
        return label
