from collections import deque
from typing import Deque, Optional, Tuple, Dict, Any, List, Union
import torch
import time
import random
from dataclasses import dataclass
from functools import wraps
from sovl_utils import safe_divide, float_lt
from sovl_logger import Logger
from threading import Lock, RLock
import traceback
from enum import Enum

class TemperamentError(Exception):
    """Base class for temperament-related errors."""
    pass

class InvalidParameterError(TemperamentError):
    """Raised when a parameter is outside valid range."""
    pass

class TemperamentStateError(TemperamentError):
    """Raised when temperament state is invalid."""
    pass

class LifecycleStage(Enum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"

def synchronized(lock):
    """Thread synchronization decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator

@dataclass
class TemperamentConfig:
    """Configuration for the temperament system."""
    # Thresholds
    eager_threshold: float = 0.8
    sluggish_threshold: float = 0.6
    
    # Influence factors
    mood_influence: float = 0.0
    curiosity_boost: float = 0.5
    restless_drop: float = 0.1
    melancholy_noise: float = 0.02
    confidence_feedback_strength: float = 0.5
    temp_smoothing_factor: float = 0.0
    
    # History tracking
    history_maxlen: int = 5
    confidence_history_maxlen: int = 5
    
    # Lifecycle boundaries
    early_lifecycle: float = 0.25
    mid_lifecycle: float = 0.75
    
    # Randomization
    randomization_factor: float = 0.05
    
    # Validation ranges (type hinted)
    _ranges: Dict[str, Tuple[float, float]] = None
    _history_ranges: Dict[str, Tuple[int, int]] = None

    def __post_init__(self):
        """Initialize and validate configuration."""
        self._ranges = {
            "eager_threshold": (0.7, 0.9),
            "sluggish_threshold": (0.4, 0.6),
            "mood_influence": (0.0, 1.0),
            "curiosity_boost": (0.0, 0.5),
            "restless_drop": (0.0, 0.5),
            "melancholy_noise": (0.0, 0.05),
            "confidence_feedback_strength": (0.0, 1.0),
            "temp_smoothing_factor": (0.0, 1.0),
            "randomization_factor": (0.0, 0.1),
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
                raise InvalidParameterError(
                    f"{key} must be between {min_val} and {max_val}, got {value}"
                )
        for key, (min_val, max_val) in self._history_ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise InvalidParameterError(
                    f"{key} must be between {min_val} and {max_val}, got {value}"
                )

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key in self._ranges:
                min_val, max_val = self._ranges[key]
                if not (min_val <= value <= max_val):
                    raise InvalidParameterError(
                        f"{key} must be between {min_val} and {max_val}, got {value}"
                    )
                setattr(self, key, value)
            elif key in self._history_ranges:
                min_val, max_val = self._history_ranges[key]
                if not (min_val <= value <= max_val):
                    raise InvalidParameterError(
                        f"{key} must be between {min_val} and {max_val}, got {value}"
                    )
                setattr(self, key, value)
            else:
                raise InvalidParameterError(f"Unknown configuration parameter: {key}")

class TemperamentSystem:
    """Manages model temperament with thread-safe operations."""
    
    STATE_VERSION = "1.0"
    
    def __init__(self, config: TemperamentConfig, logger: Logger, 
                 device: torch.device = torch.device("cpu")):
        """
        Initialize temperament system.
        
        Args:
            config: Temperament configuration
            logger: Logger instance
            device: Device for tensor operations
        """
        self.config = config
        self.logger = logger
        self.device = device
        self._score: float = 0.0
        self._history: Deque[float] = deque(maxlen=config.history_maxlen)
        self._confidence_history: Deque[float] = deque(maxlen=config.confidence_history_maxlen)
        self._sleep_confidence_sum: float = 0.0
        self._sleep_confidence_count: int = 0
        self._last_update: float = time.time()
        self._lock = RLock()  # Reentrant lock for nested calls
        self._mood_cache = None
        self._mood_cache_time = 0
        self._initialize_logging()

    def _initialize_logging(self) -> None:
        """Log system initialization."""
        self.logger.record({
            "event": "temperament_init",
            "config": vars(self.config),
            "timestamp": time.time(),
            "score": self._score,
            "mood": self.mood_label,
            "state_version": self.STATE_VERSION
        })

    @property
    @synchronized(_lock)
    def score(self) -> float:
        """Current temperament score (-1.0 to 1.0)."""
        return self._score

    @property
    @synchronized(_lock)
    def mood_label(self) -> str:
        """Human-readable mood label with caching."""
        current_time = time.time()
        if self._mood_cache and (current_time - self._mood_cache_time) < 1.0:
            return self._mood_cache
            
        mood = self._calculate_mood_label()
        self._mood_cache = mood
        self._mood_cache_time = current_time
        return mood

    def _calculate_mood_label(self) -> str:
        """Calculate mood label without caching."""
        if float_lt(self._score, -0.5):
            return "melancholic"
        elif float_lt(self._score, 0.0):
            return "restless"
        elif float_lt(self._score, self.config.sluggish_threshold):
            return "calm"
        return "curious"

    @property
    @synchronized(_lock)
    def variance(self) -> float:
        """Measure of mood stability based on history."""
        if len(self._history) < 2:
            return 0.0
        return float(torch.var(torch.tensor(list(self._history), 
                      device=self.device)).item())

    @synchronized(_lock)
    def update(self,
               confidence: float,
               lifecycle_stage: float,
               time_since_last: Optional[float] = None,
               curiosity_pressure: Optional[float] = None) -> None:
        """
        Update temperament state.
        
        Args:
            confidence: Latest confidence score (0.0-1.0)
            lifecycle_stage: Training progress (0.0-1.0)
            time_since_last: Seconds since last update
            curiosity_pressure: External curiosity signal (0.0-1.0)
        """
        try:
            # Validate inputs
            if not (0.0 <= confidence <= 1.0):
                raise InvalidParameterError(
                    f"Confidence must be between 0.0 and 1.0, got {confidence}"
                )
            if not (0.0 <= lifecycle_stage <= 1.0):
                raise InvalidParameterError(
                    f"Lifecycle stage must be between 0.0 and 1.0, got {lifecycle_stage}"
                )
            if curiosity_pressure is not None and not (0.0 <= curiosity_pressure <= 1.0):
                raise InvalidParameterError(
                    f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}"
                )

            # Update confidence tracking
            self._confidence_history.append(confidence)
            self._sleep_confidence_sum += confidence
            self._sleep_confidence_count += 1

            # Calculate metrics
            avg_confidence = safe_divide(
                self._sleep_confidence_sum,
                self._sleep_confidence_count,
                default=0.5
            )
            base_score = 2.0 * (avg_confidence - 0.5)
            bias = self._calculate_lifecycle_bias(lifecycle_stage)
            feedback = self.config.confidence_feedback_strength * (avg_confidence - 0.5)

            # Apply adjustments
            target_score = base_score + bias + feedback
            if curiosity_pressure is not None:
                target_score += self.config.curiosity_boost * curiosity_pressure

            # Add noise and clamp
            target_score += self._calculate_noise()
            target_score = max(-1.0, min(1.0, target_score))

            # Apply smoothing
            self._apply_smoothing(target_score, time_since_last)

            # Update state
            self._history.append(self._score)
            self._last_update = time.time()
            self._mood_cache = None  # Invalidate cache

            self._log_update(confidence, avg_confidence, lifecycle_stage)

        except Exception as e:
            self.logger.record({
                "error": f"Temperament update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "confidence": confidence,
                "lifecycle_stage": lifecycle_stage,
                "curiosity_pressure": curiosity_pressure
            })
            raise

    def _calculate_noise(self) -> float:
        """Calculate mood-dependent noise."""
        noise = random.uniform(
            -self.config.randomization_factor, 
            self.config.randomization_factor
        )
        if self._calculate_mood_label() == "melancholic":
            noise += random.uniform(
                -self.config.melancholy_noise, 
                self.config.melancholy_noise
            )
        return noise

    def _apply_smoothing(self, target_score: float, 
                        time_since_last: Optional[float]) -> None:
        """Apply temporal smoothing to score."""
        alpha = self.config.temp_smoothing_factor
        if time_since_last is not None:
            alpha *= min(1.0, time_since_last / 60.0)
        self._score = (1 - alpha) * self._score + alpha * target_score
        self._score = max(-1.0, min(1.0, self._score))

    def _log_update(self, confidence: float, avg_confidence: float,
                   lifecycle_stage: float) -> None:
        """Log update details."""
        self.logger.record({
            "event": "temperament_update",
            "score": self._score,
            "mood": self.mood_label,
            "confidence": confidence,
            "avg_confidence": avg_confidence,
            "lifecycle_stage": lifecycle_stage,
            "variance": self.variance,
            "timestamp": self._last_update
        })

    @synchronized(_lock)
    def _calculate_lifecycle_bias(self, lifecycle_stage: float) -> float:
        """Calculate dynamic bias based on lifecycle stage."""
        if float_lt(lifecycle_stage, self.config.early_lifecycle):
            # Early stage: encourage curiosity
            return self.config.curiosity_boost * (
                1 - lifecycle_stage / self.config.early_lifecycle
            )
        elif float_lt(lifecycle_stage, self.config.mid_lifecycle):
            # Middle stage: stabilize
            if len(self._history) >= self.config.history_maxlen:
                return -0.2 * self.variance
            return 0.0
        else:
            # Late stage: introduce melancholy
            return -self.config.curiosity_boost * (
                (lifecycle_stage - self.config.mid_lifecycle) / 
                (1.0 - self.config.mid_lifecycle)
            )

    @synchronized(_lock)
    def adjust_parameter(self,
                        base_value: float,
                        parameter_type: str = "temperature",
                        context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adjust model parameter based on current temperament.
        
        Args:
            base_value: Baseline parameter value
            parameter_type: Type of parameter to adjust
            context: Additional context dictionary
            
        Returns:
            Adjusted parameter value
        """
        try:
            adjustment = self._score * self.config.mood_influence
            if parameter_type == "temperature":
                if self.mood_label == "restless":
                    adjustment -= self.config.restless_drop
                return base_value + adjustment
            elif parameter_type == "top_k":
                scaling = 1.0 + self._score * 0.2
                return int(base_value * scaling)
            elif parameter_type == "top_p":
                return max(0.1, min(0.9, base_value * (1.0 - self._score * 0.1)))
            elif parameter_type == "repetition_penalty":
                return base_value * (1.0 + abs(self._score) * 0.1 if self._score < 0 else 1.0)
            else:
                self.logger.record({
                    "warning": f"Unknown parameter type: {parameter_type}",
                    "timestamp": time.time()
                })
                return base_value
        except Exception as e:
            self.logger.record({
                "error": f"Parameter adjustment failed: {str(e)}",
                "parameter_type": parameter_type,
                "base_value": base_value,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return base_value

    @synchronized(_lock)
    def tune(self, **kwargs) -> None:
        """Dynamically update configuration parameters."""
        try:
            old_config = vars(self.config).copy()
            self.config.update(**kwargs)
            
            # Adjust history buffers if sizes changed
            if "history_maxlen" in kwargs:
                self._history = deque(self._history, maxlen=self.config.history_maxlen)
            if "confidence_history_maxlen" in kwargs:
                self._confidence_history = deque(
                    self._confidence_history,
                    maxlen=self.config.confidence_history_maxlen
                )
                
            self.logger.record({
                "event": "temperament_tune",
                "old_config": old_config,
                "new_config": vars(self.config),
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Temperament tuning failed: {str(e)}",
                "kwargs": kwargs,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    @synchronized(_lock)
    def get_state(self) -> Dict[str, Any]:
        """Export current state for serialization."""
        return {
            "version": self.STATE_VERSION,
            "score": self._score,
            "history": list(self._history),
            "confidence_history": list(self._confidence_history),
            "sleep_confidence_sum": self._sleep_confidence_sum,
            "sleep_confidence_count": self._sleep_confidence_count,
            "last_update": self._last_update,
            "config": vars(self.config)
        }

    @synchronized(_lock)
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialized data."""
        try:
            if state.get("version") != self.STATE_VERSION:
                raise TemperamentStateError(
                    f"State version mismatch: expected {self.STATE_VERSION}, "
                    f"got {state.get('version')}"
                )
                
            self._score = state.get("score", 0.0)
            self._history = deque(
                state.get("history", []),
                maxlen=self.config.history_maxlen
            )
            self._confidence_history = deque(
                state.get("confidence_history", []),
                maxlen=self.config.confidence_history_maxlen
            )
            self._sleep_confidence_sum = state.get("sleep_confidence_sum", 0.0)
            self._sleep_confidence_count = state.get("sleep_confidence_count", 0)
            self._last_update = state.get("last_update", time.time())
            
            # Update config if present in state
            if "config" in state:
                self.config.update(**state["config"])
                
            self.logger.record({
                "event": "temperament_load_state",
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load temperament state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    @synchronized(_lock)
    def reset(self) -> None:
        """Reset to initial state."""
        self._score = 0.0
        self._history.clear()
        self._confidence_history.clear()
        self._sleep_confidence_sum = 0.0
        self._sleep_confidence_count = 0
        self._last_update = time.time()
        self._mood_cache = None
        self.logger.record({
            "event": "temperament_reset",
            "timestamp": time.time()
        })

    @synchronized(_lock)
    def get_mood_influence(self, component: str = "generation") -> float:
        """Get mood influence factor for specific component."""
        try:
            if component == "generation":
                return 1.0 + self._score * self.config.mood_influence
            elif component == "curiosity":
                return 1.0 + max(0.0, self._score) * self.config.curiosity_boost
            elif component == "training":
                return 1.0 - abs(self._score) * 0.1 if self._score < 0 else 1.0
            else:
                self.logger.record({
                    "warning": f"Unknown component: {component}",
                    "timestamp": time.time()
                })
                return 1.0
        except Exception as e:
            self.logger.record({
                "error": f"Failed to calculate mood influence: {str(e)}",
                "component": component,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return 1.0

    @synchronized(_lock)
    def check_stability(self, threshold: float = 0.1) -> bool:
        """Check if temperament is stable based on variance."""
        try:
            variance = self.variance
            is_stable = variance < threshold
            self.logger.record({
                "event": "temperament_stability_check",
                "variance": variance,
                "threshold": threshold,
                "is_stable": is_stable,
                "timestamp": time.time()
            })
            return is_stable
        except Exception as e:
            self.logger.record({
                "error": f"Stability check failed: {str(e)}",
                "threshold": threshold,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return True

    @synchronized(_lock)
    def apply_noise(self, value: float) -> float:
        """Apply mood-dependent noise to a value."""
        try:
            noise = random.uniform(
                -self.config.randomization_factor, 
                self.config.randomization_factor
            )
            if self.mood_label == "melancholic":
                noise += random.uniform(
                    -self.config.melancholy_noise, 
                    self.config.melancholy_noise
                )
            return value + noise
        except Exception as e:
            self.logger.record({
                "error": f"Failed to apply noise: {str(e)}",
                "value": value,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return value
