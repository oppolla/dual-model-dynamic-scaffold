from collections import deque
from typing import Deque, Optional, Tuple, Dict, Any
import torch
import time
import random
from dataclasses import dataclass
from sovl_utils import safe_divide, float_lt
from sovl_logger import Logger
import traceback

@dataclass
class TemperamentConfig:
    """Configuration for the temperament system, mirroring SOVLSystem controls."""
    # Thresholds for mood classification
    eager_threshold: float = 0.8
    sluggish_threshold: float = 0.6
    # Influence factors for dynamic adjustments
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
    # Randomization factor for natural fluctuations
    randomization_factor: float = 0.05
    # Validation ranges for parameters
    _ranges: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
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

class TemperamentSystem:
    """Manages the temperament system, influencing model behavior based on confidence and lifecycle."""
    def __init__(self, config: TemperamentConfig, logger: Logger, device: torch.device = torch.device("cpu")):
        """
        Initialize the temperament system.

        Args:
            config: Temperament configuration
            logger: Logger instance for recording events
            device: Device for tensor operations
        """
        self.config = config
        self.logger = logger
        self.device = device
        self._score: float = 0.0  # Temperament score in [-1.0, 1.0]
        self._history: Deque[float] = deque(maxlen=config.history_maxlen)
        self._confidence_history: Deque[float] = deque(maxlen=config.confidence_history_maxlen)
        self._last_update: float = time.time()
        self._sleep_confidence_sum: float = 0.0
        self._sleep_confidence_count: int = 0
        self._initialize_logging()

    def _initialize_logging(self) -> None:
        """Initialize logging with system startup event."""
        self.logger.record({
            "event": "temperament_init",
            "config": vars(self.config),
            "timestamp": time.time(),
            "score": self._score,
            "mood": self.mood_label
        })

    @property
    def score(self) -> float:
        """Current temperament score (-1.0 to 1.0)."""
        return self._score

    @property
    def mood_label(self) -> str:
        """Human-readable mood label based on score."""
        if float_lt(self._score, -0.5):
            return "melancholic"
        elif float_lt(self._score, 0.0):
            return "restless"
        elif float_lt(self._score, 0.5):
            return "calm"
        return "curious"

    @property
    def variance(self) -> float:
        """Measure of mood stability based on history."""
        if len(self._history) < 2:
            return 0.0
        return float(torch.var(torch.tensor(list(self._history), device=self.device)).item())

    def update(self,
               confidence: float,
               lifecycle_stage: float,
               time_since_last: Optional[float] = None,
               curiosity_pressure: Optional[float] = None) -> None:
        """
        Update temperament based on confidence, lifecycle stage, and optional curiosity pressure.

        Args:
            confidence: Latest confidence score (0.0-1.0)
            lifecycle_stage: Current training progress (0.0-1.0)
            time_since_last: Seconds since last update (optional)
            curiosity_pressure: Curiosity pressure from curiosity manager (optional, 0.0-1.0)
        """
        try:
            # Validate inputs
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
            if not (0.0 <= lifecycle_stage <= 1.0):
                raise ValueError(f"Lifecycle stage must be between 0.0 and 1.0, got {lifecycle_stage}")

            # Update confidence tracking
            self._confidence_history.append(confidence)
            self._sleep_confidence_sum += confidence
            self._sleep_confidence_count += 1

            # Calculate average confidence
            avg_confidence = safe_divide(
                self._sleep_confidence_sum,
                self._sleep_confidence_count,
                default=0.5
            )

            # Base score from confidence (maps 0.5 confidence to 0.0 score)
            base_score = 2.0 * (avg_confidence - 0.5)

            # Lifecycle-based bias
            bias = self._calculate_lifecycle_bias(lifecycle_stage)

            # Confidence feedback effect
            feedback = self.config.confidence_feedback_strength * (avg_confidence - 0.5)

            # Curiosity influence (if provided)
            curiosity_effect = 0.0
            if curiosity_pressure is not None and 0.0 <= curiosity_pressure <= 1.0:
                curiosity_effect = self.config.curiosity_boost * curiosity_pressure

            # Calculate target score
            target_score = base_score + bias + feedback + curiosity_effect

            # Add random fluctuation and noise
            noise = random.uniform(-self.config.randomization_factor, self.config.randomization_factor)
            if self.mood_label == "melancholic":
                noise += random.uniform(-self.config.melancholy_noise, self.config.melancholy_noise)
            target_score += noise

            # Clamp target score
            target_score = max(-1.0, min(1.0, target_score))

            # Apply smoothing
            alpha = self.config.temp_smoothing_factor
            if time_since_last is not None:
                alpha *= min(1.0, time_since_last / 60.0)
            self._score = (1 - alpha) * self._score + alpha * target_score
            self._score = max(-1.0, min(1.0, self._score))

            # Update history
            self._history.append(self._score)
            self._last_update = time.time()

            # Log update
            self.logger.record({
                "event": "temperament_update",
                "score": self._score,
                "mood": self.mood_label,
                "confidence": confidence,
                "avg_confidence": avg_confidence,
                "lifecycle_stage": lifecycle_stage,
                "variance": self.variance,
                "curiosity_effect": curiosity_effect,
                "timestamp": self._last_update
            })

        except Exception as e:
            self.logger.record({
                "error": f"Temperament update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "confidence": confidence,
                "lifecycle_stage": lifecycle_stage
            })
            raise

    def _calculate_lifecycle_bias(self, lifecycle_stage: float) -> float:
        """
        Calculate dynamic bias based on lifecycle stage.

        Args:
            lifecycle_stage: Current training progress (0.0-1.0)

        Returns:
            Bias value influencing temperament score
        """
        if float_lt(lifecycle_stage, self.config.early_lifecycle):
            # Early stage: encourage curiosity
            return self.config.curiosity_boost * (1 - lifecycle_stage / self.config.early_lifecycle)
        elif float_lt(lifecycle_stage, self.config.mid_lifecycle):
            # Middle stage: stabilize based on variance
            if len(self._history) >= self.config.history_maxlen:
                return -0.2 * self.variance
            return 0.0
        else:
            # Late stage: introduce melancholy
            return -self.config.curiosity_boost * (lifecycle_stage - self.config.mid_lifecycle) / (
                1.0 - self.config.mid_lifecycle)

    def adjust_parameter(self,
                        base_value: float,
                        parameter_type: str = "temperature",
                        context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adjust a model parameter based on current temperament.

        Args:
            base_value: Baseline parameter value
            parameter_type: Type of parameter ("temperature", "top_k", "top_p", etc.)
            context: Additional context (e.g., generation settings)

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
                # Wider sampling for positive moods
                scaling = 1.0 + self._score * 0.2
                return int(base_value * scaling)
            elif parameter_type == "top_p":
                # Adjust nucleus sampling based on mood
                return max(0.1, min(0.9, base_value * (1.0 - self._score * 0.1)))
            elif parameter_type == "repetition_penalty":
                # Increase penalty for melancholic moods
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
                "timestamp": time.time()
            })
            return base_value

    def tune(self, **kwargs) -> None:
        """
        Dynamically tune temperament configuration parameters.

        Args:
            **kwargs: Parameters to update (e.g., eager_threshold, curiosity_boost)
        """
        try:
            old_config = vars(self.config).copy()
            self.config.update(**kwargs)
            # Adjust history lengths if changed
            if "history_maxlen" in kwargs:
                self._history = deque(self._history, maxlen=self.config.history_maxlen)
            if "confidence_history_maxlen" in kwargs:
                self._confidence_history = deque(self._confidence_history, maxlen=self.config.confidence_history_maxlen)
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

    def get_state(self) -> Dict[str, Any]:
        """
        Export current state for serialization.

        Returns:
            Dictionary containing temperament state
        """
        return {
            "score": self._score,
            "history": list(self._history),
            "confidence_history": list(self._confidence_history),
            "last_update": self._last_update,
            "sleep_confidence_sum": self._sleep_confidence_sum,
            "sleep_confidence_count": self._sleep_confidence_count
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from serialized data.

        Args:
            state: Dictionary containing temperament state
        """
        try:
            self._score = state.get("score", 0.0)
            self._history = deque(
                state.get("history", []),
                maxlen=self.config.history_maxlen
            )
            self._confidence_history = deque(
                state.get("confidence_history", []),
                maxlen=self.config.confidence_history_maxlen
            )
            self._last_update = state.get("last_update", time.time())
            self._sleep_confidence_sum = state.get("sleep_confidence_sum", 0.0)
            self._sleep_confidence_count = state.get("sleep_confidence_count", 0)
            self.logger.record({
                "event": "temperament_load_state",
                "state": state,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load temperament state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def reset(self) -> None:
        """
        Reset temperament to initial state.
        """
        self._score = 0.0
        self._history.clear()
        self._confidence_history.clear()
        self._sleep_confidence_sum = 0.0
        self._sleep_confidence_count = 0
        self._last_update = time.time()
        self.logger.record({
            "event": "temperament_reset",
            "timestamp": time.time()
        })

    def get_mood_influence(self, parameter: str = "generation") -> float:
        """
        Calculate mood influence on a specific parameter or system component.

        Args:
            parameter: Component to influence ("generation", "curiosity", etc.)

        Returns:
            Influence factor based on mood
        """
        try:
            if parameter == "generation":
                # Positive moods increase exploration
                return 1.0 + self._score * self.config.mood_influence
            elif parameter == "curiosity":
                # Curious moods boost question generation
                return 1.0 + self._score * self.config.curiosity_boost
            elif parameter == "training":
                # Melancholic moods reduce learning rate
                return 1.0 - abs(self._score) * 0.1 if self._score < 0 else 1.0
            else:
                return 1.0
        except Exception as e:
            self.logger.record({
                "error": f"Failed to calculate mood influence: {str(e)}",
                "parameter": parameter,
                "timestamp": time.time()
            })
            return 1.0

    def check_stability(self, threshold: float = 0.1) -> bool:
        """
        Check if temperament is stable based on variance.

        Args:
            threshold: Maximum variance for stability

        Returns:
            True if temperament is stable, False otherwise
        """
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

    def apply_noise(self, value: float) -> float:
        """
        Apply mood-dependent noise to a value.

        Args:
            value: Input value to modify

        Returns:
            Noisy value
        """
        noise = random.uniform(-self.config.randomization_factor, self.config.randomization_factor)
        if self.mood_label == "melancholic":
            noise += random.uniform(-self.config.melancholy_noise, self.config.melancholy_noise)
        return value + noise
