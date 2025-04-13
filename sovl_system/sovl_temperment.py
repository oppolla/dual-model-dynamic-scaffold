# sovl_temperament.py
from collections import deque
from typing import Deque, Optional, Tuple
import torch
import time
from dataclasses import dataclass
from sovl_utils import safe_divide, float_lt
from sovl_logger import Logger

@dataclass
class TemperamentConfig:
    # Thresholds
    eager_threshold: float = 0.8
    sluggish_threshold: float = 0.6
    # Influence factors
    mood_influence: float = 0.3
    curiosity_boost: float = 0.5
    restless_drop: float = 0.1
    melancholy_noise: float = 0.02
    # Tracking
    history_maxlen: int = 5
    smoothing_factor: float = 0.1
    confidence_feedback: float = 0.5
    # Lifecycle boundaries
    early_lifecycle: float = 0.25
    mid_lifecycle: float = 0.75

class TemperamentSystem:
    def __init__(self, config: TemperamentConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self._score: float = 0.0  # Range [-1.0, 1.0]
        self._history: Deque[float] = deque(maxlen=config.history_maxlen)
        self._confidence_history: Deque[float] = deque(maxlen=config.history_maxlen)
        self._last_update: float = time.time()
        
    @property
    def score(self) -> float:
        """Current temperament score (-1.0 to 1.0)"""
        return self._score
        
    @property
    def mood_label(self) -> str:
        """Human-readable mood label"""
        if float_lt(self._score, -0.5):
            return "melancholic"
        elif float_lt(self._score, 0.0):
            return "restless"
        elif float_lt(self._score, 0.5):
            return "calm"
        return "curious"
        
    @property
    def variance(self) -> float:
        """Measure of mood stability"""
        if len(self._history) < 2:
            return 0.0
        return torch.var(torch.tensor(list(self._history))).item()
        
    def update(self, 
              confidence: float, 
              lifecycle_stage: float,
              time_since_last: Optional[float] = None) -> None:
        """
        Update temperament based on new confidence and lifecycle stage
        
        Args:
            confidence: Latest confidence score (0.0-1.0)
            lifecycle_stage: Current training progress (0.0-1.0)
            time_since_last: Seconds since last update (optional)
        """
        # Store confidence for feedback calculations
        self._confidence_history.append(confidence)
        
        # Base score from confidence (maps 0.5 confidence to 0.0 score)
        base_score = 2.0 * (confidence - 0.5)
        
        # Lifecycle-based adjustments
        bias = self._calculate_lifecycle_bias(lifecycle_stage)
        
        # Confidence feedback effect
        feedback = self.config.confidence_feedback * (confidence - 0.5)
        
        # Calculate target score with all influences
        target_score = base_score + bias + feedback
        target_score = max(-1.0, min(1.0, target_score))
        
        # Apply smoothing
        alpha = self.config.smoothing_factor * (1.0 if time_since_last is None 
                                              else min(1.0, time_since_last / 60.0))
        self._score = (1 - alpha) * self._score + alpha * target_score
        self._score = max(-1.0, min(1.0, self._score))
        
        # Update history
        self._history.append(self._score)
        self._last_update = time.time()
        
        # Log the update
        self.logger.record({
            "event": "temperament_update",
            "score": self._score,
            "mood": self.mood_label,
            "confidence": confidence,
            "lifecycle": lifecycle_stage,
            "variance": self.variance,
            "timestamp": self._last_update
        })
        
    def _calculate_lifecycle_bias(self, lifecycle_stage: float) -> float:
        """Calculate dynamic bias based on lifecycle stage"""
        if float_lt(lifecycle_stage, self.config.early_lifecycle):
            # Early stage - curiosity boost
            return self.config.curiosity_boost * (1 - lifecycle_stage / self.config.early_lifecycle)
        elif float_lt(lifecycle_stage, self.config.mid_lifecycle):
            # Middle stage - stability with variance penalty
            return -0.2 * self.variance if len(self._history) >= 5 else 0.0
        else:
            # Late stage - melancholy tendency
            return -self.config.curiosity_boost * (lifecycle_stage - self.config.mid_lifecycle) / (1.0 - self.config.mid_lifecycle)
            
    def adjust_parameter(self, 
                        base_value: float, 
                        parameter_type: str = "temperature") -> float:
        """
        Adjust a parameter based on current temperament
        
        Args:
            base_value: The baseline parameter value
            parameter_type: Type of parameter ('temperature', 'top_k', etc.)
            
        Returns:
            Adjusted parameter value
        """
        if parameter_type == "temperature":
            # Higher temp for curious, lower for melancholic
            adjustment = self._score * self.config.mood_influence
            return base_value + adjustment
        elif parameter_type == "top_k":
            # Wider sampling for positive moods
            return int(base_value * (1.0 + self._score * 0.2))
        else:
            return base_value
            
    def get_state(self) -> dict:
        """Export current state for serialization"""
        return {
            "score": self._score,
            "history": list(self._history),
            "confidence_history": list(self._confidence_history),
            "last_update": self._last_update
        }
        
    def load_state(self, state: dict) -> None:
        """Load state from serialized data"""
        self._score = state.get("score", 0.0)
        self._history = deque(state.get("history", []), maxlen=self.config.history_maxlen)
        self._confidence_history = deque(
            state.get("confidence_history", []), 
            maxlen=self.config.history_maxlen
        )
        self._last_update = state.get("last_update", time.time())
