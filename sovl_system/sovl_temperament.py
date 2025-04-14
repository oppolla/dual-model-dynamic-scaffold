import time
from typing import Deque, Optional
import torch
import traceback
from threading import RLock
from sovl_utils import synchronized, safe_divide, float_lt

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

    @synchronized(_lock)
    def adjust_temperament(self, eager_threshold=None, sluggish_threshold=None, mood_influence=None,
                          curiosity_boost=None, restless_drop=None, melancholy_noise=None,
                          conf_feedback_strength=None, temp_smoothing_factor=None, decay_rate=None):
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
            updates = {}
            
            if eager_threshold is not None and 0.7 <= eager_threshold <= 0.9:
                self.config.eager_threshold = eager_threshold
                updates["eager_threshold"] = eager_threshold
                
            if sluggish_threshold is not None and 0.4 <= sluggish_threshold <= 0.6:
                self.config.sluggish_threshold = sluggish_threshold
                updates["sluggish_threshold"] = sluggish_threshold
                
            if mood_influence is not None and 0 <= mood_influence <= 1:
                self.config.mood_influence = mood_influence
                updates["mood_influence"] = mood_influence
                
            if curiosity_boost is not None and 0 <= curiosity_boost <= 0.5:
                self.config.curiosity_boost = curiosity_boost
                updates["curiosity_boost"] = curiosity_boost
                
            if restless_drop is not None and 0 <= restless_drop <= 0.5:
                self.config.restless_drop = restless_drop
                updates["restless_drop"] = restless_drop
                
            if melancholy_noise is not None and 0 <= melancholy_noise <= 0.05:
                self.config.melancholy_noise = melancholy_noise
                updates["melancholy_noise"] = melancholy_noise
                
            if conf_feedback_strength is not None and 0 <= conf_feedback_strength <= 1:
                self.config.confidence_feedback_strength = conf_feedback_strength
                updates["confidence_feedback_strength"] = conf_feedback_strength
                
            if temp_smoothing_factor is not None and 0 <= temp_smoothing_factor <= 1:
                self.config.temp_smoothing_factor = temp_smoothing_factor
                updates["temp_smoothing_factor"] = temp_smoothing_factor
                
            if decay_rate is not None and 0.0 <= decay_rate <= 1.0:
                self.config.decay_rate = decay_rate
                updates["decay_rate"] = decay_rate

            if updates:
                self.logger.record({
                    "event": "temperament_adjusted",
                    "updates": updates,
                    "timestamp": time.time()
                })
                print(f"Temperament params updated: {updates}")

        except Exception as e:
            self.logger.record({
                "error": f"Temperament adjustment failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    @synchronized(_lock)
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

            # Calculate base score from confidence
            base_score = 2.0 * (avg_confidence - 0.5)

            # Calculate lifecycle bias
            if float_lt(lifecycle_stage, self.config.early_lifecycle):
                bias = self.config.curiosity_boost * (1 - lifecycle_stage / self.config.early_lifecycle)
            elif float_lt(lifecycle_stage, self.config.mid_lifecycle):
                bias = 0.0
                if len(self._history) >= self.config.history_maxlen:
                    variance = torch.var(torch.tensor(list(self._history), device=self.device)).item()
                    bias -= 0.2 * variance
            else:
                bias = -self.config.curiosity_boost * (
                    (lifecycle_stage - self.config.mid_lifecycle) / 
                    (1.0 - self.config.mid_lifecycle)
                )

            # Calculate target score
            target_score = base_score + bias + (
                self.config.confidence_feedback_strength * (avg_confidence - 0.5)
            )

            # Add curiosity pressure if provided
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

            # Log update
            self._log_update(confidence, avg_confidence, lifecycle_stage)

            # Print status
            print(f"Temperament score: {self._score:.3f} ({self.mood_label}, lifecycle: {lifecycle_stage:.2f}), confidence feedback: {avg_confidence:.2f}")

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