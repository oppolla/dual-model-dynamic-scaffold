import time
from typing import Deque, Optional, Dict, Any
from collections import deque
import torch
from dataclasses import dataclass
import traceback
import uuid
from threading import RLock
from sovl_utils import synchronized, safe_divide, float_lt
from sovl_config import ConfigManager, ConfigHandler
from sovl_state import SOVLState
from sovl_logger import Logger, LoggerConfig
from sovl_event import EventDispatcher

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
    """Manages the temperament state and updates."""
    
    def __init__(self, state: SOVLState, config: SOVLConfig):
        self.state = state
        self.config = config
        self.logger = LoggingManager()
        
    def update(self, new_score: float, confidence: float, lifecycle_stage: str) -> None:
        """
        Update the temperament system with new values.
        
        Args:
            new_score: New temperament score (0.0 to 1.0)
            confidence: Confidence level in the update (0.0 to 1.0)
            lifecycle_stage: Current lifecycle stage
        """
        try:
            # Validate inputs
            if not 0.0 <= new_score <= 1.0:
                raise ValueError(f"Invalid temperament score: {new_score}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Invalid confidence: {confidence}")
                
            # Update state with new score
            self.state.update_temperament(new_score)
            
            # Log the update
            self.logger.record_event(
                event_type="temperament_updated",
                message="Temperament system updated",
                level="info",
                additional_info={
                    "new_score": new_score,
                    "confidence": confidence,
                    "lifecycle_stage": lifecycle_stage,
                    "current_score": self.state.current_temperament,
                    "conversation_id": self.state.conversation_id,
                    "state_hash": self.state.state_hash
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_update_error",
                message=f"Failed to update temperament: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "conversation_id": self.state.conversation_id,
                    "state_hash": self.state.state_hash
                }
            )
            raise
            
    @property
    def current_score(self) -> float:
        """Get the current temperament score."""
        return self.state.current_temperament
        
    @property
    def mood_label(self) -> str:
        """Get a human-readable mood label based on the current score."""
        score = self.current_score
        if score < 0.3:
            return "Cautious"
        elif score < 0.7:
            return "Balanced"
        else:
            return "Curious"

    def adjust_parameter(
        self,
        base_value: float,
        parameter_type: str,
        curiosity_pressure: Optional[float] = None
    ) -> float:
        """Adjust a parameter based on current temperament and curiosity pressure."""
        try:
            # Validate inputs
            if not 0.0 <= base_value <= 1.0:
                raise ValueError(f"Base value must be between 0.0 and 1.0, got {base_value}")
            if curiosity_pressure is not None and not 0.0 <= curiosity_pressure <= 1.0:
                raise ValueError(f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}")
            
            # Get current temperament score
            current_score = self.current_score
            
            # Calculate adjustment based on parameter type
            if parameter_type == "temperature":
                # Base adjustment from temperament
                adjustment = (current_score - 0.5) * 0.3  # Scale to ±0.15
                
                # Add curiosity influence if available
                if curiosity_pressure is not None:
                    adjustment += curiosity_pressure * 0.2  # Scale to +0.2
                
                # Apply adjustment with bounds
                adjusted_value = base_value + adjustment
                adjusted_value = max(0.1, min(1.0, adjusted_value))
                
                # Log the adjustment
                self.logger.record_event(
                    event_type="parameter_adjusted",
                    message="Parameter adjusted",
                    level="info",
                    additional_info={
                        "parameter_type": parameter_type,
                        "base_value": base_value,
                        "adjusted_value": adjusted_value,
                        "temperament_score": current_score,
                        "curiosity_pressure": curiosity_pressure,
                        "adjustment": adjustment
                    }
                )
                
                return adjusted_value
                
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")
                
        except Exception as e:
            self.logger.record_event(
                event_type="parameter_adjustment_error",
                message=f"Failed to adjust parameter: {str(e)}",
                level="error",
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value,
                    "curiosity_pressure": curiosity_pressure
                }
            )
            return base_value  # Return base value on error

class TemperamentAdjuster:
    """Manages temperament adjustments and state updates."""
    
    def __init__(
        self,
        config_handler: ConfigHandler,
        state_tracker: StateTracker,
        logger: Logger,
        event_dispatcher: EventDispatcher
    ):
        """Initialize temperament adjuster with required dependencies."""
        self.config_handler = config_handler
        self.state_tracker = state_tracker
        self.logger = logger
        self.event_dispatcher = event_dispatcher
        self.temperament_system = None
        self._last_parameter_hash = None
        self._last_state_hash = None
        
        # Initialize components
        self._initialize_events()
        self._initialize_temperament_system()
        
    def _initialize_events(self) -> None:
        """Initialize event subscriptions."""
        self.event_dispatcher.subscribe("config_change", self._on_config_change)
        self.event_dispatcher.subscribe("state_update", self._on_state_update)
        
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            current_params = self._get_validated_parameters()
            current_hash = self._compute_parameter_hash(current_params)
            
            if current_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_parameters_changed",
                    message="Temperament parameters changed, reinitializing system",
                    level="info",
                    additional_info=current_params
                )
                self._initialize_temperament_system()
                
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_config_change_error",
                message=f"Failed to handle config change: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            
    def _on_state_update(self, state: SOVLState) -> None:
        """Handle state updates."""
        try:
            # Validate state consistency
            if not self._validate_state_consistency(state):
                # Reset history if inconsistent
                state.temperament_history.clear()
                self.logger.record_event(
                    event_type="temperament_history_reset",
                    message="Temperament history reset due to inconsistency",
                    level="info"
                )
            
            # Update state with current temperament
            state.temperament_score = self.temperament_system.current_score
            state.temperament_history.append(state.temperament_score)
            
            # Update state hash
            self._last_state_hash = self._compute_state_hash(state)
            
            # Notify other components
            self.event_dispatcher.notify("temperament_updated", state)
            
        except Exception as e:
            self.logger.record_event(
                event_type="state_synchronization_error",
                message=f"Failed to synchronize state: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise
            
    def _validate_state_consistency(self, state: SOVLState) -> bool:
        """Validate consistency between current state and temperament history."""
        try:
            if not state.temperament_history:
                return True
                
            # Check for significant deviation between current score and history
            if abs(state.temperament_history[-1] - state.temperament_score) > 0.5:
                self.logger.record_event(
                    event_type="temperament_inconsistency",
                    message="Temperament history inconsistent with current score",
                    level="warning",
                    additional_info={
                        "current_score": state.temperament_score,
                        "last_history_score": state.temperament_history[-1],
                        "history_length": len(state.temperament_history)
                    }
                )
                return False
                
            # Check for parameter changes that might invalidate history
            current_hash = self._compute_parameter_hash(self._get_validated_parameters())
            if current_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_history_invalidated",
                    message="Temperament parameters changed, history may be invalid",
                    level="warning",
                    additional_info={
                        "parameter_hash": current_hash,
                        "last_parameter_hash": self._last_parameter_hash
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_validation_error",
                message=f"Failed to validate state consistency: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            return False
            
    def _compute_state_hash(self, state: SOVLState) -> str:
        """Compute a hash of the current state."""
        return str({
            "temperament_score": state.temperament_score,
            "history_length": len(state.temperament_history),
            "parameter_hash": self._last_parameter_hash
        })
        
    def _initialize_temperament_system(self) -> None:
        """Initialize or reinitialize the temperament system with validated parameters."""
        try:
            # Get and validate parameters
            params = self._get_validated_parameters()
            
            # Create new temperament system
            self.temperament_system = TemperamentSystem(
                state=self.state_tracker.get_state(),
                config=params
            )
            
            # Update parameter hash
            self._last_parameter_hash = self._compute_parameter_hash(params)
            
            self.logger.record_event(
                event_type="temperament_system_initialized",
                message="Temperament system initialized with validated parameters",
                level="info",
                additional_info=params
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_system_error",
                message=f"Failed to initialize temperament system: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise
            
    def _get_validated_parameters(self) -> Dict[str, Any]:
        """Get and validate temperament parameters."""
        config = self.config_handler.config_manager
        
        # Define safe parameter ranges
        safe_ranges = {
            "temp_smoothing_factor": (0.1, 1.0),
            "temp_eager_threshold": (0.5, 0.9),
            "temp_sluggish_threshold": (0.1, 0.5),
            "temp_mood_influence": (0.1, 0.9),
            "temp_curiosity_boost": (0.1, 0.5),
            "temp_restless_drop": (0.1, 0.5),
            "temp_melancholy_noise": (0.0, 0.2),
            "conf_feedback_strength": (0.1, 0.9),
            "temperament_decay_rate": (0.1, 0.9)
        }
        
        # Get and validate parameters
        params = {}
        for key, (min_val, max_val) in safe_ranges.items():
            value = config.get(f"controls_config.{key}", (min_val + max_val) / 2)
            if not (min_val <= value <= max_val):
                self.logger.record_event(
                    event_type="temperament_parameter_warning",
                    message=f"Parameter {key} out of safe range, clamping to bounds",
                    level="warning",
                    additional_info={
                        "parameter": key,
                        "value": value,
                        "min": min_val,
                        "max": max_val
                    }
                )
                value = max(min_val, min(value, max_val))
            params[key] = value
            
        return params
        
    def _compute_parameter_hash(self, params: Dict[str, Any]) -> str:
        """Compute a hash of the current parameters."""
        return str(sorted(params.items()))
