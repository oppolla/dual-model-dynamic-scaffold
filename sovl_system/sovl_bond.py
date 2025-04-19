from typing import Optional, Dict, Any
import re
from threading import Lock
from collections import deque
from sovl_logger import Logger
import traceback
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized, NumericalGuard
from sovl_config import ConfigManager
from sovl_temperament import TemperamentSystem
from sovl_trainer import LifecycleManager

# Constants
DEFAULT_BONDING = 0.0
MIN_BONDING = -1.0
MAX_BONDING = 1.0
MIN_HISTORY_LENGTH = 3
RECOVERY_WEIGHTS = [0.5, 0.3, 0.2]
CURIOSITY_WEIGHT = 0.3
STABILITY_WEIGHT = 0.3
COHERENCE_WEIGHT = 0.2
HISTORY_WEIGHT = 0.2
NOVELTY_THRESHOLD = 0.8
PRESSURE_THRESHOLD = 0.7
PRESSURE_DROP = 0.3
ERROR_PENALTY_FACTOR = 0.2

# Temperament-based bonding adjustments
TEMPERAMENT_MOOD_MULTIPLIERS = {
    "Cautious": 0.8,  # Lower bonding in cautious mood
    "Balanced": 1.0,  # Neutral bonding
    "Curious": 1.2    # Higher bonding in curious mood
}

# Lifecycle stage adjustments
LIFECYCLE_STAGE_MULTIPLIERS = {
    "initialization": 0.9,  # Conservative bonding
    "exploration": 1.2,    # Enhanced bonding
    "consolidation": 1.0,  # Standard bonding
    "refinement": 0.95     # Slightly reduced bonding
}

class BondHistory:
    """Manages history of bonding scores."""
    
    def __init__(self, maxlen: int = 10):
        self.history = deque(maxlen=maxlen)
    
    def add_bond(self, bond: float) -> None:
        """Add a bonding score to history."""
        self.history.append(bond)
    
    def get_bond_history(self) -> deque:
        """Return the bonding history."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear the bonding history."""
        self.history.clear()

class BondCalculator:
    """Calculates bonding score between user and LLM with thread safety."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        temperament_system: Optional[TemperamentSystem] = None,
        lifecycle_manager: Optional[LifecycleManager] = None
    ):
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be a ConfigManager instance")
            
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        self.temperament_system = temperament_system
        self.lifecycle_manager = lifecycle_manager
        self.bond_history = BondHistory(maxlen=10)
        
        # Initialize configuration
        self._initialize_config()
    
    def _initialize_config(self) -> None:
        """Initialize and validate configuration from ConfigManager."""
        try:
            bond_config = self.config_manager.get_section("bond_config", {})
            controls_config = self.config_manager.get_section("controls_config", {})
            
            # Bonding parameters
            self.default_bonding = float(bond_config.get("default_bonding", DEFAULT_BONDING))
            self.min_bonding = float(bond_config.get("min_bonding", MIN_BONDING))
            self.max_bonding = float(bond_config.get("max_bonding", MAX_BONDING))
            self.min_history_length = int(bond_config.get("min_history_length", MIN_HISTORY_LENGTH))
            self.recovery_weights = [
                float(w) for w in bond_config.get("recovery_weights", RECOVERY_WEIGHTS)
            ]
            self.curiosity_weight = float(bond_config.get("curiosity_weight", CURIOSITY_WEIGHT))
            self.stability_weight = float(bond_config.get("stability_weight", STABILITY_WEIGHT))
            self.coherence_weight = float(bond_config.get("coherence_weight", COHERENCE_WEIGHT))
            self.history_weight = float(bond_config.get("history_weight", HISTORY_WEIGHT))
            self.novelty_threshold = float(controls_config.get("curiosity_novelty_threshold_response", NOVELTY_THRESHOLD))
            self.pressure_threshold = float(controls_config.get("curiosity_pressure_threshold", PRESSURE_THRESHOLD))
            self.pressure_drop = float(controls_config.get("curiosity_pressure_drop", PRESSURE_DROP))
            self.error_penalty_factor = float(bond_config.get("error_penalty_factor", ERROR_PENALTY_FACTOR))
            
            # Validate configuration
            self._validate_config_values()
            
            # Subscribe to configuration changes
            self.config_manager.subscribe(self._on_config_change)
            
            self.logger.record_event(
                event_type="bond_config_initialized",
                message="Bonding configuration initialized successfully",
                level="info",
                additional_info={
                    "default_bonding": self.default_bonding,
                    "min_bonding": self.min_bonding,
                    "max_bonding": self.max_bonding,
                    "min_history_length": self.min_history_length,
                    "recovery_weights": self.recovery_weights,
                    "curiosity_weight": self.curiosity_weight,
                    "stability_weight": self.stability_weight,
                    "coherence_weight": self.coherence_weight,
                    "history_weight": self.history_weight,
                    "novelty_threshold": self.novelty_threshold,
                    "pressure_threshold": self.pressure_threshold,
                    "pressure_drop": self.pressure_drop,
                    "error_penalty_factor": self.error_penalty_factor
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="bond_config_initialization_failed",
                message=f"Failed to initialize bonding configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
    
    def _validate_config_values(self) -> None:
        """Validate configuration values."""
        try:
            if not -1.0 <= self.min_bonding <= 1.0 or not -1.0 <= self.max_bonding <= 1.0:
                raise ValueError("Bonding bounds must be between -1.0 and 1.0")
            if self.min_bonding >= self.max_bonding:
                raise ValueError("min_bonding must be less than max_bonding")
            if not -1.0 <= self.default_bonding <= 1.0:
                raise ValueError("default_bonding must be between -1.0 and 1.0")
            if not 1 <= self.min_history_length <= 10:
                raise ValueError("min_history_length must be between 1 and 10")
            if len(self.recovery_weights) != 3 or not all(0.0 <= w <= 1.0 for w in self.recovery_weights):
                raise ValueError("Invalid recovery_weights")
            if abs(sum(self.recovery_weights) - 1.0) > 1e-6:
                raise ValueError("Recovery weights must sum to 1.0")
            if not all(0.0 <= w <= 1.0 for w in [self.curiosity_weight, self.stability_weight, self.coherence_weight, self.history_weight]):
                raise ValueError("Bonding weights must be between 0.0 and 1.0")
            if abs(self.curiosity_weight + self.stability_weight + self.coherence_weight + self.history_weight - 1.0) > 1e-6:
                raise ValueError("Bonding weights must sum to 1.0")
            if not 0.0 <= self.novelty_threshold <= 1.0:
                raise ValueError("novelty_threshold must be between 0.0 and 1.0")
            if not 0.0 <= self.pressure_threshold <= 1.0:
                raise ValueError("pressure_threshold must be between 0.0 and 1.0")
            if not 0.0 <= self.pressure_drop <= 1.0:
                raise ValueError("pressure_drop must be between 0.0 and 1.0")
            if not 0.0 <= self.error_penalty_factor <= 1.0:
                raise ValueError("error_penalty_factor must be between 0.0 and 1.0")
                
        except Exception as e:
            self.logger.record_event(
                event_type="bond_config_validation_failed",
                message=f"Bonding configuration validation failed: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
    
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            self._initialize_config()
            self.logger.record_event(
                event_type="bond_config_updated",
                message="Bonding configuration updated",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="bond_config_update_failed",
                message=f"Failed to update bonding configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
    
    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        try:
            # Preprocess: lowercase, remove special chars, split into words
            text1 = re.sub(r'[^a-zA-Z0-9\s]', '', text1.lower())
            text2 = re.sub(r'[^a-zA-Z0-9\s]', '', text2.lower())
            words1 = set(text1.split())
            words2 = set(text2.split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            self.logger.record_event(
                event_type="bond_jaccard_calculation_failed",
                message=f"Failed to compute Jaccard similarity: {str(e)}",
                level="error",
                additional_info={"text1": text1[:50], "text2": text2[:50], "error": str(e)}
            )
            return 0.0
    
    def _apply_temperament_bonding(self, base_score: float) -> float:
        """Apply temperament-based adjustments to bonding score."""
        if not self.temperament_system:
            return base_score
            
        try:
            mood_label = self.temperament_system.mood_label
            mood_multiplier = TEMPERAMENT_MOOD_MULTIPLIERS.get(mood_label, 1.0)
            adjusted_score = base_score * mood_multiplier
            
            self.logger.record_event(
                event_type="bond_temperament_adjusted",
                message="Applied temperament-based bonding adjustments",
                level="info",
                additional_info={
                    "base_score": base_score,
                    "adjusted_score": adjusted_score,
                    "mood_label": mood_label,
                    "mood_multiplier": mood_multiplier
                }
            )
            
            return adjusted_score
            
        except Exception as e:
            self.logger.record_event(
                event_type="bond_temperament_adjustment_failed",
                message=f"Failed to apply temperament bonding adjustments: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_score
    
    def _apply_lifecycle_bonding(self, base_score: float) -> float:
        """Apply lifecycle-based adjustments to bonding score."""
        if not self.lifecycle_manager:
            return base_score
            
        try:
            lifecycle_stage = self.lifecycle_manager.get_lifecycle_stage()
            stage_multiplier = LIFECYCLE_STAGE_MULTIPLIERS.get(lifecycle_stage, 1.0)
            adjusted_score = base_score * stage_multiplier
            
            self.logger.record_event(
                event_type="bond_lifecycle_adjusted",
                message="Applied lifecycle-based bonding adjustments",
                level="info",
                additional_info={
                    "base_score": base_score,
                    "adjusted_score": adjusted_score,
                    "lifecycle_stage": lifecycle_stage,
                    "stage_multiplier": stage_multiplier
                }
            )
            
            return adjusted_score
            
        except Exception as e:
            self.logger.record_event(
                event_type="bond_lifecycle_adjustment_failed",
                message=f"Failed to apply lifecycle bonding adjustments: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_score
    
    @synchronized()
    def calculate_bonding_score(
        self,
        user_input: str,
        llm_response: str,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None
    ) -> float:
        """Calculate bonding score between user and LLM on a -1 to 1 scale.
        
        Args:
            user_input: User's input text
            llm_response: LLM's response text
            state: Current SOVL state
            error_manager: Error handling manager
            context: System context
            curiosity_manager: Optional curiosity manager
            
        Returns:
            float: Bonding score between -1.0 and 1.0
        """
        try:
            # Validate inputs
            if not isinstance(user_input, str) or not isinstance(llm_response, str):
                raise ValueError("user_input and llm_response must be strings")
                
            # Curiosity alignment
            novelty = curiosity_manager.get_novelty_score(user_input) if curiosity_manager else 0.0
            novelty = novelty if novelty >= self.novelty_threshold else 0.0
            text_sim = self._compute_jaccard_similarity(user_input, llm_response)
            curiosity_score = 0.6 * novelty + 0.4 * text_sim
            
            # Interaction stability
            interaction_count = getattr(state, 'interaction_count', 0)
            engagement = min(interaction_count / 100.0, 1.0)
            error_rate = error_manager.get_error_rate(window=10) if hasattr(error_manager, 'get_error_rate') else 0.0
            stability = 1.0 - error_rate * self.error_penalty_factor
            stability_score = 0.6 * engagement + 0.4 * stability
            
            # Response coherence
            relevance = novelty if novelty >= self.novelty_threshold else 0.5
            coherence_score = 0.5 * text_sim + 0.5 * relevance
            
            # Base bonding score
            base_score = (
                self.curiosity_weight * curiosity_score +
                self.stability_weight * stability_score +
                self.coherence_weight * coherence_score
            )
            
            # Apply temperament and lifecycle adjustments
            tempered_score = self._apply_temperament_bonding(base_score)
            lifecycle_score = self._apply_lifecycle_bonding(tempered_score)
            
            # History smoothing
            history_avg = self.default_bonding
            if len(self.bond_history.get_bond_history()) >= self.min_history_length:
                recent_scores = list(self.bond_history.get_bond_history())[-self.min_history_length:]
                history_avg = sum(s * w for s, w in zip(recent_scores, self.recovery_weights))
            smoothed_score = (1.0 - self.history_weight) * lifecycle_score + self.history_weight * history_avg
            
            # Curiosity pressure adjustment
            final_score = smoothed_score
            if curiosity_manager:
                pressure = curiosity_manager.get_pressure()
                if pressure > self.pressure_threshold:
                    final_score *= (1.0 - pressure * self.pressure_drop)
            
            # Clamp and store
            final_score = max(self.min_bonding, min(self.max_bonding, final_score * 2.0 - 1.0))
            self.bond_history.add_bond(final_score)
            
            # Log the calculation
            self.logger.record_event(
                event_type="bond_calculation_complete",
                message="Bonding calculation completed",
                level="info",
                additional_info={
                    "user_input": user_input[:50],
                    "llm_response": llm_response[:50],
                    "base_score": base_score,
                    "tempered_score": tempered_score,
                    "lifecycle_score": lifecycle_score,
                    "smoothed_score": smoothed_score,
                    "final_score": final_score,
                    "curiosity_score": curiosity_score,
                    "stability_score": stability_score,
                    "coherence_score": coherence_score,
                    "novelty": novelty,
                    "text_sim": text_sim,
                    "engagement": engagement,
                    "error_rate": error_rate,
                    "mood_label": self.temperament_system.mood_label if self.temperament_system else None,
                    "lifecycle_stage": self.lifecycle_manager.get_lifecycle_stage() if self.lifecycle_manager else None
                }
            )
            
            return final_score
            
        except Exception as e:
            return self._recover_bonding(e, error_manager)
    
    def _recover_bonding(self, error: Exception, error_manager: ErrorManager) -> float:
        """Attempt to recover bonding score from history or use default."""
        try:
            history = self.bond_history.get_bond_history()
            if len(history) >= self.min_history_length:
                recent_bonds = list(history)[-self.min_history_length:]
                if any(not isinstance(b, (int, float)) or b < self.min_bonding or b > self.max_bonding for b in recent_bonds):
                    error_manager.logger.record_event(
                        event_type="bond_history_invalid",
                        message="Invalid values in bonding history",
                        level="error",
                        additional_info={"history": recent_bonds}
                    )
                    return self.default_bonding
                    
                recovered_bonding = sum(b * w for b, w in zip(recent_bonds, self.recovery_weights))
                
                error_manager.logger.record_event(
                    event_type="bond_recovered",
                    message="Recovered bonding from history",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "recovered_bonding": recovered_bonding,
                        "history_length": len(history)
                    }
                )
                
                return recovered_bonding
                
            error_manager.logger.record_event(
                event_type="bond_default_used",
                message="Using default bonding due to insufficient history",
                level="warning",
                additional_info={
                    "error": str(error),
                    "history_length": len(history)
                }
            )
            return self.default_bonding
            
        except Exception as recovery_error:
            error_manager.logger.record_event(
                event_type="bond_recovery_failed",
                message="Failed to recover bonding",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "recovery_error": str(recovery_error)
                }
            )
            return self.default_bonding

# Singleton instance
_bond_calculator = None

def calculate_bonding_score(
    user_input: str,
    llm_response: str,
    state: SOVLState,
    error_manager: ErrorManager,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager] = None
) -> float:
    """Calculate bonding score with robust error recovery."""
    global _bond_calculator
    if _bond_calculator is None:
        _bond_calculator = BondCalculator(
            config_manager=state.config_manager,
            logger=state.logger,
            temperament_system=state.temperament_system if hasattr(state, 'temperament_system') else None,
            lifecycle_manager=state.lifecycle_manager if hasattr(state, 'lifecycle_manager') else None
        )
    
    return _bond_calculator.calculate_bonding_score(
        user_input=user_input,
        llm_response=llm_response,
        state=state,
        error_manager=error_manager,
        context=context,
        curiosity_manager=curiosity_manager
    )
