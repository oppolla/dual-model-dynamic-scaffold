from typing import Optional, Dict, Any
import torch
from threading import Lock
from collections import deque
import logging

# SOVL-specific dependencies
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_context import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized, NumericalGuard

# Constants
DEFAULT_CONFIDENCE = 0.5
RECOVERY_WEIGHTS = [0.5, 0.3, 0.2]
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_HISTORY_LENGTH = 3
CURIOSITY_PRESSURE_FACTOR = 0.1
DEFAULT_TEMPERAMENT_INFLUENCE = 0.3

"""
Confidence calculation module for the SOVL system.

This module provides functionality to calculate confidence scores for model outputs,
incorporating curiosity and temperament adjustments. It is thread-safe and includes
robust error recovery mechanisms.

Primary interface: calculate_confidence_score
"""


class ConfidenceCalculator:
    """Handles confidence score calculation with thread safety."""
    
    def __init__(self):
        """Initialize the confidence calculator with a thread lock."""
        self.lock = Lock()
        
    @synchronized()
    def calculate_confidence_score(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None
    ) -> float:
        """Calculate confidence score with robust error recovery.
        
        Args:
            logits: Model output logits
            generated_ids: Generated token IDs
            state: Current SOVL state
            error_manager: Error handling manager
            context: System context
            curiosity_manager: Optional curiosity manager
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            __validate_inputs(logits, generated_ids)
            probs = __calculate_probabilities(logits)
            base_confidence = __compute_base_confidence(probs)
            adjusted_confidence = __apply_adjustments(
                base_confidence, state, context, curiosity_manager
            )
            final_confidence = __finalize_confidence(adjusted_confidence, state)
            return final_confidence
        except Exception as e:
            return __recover_confidence(e, state, error_manager)

def __validate_inputs(logits: torch.Tensor, generated_ids: torch.Tensor) -> None:
    """Validate input tensors for confidence calculation.
    
    Args:
        logits: Model output logits
        generated_ids: Generated token IDs
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(generated_ids, torch.Tensor):
        raise ValueError("logits and generated_ids must be torch.Tensor")
    if logits.dim() != 2 or generated_ids.dim() != 1:
        raise ValueError("logits must be 2D and generated_ids must be 1D")

def __calculate_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calculate softmax probabilities from logits.
    
    Args:
        logits: Model output logits
        
    Returns:
        torch.Tensor: Softmax probabilities
    """
    with NumericalGuard():
        return torch.softmax(logits, dim=-1)

def __compute_base_confidence(probs: torch.Tensor) -> float:
    """Compute base confidence from probabilities.
    
    Args:
        probs: Softmax probabilities
        
    Returns:
        float: Base confidence score
    """
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

def __apply_adjustments(
    base_confidence: float,
    state: SOVLState,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager]
) -> float:
    """Apply curiosity and temperament adjustments to confidence.
    
    Args:
        base_confidence: Initial confidence score
        state: Current SOVL state
        context: System context
        curiosity_manager: Optional curiosity manager
        
    Returns:
        float: Adjusted confidence score
    """
    confidence = base_confidence
    
    # Apply curiosity pressure adjustment if available
    if curiosity_manager is not None:
        pressure = curiosity_manager.get_pressure()
        confidence *= (1.0 - pressure * CURIOSITY_PRESSURE_FACTOR)
        
    # Apply temperament influence
    try:
        temperament_influence = context.config_handler.config_manager.get(
            "temperament_config.influence", DEFAULT_TEMPERAMENT_INFLUENCE
        )
    except Exception as e:
        logging.warning(f"Failed to get temperament influence: {str(e)}. Using default: {DEFAULT_TEMPERAMENT_INFLUENCE}")
        temperament_influence = DEFAULT_TEMPERAMENT_INFLUENCE
    confidence *= (1.0 + state.temperament_score * temperament_influence)
    
    return confidence

def __finalize_confidence(confidence: float, state: SOVLState) -> float:
    """Finalize confidence score and update history.
    
    Args:
        confidence: Adjusted confidence score
        state: Current SOVL state
        
    Returns:
        float: Final confidence score
    """
    final_confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))
    state.confidence_history.append(final_confidence)
    return final_confidence

def __recover_confidence(error: Exception, state: SOVLState, error_manager: ErrorManager) -> float:
    """Attempt to recover confidence from history or use default.
    
    Args:
        error: Original exception
        state: Current SOVL state
        error_manager: Error handling manager
        
    Returns:
        float: Recovered confidence score
    """
    try:
        # Validate confidence history
        if not hasattr(state, 'confidence_history') or not isinstance(state.confidence_history, deque):
            error_manager.logger.record_event(
                event_type="confidence_error",
                message="Invalid confidence history structure",
                level="error",
                additional_info={"error": str(error)}
            )
            return DEFAULT_CONFIDENCE

        if len(state.confidence_history) >= MIN_HISTORY_LENGTH:
            # Use weighted average of recent confidences
            recent_confidences = list(state.confidence_history)[-MIN_HISTORY_LENGTH:]
            # Validate history values
            if any(not isinstance(c, (int, float)) or c < MIN_CONFIDENCE or c > MAX_CONFIDENCE for c in recent_confidences):
                error_manager.logger.record_event(
                    event_type="confidence_error",
                    message="Invalid values in confidence history",
                    level="error",
                    additional_info={"history": recent_confidences}
                )
                return DEFAULT_CONFIDENCE
                
            recovered_confidence = sum(c * w for c, w in zip(recent_confidences, RECOVERY_WEIGHTS))
            
            # Log recovery
            error_manager.logger.record_event(
                event_type="confidence_recovery",
                message="Recovered confidence from history",
                level="warning",
                additional_info={
                    "error": str(error),
                    "recovered_confidence": recovered_confidence,
                    "history_length": len(state.confidence_history)
                }
            )
            
            return recovered_confidence
            
        # If recovery fails, use conservative default
        error_manager.logger.record_event(
            event_type="confidence_default",
            message="Using default confidence due to insufficient history",
            level="warning",
            additional_info={
                "error": str(error),
                "history_length": len(state.confidence_history)
            }
        )
        return DEFAULT_CONFIDENCE
        
    except Exception as recovery_error:
        error_manager.logger.record_event(
            event_type="confidence_recovery_failed",
            message="Failed to recover confidence",
            level="critical",
            additional_info={
                "original_error": str(error),
                "recovery_error": str(recovery_error)
            }
        )
        return DEFAULT_CONFIDENCE

# Singleton instance of ConfidenceCalculator
_confidence_calculator = ConfidenceCalculator()

def calculate_confidence_score(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    state: SOVLState,
    error_manager: ErrorManager,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager] = None
) -> float:
    """Calculate confidence score with robust error recovery.
    
    Args:
        logits: Model output logits
        generated_ids: Generated token IDs
        state: Current SOVL state
        error_manager: Error handling manager
        context: System context
        curiosity_manager: Optional curiosity manager
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    return _confidence_calculator.calculate_confidence_score(
        logits=logits,
        generated_ids=generated_ids,
        state=state,
        error_manager=error_manager,
        context=context,
        curiosity_manager=curiosity_manager
    )
