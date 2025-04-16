from typing import Optional, Dict, Any
import torch
from threading import Lock
from collections import deque
import logging

# Import necessary SOVL dependencies
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_context import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized, NumericalGuard

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
            Confidence score between 0.0 and 1.0
        """
        try:
            _validate_inputs(logits, generated_ids)
            probs = _calculate_probabilities(logits)
            base_confidence = _compute_base_confidence(probs)
            adjusted_confidence = _apply_adjustments(
                base_confidence, state, context, curiosity_manager
            )
            final_confidence = _finalize_confidence(adjusted_confidence, state)
            return final_confidence
        except Exception as e:
            return _recover_confidence(e, state, error_manager)

def _validate_inputs(logits: torch.Tensor, generated_ids: torch.Tensor) -> None:
    """Validate input tensors for confidence calculation."""
    if not isinstance(logits, torch.Tensor) or not isinstance(generated_ids, torch.Tensor):
        raise ValueError("logits and generated_ids must be tensors")
    if logits.dim() != 2 or generated_ids.dim() != 1:
        raise ValueError("Invalid tensor dimensions")

def _calculate_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calculate softmax probabilities from logits."""
    with NumericalGuard():
        return torch.softmax(logits, dim=-1)

def _compute_base_confidence(probs: torch.Tensor) -> float:
    """Compute base confidence from probabilities."""
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

def _apply_adjustments(
    base_confidence: float,
    state: SOVLState,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager]
) -> float:
    """Apply curiosity and temperament adjustments to confidence."""
    confidence = base_confidence
    
    # Apply curiosity pressure adjustment if available
    if curiosity_manager is not None:
        pressure = curiosity_manager.get_pressure()
        confidence *= (1.0 - pressure * 0.1)  # Reduce confidence under high pressure
        
    # Apply temperament influence
    temperament_influence = context.config_handler.config_manager.get("temperament_config.influence", 0.3)
    confidence *= (1.0 + state.temperament_score * temperament_influence)
    
    return confidence

def _finalize_confidence(confidence: float, state: SOVLState) -> float:
    """Finalize confidence score and update history."""
    final_confidence = max(0.0, min(1.0, confidence))
    state.confidence_history.append(final_confidence)
    return final_confidence

def _recover_confidence(error: Exception, state: SOVLState, error_manager: ErrorManager) -> float:
    """Attempt to recover confidence from history or use default."""
    try:
        if len(state.confidence_history) >= 3:
            # Use weighted average of recent confidences
            recent_confidences = list(state.confidence_history)[-3:]
            weights = [0.5, 0.3, 0.2]  # More weight to recent values
            recovered_confidence = sum(c * w for c, w in zip(recent_confidences, weights))
            
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
            level="error",
            additional_info={
                "error": str(error),
                "history_length": len(state.confidence_history)
            }
        )
        return 0.5  # Conservative default
        
    except Exception as recovery_error:
        error_manager.logger.record_event(
            event_type="confidence_error",
            message="Failed to recover confidence",
            level="critical",
            additional_info={
                "original_error": str(error),
                "recovery_error": str(recovery_error)
            }
        )
        return 0.5  # Fallback default

# Create a global instance of ConfidenceCalculator
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
        Confidence score between 0.0 and 1.0
    """
    return _confidence_calculator.calculate_confidence_score(
        logits=logits,
        generated_ids=generated_ids,
        state=state,
        error_manager=error_manager,
        context=context,
        curiosity_manager=curiosity_manager
    )
