from typing import Optional, Any, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import time
import random
import bitsandbytes as bnb
import json
import contextlib
from collections import deque, defaultdict
import traceback
import os
from threading import Lock
from sovl_curiosity import CuriosityManager, CuriosityState
from sovl_logger import Logger
from sovl_io import load_training_data, validate_quantization_mode, InsufficientDataError
from sovl_state import SOVLState, ConversationHistory
from sovl_trainer import TrainingConfig, SOVLTrainer
from sovl_config import ConfigManager
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager, CrossAttentionLayer, ScaffoldTokenMapper
from sovl_processor import LogitsProcessor, SOVLProcessor
from sovl_utils import (
    calculate_confidence,
    detect_repetitions,
    safe_compare,
    float_gt,
    synchronized
)
from sovl_temperament import TemperamentConfig, TemperamentSystem
from sovl_memory import MemoryManager
from sovl_manager import ModelManager
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner
from sovl_error import ErrorHandler
from sovl_state_manager import StateManager
from sovl_logging import LoggingManager
import logging
from sovl_training_cycle import TrainingCycleManager
from sovl_plugin import PluginManager
import sys
import math
from sovl_utils import NumericalGuard

# Remove sovl_conductor import and use TYPE_CHECKING for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

@synchronized("lock")
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
    try:
        # Input validation
        if not isinstance(logits, torch.Tensor) or not isinstance(generated_ids, torch.Tensor):
            raise ValueError("logits and generated_ids must be tensors")
            
        if logits.dim() != 2 or generated_ids.dim() != 1:
            raise ValueError("Invalid tensor dimensions")
            
        # Calculate base confidence using softmax probabilities
        with NumericalGuard():
            probs = torch.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            base_confidence = max_probs.mean().item()
            
        # Apply curiosity pressure adjustment if available
        if curiosity_manager is not None:
            pressure = curiosity_manager.get_pressure()
            base_confidence *= (1.0 - pressure * 0.1)  # Reduce confidence under high pressure
            
        # Apply temperament influence
        temperament_influence = context.config_manager.get("temperament_config.influence", 0.3)
        base_confidence *= (1.0 + state.temperament_score * temperament_influence)
        
        # Constrain final confidence
        final_confidence = max(0.0, min(1.0, base_confidence))
        
        # Update confidence history
        state.confidence_history.append(final_confidence)
        
        return final_confidence
        
    except Exception as e:
        # Attempt recovery using recent valid confidences
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
                        "error": str(e),
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
                    "error": str(e),
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
                    "original_error": str(e),
                    "recovery_error": str(recovery_error)
                }
            )
            return 0.5  # Fallback default

class SystemContext:
    """Manages system-wide context and resources."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        """
        Initialize system context.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for tensor operations
        """
        self.device = device
        self.config_manager = ConfigManager(config_path, Logger())
        self.logger = self.config_manager.logger
        self.processor = SOVLProcessor(self.config_manager, self.logger, self.device)
        self.config_handler = ConfigHandler(self)
        
        # Subscribe to configuration changes
        self.config_manager.subscribe(self._on_config_change)
        
        self.logger.record_event(
            event_type="system_init",
            message="System context initialized",
            level="info",
            additional_info={
                "device": device,
                "config_path": config_path,
                "logging_setup": True
            }
        )

    def _on_config_change(self) -> None:
        """Handle configuration changes by refreshing ConfigHandler."""
        try:
            self.config_handler._refresh_configs()
            self.logger.record_event(
                event_type="config_sync",
                message="Configuration synchronized",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="config_sync_error",
                message="Failed to synchronize configuration",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class ConfigHandler:
    """Handles configuration validation and management."""
    
    def __init__(self, context: SystemContext):
        self.context = context
        self.logger = context.logger
        self._refresh_configs()
        
    def _refresh_configs(self):
        """Refresh configuration sections from ConfigManager."""
        self.core_config = self.context.config_manager.get_section("core_config")
        self.controls_config = self.context.config_manager.get_section("controls_config")
        self.curiosity_config = self.context.config_manager.get_section("curiosity_config")
        self.training_config = self.context.config_manager.get_section("training_config")
        
        self.logger.record_event(
            event_type="config_refresh",
            message="Configuration sections refreshed",
            level="info"
        )
        
    def _validate_all_configs(self):
        """Validate all configuration sections."""
        self._validate_controls_configs()
        self._validate_curiosity_configs()
        self._validate_temperament_configs()
        self._validate_processor_configs()
        
    def _validate_controls_configs(self):
        """Validate controls configuration section."""
        for key, value in self.controls_config.items():
            is_valid, error_msg = ValidationSchema.validate_value(
                "controls_config", key, value, self.logger
            )
            if not is_valid:
                self.logger.record_event(
                    event_type="config_validation_error",
                    message=f"Invalid controls config value: {error_msg}",
                    level="error",
                    additional_info={
                        "key": key,
                        "value": value
                    }
                )
                raise ValueError(f"Invalid controls config: {error_msg}")
                
    def _validate_curiosity_configs(self):
        """Validate curiosity configuration section."""
        for key, value in self.curiosity_config.items():
            is_valid, error_msg = ValidationSchema.validate_value(
                "curiosity_config", key, value, self.logger
            )
            if not is_valid:
                self.logger.record_event(
                    event_type="config_validation_error",
                    message=f"Invalid curiosity config value: {error_msg}",
                    level="error",
                    additional_info={
                        "key": key,
                        "value": value
                    }
                )
                raise ValueError(f"Invalid curiosity config: {error_msg}")
                
    def _validate_temperament_configs(self):
        """Validate temperament configuration section."""
        for key, value in self.controls_config.items():
            if key.startswith("temp_"):
                is_valid, error_msg = ValidationSchema.validate_value(
                    "controls_config", key, value, self.logger
                )
                if not is_valid:
                    self.logger.record_event(
                        event_type="config_validation_error",
                        message=f"Invalid temperament config value: {error_msg}",
                        level="error",
                        additional_info={
                            "key": key,
                            "value": value
                        }
                    )
                    raise ValueError(f"Invalid temperament config: {error_msg}")
                    
    def _validate_processor_configs(self):
        """Validate processor configuration section."""
        for key, value in self.core_config.items():
            is_valid, error_msg = ValidationSchema.validate_value(
                "core_config", key, value, self.logger
            )
            if not is_valid:
                self.logger.record_event(
                    event_type="config_validation_error",
                    message=f"Invalid processor config value: {error_msg}",
                    level="error",
                    additional_info={
                        "key": key,
                        "value": value
                    }
                )
                raise ValueError(f"Invalid processor config: {error_msg}")
                
    def validate(self, model_config: Any = None) -> bool:
        """Validate all configurations."""
        try:
            self._validate_all_configs()
            return True
        except ValueError as e:
            self.logger.record_event(
                event_type="config_validation_failed",
                message=f"Configuration validation failed: {str(e)}",
                level="error"
            )
            return False
            
    def validate_with_model(self, model_config: Any) -> bool:
        """Validate configurations with model-specific checks."""
        try:
            self._validate_all_configs()
            # Add model-specific validation here if needed
            return True
        except ValueError as e:
            self.logger.record_event(
                event_type="config_validation_failed",
                message=f"Configuration validation failed: {str(e)}",
                level="error"
            )
            return False

class ModelLoader:
    """Handles model loading, initialization, and cross-attention injection."""
    
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        self.context = context
        self.config_handler = config_handler
        self.model = None
        self.scaffold_model = None
        self.token_map = None
        self._cross_attention_injector = None
        
    def _validate_cross_attention_weights(self, layers: List[int], weights: List[float]) -> None:
        """Validate cross-attention layer weights before injection."""
        if not weights:
            return
            
        if len(weights) != len(layers):
            self.context.logger.record_event(
                event_type="cross_attention_error",
                message="Layer weights length mismatch",
                level="error",
                additional_info={
                    "expected_layers": len(layers),
                    "provided_weights": len(weights),
                    "layers": layers,
                    "weights": weights
                }
            )
            raise ValueError(
                f"Invalid layer weights length: expected {len(layers)} weights for {len(layers)} layers, "
                f"got {len(weights)} weights"
            )
            
        # Validate weight values
        for i, weight in enumerate(weights):
            if not (0.0 <= weight <= 1.0):
                self.context.logger.record_event(
                    event_type="cross_attention_error",
                    message="Invalid layer weight value",
                    level="error",
                    additional_info={
                        "layer_index": i,
                        "weight": weight,
                        "valid_range": (0.0, 1.0)
                    }
                )
                raise ValueError(f"Layer weight {weight} at index {i} must be between 0.0 and 1.0")
                
    def inject_cross_attention(self) -> None:
        """Inject cross-attention layers with validation."""
        try:
            # Get current cross-attention configuration
            cross_attn_config = self.context.config_manager.get_section("cross_attn_config")
            layer_weights = cross_attn_config.get("layer_weights", [])
            
            # Get current cross-attention layers
            if self._cross_attention_injector:
                layers = self._cross_attention_injector.get_cross_attention_layers(
                    self.model,
                    mode=self.context.core_config.get("layer_selection_mode", "balanced")
                )
                
                # Validate weights before injection
                self._validate_cross_attention_weights(layers, layer_weights)
                
                # Log injection attempt
                self.context.logger.record_event(
                    event_type="cross_attention_injection",
                    message="Injecting cross-attention layers",
                    level="info",
                    additional_info={
                        "layer_count": len(layers),
                        "layer_weights": layer_weights,
                        "layer_selection_mode": self.context.core_config.get("layer_selection_mode", "balanced")
                    }
                )
                
                # Perform injection
                self._cross_attention_injector.inject_cross_attention(
                    model=self.model,
                    scaffold_model=self.scaffold_model,
                    core_config=self.context.core_config,
                    cross_attn_config=cross_attn_config,
                    lora_config=self.context.lora_config,
                    token_map=self.token_map,
                    device=self.context.device
                )
                
                # Log successful injection
                self.context.logger.record_event(
                    event_type="cross_attention_injected",
                    message="Successfully injected cross-attention layers",
                    level="info",
                    additional_info={
                        "layer_count": len(layers),
                        "layer_weights": layer_weights
                    }
                )
            else:
                self.context.logger.record_event(
                    event_type="cross_attention_error",
                    message="Cross attention injector not initialized",
                    level="error"
                )
                raise RuntimeError("Cross attention injector not initialized")
                
        except Exception as e:
            self.context.logger.record_event(
                event_type="cross_attention_error",
                message=f"Failed to inject cross-attention layers: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise

class StateTracker:
    """Tracks and manages the system's state."""
    
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        """Initialize the state tracker with context and configuration."""
        self.context = context
        self.config_handler = config_handler
        self.state_manager = StateManager(
            config_manager=context.config_manager,
            logger=context.logger,
            device=context.device
        )
        # Auto-initialize state
        self.state = self.state_manager.initialize_state()
        self._log_event("state_tracker_initialized", {
            "state_hash": self.state.state_hash,
            "conversation_id": self.state.history.conversation_id
        })
        
    def _log_event(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.context.logger.record_event(
            event_type=f"state_{event}",
            message=f"State event: {event}",
            level="info",
            additional_info=data
        )
        
    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with stack trace."""
        self.context.logger.log_error(
            error_msg=message,
            error_type="state_error",
            stack_trace=traceback.format_exc(),
            additional_info={"error": str(error)}
        )
        
    def load_state(self) -> None:
        """Load state from file or initialize new state if not found."""
        try:
            # Only load if state path exists, otherwise keep initialized state
            state_path = self.config_handler.config_manager.get("state_config.state_path")
            if state_path and os.path.exists(state_path):
                with self.state.lock:
                    self.state = self.state_manager.load_state(state_path)
                    self._log_event("state_loaded", {
                        "state_path": state_path,
                        "state_hash": self.state.state_hash,
                        "conversation_id": self.state.history.conversation_id
                    })
            else:
                self._log_event("state_initialized", {
                    "state_hash": self.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                })
        except Exception as e:
            self._log_error("State loading failed", e)
            raise StateError(f"Failed to load state: {str(e)}")
            
    def save_state(self) -> None:
        """Save current state to file."""
        try:
            state_path = self.config_handler.config_manager.get("state_config.state_path")
            if state_path:
                with self.state.lock:
                    self.state_manager.save_state(state_path)
                    self._log_event("state_saved", {
                        "state_path": state_path,
                        "state_hash": self.state.state_hash
                    })
        except Exception as e:
            self._log_error("State saving failed", e)
            raise StateError(f"Failed to save state: {str(e)}")
            
    def get_state(self) -> SOVLState:
        """Get the current state instance."""
        if self.state is None:
            raise StateError("State not initialized")
        return self.state

class ErrorManager:
    """Manages error handling and recovery for the SOVL system."""
    
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        self.context = context
        self.state_tracker = state_tracker
        self.logger = context.logger
        self.error_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=100)  # Track recent errors to detect duplicates
        self.error_cooldown = 1.0  # seconds
        self.severity_thresholds = {
            "warning": 3.0,  # Convert to float for safe comparison
            "error": 5.0,
            "critical": 10.0
        }
        self.recovery_actions = {
            "training": self._recover_training,
            "curiosity": self._recover_curiosity,
            "memory": self._recover_memory,
            "generation": self._recover_generation
        }
        
    def _is_duplicate_error(self, error: Exception, error_type: str) -> bool:
        """Check if this error is a duplicate within the cooldown period."""
        error_key = f"{error_type}:{type(error).__name__}"
        current_time = time.time()
        
        # Remove old errors from tracking using float_compare
        while self.recent_errors and float_gt(current_time - self.recent_errors[0]["timestamp"], self.error_cooldown):
            self.recent_errors.popleft()
            
        # Check for duplicates
        for recent_error in self.recent_errors:
            if recent_error["key"] == error_key:
                return True
                
        # Add to recent errors
        self.recent_errors.append({
            "key": error_key,
            "timestamp": current_time
        })
        
        return False
        
    def handle_training_error(self, error: Exception, batch_size: int) -> None:
        """Handle training errors with duplicate detection."""
        try:
            error_key = f"training:{type(error).__name__}"
            
            # Check for duplicate error
            if self._is_duplicate_error(error, "training"):
                self.logger.record_event(
                    event_type="duplicate_training_error",
                    message=f"Duplicate training error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "batch_size": batch_size
                    }
                )
                return
                
            # Increment error count
            self.error_counts[error_key] += 1
            
            # Log error
            self.logger.record_event(
                event_type="training_error",
                message=f"Training error: {str(error)}",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "batch_size": batch_size
                }
            )
            
            # Determine severity and take action using safe_compare
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_training(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_training_parameters(batch_size)
                
        except Exception as e:
            self.logger.record_event(
                event_type="error_handling_failed",
                message=f"Failed to handle training error: {str(e)}",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "batch_size": batch_size
                }
            )
            
    def handle_curiosity_error(self, error: Exception, pressure: float) -> None:
        """Handle curiosity errors with duplicate detection."""
        try:
            error_key = f"curiosity:{type(error).__name__}"
            
            # Check for duplicate error
            if self._is_duplicate_error(error, "curiosity"):
                self.logger.record_event(
                    event_type="duplicate_curiosity_error",
                    message=f"Duplicate curiosity error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "pressure": pressure
                    }
                )
                return
                
            # Increment error count
            self.error_counts[error_key] += 1
            
            # Log error
            self.logger.record_event(
                event_type="curiosity_error",
                message=f"Curiosity error: {str(error)}",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "pressure": pressure
                }
            )
            
            # Determine severity and take action using safe_compare
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_curiosity(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_curiosity_parameters(pressure)
                
        except Exception as e:
            self.logger.record_event(
                event_type="error_handling_failed",
                message=f"Failed to handle curiosity error: {str(e)}",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "pressure": pressure
                }
            )
            
    def handle_memory_error(self, error: Exception, memory_usage: float) -> None:
        """Handle memory errors with duplicate detection."""
        try:
            error_key = f"memory:{type(error).__name__}"
            
            # Check for duplicate error
            if self._is_duplicate_error(error, "memory"):
                self.logger.record_event(
                    event_type="duplicate_memory_error",
                    message=f"Duplicate memory error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "memory_usage": memory_usage
                    }
                )
                return
                
            # Increment error count
            self.error_counts[error_key] += 1
            
            # Log error
            self.logger.record_event(
                event_type="memory_error",
                message=f"Memory error: {str(error)}",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "memory_usage": memory_usage
                }
            )
            
            # Determine severity and take action using safe_compare
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_memory(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_memory_parameters(memory_usage)
                
        except Exception as e:
            self.logger.record_event(
                event_type="error_handling_failed",
                message=f"Failed to handle memory error: {str(e)}",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "memory_usage": memory_usage
                }
            )
            
    def handle_generation_error(self, error: Exception, temperature: float) -> None:
        """Handle generation errors with duplicate detection."""
        try:
            error_key = f"generation:{type(error).__name__}"
            
            # Check for duplicate error
            if self._is_duplicate_error(error, "generation"):
                self.logger.record_event(
                    event_type="duplicate_generation_error",
                    message=f"Duplicate generation error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "temperature": temperature
                    }
                )
                return
                
            # Increment error count
            self.error_counts[error_key] += 1
            
            # Log error
            self.logger.record_event(
                event_type="generation_error",
                message=f"Generation error: {str(error)}",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "temperature": temperature
                }
            )
            
            # Determine severity and take action using safe_compare
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_generation(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_generation_parameters(temperature)
                
        except Exception as e:
            self.logger.record_event(
                event_type="error_handling_failed",
                message=f"Failed to handle generation error: {str(e)}",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "temperature": temperature
                }
            )
            
    def _recover_training(self, error_key: str) -> None:
        """Recover from critical training errors."""
        try:
            # Reset error count
            self.error_counts[error_key] = 0
            
            # Take recovery actions
            self.context.config_manager.update("training_config.batch_size", 1)
            self.context.config_manager.update("training_config.learning_rate", 1e-5)
            
            self.logger.record_event(
                event_type="training_recovery",
                message="Recovered from critical training error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from training error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )
            
    def _adjust_training_parameters(self, batch_size: int) -> None:
        """Adjust training parameters for non-critical errors."""
        try:
            # Reduce batch size
            new_batch_size = max(1, batch_size // 2)
            self.context.config_manager.update("training_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="training_adjustment",
                message="Adjusted training parameters",
                level="info",
                additional_info={
                    "old_batch_size": batch_size,
                    "new_batch_size": new_batch_size
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="adjustment_failed",
                message=f"Failed to adjust training parameters: {str(e)}",
                level="error"
            )
            
    def _recover_curiosity(self, error_key: str) -> None:
        """Recover from critical curiosity errors."""
        try:
            self.error_counts[error_key] = 0
            
            # Reset curiosity parameters
            self.context.config_manager.update("curiosity_config.pressure_threshold", 0.5)
            self.context.config_manager.update("curiosity_config.decay_rate", 0.9)
            
            self.logger.record_event(
                event_type="curiosity_recovery",
                message="Recovered from critical curiosity error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from curiosity error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )
            
    def _recover_memory(self, error_key: str) -> bool:
        """Recover from critical memory errors."""
        try:
            self.error_counts[error_key] = 0
            
            # Clear memory and reduce usage
            self.context.config_manager.update("memory_config.max_memory_mb", 512)
            self.context.config_manager.update("memory_config.garbage_collection_threshold", 0.7)
            
            self.logger.record_event(
                event_type="memory_recovery",
                message="Recovered from critical memory error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
            return True
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from memory error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )
            return False
            
    def _adjust_curiosity_parameters(self, pressure: float) -> None:
        """Adjust curiosity parameters for non-critical errors."""
        try:
            # Adjust curiosity parameters
            current_pressure = self.context.config_manager.get("curiosity_config.pressure_threshold", 0.5)
            new_pressure = max(0.1, current_pressure - 0.05)
            self.context.config_manager.update("curiosity_config.pressure_threshold", new_pressure)
            
            self.logger.record_event(
                event_type="curiosity_adjustment",
                message="Adjusted curiosity parameters",
                level="info",
                additional_info={
                    "old_pressure": current_pressure,
                    "new_pressure": new_pressure,
                    "pressure": pressure
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="adjustment_failed",
                message=f"Failed to adjust curiosity parameters: {str(e)}",
                level="error"
            )
            
    def _recover_generation(self, error_key: str) -> str:
        """Recover from critical generation errors."""
        try:
            self.error_counts[error_key] = 0
            
            # Reset generation parameters
            self.context.config_manager.update("generation_config.temperature", 0.7)
            self.context.config_manager.update("generation_config.top_p", 0.9)
            
            self.logger.record_event(
                event_type="generation_recovery",
                message="Recovered from critical generation error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
            return "System recovered from error. Please try your request again."
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from generation error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )
            return "A critical error occurred. Please try again later."
            
    def _adjust_generation_parameters(self, temperature: float) -> str:
        """Adjust generation parameters for non-critical errors."""
        try:
            # Adjust generation parameters
            current_temp = self.context.config_manager.get("generation_config.temperature", 1.0)
            new_temp = max(0.5, current_temp - 0.05)
            self.context.config_manager.update("generation_config.temperature", new_temp)
            
            self.logger.record_event(
                event_type="generation_adjustment",
                message="Adjusted generation parameters",
                level="info",
                additional_info={
                    "old_temperature": current_temp,
                    "new_temperature": new_temp,
                    "temperature": temperature
                }
            )
            
            return "System adjusted parameters. Please try your request again."
            
        except Exception as e:
            self.logger.record_event(
                event_type="adjustment_failed",
                message=f"Failed to adjust generation parameters: {str(e)}",
                level="error"
            )
            return "An error occurred. Please try again."

class MemoryMonitor:
    """Monitors system memory health."""

    def __init__(self, context: SystemContext):
        """Initialize the memory monitor with context."""
        self.context = context
        self.memory_manager = MemoryManager(context)

    def check_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None) -> bool:
        """Check memory health and handle any errors."""
        try:
            is_healthy = self.memory_manager.check_memory_health(model_size, trainer)
            self.context.logger.log_memory_health(
                model_size=model_size,
                trainer=trainer,
                health_status="healthy" if is_healthy else "unhealthy",
                device=self.context.device
            )
            return is_healthy
        except Exception as e:
            # Handle memory errors through ErrorManager
            self.context.error_manager.handle_memory_error(e, model_size)
            return False

class TemperamentAdjuster:
    """Manages temperament adjustments and state updates."""
    
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        self.context = context
        self.state_tracker = state_tracker
        self.temperament_system = None
        self._last_parameter_hash = None
        self._last_state_hash = None
        self._initialize_temperament_system()
        
        # Subscribe to config changes
        self.context.config_manager.subscribe(self._on_config_change)
        
    def _validate_state_consistency(self, state: SOVLState) -> bool:
        """Validate consistency between current state and temperament history."""
        try:
            if not state.temperament_history:
                return True
                
            # Check for significant deviation between current score and history
            if abs(state.temperament_history[-1] - state.temperament_score) > 0.5:
                self.context.logger.record_event(
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
                self.context.logger.record_event(
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
            self.context.logger.record_event(
                event_type="temperament_validation_error",
                message=f"Failed to validate state consistency: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            return False
            
    def _synchronize_state(self, state: SOVLState) -> None:
        """Synchronize state with current temperament system."""
        try:
            with state.lock:
                # Validate state consistency
                if not self._validate_state_consistency(state):
                    # Reset history if inconsistent
                    state.temperament_history.clear()
                    self.context.logger.record_event(
                        event_type="temperament_history_reset",
                        message="Temperament history reset due to inconsistency",
                        level="info"
                    )
                
                # Update state with current temperament
                state.temperament_score = self.temperament_system.current_score
                state.temperament_history.append(state.temperament_score)
                
                # Update state hash
                self._last_state_hash = self._compute_state_hash(state)
                
        except Exception as e:
            self.context.logger.record_event(
                event_type="state_synchronization_error",
                message=f"Failed to synchronize state: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise
            
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
            
            self.context.logger.record_event(
                event_type="temperament_system_initialized",
                message="Temperament system initialized with validated parameters",
                level="info",
                additional_info=params
            )
            
        except Exception as e:
            self.context.logger.record_event(
                event_type="temperament_system_error",
                message=f"Failed to initialize temperament system: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise
            
    def _get_validated_parameters(self) -> Dict[str, Any]:
        """Get and validate temperament parameters."""
        config = self.context.config_manager
        
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
                self.context.logger.record_event(
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
        
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            current_params = self._get_validated_parameters()
            current_hash = self._compute_parameter_hash(current_params)
            
            if current_hash != self._last_parameter_hash:
                self.context.logger.record_event(
                    event_type="temperament_parameters_changed",
                    message="Temperament parameters changed, reinitializing system",
                    level="info",
                    additional_info=current_params
                )
                self._initialize_temperament_system()
                
        except Exception as e:
            self.context.logger.record_event(
                event_type="temperament_config_change_error",
                message=f"Failed to handle config change: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            
    def update_temperament(self, curiosity_manager: Optional[CuriosityManager] = None) -> None:
        """Update temperament based on current state and curiosity pressure."""
        try:
            if not self.temperament_system:
                self._initialize_temperament_system()
                
            state = self.state_tracker.get_state()
            lifecycle_stage = self._determine_lifecycle_stage(state)
            
            # Get current curiosity pressure and update state
            state.curiosity_pressure = curiosity_manager.get_pressure() if curiosity_manager else 0.0
            
            # Get confidence from history or use default
            confidence = state.confidence_history[-1] if state.confidence_history else 0.5
            
            # Update temperament with current curiosity pressure
            self.temperament_system.update(
                confidence=confidence,
                lifecycle_stage=lifecycle_stage,
                curiosity_pressure=state.curiosity_pressure
            )
            
            # Synchronize state after update
            self._synchronize_state(state)
            
            # Log the update
            self.context.logger.record_event(
                event_type="temperament_updated",
                message="Temperament updated with current state",
                level="info",
                additional_info={
                    "temperament_score": state.temperament_score,
                    "curiosity_pressure": state.curiosity_pressure,
                    "lifecycle_stage": lifecycle_stage,
                    "confidence": confidence
                }
            )
            
        except Exception as e:
            self.context.logger.record_event(
                event_type="temperament_update_error",
                message=f"Failed to update temperament: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise
            
    def _determine_lifecycle_stage(self, state: SOVLState) -> str:
        """Determine the current lifecycle stage based on state."""
        # Implementation depends on your lifecycle logic
        return "active"  # Placeholder

class CuriosityEngine:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        context: SystemContext,
        model_loader: ModelLoader,
        state_tracker: StateTracker,
        error_manager: ErrorManager
    ):
        self.context = context
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self.logger = context.logger
        
        # Initialize components
        self._initialize_curiosity_manager()
        self._initialize_training_cycle_manager()
        
    def _validate_configuration(self) -> bool:
        """Validate current configuration state."""
        try:
            if not self.context.config_handler.validate():
                self.logger.record_event(
                    event_type="config_validation_failed",
                    message="Configuration validation failed, attempting recovery",
                    level="error"
                )
                # Attempt to refresh configuration
                self.context.config_handler._refresh_configs()
                # Re-validate after refresh
                if not self.context.config_handler.validate():
                    self.logger.record_event(
                        event_type="config_recovery_failed",
                        message="Configuration recovery failed",
                        level="error"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.record_event(
                event_type="config_validation_error",
                message=f"Error during configuration validation: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return False
            
    def run_training_cycle(
        self,
        train_data: Optional[List] = None,
        valid_data: Optional[List] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """Run a training cycle with configuration validation."""
        try:
            # Validate configuration before proceeding
            if not self._validate_configuration():
                raise RuntimeError("Invalid configuration state")
                
            # Get current state
            state = self.state_tracker.get_state()
            
            # Run training cycle through cycle manager
            results = self.cycle_manager.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Log training completion
            self._log_event("training_complete", {
                "epochs": epochs,
                "batch_size": batch_size,
                "results": results
            })
            
        except Exception as e:
            self.error_manager.handle_training_error(e, batch_size or 1)
            raise

    def _initialize_training_cycle_manager(self) -> None:
        """Initialize training cycle manager."""
        try:
            self.cycle_manager = TrainingCycleManager(
                config=self.context.config_manager.get_section("sovl_config"),
                logger=self.logger,
                device=self.context.device,
                state_manager=self.state_tracker,
                curiosity_manager=self.curiosity_manager
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, "cycle_manager_init")
            raise
            
    def _initialize_curiosity_manager(self) -> None:
        """Initialize the curiosity manager with proper error handling."""
        try:
            curiosity_manager = CuriosityManager(
                config_manager=self.context.config_manager,
                logger=self.context.logger,
                device=self.context.device
            )
            self.context.logger.info("Curiosity manager initialized successfully")
            self.curiosity_manager = curiosity_manager
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to initialize curiosity manager: {str(e)}",
                error_type="initialization_error",
                stack_trace=traceback.format_exc()
            )
            raise SystemInitializationError(
                message="Failed to initialize curiosity manager",
                config_path=self.context.config_path,
                stack_trace=traceback.format_exc()
            )

    def _log_event(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=f"curiosity_{event}",
            message=f"Curiosity event: {event}",
            level="info",
            additional_info=data
        )
