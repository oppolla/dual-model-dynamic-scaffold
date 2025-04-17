from typing import Optional, Any, List, Dict, Tuple, Callable
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
from sovl_trainer import TrainingConfig, SOVLTrainer, TrainingCycleManager
from sovl_config import ConfigManager, ConfigHandler, ValidationSchema
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager, CrossAttentionLayer, ScaffoldTokenMapper
from sovl_processor import LogitsProcessor, SOVLProcessor
from sovl_utils import (
    detect_repetitions,
    safe_compare,
    float_gt,
    synchronized
)
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_memory import MemoryManager
from sovl_manager import ModelManager
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner
from sovl_error import ErrorHandler
from sovl_state_manager import StateManager
from sovl_logging import LoggingManager
from sovl_plugin import PluginManager
from sovl_utils import NumericalGuard
from sovl_confidence import calculate_confidence_score
from sovl_events import EventDispatcher

# Remove sovl_conductor import and use TYPE_CHECKING for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

class SystemContext:
    """Manages system-wide context and resources."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        """
        Initialize system context with shared resources.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for tensor operations
        """
        self.config_path = config_path
        self.device = device
        self.logger = Logger()
        self.event_dispatcher = EventDispatcher(self.logger)
        
        # Initialize config manager with event dispatcher
        self.config_handler = ConfigHandler(config_path, self.logger, self.event_dispatcher)
        
        # Validate initial configuration
        if not self.config_handler.validate():
            raise SystemInitializationError(
                "Failed to validate initial configuration",
                config_path,
                traceback.format_exc()
            )
        
        # Subscribe to configuration changes
        self.event_dispatcher.subscribe("config_change", self._on_config_change)
        
    def _on_config_change(self) -> None:
        """Handle configuration changes and propagate them to affected components."""
        try:
            # Validate configuration after change
            if not self.config_handler.validate():
                self.logger.log_error(
                    error_msg="Configuration validation failed after change",
                    error_type="config_validation_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "config_path": self.config_path,
                        "timestamp": time.time()
                    }
                )
                return
                
            self.logger.record_event(
                event_type="config_change",
                message="Configuration changed and validated successfully",
                level="info",
                additional_info={
                    "timestamp": time.time(),
                    "config_path": self.config_path
                }
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="config_change_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": self.config_path,
                    "timestamp": time.time()
                }
            )

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class ModelLoader:
    """Handles model loading and initialization."""
    
    def __init__(self, context: SystemContext):
        """
        Initialize model loader with system context.
        
        Args:
            context: System context containing shared resources
        """
        self.context = context
        self.logger = context.logger
        self.config = context.config_handler.get_section("model")
        
        # Validate model configuration
        if not self._validate_model_config():
            raise ModelLoadingError(
                "Invalid model configuration",
                self.config,
                traceback.format_exc()
            )
            
    def _validate_model_config(self) -> bool:
        """Validate model configuration section."""
        try:
            required_fields = ["model_path", "model_type", "quantization_mode"]
            for field in required_fields:
                if field not in self.config:
                    self.logger.log_error(
                        error_msg=f"Missing required model configuration field: {field}",
                        error_type="config_validation_error",
                        stack_trace=None,
                        additional_info={
                            "missing_field": field,
                            "config_section": "model"
                        }
                    )
                    return False
                    
            # Validate quantization mode
            if not validate_quantization_mode(self.config["quantization_mode"]):
                self.logger.log_error(
                    error_msg=f"Invalid quantization mode: {self.config['quantization_mode']}",
                    error_type="config_validation_error",
                    stack_trace=None,
                    additional_info={
                        "invalid_value": self.config["quantization_mode"],
                        "config_section": "model"
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "model"
                }
            )
            return False
            
    def load_model(self) -> torch.nn.Module:
        """Load and initialize the model with validated configuration."""
        try:
            if not self._validate_model_config():
                raise ModelLoadingError(
                    "Cannot load model with invalid configuration",
                    self.config,
                    traceback.format_exc()
                )
                
            # Load model with validated configuration
            model = load_model(
                self.config["model_path"],
                self.config["model_type"],
                self.config["quantization_mode"],
                self.context.device
            )
            
            self.logger.record_event(
                event_type="model_loaded",
                message="Model loaded successfully",
                level="info",
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"],
                    "device": self.context.device
                }
            )
            
            return model
            
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"]
                }
            )
            raise

class StateTracker:
    """Tracks and manages system state."""
    
    def __init__(self, context: SystemContext):
        """
        Initialize state tracker with system context.
        
        Args:
            context: System context containing shared resources
        """
        self.context = context
        self.logger = context.logger
        self.config = context.config_handler.get_section("state")
        self.state = {}
        
        # Subscribe to configuration changes
        self.context.event_dispatcher.subscribe(
            "config_changed",
            self._on_config_change
        )
        
        # Validate initial state configuration
        if not self._validate_state_config():
            raise StateTrackingError(
                "Invalid state configuration",
                self.config,
                traceback.format_exc()
            )
            
    def _validate_state_config(self) -> bool:
        """Validate state configuration section."""
        try:
            required_fields = ["state_file", "backup_interval", "max_history"]
            for field in required_fields:
                if field not in self.config:
                    self.logger.log_error(
                        error_msg=f"Missing required state configuration field: {field}",
                        error_type="config_validation_error",
                        stack_trace=None,
                        additional_info={
                            "missing_field": field,
                            "config_section": "state"
                        }
                    )
                    return False
                    
            # Validate backup interval
            if not isinstance(self.config["backup_interval"], int) or self.config["backup_interval"] <= 0:
                self.logger.log_error(
                    error_msg=f"Invalid backup interval: {self.config['backup_interval']}",
                    error_type="config_validation_error",
                    stack_trace=None,
                    additional_info={
                        "invalid_value": self.config["backup_interval"],
                        "config_section": "state"
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "state"
                }
            )
            return False
            
    def _on_config_change(self, event_data: dict) -> None:
        """Handle configuration changes."""
        try:
            if "state" in event_data:
                self.config = event_data["state"]
                
                # Validate new configuration
                if not self._validate_state_config():
                    self.logger.log_error(
                        error_msg="Invalid state configuration after change",
                        error_type="config_validation_error",
                        stack_trace=None,
                        additional_info={
                            "new_config": self.config
                        }
                    )
                    return
                    
                self.logger.log_training_event(
                    event_type="config_updated",
                    message="State configuration updated successfully",
                    additional_info={
                        "config_section": "state",
                        "changes": event_data["state"]
                    }
                )
                
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="config_update_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "state",
                    "event_data": event_data
                }
            )
            
    def update_state(self, key: str, value: Any) -> None:
        """Update state with validated configuration."""
        try:
            if not self._validate_state_config():
                raise StateTrackingError(
                    "Cannot update state with invalid configuration",
                    self.config,
                    traceback.format_exc()
                )
                
            self.state[key] = value
            
            self.logger.log_training_event(
                event_type="state_updated",
                message=f"State updated for key: {key}",
                additional_info={
                    "key": key,
                    "value": value,
                    "state_file": self.config["state_file"]
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="state_update_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "key": key,
                    "value": value
                }
            )
            raise

class ErrorManager:
    """Manages error handling and recovery for the SOVL system."""
    
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        """Initialize error manager with required dependencies."""
        self.context = context
        self.state_tracker = state_tracker
        self.logger = context.logger
        self.error_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=100)
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize error handling configuration."""
        config = self.context.config_handler.config_manager.get_section("error_config", {})
        self.error_cooldown = config.get("error_cooldown", 1.0)
        self.severity_thresholds = {
            "warning": float(config.get("warning_threshold", 3.0)),
            "error": float(config.get("error_threshold", 5.0)),
            "critical": float(config.get("critical_threshold", 10.0))
        }
        self.recovery_actions = {
            "training": self._recover_training,
            "curiosity": self._recover_curiosity,
            "memory": self._recover_memory,
            "generation": self._recover_generation,
            "data": self._recover_data
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
            
            if self._is_duplicate_error(error, "training"):
                self._log_event(
                    event_type="duplicate_training_error",
                    message=f"Duplicate training error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "batch_size": batch_size
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self._log_error(
                error=error,
                context="training",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "batch_size": batch_size
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_training(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_training_parameters(batch_size)
                
        except Exception as e:
            self._log_error(
                error=e,
                context="error_handling",
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
            
            if self._is_duplicate_error(error, "curiosity"):
                self._log_event(
                    event_type="duplicate_curiosity_error",
                    message=f"Duplicate curiosity error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "pressure": pressure
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self._log_error(
                error=error,
                context="curiosity",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "pressure": pressure
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_curiosity(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_curiosity_parameters(pressure)
                
        except Exception as e:
            self._log_error(
                error=e,
                context="error_handling",
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
            
            if self._is_duplicate_error(error, "memory"):
                self._log_event(
                    event_type="duplicate_memory_error",
                    message=f"Duplicate memory error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "memory_usage": memory_usage
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self._log_error(
                error=error,
                context="memory",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "memory_usage": memory_usage
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_memory(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_memory_parameters(memory_usage)
                
        except Exception as e:
            self._log_error(
                error=e,
                context="error_handling",
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
            
            if self._is_duplicate_error(error, "generation"):
                self._log_event(
                    event_type="duplicate_generation_error",
                    message=f"Duplicate generation error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "temperature": temperature
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self._log_error(
                error=error,
                context="generation",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "temperature": temperature
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_generation(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_generation_parameters(temperature)
                
        except Exception as e:
            self._log_error(
                error=e,
                context="error_handling",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "temperature": temperature
                }
            )
            
    def handle_data_error(self, error: Exception, context: Dict[str, Any], conversation_id: str) -> None:
        """Handle data-related errors with duplicate detection."""
        try:
            error_key = f"data:{type(error).__name__}"
            
            if self._is_duplicate_error(error, "data"):
                self._log_event(
                    event_type="duplicate_data_error",
                    message=f"Duplicate data error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "context": context,
                        "conversation_id": conversation_id
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self._log_error(
                error=error,
                context="data",
                level="error",
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "context": context,
                    "conversation_id": conversation_id
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_data(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_data_parameters(context)
                
        except Exception as e:
            self._log_error(
                error=e,
                context="error_handling",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "context": context,
                    "conversation_id": conversation_id
                }
            )

    def _recover_training(self, error_key: str) -> None:
        """Recover from critical training errors."""
        try:
            # Reset error count
            self.error_counts[error_key] = 0
            
            # Take recovery actions
            self.context.config_handler.config_manager.update("training_config.batch_size", 1)
            self.context.config_handler.config_manager.update("training_config.learning_rate", 1e-5)
            
            self._log_event(
                event_type="training_recovery",
                message="Recovered from critical training error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self._log_error(
                error=e,
                context="recovery",
                level="critical",
                additional_info={"error_key": error_key}
            )
            
    def _adjust_training_parameters(self, batch_size: int) -> None:
        """Adjust training parameters for non-critical errors."""
        try:
            # Reduce batch size
            new_batch_size = max(1, batch_size // 2)
            self.context.config_handler.config_manager.update("training_config.batch_size", new_batch_size)
            
            self._log_event(
                event_type="training_adjustment",
                message="Adjusted training parameters",
                level="info",
                additional_info={
                    "old_batch_size": batch_size,
                    "new_batch_size": new_batch_size
                }
            )
            
        except Exception as e:
            self._log_error(
                error=e,
                context="adjustment",
                level="error"
            )
            
    def _recover_curiosity(self, error_key: str) -> None:
        """Recover from critical curiosity errors."""
        try:
            self.error_counts[error_key] = 0
            
            # Reset curiosity parameters
            self.context.config_handler.config_manager.update("curiosity_config.pressure_threshold", 0.5)
            self.context.config_handler.config_manager.update("curiosity_config.decay_rate", 0.9)
            
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
            self.context.config_handler.config_manager.update("memory_config.max_memory_mb", 512)
            self.context.config_handler.config_manager.update("memory_config.garbage_collection_threshold", 0.7)
            
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
            current_pressure = self.context.config_handler.config_manager.get("curiosity_config.pressure_threshold", 0.5)
            new_pressure = max(0.1, current_pressure - 0.05)
            self.context.config_handler.config_manager.update("curiosity_config.pressure_threshold", new_pressure)
            
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
            self.context.config_handler.config_manager.update("generation_config.temperature", 0.7)
            self.context.config_handler.config_manager.update("generation_config.top_p", 0.9)
            
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
            current_temp = self.context.config_handler.config_manager.get("generation_config.temperature", 1.0)
            new_temp = max(0.5, current_temp - 0.05)
            self.context.config_handler.config_manager.update("generation_config.temperature", new_temp)
            
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

    def _recover_data(self, error_key: str) -> None:
        """Recover from critical data errors."""
        try:
            self.error_counts[error_key] = 0
            
            # Reset data parameters
            self.context.config_handler.config_manager.update("data_config.batch_size", 1)
            self.context.config_handler.config_manager.update("data_config.max_retries", 3)
            
            self.logger.record_event(
                event_type="data_recovery",
                message="Recovered from critical data error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from data error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )

    def _adjust_data_parameters(self, context: Dict[str, Any]) -> None:
        """Adjust data parameters for non-critical errors."""
        try:
            # Adjust data parameters
            current_batch_size = self.context.config_handler.config_manager.get("data_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.context.config_handler.config_manager.update("data_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="data_adjustment",
                message="Adjusted data parameters",
                level="info",
                additional_info={
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    "context": context
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="adjustment_failed",
                message=f"Failed to adjust data parameters: {str(e)}",
                level="error"
            )

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

class CuriosityEngine:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_handler: ConfigHandler,
        model_loader: ModelLoader,
        state_tracker: StateTracker,
        error_manager: ErrorManager,
        logger: Logger,
        device: str
    ):
        """
        Initialize the curiosity engine with explicit dependencies.
        
        Args:
            config_handler: Configuration handler
            model_loader: Model loader instance
            state_tracker: State tracker instance
            error_manager: Error manager instance
            logger: Logger instance
            device: Device to use for tensor operations
        """
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self.logger = logger
        self.device = device
        
        # Initialize components
        self.curiosity_manager = self._create_curiosity_manager()
        self.cycle_manager = self._create_training_cycle_manager()
        
        # Log initialization
        self.logger.record_event(
            event_type="curiosity_engine_initialized",
            message="Curiosity engine initialized successfully",
            level="info"
        )
        
    def _create_curiosity_manager(self) -> CuriosityManager:
        """Create and initialize the curiosity manager."""
        try:
            return CuriosityManager(
                config_manager=self.config_handler.config_manager,
                logger=self.logger,
                device=self.device
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, "manager_creation")
            raise
            
    def _create_training_cycle_manager(self) -> TrainingCycleManager:
        """Create and initialize the training cycle manager."""
        try:
            return TrainingCycleManager(
                config=self.config_handler.config_manager.get_section("sovl_config"),
                logger=self.logger,
                device=self.device,
                state_manager=self.state_tracker,
                curiosity_manager=self.curiosity_manager
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, "cycle_manager_creation")
            raise
            
    def _validate_configuration(self) -> bool:
        """Validate current configuration state."""
        try:
            if not self.config_handler.validate():
                self.logger.record_event(
                    event_type="config_validation_failed",
                    message="Configuration validation failed, attempting recovery",
                    level="error"
                )
                # Attempt to refresh configuration
                self.config_handler._refresh_configs()
                # Re-validate after refresh
                if not self.config_handler.validate():
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
                
            # Validate DataManager state
            if not hasattr(self.cycle_manager, 'data_manager'):
                raise RuntimeError("DataManager not initialized in cycle manager")
                
            if not hasattr(self.cycle_manager.data_manager, 'provider'):
                raise RuntimeError("DataManager provider not initialized")
                
            if not isinstance(self.cycle_manager.data_manager.provider, DataProvider):
                raise RuntimeError(f"Invalid DataManager provider type: {type(self.cycle_manager.data_manager.provider)}")
                
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
                "results": results,
                "data_manager_state": {
                    "provider_type": type(self.cycle_manager.data_manager.provider).__name__,
                    "provider_initialized": hasattr(self.cycle_manager.data_manager.provider, '_initialized')
                }
            })
            
        except Exception as e:
            self.error_manager.handle_training_error(e, batch_size or 1)
            raise
            
    def _log_event(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=f"curiosity_{event}",
            message=f"Curiosity event: {event}",
            level="info",
            additional_info=data
        )

class SOVLSystem:
    """Main SOVL system class that manages all components and state."""
    
    def __init__(
        self,
        context: SystemContext,
        config_handler: ConfigHandler,
        model_loader: ModelLoader,
        curiosity_engine: CuriosityEngine,
        memory_monitor: MemoryMonitor,
        state_tracker: StateTracker,
        error_manager: ErrorManager
    ):
        """
        Initialize the SOVL system with pre-initialized components.
        
        Args:
            context: System context containing shared resources
            config_handler: Configuration handler component
            model_loader: Model loading component
            curiosity_engine: Curiosity engine component
            memory_monitor: Memory monitoring component
            state_tracker: State tracking component
            error_manager: Error management component
        """
        try:
            # Validate required components
            self._validate_components(
                context=context,
                config_handler=config_handler,
                model_loader=model_loader,
                curiosity_engine=curiosity_engine,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            # Store injected components
            self.context = context
            self.config_handler = config_handler
            self.model_loader = model_loader
            self.curiosity_engine = curiosity_engine
            self.memory_monitor = memory_monitor
            self.state_tracker = state_tracker
            self.error_manager = error_manager
            
            # Initialize thread safety
            self._lock = Lock()
            
            # Initialize component state
            self._initialize_component_state()
            
            # Log successful initialization
            self.context.logger.record_event(
                event_type="system_initialized",
                message="SOVL system initialized successfully with dependency injection",
                level="info",
                additional_info={
                    "config_path": self.config_handler.config_path,
                    "device": self.context.device,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to initialize SOVL system: {str(e)}",
                error_type="system_initialization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": self.config_handler.config_path if hasattr(self, 'config_handler') else None,
                    "device": self.context.device if hasattr(self, 'context') else None
                }
            )
            raise

    def _validate_components(self, **components) -> None:
        """Validate that all required components are properly initialized."""
        for name, component in components.items():
            if component is None:
                raise ValueError(f"Required component {name} is None")
            if not hasattr(component, '__class__'):
                raise ValueError(f"Component {name} is not a valid object")

    def _initialize_component_state(self) -> None:
        """Initialize state for all components."""
        try:
            # Initialize state tracker if needed
            if not self.state_tracker.state:
                self.state_tracker.initialize_state()
            
            # Sync state between components
            self._sync_component_states()
            
            # Validate component states
            self._validate_component_states()
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to initialize component state: {str(e)}",
                error_type="component_state_initialization_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _sync_component_states(self) -> None:
        """Synchronize state between components."""
        try:
            # Sync state with curiosity engine
            if hasattr(self.curiosity_engine, 'state_tracker'):
                self.curiosity_engine.state_tracker = self.state_tracker
            
            # Sync state with memory monitor
            if hasattr(self.memory_monitor, 'state_tracker'):
                self.memory_monitor.state_tracker = self.state_tracker
            
            # Sync state with model loader
            if hasattr(self.model_loader, 'state_tracker'):
                self.model_loader.state_tracker = self.state_tracker
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to sync component states: {str(e)}",
                error_type="component_state_sync_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _validate_component_states(self) -> None:
        """Validate that all components have consistent state."""
        try:
            # Validate state tracker
            if not self.state_tracker.state:
                raise ValueError("State tracker state not initialized")
            
            # Validate state hashes
            state_hash = self.state_tracker.state.state_hash
            for component in [self.curiosity_engine, self.memory_monitor, self.model_loader]:
                if hasattr(component, 'state_tracker') and component.state_tracker.state.state_hash != state_hash:
                    raise ValueError(f"State hash mismatch in {component.__class__.__name__}")
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to validate component states: {str(e)}",
                error_type="component_state_validation_error",
                stack_trace=traceback.format_exc()
            )
            raise

    @classmethod
    def create_from_config(cls, config_path: str, device: str = "cuda") -> 'SOVLSystem':
        """
        Factory method to create a SOVLSystem instance from configuration.
        
        Args:
            config_path: Path to the configuration file
            device: Device to use for tensor operations
            
        Returns:
            SOVLSystem: A new instance initialized with components created from config
        """
        try:
            # Initialize shared context
            context = SystemContext(config_path, device)
            
            # Initialize state tracker and error manager first
            state_tracker = StateTracker()
            error_manager = ErrorManager()
            
            # Initialize components with explicit dependencies
            config_handler = ConfigHandler(
                config_path=config_path,
                logger=context.logger,
                state_tracker=state_tracker
            )
            
            model_loader = ModelLoader(
                config_handler=config_handler,
                logger=context.logger,
                device=context.device
            )
            
            curiosity_engine = CuriosityEngine(
                config_handler=config_handler,
                model_loader=model_loader,
                state_tracker=state_tracker,
                error_manager=error_manager,
                logger=context.logger,
                device=context.device
            )
            
            memory_monitor = MemoryMonitor(
                logger=context.logger,
                device=context.device
            )
            
            # Create and return new instance with all components
            return cls(
                context=context,
                config_handler=config_handler,
                model_loader=model_loader,
                curiosity_engine=curiosity_engine,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
        except Exception as e:
            if 'context' in locals() and hasattr(context, 'logger'):
                context.logger.log_error(
                    error_msg=f"Failed to create SOVL system from config: {str(e)}",
                    error_type="system_creation_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "config_path": config_path,
                        "device": device
                    }
                )
            raise

    @synchronized("_lock")
    def toggle_memory(self, enable: bool) -> bool:
        """Enable or disable memory management."""
        try:
            if not hasattr(self.memory_monitor, 'memory_manager'):
                self.context.logger.record_event(
                    event_type="memory_error",
                    message="Memory manager not initialized",
                    level="error"
                )
                return False
                
            self.memory_monitor.memory_manager.set_enabled(enable)
            self.context.logger.record_event(
                event_type="memory_toggle",
                message=f"Memory management {'enabled' if enable else 'disabled'}",
                level="info",
                additional_info={
                    "enabled": enable,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            return True
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, 0)  # 0 for memory size since this is a toggle operation
            self.context.logger.log_error(
                error_msg=f"Failed to toggle memory management: {str(e)}",
                error_type="memory_toggle_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "enabled": enable,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            return False

    def generate_curiosity_question(self) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            if not hasattr(self.curiosity_engine, 'curiosity_manager'):
                self.logger.record_event(
                    event_type="curiosity_error",
                    message="Curiosity manager not initialized",
                    level="error"
                )
                return None
                
            question = self.curiosity_engine.curiosity_manager.generate_question()
            self.logger.record_event(
                event_type="curiosity_question_generated",
                message="Generated curiosity question",
                level="info",
                additional_info={"question": question}
            )
            return question
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, "question_generation")
            self.logger.record_event(
                event_type="curiosity_question_error",
                message="Failed to generate curiosity question",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return None

    def dream(self) -> bool:
        """Run a dream cycle to process and consolidate memories."""
        try:
            if not hasattr(self.curiosity_engine, 'run_dream_cycle'):
                self.logger.record_event(
                    event_type="dream_error",
                    message="Dream cycle not supported",
                    level="error"
                )
                return False
                
            self.curiosity_engine.run_dream_cycle()
            self.logger.record_event(
                event_type="dream_cycle_complete",
                message="Dream cycle completed successfully",
                level="info"
            )
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, "dream_cycle")
            self.logger.record_event(
                event_type="dream_cycle_error",
                message="Failed to run dream cycle",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            if not hasattr(self.memory_monitor, 'memory_manager'):
                return {"error": "Memory manager not initialized"}
                
            stats = self.memory_monitor.memory_manager.get_stats()
            self.logger.record_event(
                event_type="memory_stats_retrieved",
                message="Retrieved memory statistics",
                level="info",
                additional_info=stats
            )
            return stats
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, 0)
            self.logger.record_event(
                event_type="memory_stats_error",
                message="Failed to get memory statistics",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return {"error": str(e)}
