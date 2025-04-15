from typing import Optional, Dict, Any, Callable
import traceback
import time
from collections import defaultdict, deque
from threading import Lock
from sovl_logger import Logger
from sovl_state import SOVLState
import torch

class ErrorHandler:
    """Handles error logging, recovery, and monitoring for the SOVL system."""

    _DEFAULT_CONFIG = {
        "error_handling.max_history_per_error": 10,
        "error_handling.critical_threshold": 5,
        "error_handling.warning_threshold": 10,
        "error_handling.retry_attempts": 3,
        "error_handling.retry_delay": 1.0,
        "error_handling.memory_recovery_attempts": 3,
        "error_handling.memory_recovery_delay": 1.0,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Logger,
        error_log_file: str = "sovl_errors.jsonl",
        max_error_log_size_mb: int = 10,
        compress_old: bool = True,
    ):
        """Initialize the error handler with configuration and logging dependencies."""
        self.config = {**self._DEFAULT_CONFIG, **config}
        self.logger = logger
        self.lock = Lock()
        self.error_counts = defaultdict(int)
        self.error_history = defaultdict(lambda: deque(maxlen=self._get_max_history_per_error()))
        self.severity_thresholds = self._load_severity_thresholds()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self._validate_config()

    def _get_max_history_per_error(self) -> int:
        """Get the maximum number of error instances to keep in history."""
        return self.config["error_handling.max_history_per_error"]

    def _load_severity_thresholds(self) -> Dict[str, int]:
        """Load severity thresholds from configuration."""
        return {
            "critical": self.config["error_handling.critical_threshold"],
            "warning": self.config["error_handling.warning_threshold"]
        }

    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            "model_loading": self._recover_model_loading,
            "training": self._recover_training,
            "generation": self._recover_generation,
            "memory": self._recover_memory,
            "default": self._recover_default
        }

    def _validate_config(self) -> None:
        """Validate error handling configuration."""
        try:
            # Ensure all required keys are present with valid values
            for key, (default, (min_val, max_val)) in self._DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = default
                    self.logger.record_event(
                        event_type="error_config_missing_key",
                        message=f"Added missing error handling key {key} with default {default}",
                        level="warning"
                    )
                value = self.config[key]
                if not (min_val <= value <= max_val):
                    self.config[key] = default
                    self.logger.record_event(
                        event_type="error_config_invalid_value",
                        message=f"Invalid {key}: {value}. Reset to default {default}",
                        level="warning"
                    )
            
            # Log final configuration
            self.logger.record_event(
                event_type="error_handler_config_validated",
                message="Error handler configuration validated successfully",
                level="info",
                additional_info={"config": self.config}
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="error_config_validation_failed",
                message=f"Failed to validate error handler configuration: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _log_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        severity: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error details using the standardized logger interface."""
        error_info = {
            "context": context,
            "phase": phase,
            "severity": severity,
            "timestamp": time.time(),
            **(additional_info or {})
        }
        
        self.logger.log_error(
            error_msg=str(error),
            error_type=f"{context}_{phase}_error",
            stack_trace=traceback.format_exc(),
            additional_info=error_info
        )

    def _get_state_hash(self) -> Optional[str]:
        """Safely get the current state hash with proper locking."""
        if not self.state:
            return None
        with self.state.lock:
            return self.state.state_hash()

    def _get_state_info(self) -> Dict[str, Any]:
        """Safely get state information with proper locking."""
        if not self.state:
            return {}
        with self.state.lock:
            return {
                "state_hash": self.state.state_hash(),
                "conversation_id": self.state.history.conversation_id if hasattr(self.state, 'history') else None
            }

    def handle_generation_error(self, error: Exception, prompt: str, state: Optional[SOVLState] = None) -> str:
        """Handle generation errors and return a fallback response."""
        with self.lock:
            error_key = f"generation:generate:{type(error).__name__}"
            self.record_error(
                error=error,
                context="generation",
                phase="generate",
                additional_info={
                    "prompt": prompt[:200],  # Truncate for logging
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_generation(error_key)
                
        return "An error occurred during generation"

    def handle_training_error(self, error: Exception, batch_size: int, state: Optional[SOVLState] = None) -> None:
        """Handle training errors."""
        with self.lock:
            error_key = f"training:train_step:{type(error).__name__}"
            self.record_error(
                error=error,
                context="training",
                phase="train_step",
                additional_info={
                    "batch_size": batch_size,
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_training(error_key)

    def handle_data_loading_error(self, error: Exception, file_path: str) -> None:
        """Handle data loading errors."""
        self.record_error(
            error=error,
            context="data_loading",
            phase="load_training_data",
            additional_info={
                "file_path": file_path,
                "is_error_prompt": True
            },
            severity="error"
        )

    def handle_curiosity_error(self, error: Exception, event_type: str, state: Optional[SOVLState] = None) -> None:
        """Handle curiosity-related errors."""
        with self.lock:
            error_key = f"curiosity:{event_type}:{type(error).__name__}"
            self.record_error(
                error=error,
                context="curiosity",
                phase=event_type,
                additional_info={
                    "error_type": "generation_error" if event_type == "question_generation" else "curiosity_error",
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_curiosity(error_key)

    def handle_memory_error(self, error: Exception, model_size: int, state: Optional[SOVLState] = None) -> None:
        """Handle memory errors and attempt recovery."""
        with self.lock:
            error_key = f"memory:check_health:{type(error).__name__}"
            self.record_error(
                error=error,
                context="memory",
                phase="check_health",
                additional_info={
                    "model_size": model_size,
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None,
                    "device": str(state.device) if state else None,
                    "memory_stats": self._get_memory_stats() if torch.cuda.is_available() else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_memory(error_key)

    def record_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        additional_info: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        reraise: bool = False
    ) -> None:
        """
        Record an error with detailed context and handle it according to severity.

        Args:
            error: The exception that was raised.
            context: The context in which the error occurred (e.g., "training", "generation").
            phase: The specific phase or operation where the error occurred.
            additional_info: Additional context-specific information.
            severity: The severity level of the error ("error", "warning", "critical").
            reraise: Whether to re-raise the exception after handling.
        """
        with self.lock:
            error_key = f"{context}:{phase}:{type(error).__name__}"
            self.error_counts[error_key] += 1
            self.error_history[error_key].append({
                "timestamp": time.time(),
                "error": str(error),
                "stack_trace": traceback.format_exc(),
                "severity": severity,
                "additional_info": additional_info or {}
            })

            # Log the error using standardized format
            self._log_error(error, context, phase, severity, additional_info)

            # Log the event using record_event
            self.logger.record_event(
                event_type=f"error_{context}_{phase}",
                message=f"Error recorded: {str(error)}",
                level=severity,
                additional_info={
                    "context": context,
                    "phase": phase,
                    "stack_trace": traceback.format_exc(),
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    **(additional_info or {})
                }
            )

            # Check if we've exceeded thresholds
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._handle_critical_error(error_key)
            elif self.error_counts[error_key] >= self.severity_thresholds["warning"]:
                self._handle_warning(error_key)

            if reraise:
                raise error

    def _handle_critical_error(self, error_key: str) -> None:
        """Handle critical errors that exceed the threshold."""
        error_history = list(self.error_history[error_key])
        self.logger.record_event(
            event_type="critical_error_threshold_exceeded",
            message=f"Critical error threshold exceeded for {error_key}",
            level="critical",
            additional_info={
                "error_key": error_key,
                "count": self.error_counts[error_key],
                "recent_errors": error_history[-5:],  # Last 5 errors
                "timestamp": time.time()
            }
        )

        # Attempt recovery based on error context
        context = error_key.split(":")[0]
        recovery_strategy = self.recovery_strategies.get(context, self.recovery_strategies["default"])
        try:
            recovery_strategy(error_key)
        except Exception as e:
            self._log_error(
                error=e,
                context="error_recovery",
                phase="critical_error",
                severity="error",
                additional_info={"error_key": error_key}
            )

    def _handle_warning(self, error_key: str) -> None:
        """Handle warnings that exceed the threshold."""
        self.logger.record_event(
            event_type="warning_threshold_exceeded",
            message=f"Warning threshold exceeded for {error_key}",
            level="warning",
            additional_info={
                "error_key": error_key,
                "count": self.error_counts[error_key],
                "timestamp": time.time()
            }
        )

    def _recover_model_loading(self, error_key: str) -> None:
        """Recovery strategy for model loading errors."""
        try:
            self.logger.record_event(
                event_type="model_loading_recovery_attempt",
                message=f"Attempting model loading recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Clear model cache and reload
                    if hasattr(self.state, 'model'):
                        del self.state.model
                    if hasattr(self.state, 'tokenizer'):
                        del self.state.tokenizer
                        
                    # Reset model configuration
                    self.state.model_config = self.state.initial_model_config.copy()
                    
                    self.logger.record_event(
                        event_type="model_loading_recovery",
                        message="Model cache cleared and configuration reset",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="model_loading_recovery_failed",
                message=f"Model loading recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_training(self, error_key: str) -> None:
        """Recovery strategy for training errors."""
        try:
            self.logger.record_event(
                event_type="training_recovery_attempt",
                message=f"Attempting training recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Reduce batch size
                    current_batch = self.state.training_config.get("batch_size", 4)
                    new_batch = max(1, current_batch // 2)
                    self.state.training_config["batch_size"] = new_batch
                    
                    # Adjust learning rate
                    current_lr = self.state.training_config.get("learning_rate", 0.0003)
                    new_lr = current_lr * 0.5
                    self.state.training_config["learning_rate"] = new_lr
                    
                    self.logger.record_event(
                        event_type="training_recovery",
                        message=f"Reduced batch size to {new_batch} and learning rate to {new_lr}",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="training_recovery_failed",
                message=f"Training recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_generation(self, error_key: str) -> None:
        """Recovery strategy for generation errors."""
        try:
            self.logger.record_event(
                event_type="generation_recovery_attempt",
                message=f"Attempting generation recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Reset generation parameters to safer values
                    self.state.generation_config["temperature"] = 0.7
                    self.state.generation_config["top_p"] = 0.9
                    self.state.generation_config["max_length"] = 128
                    
                    # Clear generation cache
                    if hasattr(self.state, 'generation_cache'):
                        self.state.generation_cache.clear()
                    
                    self.logger.record_event(
                        event_type="generation_recovery",
                        message="Generation parameters reset to safer values",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="generation_recovery_failed",
                message=f"Generation recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_memory(self, error_key: str) -> None:
        """Recovery strategy for memory errors."""
        try:
            self.logger.record_event(
                event_type="memory_recovery_attempt",
                message=f"Attempting memory recovery for {error_key}",
                level="info"
            )
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if self.state:
                with self.state.lock:
                    # Reduce memory-intensive parameters
                    if "batch_size" in self.state.training_config:
                        self.state.training_config["batch_size"] = max(1, self.state.training_config["batch_size"] // 2)
                    
                    # Clear caches
                    if hasattr(self.state, 'cache'):
                        self.state.cache.clear()
                    
                    self.logger.record_event(
                        event_type="memory_recovery",
                        message="Memory cache cleared and batch size reduced",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="memory_recovery_failed",
                message=f"Memory recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_default(self, error_key: str) -> None:
        """Default recovery strategy for unhandled error types."""
        self.logger.record({
            "event": "attempting_default_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement default recovery logic here

    def _recover_curiosity(self, error_key: str) -> None:
        """Recovery strategy for curiosity errors."""
        try:
            self.logger.record_event(
                event_type="curiosity_recovery_attempt",
                message=f"Attempting curiosity recovery for {error_key}",
                level="info"
            )
            
            if self.state and hasattr(self.state, 'curiosity'):
                with self.state.lock:
                    # Reset curiosity parameters
                    self.state.curiosity.pressure = 0.5
                    self.state.curiosity.novelty_threshold_spontaneous = 0.7
                    self.state.curiosity.novelty_threshold_response = 0.6
                    
                    # Clear question queue
                    if hasattr(self.state.curiosity, 'unanswered_questions'):
                        self.state.curiosity.unanswered_questions.clear()
                    
                    self.logger.record_event(
                        event_type="curiosity_recovery",
                        message="Curiosity parameters reset to default values",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_recovery_failed",
                message=f"Curiosity recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """Return a summary of recorded errors."""
        with self.lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_types": dict(self.error_counts),
                "recent_errors": {
                    error_type: list(history)[-5:]
                    for error_type, history in self.error_history.items()
                },
            }

    def clear_error_history(self) -> None:
        """Clear error counts and history."""
        with self.lock:
            self.error_counts.clear()
            self.error_history.clear()
            self.logger.record({
                "event": "error_history_cleared",
                "timestamp": time.time(),
            })

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics if CUDA is available."""
        if not torch.cuda.is_available():
            return {}
            
        try:
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved(),
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name()
            }
        except Exception as e:
            self.logger.record_event(
                event_type="memory_stats_error",
                message=f"Failed to get memory stats: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return {}
