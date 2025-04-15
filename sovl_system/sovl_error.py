from typing import Optional, Dict, Any, Callable
import traceback
import time
from collections import defaultdict, deque
from threading import Lock
from sovl_logger import Logger
from sovl_state import SOVLState
from sovl_config import ConfigManager

class ErrorHandler:
    """Handles error logging, recovery, and monitoring for the SOVL system."""

    _DEFAULT_CONFIG = {
        "error_handling.max_history_per_error": 10,
        "error_handling.critical_threshold": 5,
        "error_handling.warning_threshold": 10,
        "error_handling.retry_attempts": 3,
        "error_handling.retry_delay": 1.0,
    }

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_log_file: str = "sovl_errors.jsonl",
        max_error_log_size_mb: int = 10,
        compress_old: bool = True,
        state: Optional[SOVLState] = None,
    ):
        """
        Initialize the error handler with configuration and logging dependencies.

        Args:
            config_manager: ConfigManager instance for accessing error handling settings.
            logger: Logger instance for recording errors and warnings.
            error_log_file: File path for dedicated error logs.
            max_error_log_size_mb: Maximum size for error log file before rotation.
            compress_old: Whether to compress rotated log files.
            state: Optional SOVLState instance for context-aware error handling.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.state = state
        self.error_logger = Logger(
            log_file=error_log_file,
            max_size_mb=max_error_log_size_mb,
            compress_old=compress_old,
        )
        self.lock = Lock()
        self.error_counts = defaultdict(int)
        self.error_history = defaultdict(lambda: deque(maxlen=self._get_max_history_per_error()))
        self.severity_thresholds = self._load_severity_thresholds()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self._validate_config()

    def _get_max_history_per_error(self) -> int:
        """Get the maximum number of error instances to keep in history."""
        return self.config_manager.get(
            "error_handling.max_history_per_error",
            self._DEFAULT_CONFIG["error_handling.max_history_per_error"]
        )

    def _load_severity_thresholds(self) -> Dict[str, int]:
        """Load severity thresholds from configuration."""
        return {
            "critical": self.config_manager.get(
                "error_handling.critical_threshold",
                self._DEFAULT_CONFIG["error_handling.critical_threshold"]
            ),
            "warning": self.config_manager.get(
                "error_handling.warning_threshold",
                self._DEFAULT_CONFIG["error_handling.warning_threshold"]
            )
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
            thresholds = self._load_severity_thresholds()
            if thresholds["critical"] <= thresholds["warning"]:
                raise ValueError("Critical threshold must be greater than warning threshold")
        except Exception as e:
            self.logger.error(f"Error handling configuration validation failed: {str(e)}")
            raise

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

            # Log the error
            self._log_error(error, context, phase, severity, additional_info)

            # Check if we've exceeded thresholds
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._handle_critical_error(error_key)
            elif self.error_counts[error_key] >= self.severity_thresholds["warning"]:
                self._handle_warning(error_key)

            if reraise:
                raise error

    def _log_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        severity: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error details to both system and error logs."""
        error_entry = {
            "error": str(error),
            "context": context,
            "phase": phase,
            "severity": severity,
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc(),
            "additional_info": additional_info or {}
        }

        # Log to system logger
        self.logger.record(error_entry)

        # Log to dedicated error logger
        self.error_logger.record(error_entry)

    def _handle_critical_error(self, error_key: str) -> None:
        """Handle critical errors that exceed the threshold."""
        error_history = list(self.error_history[error_key])
        self.logger.record({
            "event": "critical_error_threshold_exceeded",
            "error_key": error_key,
            "count": self.error_counts[error_key],
            "recent_errors": error_history[-5:],  # Last 5 errors
            "timestamp": time.time()
        })

        # Attempt recovery based on error context
        context = error_key.split(":")[0]
        recovery_strategy = self.recovery_strategies.get(context, self.recovery_strategies["default"])
        try:
            recovery_strategy(error_key)
        except Exception as e:
            self.logger.record({
                "error": f"Recovery failed for critical error {error_key}: {str(e)}",
                "timestamp": time.time()
            })

    def _handle_warning(self, error_key: str) -> None:
        """Handle warnings that exceed the threshold."""
        self.logger.record({
            "event": "warning_threshold_exceeded",
            "error_key": error_key,
            "count": self.error_counts[error_key],
            "timestamp": time.time()
        })

    def _recover_model_loading(self, error_key: str) -> None:
        """Recovery strategy for model loading errors."""
        self.logger.record({
            "event": "attempting_model_loading_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement model loading recovery logic here

    def _recover_training(self, error_key: str) -> None:
        """Recovery strategy for training errors."""
        self.logger.record({
            "event": "attempting_training_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement training recovery logic here

    def _recover_generation(self, error_key: str) -> None:
        """Recovery strategy for generation errors."""
        self.logger.record({
            "event": "attempting_generation_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement generation recovery logic here

    def _recover_memory(self, error_key: str) -> None:
        """Recovery strategy for memory-related errors."""
        self.logger.record({
            "event": "attempting_memory_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement memory recovery logic here

    def _recover_default(self, error_key: str) -> None:
        """Default recovery strategy for unhandled error types."""
        self.logger.record({
            "event": "attempting_default_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement default recovery logic here

    def handle_generation_error(self, error: Exception, prompt: str) -> str:
        """Handle generation errors and return a fallback response."""
        self.record_error(
            error=error,
            context="generation",
            phase="generate",
            additional_info={
                "prompt": prompt[:200],  # Truncate for logging
                "state_hash": self.state.state_hash() if self.state else None
            },
            severity="error"
        )
        return "An error occurred during generation"

    def handle_training_error(self, error: Exception, batch_size: int) -> None:
        """Handle training errors."""
        self.record_error(
            error=error,
            context="training",
            phase="train_step",
            additional_info={
                "batch_size": batch_size,
                "phase": "training",
                "state_hash": self.state.state_hash() if self.state else None
            },
            severity="error",
            reraise=True
        )

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

    def handle_curiosity_error(self, error: Exception, event_type: str) -> None:
        """Handle curiosity-related errors."""
        self.record_error(
            error=error,
            context="curiosity",
            phase=event_type,
            additional_info={
                "error_type": "generation_error" if event_type == "question_generation" else "curiosity_error",
                "state_hash": self.state.state_hash() if self.state else None
            },
            severity="error"
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
