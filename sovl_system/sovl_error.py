from typing import Optional, Dict, Any, Callable
import traceback
import time
from collections import defaultdict
import json
import os
from threading import Lock
from sovl_logger import Logger
from sovl_state import SOVLState
from sovl_config import ConfigManager

class ErrorHandler:
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_log_file: str = "sovl_errors.jsonl",
        max_error_log_size_mb: int = 10,
        compress_old: bool = True,
        state: Optional[SOVLState] = None
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
            compress_old=compress_old
        )
        self.lock = Lock()
        self.error_counts = defaultdict(int)
        self.error_history = defaultdict(list)
        self.max_history_per_error = config_manager.get(
            "error_handling.max_history_per_error", 10, expected_type=int
        )
        self.severity_thresholds = {
            "critical": config_manager.get("error_handling.critical_threshold", 5, expected_type=int),
            "warning": config_manager.get("error_handling.warning_threshold", 10, expected_type=int),
        }
        self.recovery_strategies = {
            "training": self._default_training_recovery,
            "generation": self._default_generation_recovery,
            "config_validation": self._default_config_recovery,
            "data_loading": self._default_data_recovery,
            "cross_attention": self._default_cross_attention_recovery,
        }
        self._validate_config()

    def _validate_config(self):
        """Validate error handling configuration."""
        required_keys = [
            "error_handling.max_history_per_error",
            "error_handling.critical_threshold",
            "error_handling.warning_threshold",
        ]
        try:
            self.config_manager.validate_keys(required_keys)
        except KeyError as e:
            self.logger.record({
                "warning": f"Missing error handling config key: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id if self.state else "init",
            })

    def record_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        conversation_id: str = "unknown",
        additional_info: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        reraise: bool = False
    ) -> None:
        """
        Record an error with context and log it.

        Args:
            error: The exception instance.
            context: Context of the error (e.g., 'training', 'generation').
            phase: Specific phase (e.g., 'train_step', 'generate').
            conversation_id: ID for tracking conversation context.
            additional_info: Extra metadata to include in the log.
            severity: Severity level ('critical', 'error', 'warning').
            reraise: Whether to re-raise the exception after logging.
        """
        with self.lock:
            error_type = type(error).__name__
            error_msg = str(error)
            self.error_counts[error_type] += 1
            state_hash = self.state.state_hash() if self.state else None

            log_entry = {
                "error": error_msg,
                "type": error_type,
                "context": context,
                "phase": phase,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "stack_trace": traceback.format_exc(),
                "severity": severity,
                "state_hash": state_hash,
            }
            if additional_info:
                log_entry.update(additional_info)

            # Update error history
            self.error_history[error_type].append(log_entry)
            if len(self.error_history[error_type]) > self.max_history_per_error:
                self.error_history[error_type].pop(0)

            # Log to error-specific logger
            self.error_logger.record(log_entry)
            # Log to main logger with severity
            self.logger.record({
                severity: error_msg,
                "context": context,
                "phase": phase,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "error_type": error_type,
            })

            # Check if error exceeds severity threshold
            if self.error_counts[error_type] >= self.severity_thresholds.get("critical", 5):
                self._handle_critical_error(error_type, context, phase)

            if reraise:
                raise error

    def handle_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        recovery_fn: Optional[Callable] = None,
        default_recovery_value: Any = None,
        conversation_id: str = "unknown",
        additional_info: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ) -> Any:
        """
        Handle an error by logging it and applying a recovery strategy.

        Args:
            error: The exception instance.
            context: Context of the error.
            phase: Specific phase.
            recovery_fn: Custom recovery function to execute.
            default_recovery_value: Value to return if recovery fails.
            conversation_id: ID for tracking conversation context.
            additional_info: Extra metadata to include in the log.
            severity: Severity level.

        Returns:
            Result of the recovery function or default_recovery_value.
        """
        self.record_error(
            error=error,
            context=context,
            phase=phase,
            conversation_id=conversation_id,
            additional_info=additional_info,
            severity=severity,
        )

        # Apply recovery strategy
        recovery_fn = recovery_fn or self.recovery_strategies.get(context, lambda e, c, p: default_recovery_value)
        try:
            return recovery_fn(error, context, phase)
        except Exception as recovery_error:
            self.record_error(
                error=recovery_error,
                context=context,
                phase=f"{phase}_recovery",
                conversation_id=conversation_id,
                severity="critical",
                additional_info={"original_error": str(error)},
            )
            return default_recovery_value

    def _handle_critical_error(self, error_type: str, context: str, phase: str):
        """Handle critical errors that exceed thresholds."""
        self.logger.record({
            "critical": f"Error type {error_type} exceeded threshold in {context}/{phase}",
            "timestamp": time.time(),
            "conversation_id": self.state.conversation_id if self.state else "init",
            "error_count": self.error_counts[error_type],
        })
        # Example: Trigger system pause or reset
        if self.state:
            self.state.is_sleeping = True
            self.logger.record({
                "event": "system_pause",
                "reason": f"Critical error: {error_type}",
                "timestamp": time.time(),
            })

    def _default_training_recovery(self, error: Exception, context: str, phase: str) -> None:
        """Default recovery for training errors."""
        self.logger.record({
            "warning": f"Training recovery triggered for {phase}: {str(error)}",
            "timestamp": time.time(),
        })
        return None

    def _default_generation_recovery(self, error: Exception, context: str, phase: str) -> str:
        """Default recovery for generation errors."""
        return "An error occurred during generation."

    def _default_config_recovery(self, error: Exception, context: str, phase: str) -> None:
        """Default recovery for configuration errors."""
        self.logger.record({
            "warning": f"Using default config values due to error: {str(error)}",
            "timestamp": time.time(),
        })
        return None

    def _default_data_recovery(self, error: Exception, context: str, phase: str) -> list:
        """Default recovery for data loading errors."""
        self.logger.record({
            "warning": f"Returning empty dataset due to error: {str(error)}",
            "timestamp": time.time(),
        })
        return []

    def _default_cross_attention_recovery(self, error: Exception, context: str, phase: str) -> bool:
        """Default recovery for cross-attention errors."""
        self.logger.record({
            "warning": f"Disabling cross-attention due to error: {str(error)}",
            "timestamp": time.time(),
        })
        return False

    def get_error_summary(self) -> Dict[str, Any]:
        """Return a summary of recorded errors."""
        with self.lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_types": dict(self.error_counts),
                "recent_errors": {
                    error_type: history[-5:] for error_type, history in self.error_history.items()
                },
            }

    def clear_error_history(self):
        """Clear error counts and history."""
        with self.lock:
            self.error_counts.clear()
            self.error_history.clear()
            self.logger.record({
                "event": "error_history_cleared",
                "timestamp": time.time(),
            })
