from typing import Optional, Dict, Any, Deque
from collections import deque
import time
import hashlib
import json
from threading import Lock
from dataclasses import dataclass
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_utils import NumericalGuard, synchronized
import traceback


"""
Module for managing confidence history to resolve mutual dependency between
sovl_state.py and sovl_processor.py. Provides a ConfidenceHistory class to store
and manage confidence scores, with thread-safe operations and serialization.
"""

class HistoryError(Exception):
    """Raised for invalid history operations or data."""
    pass

@dataclass
class ConfidenceHistoryConfig:
    """Configuration for confidence history parameters."""
    max_confidence_history: int
    confidence_history_file: str = "confidence_history.json"

    def validate(self) -> None:
        """Validate configuration parameters."""
        try:
            assert self.max_confidence_history > 0, "max_confidence_history must be positive"
            assert isinstance(self.confidence_history_file, str) and self.confidence_history_file.endswith(".json"), \
                "confidence_history_file must be a valid JSON file path"
        except AssertionError as e:
            raise HistoryError(f"Configuration validation failed: {str(e)}")

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ConfidenceHistoryConfig':
        """Create ConfidenceHistoryConfig from ConfigManager."""
        try:
            config = cls(
                max_confidence_history=config_manager.get("controls_config.confidence_history_maxlen", 5),
                confidence_history_file=config_manager.get("controls_config.confidence_history_file", "confidence_history.json")
            )
            config.validate()
            return config
        except Exception as e:
            raise HistoryError(f"Failed to create ConfidenceHistoryConfig: {str(e)}")

class ConfidenceHistory:
    """Manages confidence history with thread-safe operations and persistence."""
    
    VERSION = "1.0"

    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize the confidence history manager.

        Args:
            config_manager: Configuration manager instance.
            logger: Logger for event recording.
        """
        self.config_manager = config_manager
        self.logger = logger
        self._config = ConfidenceHistoryConfig.from_config_manager(config_manager)
        self._confidence_history: Deque[float] = deque(maxlen=self._config.max_confidence_history)
        self._lock = Lock()
        self._last_update_time: float = time.time()
        self._initialize_history()
        self._log_event("history_initialized", {
            "max_length": self._config.max_confidence_history,
            "history_file": self._config.confidence_history_file
        })

    @synchronized("_lock")
    def _initialize_history(self) -> None:
        """Initialize or load confidence history."""
        try:
            history_file = self._config.confidence_history_file
            if history_file and os.path.exists(history_file):
                self._load_history(history_file)
            self._validate_history()
            self._log_event("history_initialized", {
                "history_size": len(self._confidence_history),
                "history_hash": self.get_history_hash()
            })
        except Exception as e:
            self._log_error("Failed to initialize confidence history", e)
            raise HistoryError(f"Confidence history initialization failed: {str(e)}")

    @synchronized("_lock")
    def add_confidence(self, confidence: float) -> None:
        """
        Add a confidence score to the history.

        Args:
            confidence: Confidence score (0.0 to 1.0).

        Raises:
            HistoryError: If the confidence score is invalid.
        """
        try:
            with NumericalGuard():
                if not isinstance(confidence, (int, float)):
                    raise HistoryError(f"Confidence must be a number, got {type(confidence)}")
                if not 0.0 <= confidence <= 1.0:
                    raise HistoryError(f"Confidence must be in [0.0, 1.0], got {confidence}")
                self._confidence_history.append(float(confidence))
                self._last_update_time = time.time()
                self._validate_history()
                self._log_event("confidence_added", {
                    "confidence": confidence,
                    "history_size": len(self._confidence_history),
                    "history_hash": self.get_history_hash()
                })
        except Exception as e:
            self._log_error("Failed to add confidence score", e)
            raise HistoryError(f"Add confidence failed: {str(e)}")

    @synchronized("_lock")
    def get_confidence_history(self) -> Deque[float]:
        """
        Get the current confidence history.

        Returns:
            Deque of confidence scores.
        """
        try:
            return self._confidence_history
        except Exception as e:
            self._log_error("Failed to get confidence history", e)
            raise HistoryError(f"Get confidence history failed: {str(e)}")

    @synchronized("_lock")
    def clear_history(self) -> None:
        """
        Clear the confidence history.

        Raises:
            HistoryError: If clearing fails.
        """
        try:
            self._confidence_history.clear()
            self._last_update_time = time.time()
            self._log_event("history_cleared", {
                "history_size": len(self._confidence_history),
                "history_hash": self.get_history_hash()
            })
        except Exception as e:
            self._log_error("Failed to clear confidence history", e)
            raise HistoryError(f"Clear confidence history failed: {str(e)}")

    @synchronized("_lock")
    def save_history(self, file_path: Optional[str] = None) -> None:
        """
        Save the confidence history to a file.

        Args:
            file_path: Optional file path to save to. Uses config path if None.

        Raises:
            HistoryError: If saving fails.
        """
        try:
            file_path = file_path or self._config.confidence_history_file
            history_dict = self.to_dict()
            with open(file_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
            self._log_event("history_saved", {
                "file_path": file_path,
                "history_size": len(self._confidence_history),
                "history_hash": self.get_history_hash()
            })
        except Exception as e:
            self._log_error("Failed to save confidence history", e)
            raise HistoryError(f"Save confidence history failed: {str(e)}")

    @synchronized("_lock")
    def _load_history(self, file_path: str) -> None:
        """
        Load the confidence history from a file.

        Args:
            file_path: File path to load from.

        Raises:
            HistoryError: If loading fails.
        """
        try:
            with open(file_path, 'r') as f:
                history_dict = json.load(f)
            self.from_dict(history_dict)
            self._validate_history()
            self._log_event("history_loaded", {
                "file_path": file_path,
                "history_size": len(self._confidence_history),
                "history_hash": self.get_history_hash()
            })
        except Exception as e:
            self._log_error("Failed to load confidence history", e)
            raise HistoryError(f"Load confidence history failed: {str(e)}")

    @synchronized("_lock")
    def _validate_history(self) -> None:
        """
        Validate the integrity of the confidence history.

        Raises:
            HistoryError: If validation fails.
        """
        try:
            assert isinstance(self._confidence_history, deque), "confidence_history must be a deque"
            assert len(self._confidence_history) <= self._config.max_confidence_history, \
                f"History size {len(self._confidence_history)} exceeds max {self._config.max_confidence_history}"
            for score in self._confidence_history:
                assert isinstance(score, float), f"Confidence score must be float, got {type(score)}"
                assert 0.0 <= score <= 1.0, f"Confidence score {score} must be in [0.0, 1.0]"
        except AssertionError as e:
            self._log_error("Confidence history validation failed", e)
            raise HistoryError(f"History validation failed: {str(e)}")

    def get_history_hash(self) -> str:
        """
        Generate a hash of the current confidence history.

        Returns:
            String hash of the history.
        """
        try:
            history_dict = self.to_dict()
            return hashlib.md5(json.dumps(history_dict, sort_keys=True).encode()).hexdigest()
        except Exception as e:
            self._log_error("Failed to generate history hash", e)
            raise HistoryError(f"Generate history hash failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize confidence history to dictionary.

        Returns:
            Dictionary containing serialized history.
        """
        try:
            with self._lock:
                return {
                    "version": self.VERSION,
                    "confidence_history": list(self._confidence_history),
                    "last_update_time": self._last_update_time
                }
        except Exception as e:
            self._log_error("Confidence history serialization failed", e)
            raise HistoryError(f"History serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load confidence history from dictionary.

        Args:
            data: Dictionary containing serialized history.

        Raises:
            HistoryError: If loading fails.
        """
        try:
            with self._lock:
                if data.get("version") != self.VERSION:
                    self._log_error(
                        f"History version mismatch: expected {self.VERSION}, got {data.get('version')}",
                        HistoryError("Version mismatch")
                    )
                    raise HistoryError("History version mismatch")
                self._confidence_history = deque(
                    [float(score) for score in data.get("confidence_history", [])],
                    maxlen=self._config.max_confidence_history
                )
                self._last_update_time = float(data.get("last_update_time", time.time()))
                self._validate_history()
                self._log_event("history_loaded_from_dict", {
                    "history_size": len(self._confidence_history),
                    "history_hash": self.get_history_hash()
                })
        except Exception as e:
            self._log_error("Failed to load confidence history from dictionary", e)
            raise HistoryError(f"Load history from dictionary failed: {str(e)}")

    def _log_event(self, event_type: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an event with standardized metadata.

        Args:
            event_type: Type of the event.
            additional_info: Additional event data.
        """
        try:
            metadata = {
                "timestamp": time.time(),
                "history_size": len(self._confidence_history)
            }
            if additional_info:
                metadata.update(additional_info)
            self.logger.record_event(
                event_type=f"confidence_history_{event_type}",
                message=f"Confidence history event: {event_type}",
                level="info",
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log event: {str(e)}")

    def _log_error(self, message: str, error: Exception) -> None:
        """
        Log an error with standardized metadata.

        Args:
            message: Error message.
            error: Exception instance.
        """
        try:
            metadata = {
                "error": str(error),
                "stack_trace": traceback.format_exc()
            }
            self.logger.log_error(
                error_msg=message,
                error_type="confidence_history_error",
                stack_trace=traceback.format_exc(),
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")

# Example usage
if __name__ == "__main__":
    import os
    from sovl_config import ConfigManager
    from sovl_logger import LoggingManager

    # Initialize dependencies
    config_path = "sovl_config.json"
    log_file = "sovl_records_logs.jsonl"
    logger = LoggingManager(log_file=log_file)
    config_manager = ConfigManager(config_path=config_path, logger=logger)

    # Create confidence history instance
    history = ConfidenceHistory(config_manager=config_manager, logger=logger)

    try:
        # Add some confidence scores
        history.add_confidence(0.85)
        history.add_confidence(0.92)
        history.add_confidence(0.78)

        # Save history
        history.save_history()

        # Load history
        new_history = ConfidenceHistory(config_manager=config_manager, logger=logger)
        new_history._load_history(history._config.confidence_history_file)

        # Verify loaded history
        assert list(new_history.get_confidence_history()) == list(history.get_confidence_history())
        print("Confidence history test passed!")
    except Exception as e:
        logger.log_error(
            error_msg="Confidence history test failed",
            error_type="test_error",
            stack_trace=traceback.format_exc(),
            additional_info={"error": str(e)}
        )
        raise
