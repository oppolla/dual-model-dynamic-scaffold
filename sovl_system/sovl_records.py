from typing import Optional, Dict, Any, Deque
from collections import deque
import time
import hashlib
import json
import os
import threading
import traceback
from dataclasses import dataclass
from threading import Lock
from sovl_config import ConfigManager
from sovl_logger import Logger, LoggingManager
from sovl_utils import NumericalGuard, synchronized

"""
Module for managing confidence history, decoupling it from sovl_state.py and
sovl_processor.py to resolve mutual dependencies. Provides thread-safe storage,
serialization, and backup of confidence scores.
"""

class HistoryError(Exception):
    """Base exception for confidence history errors."""
    pass

class SerializationError(HistoryError):
    """Raised for serialization/deserialization failures."""
    pass

class ValidationError(HistoryError):
    """Raised for history validation failures."""
    pass

@dataclass
class ConfidenceHistoryConfig:
    """Configuration for confidence history management."""
    max_confidence_history: int
    confidence_history_file: str = "confidence_history.json"
    backup_interval: float = 300.0  # Seconds between backups
    enable_persistence: bool = True  # Whether to save to disk
    strict_validation: bool = True  # Enforce strict validation on load

    def validate(self) -> None:
        """Validate configuration parameters."""
        try:
            if self.max_confidence_history <= 0:
                raise ValidationError("max_confidence_history must be positive")
            if not isinstance(self.confidence_history_file, str) or not self.confidence_history_file.endswith(".json"):
                raise ValidationError("confidence_history_file must be a valid JSON file path")
            if self.backup_interval <= 0:
                raise ValidationError("backup_interval must be positive")
            if not isinstance(self.enable_persistence, bool):
                raise ValidationError("enable_persistence must be boolean")
            if not isinstance(self.strict_validation, bool):
                raise ValidationError("strict_validation must be boolean")
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {str(e)}")

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ConfidenceHistoryConfig':
        """Create configuration from ConfigManager."""
        try:
            config = cls(
                max_confidence_history=config_manager.get("controls_config.confidence_history_maxlen", 5),
                confidence_history_file=config_manager.get("controls_config.confidence_history_file", "confidence_history.json"),
                backup_interval=config_manager.get("controls_config.history_backup_interval", 300.0),
                enable_persistence=config_manager.get("controls_config.history_enable_persistence", True),
                strict_validation=config_manager.get("controls_config.history_strict_validation", True)
            )
            config.validate()
            return config
        except Exception as e:
            raise HistoryError(f"Failed to create ConfidenceHistoryConfig: {str(e)}")

class ConfidenceHistory:
    """Thread-safe manager for confidence history with persistence and backups."""
    
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
        self._cached_hash: Optional[str] = None
        self._initialize_history()
        if self._config.enable_persistence:
            self._start_backup_thread()
        self._log_training_event("history_initialized", {
            "max_length": self._config.max_confidence_history,
            "history_file": self._config.confidence_history_file,
            "persistence_enabled": self._config.enable_persistence
        })

    @synchronized("_lock")
    def _initialize_history(self) -> None:
        """Initialize or load confidence history from file."""
        try:
            if not self._config.enable_persistence:
                self._log_training_event("history_init_no_persistence", {"message": "Persistence disabled"})
                return
            history_file = self._config.confidence_history_file
            if os.path.exists(history_file):
                try:
                    self._load_history(history_file)
                except SerializationError:
                    self._log_error("Corrupted history file, falling back to backup or empty history")
                    backup_file = f"{history_file}.backup"
                    if os.path.exists(backup_file):
                        self._load_history(backup_file)
                    else:
                        self._log_training_event("history_init_empty", {"message": "No valid history or backup, using empty history"})
            self._validate_history()
            self._cached_hash = None  # Invalidate cache
            self._log_training_event("history_initialized", {
                "history_size": len(self._confidence_history),
                "history_hash": self.get_history_hash()
            })
        except Exception as e:
            self._log_error("History initialization failed", e)
            raise HistoryError(f"History initialization failed: {str(e)}")

    @synchronized("_lock")
    def add_confidence(self, confidence: float) -> None:
        """
        Add a confidence score to the history.

        Args:
            confidence: Confidence score in [0.0, 1.0].

        Raises:
            ValidationError: If the confidence score is invalid.
        """
        try:
            with NumericalGuard():
                if not isinstance(confidence, (int, float)):
                    raise ValidationError(f"Confidence must be a number, got {type(confidence)}")
                if not 0.0 <= confidence <= 1.0:
                    raise ValidationError(f"Confidence must be in [0.0, 1.0], got {confidence}")
                self._confidence_history.append(float(confidence))
                self._last_update_time = time.time()
                self._cached_hash = None  # Invalidate cache
                self._log_training_event("confidence_added", {
                    "confidence": confidence,
                    "history_size": len(self._confidence_history),
                    "timestamp": self._last_update_time
                }, level="debug")
        except Exception as e:
            self._log_error("Failed to add confidence score", e)
            raise ValidationError(f"Add confidence failed: {str(e)}")

    @synchronized("_lock")
    def get_confidence_history(self) -> Deque[float]:
        """
        Retrieve the current confidence history.

        Returns:
            Deque of confidence scores.
        """
        return self._confidence_history

    @synchronized("_lock")
    def clear_history(self) -> None:
        """Clear the confidence history."""
        try:
            self._confidence_history.clear()
            self._last_update_time = time.time()
            self._cached_hash = None
            self._log_training_event("history_cleared", {
                "history_size": len(self._confidence_history),
                "timestamp": self._last_update_time
            })
        except Exception as e:
            self._log_error("Failed to clear history", e)
            raise HistoryError(f"Clear history failed: {str(e)}")

    @synchronized("_lock")
    def save_history(self, file_path: Optional[str] = None) -> None:
        """
        Save the confidence history to a file.

        Args:
            file_path: Optional file path. Uses config path if None.

        Raises:
            SerializationError: If saving fails.
        """
        if not self._config.enable_persistence:
            return
        try:
            file_path = file_path or self._config.confidence_history_file
            history_dict = self.to_dict()
            backup_file = f"{file_path}.backup"
            # Use orjson if available for faster serialization
            if orjson:
                serialized = orjson.dumps(history_dict, option=orjson.OPT_INDENT_2)
                with open(backup_file, 'wb') as f:
                    f.write(serialized)
            else:
                with open(backup_file, 'w') as f:
                    json.dump(history_dict, f, indent=2)
            os.replace(backup_file, file_path)
            self._log_training_event("history_saved", {
                "file_path": file_path,
                "history_size": len(self._confidence_history),
                "timestamp": time.time()
            })
        except Exception as e:
            self._log_error("Failed to save history", e)
            raise SerializationError(f"Save history failed: {str(e)}")

    @synchronized("_lock")
    def _load_history(self, file_path: str) -> None:
        """
        Load confidence history from a file.

        Args:
            file_path: File path to load from.

        Raises:
            SerializationError: If loading or validation fails.
        """
        try:
            with open(file_path, 'rb' if orjson else 'r') as f:
                history_dict = orjson.loads(f.read()) if orjson else json.load(f)
            self.from_dict(history_dict)
            self._validate_history()
            self._cached_hash = None
            self._log_training_event("history_loaded", {
                "file_path": file_path,
                "history_size": len(self._confidence_history),
                "timestamp": time.time()
            })
        except Exception as e:
            self._log_error("Failed to load history", e)
            raise SerializationError(f"Load history failed: {str(e)}")

    @synchronized("_lock")
    def _validate_history(self) -> None:
        """
        Validate the confidence history integrity.

        Raises:
            ValidationError: If validation fails and strict_validation is enabled.
        """
        try:
            if not isinstance(self._confidence_history, deque):
                raise ValidationError("confidence_history must be a deque")
            if len(self._confidence_history) > self._config.max_confidence_history:
                raise ValidationError(f"History size {len(self._confidence_history)} exceeds max {self._config.max_confidence_history}")
            for score in self._confidence_history:
                if not isinstance(score, float):
                    raise ValidationError(f"Confidence score must be float, got {type(score)}")
                if not 0.0 <= score <= 1.0:
                    raise ValidationError(f"Confidence score {score} must be in [0.0, 1.0]")
        except ValidationError as e:
            if self._config.strict_validation:
                raise ValidationError(f"History validation failed: {str(e)}")
            self._log_error("Validation failed, continuing with partial history", e, level="warning")
            self._confidence_history = deque(
                [score for score in self._confidence_history if isinstance(score, float) and 0.0 <= score <= 1.0],
                maxlen=self._config.max_confidence_history
            )

    def get_history_hash(self) -> str:
        """
        Generate a hash of the current confidence history.

        Returns:
            String hash of the history.
        """
        try:
            with self._lock:
                if self._cached_hash:
                    return self._cached_hash
                history_dict = self.to_dict()
                # Use orjson for faster hashing if available
                serialized = orjson.dumps(history_dict, option=orjson.OPT_SORT_KEYS) if orjson else json.dumps(history_dict, sort_keys=True)
                self._cached_hash = hashlib.md5(serialized).hexdigest()
                return self._cached_hash
        except Exception as e:
            self._log_error("Failed to generate history hash", e)
            raise HistoryError(f"Generate history hash failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize confidence history to dictionary.

        Returns:
            Dictionary with serialized history.
        """
        try:
            with self._lock:
                return {
                    "version": self.VERSION,
                    "confidence_history": list(self._confidence_history),
                    "last_update_time": self._last_update_time
                }
        except Exception as e:
            self._log_error("History serialization failed", e)
            raise SerializationError(f"History serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load confidence history from dictionary.

        Args:
            data: Dictionary with serialized history.

        Raises:
            SerializationError: If loading or validation fails.
        """
        try:
            with self._lock:
                version = data.get("version", "1.0")
                if version != self.VERSION:
                    self._migrate_history(data, version)
                self._confidence_history = deque(
                    [float(score) for score in data.get("confidence_history", []) if 0.0 <= float(score) <= 1.0],
                    maxlen=self._config.max_confidence_history
                )
                self._last_update_time = float(data.get("last_update_time", time.time()))
                self._cached_hash = None
                self._validate_history()
                self._log_training_event("history_loaded_from_dict", {
                    "history_size": len(self._confidence_history),
                    "version": version
                })
        except Exception as e:
            self._log_error("Failed to load history from dictionary", e)
            raise SerializationError(f"Load history from dictionary failed: {str(e)}")

    def _migrate_history(self, data: Dict[str, Any], version: str) -> None:
        """
        Migrate history data from older versions.

        Args:
            data: Dictionary with serialized history.
            version: Version of the loaded data.
        """
        try:
            self._log_training_event("history_migration", {
                "message": f"Migrating history from version {version} to {self.VERSION}",
                "version": version
            }, level="warning")
            # Placeholder for future migration logic
            # Example: Adjust score ranges or format if needed
        except Exception as e:
            self._log_error("History migration failed", e)
            raise SerializationError(f"History migration failed: {str(e)}")

    def reset(self) -> None:
        """
        Reset the confidence history for testing or new sessions.

        Raises:
            HistoryError: If reset fails.
        """
        try:
            with self._lock:
                self._confidence_history.clear()
                self._last_update_time = time.time()
                self._cached_hash = None
                self._log_training_event("history_reset", {
                    "history_size": len(self._confidence_history),
                    "timestamp": self._last_update_time
                })
        except Exception as e:
            self._log_error("Failed to reset history", e)
            raise HistoryError(f"Reset history failed: {str(e)}")

    def _start_backup_thread(self) -> None:
        """Start a thread for periodic history backups."""
        try:
            if self._config.backup_interval <= 0:
                self._log_training_event("backup_disabled", {
                    "message": "History backup disabled due to invalid interval",
                    "backup_interval": self._config.backup_interval
                })
                return
            def backup_loop():
                while True:
                    try:
                        time.sleep(self._config.backup_interval)
                        self.save_history()
                    except Exception as e:
                        self._log_error("Backup thread error", e)
            backup_thread = threading.Thread(target=backup_loop, daemon=True)
            backup_thread.start()
            self._log_training_event("backup_thread_started", {
                "backup_interval": self._config.backup_interval
            })
        except Exception as e:
            self._log_error("Failed to start backup thread", e)
            raise HistoryError(f"Start backup thread failed: {str(e)}")

    def _log_training_event(self, event_type: str, additional_info: Dict[str, Any], level: str = "info") -> None:
        """
        Log a training event with standardized metadata.

        Args:
            event_type: Type of the event.
            additional_info: Additional event data.
            level: Log level (debug, info, warning, error).
        """
        try:
            metadata = {
                "timestamp": time.time(),
                "history_size": len(self._confidence_history),
                **additional_info
            }
            self.logger.log_training_event(
                event_type=f"confidence_history_{event_type}",
                message=f"Confidence history event: {event_type}",
                level=level,
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log training event: {str(e)}")

    def _log_error(self, message: str, error: Exception, level: str = "error") -> None:
        """
        Log an error with standardized metadata.

        Args:
            message: Error message.
            error: Exception instance.
            level: Log level (error or warning).
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
                additional_info=metadata,
                level=level
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    import unittest

    class TestConfidenceHistory(unittest.TestCase):
        def setUp(self):
            self.logger = LoggingManager("test_logs.jsonl")
            self.config_manager = ConfigManager("sovl_config.json", self.logger)
            self.history = ConfidenceHistory(self.config_manager, self.logger)

        def test_add_confidence(self):
            self.history.add_confidence(0.85)
            self.assertEqual(len(self.history.get_confidence_history()), 1)
            self.assertEqual(self.history.get_confidence_history()[0], 0.85)

        def test_save_load(self):
            self.history.add_confidence(0.92)
            self.history.save_history("test_history.json")
            new_history = ConfidenceHistory(self.config_manager, self.logger)
            new_history._load_history("test_history.json")
            self.assertEqual(list(new_history.get_confidence_history()), [0.92])

        def test_reset(self):
            self.history.add_confidence(0.78)
            self.history.reset()
            self.assertEqual(len(self.history.get_confidence_history()), 0)

        def test_invalid_confidence(self):
            with self.assertRaises(ValidationError):
                self.history.add_confidence(1.5)
            with self.assertRaises(ValidationError):
                self.history.add_confidence("invalid")

    if __name__ == "__main__":
        unittest.main()
