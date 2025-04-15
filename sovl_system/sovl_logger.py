import json
import os
import gzip
import uuid
import time
import logging
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import torch
from logging.handlers import RotatingFileHandler
import traceback

@dataclass
class LoggerConfig:
    """Configuration for Logger with validation."""
    log_file: str = "sovl_logs.jsonl"
    max_size_mb: int = 10
    compress_old: bool = False
    max_in_memory_logs: int = 1000
    rotation_count: int = 5

    _RANGES = {
        "max_size_mb": (0, 100),
        "max_in_memory_logs": (100, 10000),
    }

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.log_file, str) or not self.log_file.endswith(".jsonl"):
            raise ValueError("log_file must be a .jsonl file path")
        if not isinstance(self.compress_old, bool):
            raise ValueError("compress_old must be a boolean")
        for key, (min_val, max_val) in self._RANGES.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key == "log_file":
                if not isinstance(value, str) or not value.endswith(".jsonl"):
                    raise ValueError("log_file must be a .jsonl file path")
            elif key == "compress_old":
                if not isinstance(value, bool):
                    raise ValueError("compress_old must be a boolean")
            elif key in self._RANGES:
                min_val, max_val = self._RANGES[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            elif key == "rotation_count":
                if not isinstance(value, int) or value < 0:
                    raise ValueError("rotation_count must be a non-negative integer")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)

class _LogValidator:
    """Handles log entry validation logic."""
    
    REQUIRED_FIELDS = {'timestamp', 'conversation_id'}
    OPTIONAL_FIELDS = {'prompt', 'response', 'confidence_score', 'error', 'warning', 'mood', 'variance', 'logits_shape'}
    FIELD_VALIDATORS = {
        'timestamp': lambda x: isinstance(x, (str, float, int)),
        'conversation_id': lambda x: isinstance(x, str),
        'confidence_score': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
        'is_error_prompt': lambda x: isinstance(x, bool),
        'mood': lambda x: x in {'melancholic', 'restless', 'calm', 'curious'},
        'variance': lambda x: isinstance(x, (int, float)) and x >= 0.0,
        'logits_shape': lambda x: isinstance(x, (tuple, list, str))
    }

    def __init__(self, fallback_logger: logging.Logger):
        self.fallback_logger = fallback_logger

    def validate_entry(self, entry: Dict) -> bool:
        """Validate log entry structure and types."""
        if not isinstance(entry, dict):
            self.fallback_logger.warning("Log entry is not a dictionary")
            return False

        try:
            # Ensure required fields
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
            if 'conversation_id' not in entry:
                entry['conversation_id'] = str(uuid.uuid4())

            # Validate field types
            for field, validator in self.FIELD_VALIDATORS.items():
                if field in entry and not validator(entry[field]):
                    self.fallback_logger.warning(f"Invalid value for field {field}: {entry[field]}")
                    return False

            return True
        except Exception as e:
            self.fallback_logger.error(f"Validation failed: {str(e)}")
            return False

class _FileHandler:
    """Manages file operations for logging."""
    
    def __init__(self, config: LoggerConfig, fallback_logger: logging.Logger):
        self.config = config
        self.fallback_logger = fallback_logger

    def safe_file_op(self, operation: Callable, *args, **kwargs):
        """Execute file operation with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    self.fallback_logger.error(f"File operation failed after {max_retries} retries: {str(e)}")
                    raise
                time.sleep(0.1 * (attempt + 1))

    def atomic_write(self, filename: str, content: str) -> None:
        """Perform atomic file write using temporary file."""
        temp_file = f"{filename}.tmp"
        try:
            with self.safe_file_op(open, temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.safe_file_op(os.replace, temp_file, filename)
        except Exception as e:
            self.fallback_logger.error(f"Atomic write failed: {str(e)}")
            if os.path.exists(temp_file):
                self.safe_file_op(os.remove, temp_file)
            raise

    def rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if self.config.max_size_mb <= 0 or not os.path.exists(self.config.log_file):
            return

        file_size = os.path.getsize(self.config.log_file)
        if file_size < self.config.max_size_mb * 1024 * 1024:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = f"{self.config.log_file}.{timestamp}"

            if self.config.compress_old:
                rotated_file += ".gz"
                with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                    with self.safe_file_op(gzip.open, rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                self.safe_file_op(os.rename, self.config.log_file, rotated_file)

            self.fallback_logger.info(f"Rotated logs to {rotated_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to rotate log file: {str(e)}")

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file."""
        if not os.path.exists(self.config.log_file):
            return None

        compressed_file = f"{self.config.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                with self.safe_file_op(gzip.open, compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            if not keep_original:
                self.safe_file_op(os.remove, self.config.log_file)

            self.fallback_logger.info(f"Compressed logs to {compressed_file}")
            return compressed_file
        except Exception as e:
            self.fallback_logger.error(f"Failed to compress logs: {str(e)}")
            return None

    def manage_rotation(self, max_files: int = 5) -> None:
        """Manage rotated log files, keeping only max_files most recent."""
        if not os.path.exists(self.config.log_file):
            return

        try:
            base_name = os.path.basename(self.config.log_file)
            log_dir = os.path.dirname(self.config.log_file) or '.'

            rotated_files = [
                os.path.join(log_dir, f) for f in os.listdir(log_dir)
                if f.startswith(base_name) and f != base_name
            ]

            rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            for old_file in rotated_files[max_files:]:
                try:
                    self.safe_file_op(os.remove, old_file)
                    self.fallback_logger.info(f"Removed old log file {old_file}")
                except OSError:
                    self.fallback_logger.error(f"Failed to remove old log file {old_file}")
        except Exception as e:
            self.fallback_logger.error(f"Error managing log rotation: {str(e)}")

class Logger:
    """Thread-safe logger for SOVLSystem interactions."""
    
    def __init__(self, config: LoggerConfig, fallback_logger: logging.Logger = None):
        """
        Initialize the logger with configurable file handling.

        Args:
            config: Logger configuration
            fallback_logger: Python logger for fallback logging (optional)
        """
        self.config = config
        self.fallback_logger = fallback_logger or logging.getLogger(__name__)
        self.logs: List[Dict] = []
        self.lock = Lock()
        self.file_lock = Lock()  # Separate lock for file operations
        self.validator = _LogValidator(self.fallback_logger)
        self.file_handler = _FileHandler(config, self.fallback_logger)
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing logs from file with memory constraints and format detection."""
        if not os.path.exists(self.config.log_file):
            return

        try:
            with self.file_lock:  # Ensure thread-safe file reading
                with self.file_handler.safe_file_op(open, self.config.log_file, 'r', encoding='utf-8') as f:
                    temp_logs = []
                    for line_num, line in enumerate(f, 1):
                        if len(temp_logs) >= self.config.max_in_memory_logs:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to parse as JSON first
                        try:
                            entry = json.loads(line)
                            if isinstance(entry, dict) and self.validator.validate_entry(entry):
                                temp_logs.append(entry)
                                continue
                        except json.JSONDecodeError:
                            pass
                        
                        # If not valid JSON, try to parse as text log
                        try:
                            # Extract timestamp if present
                            timestamp = None
                            if line.startswith('[') and ']' in line:
                                timestamp_str = line[1:line.index(']')]
                                try:
                                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').timestamp()
                                except ValueError:
                                    pass
                            
                            # Create structured log entry
                            entry = {
                                'timestamp': timestamp or time.time(),
                                'conversation_id': str(uuid.uuid4()),
                                'message': line,
                                'log_type': 'text',
                                'original_line': line_num
                            }
                            
                            if self.validator.validate_entry(entry):
                                temp_logs.append(entry)
                        except Exception as e:
                            self.fallback_logger.warning(
                                f"Failed to parse line {line_num} as text log: {str(e)}"
                            )

            with self.lock:  # Ensure thread-safe memory update
                self.logs = temp_logs[-self.config.max_in_memory_logs:]
                self.fallback_logger.info(
                    f"Loaded {len(self.logs)} log entries from {self.config.log_file}"
                )
                
        except Exception as e:
            self.fallback_logger.error(
                f"Failed to load logs from {self.config.log_file}: {str(e)}"
            )
            # Preserve corrupted file for debugging
            corrupted_file = f"{self.config.log_file}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with self.file_lock:
                    os.rename(self.config.log_file, corrupted_file)
                self.fallback_logger.info(f"Preserved corrupted log file as {corrupted_file}")
            except OSError:
                self.fallback_logger.error("Failed to preserve corrupted log file")
            with self.lock:
                self.logs = []

    def record(self, entry: Dict) -> None:
        """Write a validated log entry (alias for write)."""
        self.write(entry)

    def write(self, entry: Dict) -> None:
        """
        Write a validated log entry.

        Args:
            entry: Single log entry to write
        """
        if not self.validator.validate_entry(entry):
            raise ValueError("Invalid log entry structure")

        # Ensure entry has required fields
        if "timestamp" not in entry:
            entry["timestamp"] = time.time()
        if "conversation_id" not in entry:
            entry["conversation_id"] = str(uuid.uuid4())

        with self.lock:  # Ensure thread-safe memory update
            self.logs.append(entry)
            self.logs = self.logs[-self.config.max_in_memory_logs:]
            
            # Check if rotation is needed
            if os.path.exists(self.config.log_file):
                file_size = os.path.getsize(self.config.log_file)
                if file_size >= self.config.max_size_mb * 1024 * 1024:
                    self._rotate_logs()

            try:
                with self.file_lock:  # Ensure thread-safe file writing
                    # Write as JSONL for structured logs
                    if entry.get('log_type') != 'text':
                        self.file_handler.atomic_write(
                            self.config.log_file,
                            json.dumps(entry) + '\n'
                        )
                    else:
                        # Write as text log for text-type entries
                        timestamp = datetime.fromtimestamp(entry['timestamp'])
                        formatted_time = timestamp.strftime('[%Y-%m-%d %H:%M:%S]')
                        self.file_handler.atomic_write(
                            self.config.log_file,
                            f"{formatted_time} {entry['message']}\n"
                        )
            except Exception as e:
                self.fallback_logger.error(
                    f"Failed to write to {self.config.log_file}: {str(e)}"
                )
                entry['_write_failed'] = True

    def _rotate_logs(self) -> None:
        """Rotate log files with compression if enabled."""
        try:
            with self.file_lock:  # Ensure thread-safe file operations
                # Get list of existing rotated files
                base_name = os.path.basename(self.config.log_file)
                log_dir = os.path.dirname(self.config.log_file) or '.'
                rotated_files = [
                    os.path.join(log_dir, f) for f in os.listdir(log_dir)
                    if f.startswith(base_name) and f != base_name
                ]

                # Sort rotated files by modification time
                rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                # Remove excess rotated files
                for old_file in rotated_files[self.config.rotation_count:]:
                    try:
                        os.remove(old_file)
                    except OSError:
                        self.fallback_logger.error(f"Failed to remove old log file {old_file}")

                # Create new rotated file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_file = f"{self.config.log_file}.{timestamp}"

                if self.config.compress_old:
                    rotated_file += ".gz"
                    with open(self.config.log_file, 'rb') as f_in:
                        with gzip.open(rotated_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                else:
                    os.rename(self.config.log_file, rotated_file)

                # Create new empty log file
                open(self.config.log_file, 'w').close()

                self.fallback_logger.info(f"Rotated logs to {rotated_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to rotate log file: {str(e)}")

    def write_batch(self, entries: List[Dict]) -> None:
        """Optimized batch writing with validation and atomic write."""
        if not entries:
            return

        valid_entries = []
        for entry in entries:
            if self.validator.validate_entry(entry):
                if "error" in entry or "warning" in entry:
                    entry["is_error_prompt"] = True
                valid_entries.append(entry)
            else:
                self.fallback_logger.warning(f"Invalid log entry skipped: {entry}")

        if not valid_entries:
            return

        with self.lock:  # Ensure thread-safe memory update
            self.logs.extend(valid_entries)
            self.logs = self.logs[-self.config.max_in_memory_logs:]
            
            # Check if rotation is needed
            if os.path.exists(self.config.log_file):
                file_size = os.path.getsize(self.config.log_file)
                if file_size >= self.config.max_size_mb * 1024 * 1024:
                    self._rotate_logs()

            try:
                with self.file_lock:  # Ensure thread-safe file writing
                    content = '\n'.join(json.dumps(e) for e in valid_entries) + '\n'
                    self.file_handler.atomic_write(self.config.log_file, content)
            except Exception as e:
                self.fallback_logger.error(f"Error writing batch: {str(e)}")
                for entry in valid_entries:
                    entry['_write_failed'] = True

    def recover_failed_writes(self) -> int:
        """Attempt to recover any log entries that failed to write to disk."""
        with self.lock:
            failed_entries = [entry for entry in self.logs if entry.get('_write_failed')]
            if not failed_entries:
                return 0

            try:
                content = '\n'.join(json.dumps(e) for e in failed_entries) + '\n'
                self.file_handler.atomic_write(self.config.log_file, content)

                for entry in failed_entries:
                    entry.pop('_write_failed', None)

                self.fallback_logger.info(f"Recovered {len(failed_entries)} failed log entries")
                return len(failed_entries)
            except Exception as e:
                self.fallback_logger.error(f"Failed to recover log entries: {str(e)}")
                return 0

    def query(self, conditions: Dict[str, Union[str, List, Callable]], 
              sort_by: str = None, reverse: bool = False) -> List[Dict]:
        """
        Query logs with flexible conditions.

        Args:
            conditions: Field:condition pairs for filtering
            sort_by: Field to sort results by
            reverse: Reverse sort order

        Returns:
            List of matching log entries
        """
        def matches_condition(entry, field, condition):
            value = entry.get(field)
            if value is None:
                return False
            if callable(condition):
                return condition(value)
            if isinstance(condition, list):
                return value in condition
            return value == condition

        with self.lock:
            results = [
                entry for entry in self.logs
                if all(matches_condition(entry, field, cond) 
                       for field, cond in conditions.items())
            ]

            if sort_by:
                results.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)

            return results

    def read(self, limit: Optional[int] = None, search: Optional[Dict] = None) -> List[Dict]:
        """
        Read logs with optional filtering and limiting.

        Args:
            limit: Maximum number of logs to return
            search: Key-value pairs to filter by

        Returns:
            List of matching log entries
        """
        results = self.query(conditions=search or {}, sort_by="timestamp", reverse=True)
        return results[:limit] if limit is not None else results

    def clear(self) -> None:
        """Clear all logs from memory and file."""
        with self.lock:
            self.logs.clear()
            try:
                if os.path.exists(self.config.log_file):
                    self.file_handler.safe_file_op(os.remove, self.config.log_file)
                self.fallback_logger.info(f"Cleared logs from {self.config.log_file}")
            except Exception as e:
                self.fallback_logger.error(f"Failed to clear {self.config.log_file}: {str(e)}")

    def size(self) -> int:
        """Return the number of log entries."""
        with self.lock:
            return len(self.logs)

    def stats(self) -> Dict:
        """Return basic statistics about the logs."""
        with self.lock:
            if not self.logs:
                return {}

            mood_counts = {}
            for mood in ["melancholic", "restless", "calm", "curious"]:
                mood_counts[mood] = sum(1 for log in self.logs if log.get("mood") == mood)

            return {
                "total_entries": len(self.logs),
                "first_entry": self.logs[0].get("timestamp"),
                "last_entry": self.logs[-1].get("timestamp"),
                "error_count": sum(1 for log in self.logs if "error" in log),
                "warning_count": sum(1 for log in self.logs if "warning" in log),
                "file_size": os.path.getsize(self.config.log_file) if os.path.exists(self.config.log_file) else 0,
                "failed_writes": sum(1 for log in self.logs if log.get('_write_failed')),
                "mood_distribution": mood_counts
            }

    def compress_entries(self, entries: List[Dict]) -> bytes:
        """Compress log entries to bytes."""
        try:
            json_str = '\n'.join(json.dumps(e) for e in entries)
            compressed = gzip.compress(json_str.encode('utf-8'))
            self.fallback_logger.debug(f"Compressed {len(entries)} log entries")
            return compressed
        except Exception as e:
            self.fallback_logger.error(f"Compression failed: {str(e)}")
            return b""

    def decompress_entries(self, compressed: bytes) -> List[Dict]:
        """Decompress bytes back to log entries."""
        try:
            json_str = gzip.decompress(compressed).decode('utf-8')
            entries = [json.loads(line) for line in json_str.splitlines() if line.strip()]
            self.fallback_logger.debug(f"Decompressed {len(entries)} log entries")
            return entries
        except Exception as e:
            self.fallback_logger.error(f"Decompression failed: {str(e)}")
            return []

    def get_state(self) -> Dict[str, Any]:
        """Export current state for serialization."""
        with self.lock:
            return {
                "logs": self.logs[:self.config.max_in_memory_logs],
                "log_file": self.config.log_file
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialized data."""
        try:
            with self.lock:
                self.logs.clear()
                for entry in state.get("logs", []):
                    if self.validator.validate_entry(entry):
                        self.logs.append(entry)
                self.logs = self.logs[-self.config.max_in_memory_logs:]
                self.config.log_file = state.get("log_file", self.config.log_file)
                self.fallback_logger.info(f"Loaded logger state with {len(self.logs)} entries")
        except Exception as e:
            self.fallback_logger.error(f"Failed to load logger state: {str(e)}")
            raise

    def tune(self, **kwargs) -> None:
        """Dynamically tune logger configuration."""
        try:
            with self.lock:
                self.config.update(**kwargs)
                self.logs = self.logs[-self.config.max_in_memory_logs:]
                self.file_handler.config = self.config
                self.fallback_logger.info(f"Tuned logger config: {kwargs}")
        except Exception as e:
            self.fallback_logger.error(f"Logger tuning failed: {str(e)}")
            raise

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, 
                  conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log an error with standardized format."""
        error_entry = {
            "error": error_msg,
            "type": error_type or "unknown",
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "state_hash": state_hash,
            "stack_trace": stack_trace,
            **kwargs
        }
        self.record(error_entry)

    def log_memory_usage(self, phase: str, device: torch.device, **kwargs) -> None:
        """
        Log memory usage statistics.

        Args:
            phase: The phase or operation being logged (e.g., "training", "generation")
            device: The torch device to get memory stats from
            **kwargs: Additional memory-related information to log
        """
        memory_stats = None
        if torch.cuda.is_available() and device.type == "cuda":
            memory_stats = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }

        memory_entry = {
            "event": "memory_usage",
            "phase": phase,
            "timestamp": time.time(),
            "memory_stats": memory_stats,
            **kwargs
        }

        self.record(memory_entry)

    def log_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None, **kwargs) -> None:
        """
        Log memory health check results.

        Args:
            model_size: Size of the model in bytes
            trainer: Optional trainer instance for additional memory stats
            **kwargs: Additional health-related information to log
        """
        health_entry = {
            "event": "memory_health_check",
            "timestamp": time.time(),
            "model_size": model_size,
            "health_status": "healthy"  # Default status
        }

        if trainer is not None:
            health_entry.update({
                "trainer_memory_usage": trainer.get_memory_usage(),
                "trainer_batch_size": trainer.current_batch_size
            })

        health_entry.update(kwargs)
        self.record(health_entry)

    def log_training_event(self, event_type: str, epoch: int = None, loss: float = None,
                          batch_size: int = None, data_exposure: float = None,
                          conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log training-related events."""
        self.record({
            "event": event_type,
            "epoch": epoch,
            "loss": loss,
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "state_hash": state_hash,
            **kwargs
        })

    def log_generation_event(self, prompt: str, response: str, confidence_score: float,
                            generation_params: dict = None, conversation_id: str = None,
                            state_hash: str = None, **kwargs) -> None:
        """Log generation-related events."""
        self.record({
            "event": "generation",
            "prompt": prompt,
            "response": response,
            "confidence_score": confidence_score,
            "generation_params": generation_params,
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "state_hash": state_hash,
            **kwargs
        })

    def log_cleanup_event(self, phase: str, success: bool, error: str = None,
                         conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log cleanup-related events."""
        self.record({
            "event": "cleanup",
            "phase": phase,
            "success": success,
            "error": error,
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "state_hash": state_hash,
            **kwargs
        })

    def log_curiosity_event(self, event_type: str, question: str = None, score: float = None,
                           spontaneous: bool = False, answered: bool = False,
                           conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log curiosity-related events."""
        self.record({
            "event": event_type,
            "question": question,
            "score": score,
            "spontaneous": spontaneous,
            "answered": answered,
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "state_hash": state_hash,
            **kwargs
        })

    def cleanup(self) -> None:
        """Clean up logging resources."""
        try:
            # Rotate logs one final time
            if os.path.exists(self.config.log_file):
                with self.file_lock:  # Ensure thread-safe file operations
                    self._rotate_logs()
        except Exception as e:
            self.fallback_logger.error(f"Error during cleanup: {str(e)}")

class LoggingManager:
    """Manages logging setup and configuration for the SOVL system."""

    _DEFAULT_CONFIG = {
        "logging.max_size_mb": 10,
        "logging.compress_old": True,
        "logging.rotation_count": 5,
        "logging.level": "INFO"
    }

    def __init__(
        self,
        config_manager: ConfigManager,
        log_dir: str = "logs",
        system_log_file: str = "sovl_system.log",
        error_log_file: str = "sovl_errors.log",
        debug_log_file: str = "sovl_debug.log"
    ):
        """
        Initialize the logging manager with configuration and file paths.

        Args:
            config_manager: ConfigManager instance for accessing logging settings.
            log_dir: Directory to store log files.
            system_log_file: Name of the system log file.
            error_log_file: Name of the error log file.
            debug_log_file: Name of the debug log file.
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be an instance of ConfigManager")
            
        self.config_manager = config_manager
        self.log_dir = log_dir
        self.system_log_file = os.path.join(log_dir, system_log_file)
        self.error_log_file = os.path.join(log_dir, error_log_file)
        self.debug_log_file = os.path.join(log_dir, debug_log_file)
        self.loggers = {}
        
        # Ensure logging configuration exists
        self._ensure_logging_config()
        self._validate_config()
        self._setup_logging()

    def _ensure_logging_config(self) -> None:
        """Ensure logging configuration exists in config manager."""
        try:
            # Get or create logging section
            logging_config = self.config_manager.get_section("logging", {})
            
            # Set default values if not present
            for key, default_value in self._DEFAULT_CONFIG.items():
                if key not in logging_config:
                    self.config_manager.set(key, default_value)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to ensure logging configuration: {str(e)}")

    def _validate_config(self) -> None:
        """Validate logging configuration."""
        try:
            max_size = self.config_manager.get(
                "logging.max_size_mb",
                self._DEFAULT_CONFIG["logging.max_size_mb"]
            )
            if max_size <= 0:
                raise ValueError("Log file size must be positive")
        except Exception as e:
            print(f"Logging configuration validation failed: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """Set up logging configuration and handlers."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure system logger with rotation settings
        system_config = LoggerConfig(
            log_file=self.system_log_file,
            max_size_mb=self.config_manager.get("logging.max_size_mb", self._DEFAULT_CONFIG["logging.max_size_mb"]),
            compress_old=self.config_manager.get("logging.compress_old", self._DEFAULT_CONFIG["logging.compress_old"]),
            max_in_memory_logs=1000,
            rotation_count=self.config_manager.get("logging.rotation_count", self._DEFAULT_CONFIG["logging.rotation_count"])
        )
        self.loggers["system"] = Logger(system_config)

        # Configure debug logger with rotation settings
        debug_config = LoggerConfig(
            log_file=self.debug_log_file,
            max_size_mb=self.config_manager.get("logging.max_size_mb", self._DEFAULT_CONFIG["logging.max_size_mb"]),
            compress_old=self.config_manager.get("logging.compress_old", self._DEFAULT_CONFIG["logging.compress_old"]),
            max_in_memory_logs=1000,
            rotation_count=self.config_manager.get("logging.rotation_count", self._DEFAULT_CONFIG["logging.rotation_count"])
        )
        self.loggers["debug"] = Logger(debug_config)

        # Log successful setup
        self.loggers["system"].record({
            "event": "logging_initialized",
            "message": "Logging system initialized successfully",
            "timestamp": time.time()
        })

    def setup_logging(self) -> Logger:
        """
        Set up and return the main logger.

        Returns:
            The main logger instance
        """
        if not self.loggers:
            self._setup_logging()
        return self.loggers["system"]

    def get_logger(self, name: str) -> Logger:
        """
        Get a logger instance by name.

        Args:
            name: The name of the logger to retrieve.

        Returns:
            The requested logger instance.

        Raises:
            KeyError: If the logger name is not found.
        """
        if name not in self.loggers:
            raise KeyError(f"Logger '{name}' not found")
        return self.loggers[name]

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update logging configuration and reconfigure loggers.

        Args:
            new_config: Dictionary containing new configuration values.
        """
        # Update configuration
        for key, value in new_config.items():
            if key.startswith("logging."):
                self.config_manager.set(key, value)

        # Revalidate and reconfigure
        self._validate_config()
        self._setup_logging()

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, 
                  conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log an error with standardized format."""
        self.get_logger("system").log_error(
            error_msg=error_msg,
            error_type=error_type,
            stack_trace=stack_trace,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_memory_usage(self, phase: str, device: torch.device, **kwargs) -> None:
        """
        Log memory usage statistics.

        Args:
            phase: The phase or operation being logged (e.g., "training", "generation")
            device: The torch device to get memory stats from
            **kwargs: Additional memory-related information to log
        """
        self.get_logger("system").log_memory_usage(
            phase=phase,
            device=device,
            **kwargs
        )

    def log_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None, **kwargs) -> None:
        """
        Log memory health check results.

        Args:
            model_size: Size of the model in bytes
            trainer: Optional trainer instance for additional memory stats
            **kwargs: Additional health-related information to log
        """
        self.get_logger("system").log_memory_health(
            model_size=model_size,
            trainer=trainer,
            **kwargs
        )

    def log_training_event(self, event_type: str, epoch: int = None, loss: float = None,
                          batch_size: int = None, data_exposure: float = None,
                          conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log training-related events."""
        self.get_logger("system").log_training_event(
            event_type=event_type,
            epoch=epoch,
            loss=loss,
            batch_size=batch_size,
            data_exposure=data_exposure,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_generation_event(self, prompt: str, response: str, confidence_score: float,
                            generation_params: dict = None, conversation_id: str = None,
                            state_hash: str = None, **kwargs) -> None:
        """Log generation-related events."""
        self.get_logger("system").log_generation_event(
            prompt=prompt,
            response=response,
            confidence_score=confidence_score,
            generation_params=generation_params,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_cleanup_event(self, phase: str, success: bool, error: str = None,
                         conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log cleanup-related events."""
        self.get_logger("system").log_cleanup_event(
            phase=phase,
            success=success,
            error=error,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_curiosity_event(self, event_type: str, question: str = None, score: float = None,
                           spontaneous: bool = False, answered: bool = False,
                           conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log curiosity-related events."""
        self.get_logger("system").log_curiosity_event(
            event_type=event_type,
            question=question,
            score=score,
            spontaneous=spontaneous,
            answered=answered,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def record(self, entry: Dict) -> None:
        """Write a validated log entry."""
        self.get_logger("system").record(entry)

    def cleanup(self) -> None:
        """Clean up logging resources."""
        for logger in self.loggers.values():
            logger.cleanup()

class LoggingError(Exception):
    """Raised for logging-related errors."""
    pass
