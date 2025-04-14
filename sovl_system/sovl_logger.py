import json
import os
import gzip
import uuid
import time
import logging
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional, Callable
from dataclasses import dataclass
import torch

@dataclass
class LoggerConfig:
    """Configuration for Logger, aligned with ConfigManager."""
    log_file: str = "sovl_logs.jsonl"
    max_size_mb: int = 10
    compress_old: bool = False
    max_in_memory_logs: int = 1000
    _ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        self._ranges = {
            "max_size_mb": (0, 100),
            "max_in_memory_logs": (100, 10000),
        }
        for key, (min_val, max_val) in self._ranges.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
        if not isinstance(self.log_file, str) or not self.log_file.endswith(".jsonl"):
            raise ValueError("log_file must be a .jsonl file path")
        if not isinstance(self.compress_old, bool):
            raise ValueError("compress_old must be a boolean")

    def update(self, **kwargs) -> None:
        """Dynamically update configuration parameters with validation."""
        for key, value in kwargs.items():
            if key == "log_file":
                if not isinstance(value, str) or not value.endswith(".jsonl"):
                    raise ValueError("log_file must be a .jsonl file path")
            elif key == "compress_old":
                if not isinstance(value, bool):
                    raise ValueError("compress_old must be a boolean")
            elif key in self._ranges:
                min_val, max_val = self._ranges[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)

class Logger:
    """A thread-safe logger for SOVLSystem interactions, aligned with system components."""
    
    REQUIRED_FIELDS = ['timestamp', 'conversation_id']
    OPTIONAL_FIELDS = ['prompt', 'response', 'confidence_score', 'error', 'warning', 'mood', 'variance', 'logits_shape']
    FIELD_VALIDATORS = {
        'timestamp': lambda x: isinstance(x, (str, float, int)),
        'conversation_id': lambda x: isinstance(x, str),
        'confidence_score': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
        'is_error_prompt': lambda x: isinstance(x, bool),
        'mood': lambda x: x in ['melancholic', 'restless', 'calm', 'curious'],
        'variance': lambda x: isinstance(x, (int, float)) and x >= 0.0,
        'logits_shape': lambda x: isinstance(x, (tuple, list, str))
    }
    
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
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing logs from file with memory constraints."""
        if not os.path.exists(self.config.log_file):
            return
            
        try:
            with self._safe_file_op(open, self.config.log_file, 'r', encoding='utf-8') as f:
                temp_logs = []
                for line_num, line in enumerate(f, 1):
                    if len(temp_logs) >= self.config.max_in_memory_logs:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict) and self._validate_entry(entry):
                            temp_logs.append(entry)
                    except json.JSONDecodeError:
                        self.fallback_logger.warning(f"Corrupted log entry at line {line_num}, skipping")
                        continue
                        
            self.logs = temp_logs[-self.config.max_in_memory_logs:]  # Keep only recent logs
            self.fallback_logger.info(f"Loaded {len(self.logs)} log entries from {self.config.log_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to load logs from {self.config.log_file}: {str(e)}")
            corrupted_file = f"{self.config.log_file}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(self.config.log_file, corrupted_file)
                self.fallback_logger.info(f"Preserved corrupted log file as {corrupted_file}")
            except OSError:
                self.fallback_logger.error(f"Failed to preserve corrupted log file")
            self.logs = []

    def _safe_file_op(self, operation: Callable, *args, **kwargs):
        """Wrapper for file operations with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    self.fallback_logger.error(f"File operation failed after {max_retries} retries: {str(e)}")
                    raise
                time.sleep(0.1 * (attempt + 1))

    def _atomic_write(self, filename: str, content: str):
        """Atomic file write using temp file pattern."""
        temp_file = f"{filename}.tmp"
        try:
            with self._safe_file_op(open, temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self._safe_file_op(os.replace, temp_file, filename)
        except Exception as e:
            self.fallback_logger.error(f"Atomic write failed: {str(e)}")
            if os.path.exists(temp_file):
                self._safe_file_op(os.remove, temp_file)
            raise

    def _rotate_if_needed(self) -> None:
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
                with self._safe_file_op(open, self.config.log_file, 'rb') as f_in:
                    with self._safe_file_op(gzip.open, rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                self._safe_file_op(os.rename, self.config.log_file, rotated_file)
                
            self.logs = self.logs[-self.config.max_in_memory_logs:]  # Keep recent logs
            self.manage_rotation()
            self.fallback_logger.info(f"Rotated logs to {rotated_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to rotate log file: {str(e)}")

    def _validate_entry(self, entry: Dict) -> bool:
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

    def record(self, entry: Dict) -> None:
        """Write a validated log entry (alias for write)."""
        self.write(entry)

    def write(self, entry: Dict) -> None:
        """
        Write a validated log entry.

        Args:
            entry: Single log entry to write
        """
        if not self._validate_entry(entry):
            raise ValueError("Invalid log entry structure")
            
        # Add error metadata
        if "error" in entry or "warning" in entry:
            entry["is_error_prompt"] = True
            
        with self.lock:
            self.logs.append(entry)
            self.logs = self.logs[-self.config.max_in_memory_logs:]  # Trim memory
            self._rotate_if_needed()
            
            try:
                self._atomic_write(self.config.log_file, json.dumps(entry) + '\n')
            except Exception as e:
                self.fallback_logger.error(f"Failed to write to {self.config.log_file}: {str(e)}")
                entry['_write_failed'] = True

    def write_batch(self, entries: List[Dict]) -> None:
        """Optimized batch writing with validation and atomic write."""
        if not entries:
            return
            
        valid_entries = []
        for entry in entries:
            if self._validate_entry(entry):
                if "error" in entry or "warning" in entry:
                    entry["is_error_prompt"] = True
                valid_entries.append(entry)
            else:
                self.fallback_logger.warning(f"Invalid log entry skipped: {entry}")
                
        if not valid_entries:
            return
            
        with self.lock:
            self.logs.extend(valid_entries)
            self.logs = self.logs[-self.config.max_in_memory_logs:]  # Trim memory
            self._rotate_if_needed()
            
            try:
                content = '\n'.join(json.dumps(e) for e in valid_entries) + '\n'
                self._atomic_write(self.config.log_file, content)
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
                self._atomic_write(self.config.log_file, content)
                
                for entry in failed_entries:
                    if '_write_failed' in entry:
                        del entry['_write_failed']
                        
                self.fallback_logger.info(f"Recovered {len(failed_entries)} failed log entries")
                return len(failed_entries)
            except Exception as e:
                self.fallback_logger.error(f"Failed to recover log entries: {str(e)}")
                return 0

    def query(self, conditions: Dict[str, Union[str, List, Callable]], 
             sort_by: str = None, 
             reverse: bool = False) -> List[Dict]:
        """
        Advanced log querying with multiple condition types.

        Args:
            conditions: Dict of field:condition pairs
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
            search: Dictionary of key-value pairs to filter by

        Returns:
            List of matching log entries
        """
        return self.query(conditions=search or {}, sort_by="timestamp", reverse=True)[:limit or len(self.logs)]

    def clear(self) -> None:
        """Clear all logs from memory and file."""
        with self.lock:
            self.logs = []
            try:
                if os.path.exists(self.config.log_file):
                    self._safe_file_op(os.remove, self.config.log_file)
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
                
            return {
                "total_entries": len(self.logs),
                "first_entry": self.logs[0].get("timestamp"),
                "last_entry": self.logs[-1].get("timestamp"),
                "error_count": sum(1 for log in self.logs if "error" in log),
                "warning_count": sum(1 for log in self.logs if "warning" in log),
                "file_size": os.path.getsize(self.config.log_file) if os.path.exists(self.config.log_file) else 0,
                "failed_writes": sum(1 for log in self.logs if log.get('_write_failed')),
                "mood_distribution": {
                    mood: sum(1 for log in self.logs if log.get("mood") == mood)
                    for mood in ["melancholic", "restless", "calm", "curious"]
                }
            }

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file and optionally keep original."""
        if not os.path.exists(self.config.log_file):
            return None
            
        compressed_file = f"{self.config.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with self._safe_file_op(open, self.config.log_file, 'rb') as f_in:
                with self._safe_file_op(gzip.open, compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
            if not keep_original:
                self._safe_file_op(os.remove, self.config.log_file)
                
            self.fallback_logger.info(f"Compressed logs to {compressed_file}")
            return compressed_file
        except Exception as e:
            self.fallback_logger.error(f"Failed to compress logs: {str(e)}")
            return None

    def compress_entries(self, entries: List[Dict]) -> bytes:
        """Compress log entries to bytes for efficient storage/transfer."""
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
            entries = [json.loads(line) for line in json_str.splitlines()]
            self.fallback_logger.debug(f"Decompressed {len(entries)} log entries")
            return entries
        except Exception as e:
            self.fallback_logger.error(f"Decompression failed: {str(e)}")
            return []

    def manage_rotation(self, max_files: int = 5) -> None:
        """Manage rotated log files, keeping only max_files most recent."""
        if not os.path.exists(self.config.log_file):
            return
            
        try:
            base_name = os.path.basename(self.config.log_file)
            log_dir = os.path.dirname(self.config.log_file) or '.'
            
            rotated_files = []
            for f in os.listdir(log_dir):
                if f.startswith(base_name) and f != base_name:
                    rotated_files.append(os.path.join(log_dir, f))
                    
            rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            for old_file in rotated_files[max_files:]:
                try:
                    self._safe_file_op(os.remove, old_file)
                    self.fallback_logger.info(f"Removed old log file {old_file}")
                except OSError:
                    self.fallback_logger.error(f"Failed to remove old log file {old_file}")
        except Exception as e:
            self.fallback_logger.error(f"Error managing log rotation: {str(e)}")

    def get_state(self) -> Dict[str, Any]:
        """
        Export current state for serialization.

        Returns:
            Dictionary containing logger state
        """
        with self.lock:
            return {
                "logs": self.logs[:self.config.max_in_memory_logs],
                "log_file": self.config.log_file
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from serialized data.

        Args:
            state: Dictionary containing logger state
        """
        try:
            with self.lock:
                self.logs = []
                for entry in state.get("logs", []):
                    if self._validate_entry(entry):
                        self.logs.append(entry)
                self.logs = self.logs[-self.config.max_in_memory_logs:]
                self.config.log_file = state.get("log_file", self.config.log_file)
                self.fallback_logger.info(f"Loaded logger state with {len(self.logs)} entries")
        except Exception as e:
            self.fallback_logger.error(f"Failed to load logger state: {str(e)}")
            raise

    def tune(self, **kwargs) -> None:
        """
        Dynamically tune logger configuration parameters.

        Args:
            **kwargs: Parameters to update (e.g., max_size_mb, compress_old)
        """
        try:
            with self.lock:
                old_config = vars(self.config).copy()
                self.config.update(**kwargs)
                self.logs = self.logs[-self.config.max_in_memory_logs:]
                self.fallback_logger.info(f"Tuned logger config: {kwargs}")
        except Exception as e:
            self.fallback_logger.error(f"Logger tuning failed: {str(e)}")
            raise

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, 
                 conversation_id: str = None, state_hash: str = None, **kwargs):
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

    def log_memory_usage(self, phase: str, device: torch.device, **kwargs):
        """Log memory usage statistics."""
        if torch.cuda.is_available() and device.type == "cuda":
            memory_stats = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        else:
            memory_stats = None

        self.record({
            "event": "memory_usage",
            "phase": phase,
            "timestamp": time.time(),
            "memory_stats": memory_stats,
            **kwargs
        })

    def log_training_event(self, event_type: str, epoch: int = None, loss: float = None,
                         batch_size: int = None, data_exposure: float = None,
                         conversation_id: str = None, state_hash: str = None, **kwargs):
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
                           state_hash: str = None, **kwargs):
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
                         conversation_id: str = None, state_hash: str = None, **kwargs):
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
                          conversation_id: str = None, state_hash: str = None, **kwargs):
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
