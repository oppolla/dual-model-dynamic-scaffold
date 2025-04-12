import json
import os
import gzip
import uuid
import time
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional, Callable

class Logger:
    """A thread-safe logger for storing and retrieving system interactions with enhanced features."""
    
    # Field configuration constants
    REQUIRED_FIELDS = ['timestamp', 'conversation_id']
    OPTIONAL_FIELDS = ['prompt', 'response', 'confidence_score', 'error', 'warning']
    FIELD_VALIDATORS = {
        'timestamp': lambda x: isinstance(x, (str, float, int)),
        'conversation_id': lambda x: isinstance(x, str),
        'confidence_score': lambda x: isinstance(x, (int, float)),
        'is_error_prompt': lambda x: isinstance(x, bool)
    }
    
    def __init__(self, log_file: str = "sovl_logs.jsonl", max_size_mb: int = 10, compress_old: bool = False):
        """
        Initialize the logger with configurable file handling.
        
        Args:
            log_file: Path to log file
            max_size_mb: Maximum size in MB before rotation (0 for no rotation)
            compress_old: Whether to compress rotated logs
        """
        self.logs: List[Dict] = []
        self.log_file = log_file
        self.max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.compress_old = compress_old
        self.lock = Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing logs from file with improved error recovery."""
        if not os.path.exists(self.log_file):
            return
            
        temp_logs = []
        try:
            with self._safe_file_op(open, self.log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict):  # Validate entry structure
                            # Ensure required fields exist
                            if 'timestamp' not in entry:
                                entry['timestamp'] = datetime.now().isoformat()
                            if 'conversation_id' not in entry:
                                entry['conversation_id'] = str(uuid.uuid4())
                            temp_logs.append(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted log entry at line {line_num}, skipping")
                        continue
                        
            self.logs = temp_logs
        except Exception as e:
            print(f"Critical: Failed to load logs from {self.log_file}: {str(e)}")
            # Attempt to preserve corrupted logs for analysis
            corrupted_file = f"{self.log_file}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.rename(self.log_file, corrupted_file)
            except OSError:
                print(f"Failed to preserve corrupted log file")
            self.logs = []

    def _safe_file_op(self, operation: Callable, *args, **kwargs):
        """Wrapper for file operations with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
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
            if os.path.exists(temp_file):
                self._safe_file_op(os.remove, temp_file)
            raise

    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if self.max_size <= 0 or not os.path.exists(self.log_file):
            return
            
        file_size = os.path.getsize(self.log_file)
        if file_size < self.max_size:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = f"{self.log_file}.{timestamp}"
            
            if self.compress_old:
                rotated_file += ".gz"
                with self._safe_file_op(open, self.log_file, 'rb') as f_in:
                    with self._safe_file_op(gzip.open, rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                self._safe_file_op(os.rename, self.log_file, rotated_file)
                
            self.logs = []  # Clear in-memory logs after rotation
            self.manage_rotation()  # Clean up old rotated files
        except Exception as e:
            print(f"Warning: Failed to rotate log file: {str(e)}")

    def _validate_entry(self, entry: Dict) -> bool:
        """Validate log entry structure and types."""
        if not isinstance(entry, dict):
            return False
            
        try:
            # Ensure required fields
            for field in self.REQUIRED_FIELDS:
                if field not in entry:
                    if field == 'timestamp':
                        entry['timestamp'] = datetime.now().isoformat()
                    elif field == 'conversation_id':
                        entry['conversation_id'] = str(uuid.uuid4())
            
            # Validate field types
            for field, validator in self.FIELD_VALIDATORS.items():
                if field in entry and not validator(entry[field]):
                    return False
                    
            return True
        except Exception:
            return False

    def record(self, entry: Dict) -> None:
        """Alias for write() to maintain backward compatibility."""
        self.write(entry)

    def write(self, entry: Dict) -> None:
        """
        Write a validated log entry.
        
        Args:
            entry: Single log entry to write
        """
        if not self._validate_entry(entry):
            raise ValueError("Invalid log entry structure")
            
        # Add error metadata if applicable
        if "error" in entry or "warning" in entry:
            entry["is_error_prompt"] = True
            
        with self.lock:
            self.logs.append(entry)
            self._rotate_if_needed()
            
            try:
                self._atomic_write(self.log_file, json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Warning: Failed to write to {self.log_file}: {str(e)}")
                # Store in memory only if file write fails
                self.logs[-1]['_write_failed'] = True

    def write_batch(self, entries: List[Dict]) -> None:
        """Optimized batch writing with validation and atomic write."""
        if not entries:
            return
            
        # Validate all entries first
        valid_entries = []
        for entry in entries:
            if self._validate_entry(entry):
                valid_entries.append(entry)
            else:
                print(f"Warning: Invalid log entry skipped: {entry}")
                
        if not valid_entries:
            return
            
        with self.lock:
            self.logs.extend(valid_entries)
            self._rotate_if_needed()
            
            try:
                content = '\n'.join(json.dumps(e) for e in valid_entries) + '\n'
                self._atomic_write(self.log_file, content)
            except Exception as e:
                print(f"Error writing batch: {str(e)}")
                # Mark all batch entries as failed
                for entry in valid_entries:
                    entry['_write_failed'] = True

    def recover_failed_writes(self) -> int:
        """Attempt to recover any log entries that failed to write to disk."""
        with self.lock:
            failed_entries = [i for i, entry in enumerate(self.logs) 
                            if entry.get('_write_failed')]
            if not failed_entries:
                return 0
                
            try:
                # Write all failed entries at once
                entries_to_write = [self.logs[i] for i in failed_entries]
                content = '\n'.join(json.dumps(e) for e in entries_to_write) + '\n'
                
                self._atomic_write(self.log_file, content)
                
                # Clear failure flags
                for i in failed_entries:
                    if '_write_failed' in self.logs[i]:
                        del self.logs[i]['_write_failed']
                        
                return len(failed_entries)
            except Exception as e:
                print(f"Failed to recover log entries: {str(e)}")
                return 0

    def query(self, conditions: Dict[str, Union[str, List, Callable]], 
             sort_by: str = None, 
             reverse: bool = False) -> List[Dict]:
        """
        Advanced log querying with multiple condition types.
        
        Args:
            conditions: Dict of field:condition pairs where condition can be:
                       - Exact match value
                       - List of possible values
                       - Callable function that takes value returns bool
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
                if os.path.exists(self.log_file):
                    self._safe_file_op(os.remove, self.log_file)
            except Exception as e:
                print(f"Warning: Failed to clear {self.log_file}: {str(e)}")

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
                "file_size": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0,
                "failed_writes": sum(1 for log in self.logs if log.get('_write_failed'))
            }

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file and optionally keep original.
        
        Returns path to compressed file or None if failed.
        """
        if not os.path.exists(self.log_file):
            return None
            
        compressed_file = f"{self.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with self._safe_file_op(open, self.log_file, 'rb') as f_in:
                with self._safe_file_op(gzip.open, compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
            if not keep_original:
                self._safe_file_op(os.remove, self.log_file)
                
            return compressed_file
        except Exception as e:
            print(f"Failed to compress logs: {str(e)}")
            return None

    def compress_entries(self, entries: List[Dict]) -> bytes:
        """Compress log entries to bytes for efficient storage/transfer."""
        try:
            json_str = '\n'.join(json.dumps(e) for e in entries)
            return gzip.compress(json_str.encode('utf-8'))
        except Exception as e:
            print(f"Compression failed: {str(e)}")
            return None

    def decompress_entries(self, compressed: bytes) -> List[Dict]:
        """Decompress bytes back to log entries."""
        try:
            json_str = gzip.decompress(compressed).decode('utf-8')
            return [json.loads(line) for line in json_str.splitlines()]
        except Exception as e:
            print(f"Decompression failed: {str(e)}")
            return []

    def manage_rotation(self, max_files: int = 5) -> None:
        """Manage rotated log files, keeping only max_files most recent."""
        if not os.path.exists(self.log_file):
            return
            
        try:
            # Find all rotated log files
            base_name = os.path.basename(self.log_file)
            log_dir = os.path.dirname(self.log_file) or '.'
            
            rotated_files = []
            for f in os.listdir(log_dir):
                if f.startswith(base_name) and f != base_name:
                    rotated_files.append(os.path.join(log_dir, f))
                    
            # Sort by modification time (newest first)
            rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Remove oldest files beyond max_files limit
            for old_file in rotated_files[max_files:]:
                try:
                    self._safe_file_op(os.remove, old_file)
                except OSError:
                    print(f"Failed to remove old log file {old_file}")
        except Exception as e:
            print(f"Error managing log rotation: {str(e)}")
