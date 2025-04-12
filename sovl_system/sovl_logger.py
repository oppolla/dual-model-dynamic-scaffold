import json
import os
import gzip
import uuid
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional, Callable

class Logger:
    """A thread-safe logger for storing and retrieving system interactions with enhanced features."""
    
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
            with self.lock, open(self.log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict):  # Validate entry structure
                            temp_logs.append(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted log entry at line {line_num}, skipping")
                        continue
                        
            self.logs = temp_logs
        except Exception as e:
            print(f"Critical: Failed to load logs from {self.log_file}: {str(e)}")
            # Attempt to preserve corrupted logs for analysis
            corrupted_file = f"{self.log_file}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.log_file, corrupted_file)
            self.logs = []

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
                with open(self.log_file, 'rb') as f_in:
                    with gzip.open(rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                os.rename(self.log_file, rotated_file)
                
            self.logs = []  # Clear in-memory logs after rotation
        except Exception as e:
            print(f"Warning: Failed to rotate log file: {str(e)}")

    def _validate_entry(self, entry: Dict) -> bool:
        """Validate log entry structure and types."""
        if not isinstance(entry, dict):
            return False
            
        try:
            # Required fields
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
                
            # Type checking
            if 'confidence_score' in entry and not isinstance(entry['confidence_score'], (int, float)):
                return False
                
            # Conversation ID formatting
            if 'conversation_id' not in entry:
                entry['conversation_id'] = str(uuid.uuid4())
                
            return True
        except Exception:
            return False

    def write(self, entry: Dict, batch: bool = False) -> None:
        """
        Write a validated log entry or batch of entries.
        
        Args:
            entry: Single log entry or list of entries
            batch: Whether writing multiple entries at once
        """
        if not self._validate_entry(entry):
            raise ValueError("Invalid log entry structure")
            
        # Add error metadata if applicable
        if "error" in entry or "warning" in entry:
            entry["is_error_prompt"] = True
            
        with self.lock:
            if batch:
                if not all(self._validate_entry(e) for e in entry):
                    raise ValueError("Batch contains invalid log entries")
                self.logs.extend(entry)
            else:
                self.logs.append(entry)
                
            self._rotate_if_needed()
            try:
                mode = 'a' if os.path.exists(self.log_file) else 'w'
                with open(self.log_file, mode, encoding='utf-8') as f:
                    if batch:
                        f.write('\n'.join(json.dumps(e) for e in entry) + '\n')
                    else:
                        f.write(json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Warning: Failed to write to {self.log_file}: {str(e)}")

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
                    for field, cond in conditions.items()
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
                    os.remove(self.log_file)
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
                "file_size": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
            }

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file and optionally keep original.
        
        Returns path to compressed file or None if failed.
        """
        if not os.path.exists(self.log_file):
            return None
            
        compressed_file = f"{self.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
                    
            if not keep_original:
                os.remove(self.log_file)
                
            return compressed_file
        except Exception as e:
            print(f"Failed to compress logs: {str(e)}")
            return None
