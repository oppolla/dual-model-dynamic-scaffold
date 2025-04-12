import json
import os
import gzip
import uuid
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional

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
        """Load existing logs from file in a thread-safe manner."""
        if not os.path.exists(self.log_file):
            return
            
        try:
            with self.lock, open(self.log_file, 'r', encoding='utf-8') as f:
                self.logs = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Failed to load logs from {self.log_file}: {str(e)}")
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

    def write(self, entry: Dict, batch: bool = False) -> None:
        """
        Write a log entry or batch of entries.
        
        Args:
            entry: Single log entry or list of entries
            batch: Whether writing multiple entries at once
        """
        if not isinstance(entry, dict):
            raise ValueError("Log entry must be a dictionary")
            
        # Add standard metadata
        entry.setdefault("timestamp", datetime.now().isoformat())
        if "conversation_id" not in entry:
            entry["conversation_id"] = str(uuid.uuid4())
        if "error" in entry or "warning" in entry:
            entry["is_error_prompt"] = True
            
        with self.lock:
            if batch:
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

    def read(self, limit: Optional[int] = None, search: Optional[Dict] = None) -> List[Dict]:
        """
        Read logs with optional filtering and limiting.
        
        Args:
            limit: Maximum number of logs to return
            search: Dictionary of key-value pairs to filter by
            
        Returns:
            List of matching log entries
        """
        with self.lock:
            logs = self.logs[:]
            
        if search:
            logs = [log for log in logs if all(log.get(k) == v for k, v in search.items())]
            
        if limit is not None and limit > 0:
            return logs[-limit:]
        return logs

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
