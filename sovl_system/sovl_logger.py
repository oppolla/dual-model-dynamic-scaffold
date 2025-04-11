import json
import os
from datetime import datetime
from threading import Lock  # Add for thread safety

class Logger:
    """A thread-safe logger for storing and retrieving system interactions."""
    
    def __init__(self, log_file="sovl_logs.jsonl"):
        """Initialize the logger with an optional file path."""
        self.logs = []  # In-memory log storage
        self.log_file = log_file
        self.lock = Lock()  # Add thread safety
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.logs.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Warning: Failed to load logs from {log_file}: {e}")

    def write(self, entry):
        """Write a log entry to memory and file."""
        if not isinstance(entry, dict):
            raise ValueError("Log entry must be a dictionary")
        # Add timestamp if missing
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        # Add error prompt flag and default conversation_id (from ThreadSafeLogger)
        if "error" in entry or "warning" in entry:
            entry["is_error_prompt"] = True
            if "conversation_id" not in entry:
                entry["conversation_id"] = str(uuid.uuid4())
        
        with self.lock:  # Thread-safe write
            self.logs.append(entry)
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Warning: Failed to write to {self.log_file}: {e}")

    def read(self, limit=None):
        """Read all logs or up to a limit from memory."""
        with self.lock:  # Thread-safe read
            if limit is None:
                return self.logs[:]
            return self.logs[-limit:] if limit > 0 else []

    def clear(self):
        """Clear all logs from memory and file."""
        with self.lock:  # Thread-safe clear
            self.logs = []
            try:
                if os.path.exists(self.log_file):
                    os.remove(self.log_file)
            except Exception as e:
                print(f"Warning: Failed to clear {self.log_file}: {e}")

    def size(self):
        """Return the number of log entries."""
        with self.lock:  # Thread-safe size check
            return len(self.logs)

if __name__ == "__main__":
    # Quick test
    logger = Logger("test_logs.jsonl")
    logger.write({"prompt": "Hello", "response": "Hi there", "confidence_score": 0.8})
    logger.write({"error": "Test error", "details": "Something broke"})
    print("Logs:", logger.read())
    print("Last log:", logger.read(1))
    print("Size:", logger.size())
    logger.clear()
    print("After clear:", logger.read())
