import json
import os
from datetime import datetime

class Logger:
    """A simple logger for storing and retrieving system interactions."""
    
    def __init__(self, log_file="sovl_logs.jsonl"):
        """Initialize the logger with an optional file path."""
        self.logs = []  # In-memory log storage
        self.log_file = log_file  # File to persist logs
        # Load existing logs if file exists
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.logs.append(json.loads(line.strip()))
            except Exception as e:
                print(f"Warning: Failed to load logs from {log_file}: {e}")

    def write(self, entry):
        """Write a log entry to memory and file."""
        # Ensure entry is a dict and add timestamp if missing
        if not isinstance(entry, dict):
            raise ValueError("Log entry must be a dictionary")
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        
        # Append to in-memory logs
        self.logs.append(entry)
        
        # Append to file (JSONL format: one JSON object per line)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write to {self.log_file}: {e}")

    def read(self, limit=None):
        """Read all logs or up to a limit from memory."""
        if limit is None:
            return self.logs
        return self.logs[-limit:] if limit > 0 else []

    def clear(self):
        """Clear all logs from memory and file."""
        self.logs = []
        try:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
        except Exception as e:
            print(f"Warning: Failed to clear {self.log_file}: {e}")

    def size(self):
        """Return the number of log entries."""
        return len(self.logs)

if __name__ == "__main__":
    # Quick test
    logger = Logger("test_logs.jsonl")
    logger.write({"prompt": "Hello", "response": "Hi there", "confidence_score": 0.8})
    logger.write({"prompt": "Bye", "response": "See ya", "confidence_score": 0.7})
    print("Logs:", logger.read())
    print("Last log:", logger.read(1))
    print("Size:", logger.size())
    logger.clear()
    print("After clear:", logger.read())
