import json
import os
import gzip
from typing import Optional, List, Dict, Any, Callable
from threading import Lock
import traceback
from sovl_logger import Logger
from sovl_config import ConfigManager

class InsufficientDataError(Exception):
    """Raised when loaded data doesn't meet minimum entry requirements."""
    pass

class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

class JSONLLoader:
    """Thread-safe JSONL data loader with configurable validation."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize loader with configuration and logger.

        Args:
            config_manager: ConfigManager instance for validation rules
            logger: Logger instance for recording events
        """
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        self.required_fields = self.config_manager.get("controls_config.data_required_fields", ["prompt", "response"])
        self.min_string_length = self.config_manager.get("controls_config.data_min_string_length", 1)
        self.max_string_length = self.config_manager.get("controls_config.data_max_string_length", 10000)
        self.field_validators = {
            "prompt": lambda x: isinstance(x, str) and self.min_string_length <= len(x.strip()) <= self.max_string_length,
            "response": lambda x: isinstance(x, str) and self.min_string_length <= len(x.strip()) <= self.max_string_length,
            "conversation_id": lambda x: isinstance(x, str) and len(x) > 0,
            "timestamp": lambda x: isinstance(x, (str, float, int)) and (isinstance(x, str) and len(x) > 0 or x > 0)
        }

    def load_jsonl(
        self,
        file_path: str,
        min_entries: int = 0,
        field_mapping: Optional[Dict[str, str]] = None,
        custom_validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a JSONL file into a list of dictionaries with validation.

        Args:
            file_path: Path to the JSONL file (supports .jsonl and .jsonl.gz)
            min_entries: Minimum number of valid entries required (0 to disable)
            field_mapping: Optional mapping of input fields to output fields (e.g., {"response": "completion"})
            custom_validators: Optional custom validation functions for fields

        Returns:
            List of validated dictionaries

        Raises:
            InsufficientDataError: If fewer than min_entries valid entries are loaded
            DataValidationError: If file is invalid or corrupted
        """
        data = []
        errors = []
        field_mapping = field_mapping or {"response": "completion"}
        validators = self.field_validators.copy()
        if custom_validators:
            validators.update(custom_validators)

        try:
            with self.lock:
                if not os.path.exists(file_path):
                    self.logger.record({
                        "error": f"File not found: {file_path}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    })
                    return []

                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    self.logger.record({
                        "warning": f"File {file_path} is empty",
                        "timestamp": time.time()
                    })
                    return []

                open_func = gzip.open if file_path.endswith('.gz') else open
                mode = 'rt' if file_path.endswith('.gz') else 'r'

                with open_func(file_path, mode, encoding='utf-8') as file:
                    for line_number, line in enumerate(file, start=1):
                        line = line.strip()
                        if not line:
                            errors.append(f"Line {line_number}: Empty line")
                            continue
                        try:
                            entry = json.loads(line)
                            validated_entry = {}
                            for field in self.required_fields:
                                if field not in entry:
                                    errors.append(f"Line {line_number}: Missing field '{field}'")
                                    continue
                                if field in validators and not validators[field](entry[field]):
                                    errors.append(f"Line {line_number}: Invalid value for '{field}': {entry[field]}")
                                    continue
                                output_field = field_mapping.get(field, field)
                                validated_entry[output_field] = entry[field]
                            if len(validated_entry) == len(self.required_fields):
                                data.append(validated_entry)
                            else:
                                errors.append(f"Line {line_number}: Incomplete entry after validation")
                        except json.JSONDecodeError as e:
                            errors.append(f"Line {line_number}: JSON decode error: {str(e)}")
                            continue
                        except Exception as e:
                            errors.append(f"Line {line_number}: Unexpected error: {str(e)}")
                            continue

                # Log errors in batches to reduce overhead
                if errors:
                    self.logger.record({
                        "warning": "JSONL loading errors",
                        "errors": errors[:100],  # Limit for performance
                        "total_errors": len(errors),
                        "file_path": file_path,
                        "timestamp": time.time()
                    })

                if min_entries > 0 and len(data) < min_entries:
                    error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}"
                    self.logger.record({
                        "error": error_msg,
                        "timestamp": time.time()
                    })
                    raise InsufficientDataError(error_msg)

                self.logger.record({
                    "event": "jsonl_load",
                    "file_path": file_path,
                    "entries_loaded": len(data),
                    "file_size_bytes": file_size,
                    "timestamp": time.time()
                })
                return data

        except Exception as e:
            self.logger.record({
                "error": f"Failed to load JSONL file {file_path}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise DataValidationError(f"Failed to load JSONL file: {str(e)}")

if __name__ == "__main__":
    from sovl_logger import Logger, LoggerConfig
    from sovl_config import ConfigManager
    logger = Logger(LoggerConfig())
    config_manager = ConfigManager("sovl_config.json", logger)
    loader = JSONLLoader(config_manager, logger)
    try:
        data = loader.load_jsonl("sample.jsonl", min_entries=1)
        print(f"Loaded {len(data)} entries")
    except (InsufficientDataError, DataValidationError) as e:
        print(f"Error: {str(e)}")
