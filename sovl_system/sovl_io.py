import json
import os
import gzip
from typing import Optional, List, Dict, Any, Callable, Tuple
from threading import Lock
import traceback
from sovl_logger import Logger
from sovl_config import ConfigManager
import random
import time

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
        
        # Get field mapping from config with defaults
        self.field_mapping = self.config_manager.get(
            "core_config.data_field_mapping",
            {"response": "completion", "prompt": "prompt"},
            expected_type=dict
        )
        
        # Get required fields from config with defaults
        self.required_fields = self.config_manager.get(
            "core_config.data_required_fields",
            ["prompt", "response"],
            expected_type=list
        )
        
        # Get string length constraints from config
        self.min_string_length = self.config_manager.get(
            "controls_config.data_min_string_length",
            1,
            expected_type=int
        )
        self.max_string_length = self.config_manager.get(
            "controls_config.data_max_string_length",
            10000,
            expected_type=int
        )
        
        # Initialize field validators
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
            field_mapping: Optional mapping of input fields to output fields
            custom_validators: Optional custom validation functions for fields

        Returns:
            List of validated dictionaries

        Raises:
            InsufficientDataError: If fewer than min_entries valid entries are loaded
            DataValidationError: If file is invalid or corrupted
        """
        data = []
        errors = []
        
        # Use provided field mapping or fall back to config
        field_mapping = field_mapping or self.field_mapping
        
        # Combine default and custom validators
        validators = self.field_validators.copy()
        if custom_validators:
            validators.update(custom_validators)

        try:
            with self.lock:
                if not os.path.exists(file_path):
                    self.logger.log_error(
                        error_msg=f"File not found: {file_path}",
                        error_type="file_not_found",
                        stack_trace=traceback.format_exc(),
                        additional_info={
                            "file_path": file_path
                        }
                    )
                    return []

                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    self.logger.log_error(
                        error_msg=f"File {file_path} is empty",
                        error_type="empty_file",
                        additional_info={
                            "file_path": file_path
                        }
                    )
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
                            
                            # Validate required fields
                            for field in self.required_fields:
                                if field not in entry:
                                    errors.append(f"Line {line_number}: Missing required field '{field}'")
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
                    self.logger.log_error(
                        error_msg="JSONL loading errors",
                        error_type="data_validation_error",
                        additional_info={
                            "errors": errors[:100],  # Limit for performance
                            "total_errors": len(errors),
                            "file_path": file_path,
                            "field_mapping": field_mapping,
                            "required_fields": self.required_fields
                        }
                    )

                if min_entries > 0 and len(data) < min_entries:
                    error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}"
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="insufficient_data",
                        additional_info={
                            "entries_loaded": len(data),
                            "min_required": min_entries,
                            "file_path": file_path,
                            "field_mapping": field_mapping,
                            "required_fields": self.required_fields
                        }
                    )
                    raise InsufficientDataError(error_msg)

                self.logger.record({
                    "event": "jsonl_load",
                    "file_path": file_path,
                    "entries_loaded": len(data),
                    "file_size_bytes": file_size,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields,
                    "timestamp": time.time()
                })
                return data

        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to load JSONL file {file_path}: {str(e)}",
                error_type="data_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "file_path": file_path,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields
                }
            )
            raise DataValidationError(f"Failed to load JSONL file: {str(e)}")

def load_and_split_data(config_manager: ConfigManager, logger: Logger, train_data: List, valid_split_ratio: float) -> Tuple[List, List]:
    """
    Load and split the training data into training and validation sets.

    Args:
        config_manager: ConfigManager instance for configuration settings.
        logger: Logger instance for recording events.
        train_data: List of training data samples.
        valid_split_ratio: Ratio for splitting validation data.

    Returns:
        A tuple containing the training and validation data lists.
    """
    random_seed = config_manager.get("core_config.random_seed", 42, expected_type=int)
    random.seed(random_seed)
    random.shuffle(train_data)
    split_idx = int(len(train_data) * (1 - valid_split_ratio))
    train_data, valid_data = train_data[:split_idx], train_data[split_idx:]

    logger.write({
        "event": "data_split",
        "train_samples": len(train_data),
        "valid_samples": len(valid_data),
        "timestamp": time.time()
    })
    return train_data, valid_data

def validate_quantization_mode(mode: str, logger: Logger) -> str:
    """
    Validate and normalize the quantization mode.

    Args:
        mode: The quantization mode to validate
        logger: Logger instance for recording events

    Returns:
        The normalized quantization mode

    Raises:
        ValueError: If the mode is invalid
    """
    valid_modes = ["fp16", "int8", "int4"]
    normalized_mode = mode.lower()
    
    if normalized_mode not in valid_modes:
        logger.write({
            "warning": f"Invalid quantization mode '{mode}'. Defaulting to 'fp16'.",
            "timestamp": time.time()
        })
        return "fp16"
    
    return normalized_mode

def load_training_data(config_manager: ConfigManager, logger: Logger) -> Tuple[List, List]:
    """
    Load and validate training data from the seed file.

    Args:
        config_manager: ConfigManager instance for configuration settings
        logger: Logger instance for recording events

    Returns:
        A tuple containing the training and validation data lists

    Raises:
        InsufficientDataError: If fewer than minimum required entries are loaded
        DataValidationError: If data validation fails
    """
    try:
        # Initialize JSONL loader
        loader = JSONLLoader(config_manager, logger)
        
        # Get minimum required entries from config
        min_entries = config_manager.get("core_config.min_training_entries", 10, expected_type=int)
        
        # Load training data with minimum entries requirement
        train_data = loader.load_jsonl("sovl_seed.jsonl", min_entries=min_entries)
        
        # Get validation split ratio from config
        valid_split_ratio = config_manager.get("core_config.valid_split_ratio", 0.2, expected_type=float)
        
        # Split data into training and validation sets
        train_data, valid_data = load_and_split_data(config_manager, logger, train_data, valid_split_ratio)
        
        # Log successful data loading
        logger.record({
            "event": "training_data_loaded",
            "train_samples": len(train_data),
            "valid_samples": len(valid_data),
            "min_entries": min_entries,
            "timestamp": time.time(),
            "conversation_id": "init"
        })
        
        return train_data, valid_data

    except InsufficientDataError as e:
        logger.log_error(
            error_msg=str(e),
            error_type="insufficient_data",
            stack_trace=traceback.format_exc(),
            additional_info={
                "min_entries": min_entries
            }
        )
        raise  # Re-raise to force caller to handle

    except Exception as e:
        logger.log_error(
            error_msg=f"Unexpected error during data loading: {str(e)}",
            error_type="data_loading_error",
            stack_trace=traceback.format_exc()
        )
        raise DataValidationError(f"Failed to load training data: {str(e)}")

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
