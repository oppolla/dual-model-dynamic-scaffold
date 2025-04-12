import json
import os
import warnings
from typing import Optional, Callable
from sovl_config import ConfigManager

class InsufficientDataError(Exception):
    pass

def load_jsonl(file_path: str, min_entries: int = 0, logger: Optional[Callable] = None) -> list:
    """
    Load a JSONL file into a list of dictionaries with validation.

    Args:
        file_path (str): Path to the JSONL file.
        min_entries (int): Minimum number of valid entries required (0 to disable).
        logger (Callable, optional): Logger function to record errors (e.g., sovl_logger.Logger.record).

    Returns:
        list: List of dictionaries with 'prompt' and 'completion' keys.

    Raises:
        InsufficientDataError: If fewer than min_entries valid entries are loaded and min_entries > 0.
    """
    data = []
    error_log = []

    # Default to print if no logger is provided
    log_fn = logger if logger else lambda x: print(f"INFO: {x}")

    if not os.path.exists(file_path):
        log_fn(f"File not found: {file_path}. Returning empty list.")
        return []

    if os.path.getsize(file_path) == 0:
        log_fn(f"File {file_path} is empty. Returning empty list.")
        return []

    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    error_log.append(f"Line {line_number}: Empty line. Skipping.")
                    continue
                try:
                    entry = json.loads(line.strip())
                    if not isinstance(entry.get("prompt"), str) or not isinstance(entry.get("response"), str):
                        error_log.append(f"Line {line_number}: Missing or invalid 'prompt' or 'response'. Skipping.")
                        continue
                    if not entry["prompt"].strip() or not entry["response"].strip():
                        error_log.append(f"Line {line_number}: Empty 'prompt' or 'response'. Skipping.")
                        continue
                    data.append({"prompt": entry["prompt"], "completion": entry["response"]})
                except json.JSONDecodeError as e:
                    error_log.append(f"Line {line_number}: JSON decode error: {e}. Skipping.")

        if error_log:
            log_fn("Warnings encountered during data loading:")
            for error in error_log:
                log_fn(f"WARNING: {error}")

        if min_entries > 0 and len(data) < min_entries:
            error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}."
            log_fn(f"ERROR: {error_msg}")
            raise InsufficientDataError(error_msg)

    except Exception as e:
        log_fn(f"Unexpected error loading {file_path}: {e}. Returning empty list.")
        return []

    log_fn(f"Data Validation: {len(data)} entries loaded successfully from {file_path}.")
    return data

def load_config(config_file="sovl_config.json", defaults=None):
    """DEPRECATED: Use ConfigManager instead."""
    warnings.warn(
        "load_config is deprecated. Use sovl_config.ConfigManager instead.",
        DeprecationWarning,
        stacklevel=2
    )
    config_manager = ConfigManager(config_file)
    if defaults:
        for key, value in defaults.items():
            config_manager.update(key, value)
    return config_manager.config

def get_config_value(config, key, default=None):
    """DEPRECATED: Use ConfigManager.get instead."""
    warnings.warn(
        "get_config_value is deprecated. Use ConfigManager.get instead.",
        DeprecationWarning,
        stacklevel=2
    )
    config_manager = ConfigManager("dummy.json")
    config_manager.config = config
    return config_manager.get(key, default)
