import json
import os
import logging
from typing import Optional, Callable, Dict, List, Any

class InsufficientDataError(Exception):
    pass

def load_jsonl(file_path: str, min_entries: int = 0, logger: Optional[logging.Logger] = None) -> List[Dict[str, str]]:
    """
    Load a JSONL file into a list of dictionaries with validation.

    Args:
        file_path (str): Path to the JSONL file.
        min_entries (int): Minimum number of valid entries required (0 to disable).
        logger (logging.Logger, optional): Logger object to record errors.

    Returns:
        list: List of dictionaries with 'prompt' and 'completion' keys.

    Raises:
        InsufficientDataError: If fewer than min_entries valid entries are loaded and min_entries > 0.
    """
    data = []
    errors = []

    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    if os.path.getsize(file_path) == 0:
        logger.warning(f"File {file_path} is empty.")
        return []

    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    errors.append(f"Line {line_number}: Empty line. Skipping.")
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry.get("prompt"), str) or not isinstance(entry.get("response"), str):
                        errors.append(f"Line {line_number}: Missing or invalid 'prompt' or 'response'. Skipping.")
                        continue
                    if not entry["prompt"].strip() or not entry["response"].strip():
                        errors.append(f"Line {line_number}: Empty 'prompt' or 'response'. Skipping.")
                        continue
                    data.append({"prompt": entry["prompt"], "completion": entry["response"]})
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_number}: JSON decode error: {e}. Skipping.")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except OSError as e:
        logger.error(f"OSError when loading {file_path}: {e}")
        return []

    for error in errors:
        logger.warning(error)

    if min_entries > 0 and len(data) < min_entries:
        error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}."
        logger.error(error_msg)
        raise InsufficientDataError(error_msg)

    logger.info(f"Data Validation: {len(data)} entries loaded successfully from {file_path}.")
    return data

import warnings
from sovl_config import ConfigManager

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
