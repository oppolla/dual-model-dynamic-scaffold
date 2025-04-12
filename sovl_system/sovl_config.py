import json
import logging
from typing import Any, Optional

class ConfigManager:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = {}
        self._load_config()

    def _load_config(self):
        """Load the configuration file into memory."""
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            logging.info(f"Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_file} not found.")
            self.config = {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {self.config_file}: {e}")
            self.config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the configuration using a dot-separated key.
        Returns a default value if the key is missing.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        if value == {}:  # Handle empty dictionaries
            logging.warning(f"Key '{key}' is empty or missing. Using default: {default}")
            return default
        return value

    def validate_keys(self, required_keys: list[str]):
        """
        Validate that all required keys exist in the configuration.
        Logs an error for any missing keys.
        """
        missing_keys = []
        for key in required_keys:
            if self.get(key, None) is None:
                missing_keys.append(key)
        if missing_keys:
            logging.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager("sovl_config.json")
    try:
        config_manager.validate_keys(["core_config.base_model_name", "training_config.learning_rate"])
    except ValueError as e:
        print(e)
