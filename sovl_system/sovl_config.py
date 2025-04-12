import json
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

class ConfigManager:
    def __init__(self, config_file: str):
        """
        Initialize ConfigManager with configuration file path.
        Maintains 100% backward compatibility with original interface.
        """
        self.config_file = config_file
        self.config = {}  # Preserve original flat dict structure
        self._structured_config = None  # New structured config
        self._load_config()

    def _load_config(self):
        """Load the configuration file into memory (original implementation)."""
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            logging.info(f"Configuration loaded from {self.config_file}")
            
            # New: Build structured config while maintaining original flat config
            self._build_structured_config()
            
        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_file} not found.")
            self.config = {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {self.config_file}: {e}")
            self.config = {}

    def _build_structured_config(self):
        """New: Create structured config while preserving original flat config"""
        self._structured_config = {
            "core_config": {
                "base_model_name": self.get("core_config.base_model_name", "gpt2"),
                "scaffold_model_name": self.get("core_config.scaffold_model_name", "gpt2"),
                "cross_attn_layers": self.get("core_config.cross_attn_layers", [0, 1, 2]),
                "use_dynamic_layers": self.get("core_config.use_dynamic_layers", False),
                "layer_selection_mode": self.get("core_config.layer_selection_mode", "balanced"),
                "custom_layers": self.get("core_config.custom_layers", []),
                "valid_split_ratio": self.get("core_config.valid_split_ratio", 0.2),
                "random_seed": self.get("core_config.random_seed", 42),
                "quantization": self.get("core_config.quantization", "fp16"),
                "hidden_size": self.get("core_config.hidden_size", 768)
            },
            # Include all other sections similarly...
        }

    # Original methods (unchanged)
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

    def validate_keys(self, required_keys: List[str]):
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

    # New methods (additive, don't break existing code)
    def get_section(self, section: str) -> Dict[str, Any]:
        """New: Get entire configuration section as dict"""
        if self._structured_config:
            return self._structured_config.get(section, {})
        return {k.split('.')[1]: v for k, v in self.config.items() 
                if k.startswith(f"{section}.")}

    def update(self, key: str, value: Any) -> bool:
        """New: Update a configuration value"""
        keys = key.split('.')
        if len(keys) != 2:
            return False
            
        # Update both flat and structured config
        section, field = keys
        if section not in self.config:
            self.config[section] = {}
        self.config[section][field] = value
        
        if self._structured_config and section in self._structured_config:
            self._structured_config[section][field] = value
            
        return True

    def save_config(self, file_path: Optional[str] = None) -> bool:
        """New: Save current configuration to file"""
        save_path = file_path or self.config_file
        try:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save config: {str(e)}")
            return False

    # Preserve original example usage
    if __name__ == "__main__":
        config_manager = ConfigManager("sovl_config.json")
        try:
            config_manager.validate_keys(["core_config.base_model_name", "training_config.learning_rate"])
        except ValueError as e:
            print(e)
