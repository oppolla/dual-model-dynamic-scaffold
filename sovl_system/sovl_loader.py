from sovl_config import ConfigManager
from sovl_logger import Logger
import time

class ConfigLoader:
    """Handles loading and organizing SOVL system configurations."""
    
    def __init__(self, config_file="sovl_config.json"):
        # Initialize logger for config-related events
        self.logger = Logger(
            log_file="sovl_config_logs.jsonl",
            max_size_mb=10,
            compress_old=True
        )
        # Initialize ConfigManager
        self.config_manager = ConfigManager(config_file, self.logger)
        # Load and organize configurations
        self.configs = self._load_configs()
    
    def _load_configs(self):
        """Load and organize configurations into sections."""
        configs = {
            "core": self.config_manager.get_section("core_config"),
            "lora": self.config_manager.get_section("lora_config"),
            "training": self.config_manager.get_section("training_config"),
            "curiosity": self.config_manager.get_section("curiosity_config"),
            "cross_attn": self.config_manager.get_section("cross_attn_config"),
            "controls": self.config_manager.get_section("controls_config"),
            "logging": self.config_manager.get_section("logging_config")
        }
        
        # Log successful config loading
        self.logger.record({
            "event": "config_loaded",
            "config_summary": {
                k: list(v.keys()) for k, v in configs.items()
            },
            "timestamp": time.time(),
            "conversation_id": "init"
        })
        
        return configs
    
    def get_config(self, section):
        """Get configuration for a specific section."""
        return self.configs.get(section, {})
    
    def get_all_configs(self):
        """Get all configurations."""
        return self.configs
    
    def get_config_manager(self):
        """Get the underlying ConfigManager instance."""
        return self.config_manager
