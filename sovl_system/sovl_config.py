import json
import os
import gzip
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from threading import Lock
import traceback
import re
import time
from sovl_logger import Logger
from transformers import AutoConfig

@dataclass
class ConfigSchema:
    """Defines validation rules for configuration fields."""
    field: str
    type: type
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    range: Optional[tuple] = None
    required: bool = False
    nullable: bool = False

class _SchemaValidator:
    """Handles configuration schema validation logic."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.schemas: Dict[str, ConfigSchema] = {}

    def register(self, schemas: List[ConfigSchema]):
        """Register new schemas."""
        self.schemas.update({s.field: s for s in schemas})

    def validate(self, key: str, value: Any, conversation_id: str = "init") -> tuple[bool, Any]:
        """Validate a value against its schema."""
        schema = self.schemas.get(key)
        if not schema:
            self.logger.record({
                "error": f"Unknown configuration key: {key}",
                "timestamp": time.time(),
                "conversation_id": conversation_id
            })
            return False, None

        if value is None:
            if schema.required:
                self.logger.record({
                    "error": f"Required field {key} is missing",
                    "suggested": f"Set to default: {schema.default}",
                    "timestamp": time.time(),
                    "conversation_id": conversation_id
                })
                return False, schema.default
            if schema.nullable:
                return True, value
            return False, schema.default

        if not isinstance(value, schema.type):
            self.logger.record({
                "warning": f"Invalid type for {key}: expected {schema.type.__name__}, got {type(value).__name__}",
                "suggested": f"Set to default: {schema.default}",
                "timestamp": time.time(),
                "conversation_id": conversation_id
            })
            return False, schema.default

        if schema.validator and not schema.validator(value):
            valid_options = getattr(schema.validator, '__doc__', '') or str(schema.validator)
            self.logger.record({
                "warning": f"Invalid value for {key}: {value}",
                "suggested": f"Valid options: {valid_options}, default: {schema.default}",
                "timestamp": time.time(),
                "conversation_id": conversation_id
            })
            return False, schema.default

        if schema.range and not (schema.range[0] <= value <= schema.range[1]):
            self.logger.record({
                "warning": f"Value for {key} out of range {schema.range}: {value}",
                "suggested": f"Set to default: {schema.default}",
                "timestamp": time.time(),
                "conversation_id": conversation_id
            })
            return False, schema.default

        return True, value

class _ConfigStore:
    """Manages configuration storage and structure."""

    def __init__(self):
        self.flat_config: Dict[str, Any] = {}
        self.structured_config: Dict[str, Any] = {
            "core_config": {},
            "lora_config": {},
            "training_config": {"dry_run_params": {}},
            "curiosity_config": {},
            "cross_attn_config": {},
            "controls_config": {},
            "logging_config": {},
        }
        self.cache: Dict[str, Any] = {}

    def set_value(self, key: str, value: Any):
        """Set a value in flat and structured configs."""
        self.cache[key] = value
        keys = key.split('.')
        if len(keys) == 2:
            section, field = keys
            self.flat_config.setdefault(section, {})[field] = value
            self.structured_config[section][field] = value
        elif len(keys) == 3 and keys[0] == "training_config" and keys[1] == "dry_run_params":
            section, sub_section, field = keys
            self.flat_config.setdefault(section, {}).setdefault(sub_section, {})[field] = value
            self.structured_config[section][sub_section][field] = value

    def get_value(self, key: str, default: Any) -> Any:
        """Retrieve a value from the configuration."""
        if key in self.cache:
            return self.cache[key]
        keys = key.split('.')
        value = self.flat_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value != {} and value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self.structured_config.get(section, {})

    def rebuild_structured(self, schemas: List[ConfigSchema]):
        """Rebuild structured config from flat config."""
        for schema in schemas:
            keys = schema.field.split('.')
            section = keys[0]
            if len(keys) == 2:
                field = keys[1]
                self.structured_config[section][field] = self.get_value(schema.field, schema.default)
            elif len(keys) == 3 and section == "training_config" and keys[1] == "dry_run_params":
                field = keys[2]
                self.structured_config[section]["dry_run_params"][field] = self.get_value(schema.field, schema.default)

    def update_cache(self, schemas: List[ConfigSchema]):
        """Update cache with current config values."""
        self.cache = {schema.field: self.get_value(schema.field, schema.default) for schema in schemas}

class _FileHandler:
    """Handles configuration file operations."""

    def __init__(self, config_file: str, logger: Logger):
        self.config_file = config_file
        self.logger = logger

    def load(self, max_retries: int = 3) -> Dict[str, Any]:
        """Load configuration file with retry logic."""
        for attempt in range(max_retries):
            try:
                if not os.path.exists(self.config_file):
                    return {}
                if self.config_file.endswith('.gz'):
                    with gzip.open(self.config_file, 'rt', encoding='utf-8') as f:
                        return json.load(f)
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile) as e:
                self.logger.record({
                    "error": f"Attempt {attempt + 1} failed to load config {self.config_file}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": "init"
                })
                if attempt == max_retries - 1:
                    return {}
            time.sleep(0.1)
        return {}

    def save(self, config: Dict[str, Any], file_path: Optional[str] = None, compress: bool = False, max_retries: int = 3) -> bool:
        """Save configuration to file atomically."""
        save_path = file_path or self.config_file
        temp_file = f"{save_path}.tmp"
        for attempt in range(max_retries):
            try:
                if compress:
                    with gzip.open(temp_file, 'wt', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                else:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
                os.replace(temp_file, save_path)
                self.logger.record({
                    "event": "config_save",
                    "file_path": save_path,
                    "compressed": compress,
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                return True
            except Exception as e:
                self.logger.record({
                    "error": f"Attempt {attempt + 1} failed to save config to {save_path}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": "init"
                })
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if attempt == max_retries - 1:
                    return False
                time.sleep(0.1)
        return False

class ConfigManager:
    """Manages SOVLSystem configuration with validation, thread safety, and persistence."""

    DEFAULT_SCHEMA = [
        # core_config
        ConfigSchema("core_config.base_model_name", str, "gpt2", required=True),
        ConfigSchema("core_config.scaffold_model_name", str, "gpt2", required=True),
        ConfigSchema("core_config.cross_attn_layers", list, [5, 7], lambda x: all(isinstance(i, int) for i in x)),
        ConfigSchema("core_config.use_dynamic_layers", bool, False),
        ConfigSchema("core_config.layer_selection_mode", str, "balanced", lambda x: x in ["balanced", "random", "fixed"]),
        ConfigSchema("core_config.custom_layers", list, None, lambda x: x is None or all(isinstance(i, int) for i in x), nullable=True),
        ConfigSchema("core_config.valid_split_ratio", float, 0.2, range=(0.0, 1.0)),
        ConfigSchema("core_config.random_seed", int, 42, range=(0, 2**32)),
        ConfigSchema("core_config.quantization", str, "fp16", lambda x: x in ["fp16", "int8", "fp32"]),
        ConfigSchema("core_config.hidden_size", int, 768, range=(128, 4096)),
        # lora_config
        ConfigSchema("lora_config.lora_rank", int, 8, range=(1, 64)),
        ConfigSchema("lora_config.lora_alpha", int, 16, range=(1, 128)),
        ConfigSchema("lora_config.lora_dropout", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("lora_config.lora_target_modules", list, ["c_attn", "c_proj", "c_fc"], lambda x: all(isinstance(i, str) for i in x)),
        # training_config
        ConfigSchema("training_config.learning_rate", float, 0.0003, range=(0.0, 0.01)),
        ConfigSchema("training_config.train_epochs", int, 3, range=(1, 10)),
        ConfigSchema("training_config.batch_size", int, 1, range=(1, 64)),
        ConfigSchema("training_config.max_seq_length", int, 128, range=(64, 2048)),
        ConfigSchema("training_config.sigmoid_scale", float, 0.5, range=(0.1, 10.0)),
        ConfigSchema("training_config.sigmoid_shift", float, 5.0, range=(0.0, 10.0)),
        ConfigSchema("training_config.lifecycle_capacity_factor", float, 0.01, range=(0.0, 1.0)),
        ConfigSchema("training_config.lifecycle_curve", str, "sigmoid_linear", lambda x: x in ["sigmoid_linear", "linear", "exponential"]),
        ConfigSchema("training_config.accumulation_steps", int, 4, range=(1, 16)),
        ConfigSchema("training_config.exposure_gain_eager", int, 3, range=(1, 10)),
        ConfigSchema("training_config.exposure_gain_default", int, 2, range=(1, 10)),
        ConfigSchema("training_config.max_patience", int, 2, range=(1, 5)),
        ConfigSchema("training_config.sleep_max_steps", int, 100, range=(10, 1000)),
        ConfigSchema("training_config.lora_capacity", int, 0, range=(0, 1000)),
        ConfigSchema("training_config.dry_run", bool, False),
        ConfigSchema("training_config.dry_run_params.max_samples", int, 2, range=(1, 100)),
        ConfigSchema("training_config.dry_run_params.max_length", int, 128, range=(64, 2048)),
        ConfigSchema("training_config.dry_run_params.validate_architecture", bool, True),
        ConfigSchema("training_config.dry_run_params.skip_training", bool, True),
        # curiosity_config
        ConfigSchema("curiosity_config.queue_maxlen", int, 10, range=(1, 50)),
        ConfigSchema("curiosity_config.novelty_history_maxlen", int, 20, range=(5, 100)),
        ConfigSchema("curiosity_config.decay_rate", float, 0.9, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.attention_weight", float, 0.5, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.question_timeout", float, 3600.0, range=(60.0, 86400.0)),
        ConfigSchema("curiosity_config.novelty_threshold_spontaneous", float, 0.9, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.novelty_threshold_response", float, 0.8, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.pressure_threshold", float, 0.7, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.pressure_drop", float, 0.3, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.silence_threshold", float, 20.0, range=(0.0, 3600.0)),
        ConfigSchema("curiosity_config.question_cooldown", float, 60.0, range=(0.0, 3600.0)),
        ConfigSchema("curiosity_config.weight_ignorance", float, 0.7, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.weight_novelty", float, 0.3, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.max_new_tokens", int, 8, range=(1, 100)),
        ConfigSchema("curiosity_config.base_temperature", float, 1.1, range=(0.1, 2.0)),
        ConfigSchema("curiosity_config.temperament_influence", float, 0.4, range=(0.0, 1.0)),
        ConfigSchema("curiosity_config.top_k", int, 30, range=(1, 100)),
        ConfigSchema("curiosity_config.enable_curiosity", bool, True),
        # cross_attn_config
        ConfigSchema("cross_attn_config.memory_weight", float, 0.5, range=(0.0, 1.0)),
        ConfigSchema("cross_attn_config.dynamic_scale", float, 0.3, range=(0.0, 1.0)),
        ConfigSchema("cross_attn_config.enable_dynamic", bool, True),
        ConfigSchema("cross_attn_config.enable_memory", bool, True),
        # controls_config
        ConfigSchema("controls_config.sleep_conf_threshold", float, 0.7, range=(0.0, 1.0)),
        ConfigSchema("controls_config.sleep_time_factor", float, 1.0, range=(0.1, 10.0)),
        ConfigSchema("controls_config.sleep_log_min", int, 10, range=(1, 100)),
        ConfigSchema("controls_config.dream_swing_var", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("controls_config.dream_lifecycle_delta", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("controls_config.dream_temperament_on", bool, True),
        ConfigSchema("controls_config.dream_noise_scale", float, 0.05, range=(0.0, 0.1)),
        ConfigSchema("controls_config.temp_eager_threshold", float, 0.8, range=(0.7, 0.9)),
        ConfigSchema("controls_config.temp_sluggish_threshold", float, 0.6, range=(0.4, 0.6)),
        ConfigSchema("controls_config.temp_mood_influence", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("controls_config.scaffold_weight_cap", float, 0.9, range=(0.0, 1.0)),
        ConfigSchema("controls_config.base_temperature", float, 0.7, range=(0.1, 2.0)),
        ConfigSchema("controls_config.save_path_prefix", str, "state", lambda x: bool(re.match(r'^[a-zA-Z0-9_/.-]+$', x))),
        ConfigSchema("controls_config.dream_memory_weight", float, 0.1, range=(0.0, 1.0)),
        ConfigSchema("controls_config.dream_memory_maxlen", int, 10, range=(1, 50)),
        ConfigSchema("controls_config.dream_prompt_weight", float, 0.5, range=(0.0, 1.0)),
        ConfigSchema("controls_config.dream_novelty_boost", float, 0.03, range=(0.0, 0.1)),
        ConfigSchema("controls_config.temp_curiosity_boost", float, 0.5, range=(0.0, 0.5)),
        ConfigSchema("controls_config.temp_restless_drop", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("controls_config.temp_melancholy_noise", float, 0.02, range=(0.0, 0.05)),
        ConfigSchema("controls_config.conf_feedback_strength", float, 0.5, range=(0.0, 1.0)),
        ConfigSchema("controls_config.temp_smoothing_factor", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("controls_config.dream_memory_decay", float, 0.95, range=(0.0, 1.0)),
        ConfigSchema("controls_config.dream_prune_threshold", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("controls_config.use_scaffold_memory", bool, True),
        ConfigSchema("controls_config.use_token_map_memory", bool, True),
        ConfigSchema("controls_config.memory_decay_rate", float, 0.95, range=(0.0, 1.0)),
        ConfigSchema("controls_config.dynamic_cross_attn_mode", str, None, lambda x: x is None or x in ["adaptive", "fixed"], nullable=True),
        ConfigSchema("controls_config.has_woken", bool, False),
        ConfigSchema("controls_config.is_sleeping", bool, False),
        ConfigSchema("controls_config.confidence_history_maxlen", int, 5, range=(3, 10)),
        ConfigSchema("controls_config.temperament_history_maxlen", int, 5, range=(3, 10)),
        ConfigSchema("controls_config.conversation_history_maxlen", int, 10, range=(5, 50)),
        ConfigSchema("controls_config.max_seen_prompts", int, 1000, range=(100, 10000)),
        ConfigSchema("controls_config.prompt_timeout", float, 86400.0, range=(3600.0, 604800.0)),
        ConfigSchema("controls_config.temperament_decay_rate", float, 0.95, range=(0.0, 1.0)),
        ConfigSchema("controls_config.scaffold_unk_id", int, 0, range=(0, 100000)),
        ConfigSchema("controls_config.enable_dreaming", bool, True),
        ConfigSchema("controls_config.enable_temperament", bool, True),
        ConfigSchema("controls_config.enable_confidence_tracking", bool, True),
        ConfigSchema("controls_config.enable_gestation", bool, True),
        ConfigSchema("controls_config.enable_sleep_training", bool, True),
        ConfigSchema("controls_config.enable_cross_attention", bool, True),
        ConfigSchema("controls_config.enable_dynamic_cross_attention", bool, True),
        ConfigSchema("controls_config.enable_lora_adapters", bool, True),
        ConfigSchema("controls_config.enable_repetition_check", bool, True),
        ConfigSchema("controls_config.enable_prompt_driven_dreams", bool, True),
        ConfigSchema("controls_config.enable_lifecycle_weighting", bool, True),
        ConfigSchema("controls_config.memory_threshold", float, 0.85, range=(0.0, 1.0)),
        ConfigSchema("controls_config.enable_error_listening", bool, True),
        # logging_config
        ConfigSchema("logging_config.log_file", str, "sovl_logs.jsonl", lambda x: x.endswith(".jsonl")),
        ConfigSchema("logging_config.max_size_mb", int, 10, range=(0, 100)),
        ConfigSchema("logging_config.compress_old", bool, False),
        ConfigSchema("logging_config.max_in_memory_logs", int, 1000, range=(100, 10000)),
        ConfigSchema("logging_config.schema_version", str, "1.1"),
    ]

    def __init__(self, config_file: str, logger: Logger):
        """
        Initialize ConfigManager with configuration file path and logger.

        Args:
            config_file: Path to configuration file
            logger: Logger instance for recording events
        """
        self.config_file = os.getenv("SOVL_CONFIG_FILE", config_file)
        self.logger = logger
        self.store = _ConfigStore()
        self.validator = _SchemaValidator(logger)
        self.file_handler = _FileHandler(self.config_file, logger)
        self.lock = Lock()
        self._frozen = False
        self._last_config_hash = ""
        self.validator.register(self.DEFAULT_SCHEMA)
        self._initialize_config()

    def _initialize_config(self):
        """Initialize configuration by loading and validating."""
        with self.lock:
            self.store.flat_config = self.file_handler.load()
            self._validate_and_set_defaults()
            self.store.rebuild_structured(self.DEFAULT_SCHEMA)
            self.store.update_cache(self.DEFAULT_SCHEMA)
            self._last_config_hash = self._compute_config_hash()
            self.logger.record({
                "event": "config_load",
                "config_file": self.config_file,
                "config_hash": self._last_config_hash,
                "timestamp": time.time(),
                "conversation_id": "init"
            })

    def _compute_config_hash(self) -> str:
        """Compute a hash of the current config for change tracking."""
        try:
            config_str = json.dumps(self.store.flat_config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.record({
                "error": f"Config hash computation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            return ""

    def _validate_and_set_defaults(self):
        """Validate entire configuration and set defaults where needed."""
        for schema in self.DEFAULT_SCHEMA:
            value = self.store.get_value(schema.field, schema.default)
            is_valid, corrected_value = self.validator.validate(schema.field, value)
            if not is_valid:
                self.store.set_value(schema.field, corrected_value)

    def freeze(self):
        """Prevent further updates to the configuration."""
        with self.lock:
            self._frozen = True
            self.logger.record({
                "event": "config_frozen",
                "timestamp": time.time(),
                "conversation_id": "init"
            })

    def unfreeze(self):
        """Allow updates to the configuration."""
        with self.lock:
            self._frozen = False
            self.logger.record({
                "event": "config_unfrozen",
                "timestamp": time.time(),
                "conversation_id": "init"
            })

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the configuration using a dot-separated key.

        Args:
            key: Dot-separated configuration key
            default: Default value if key is missing

        Returns:
            Configuration value or default
        """
        with self.lock:
            value = self.store.get_value(key, default)
            if value == {} or value is None:
                self.logger.record({
                    "warning": f"Key '{key}' is empty or missing. Using default: {default}",
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                return default
            return value

    def validate_keys(self, required_keys: List[str]):
        """
        Validate that all required keys exist in the configuration.

        Args:
            required_keys: List of required configuration keys

        Raises:
            ValueError: If any required keys are missing
        """
        with self.lock:
            missing_keys = [key for key in required_keys if self.get(key, None) is None]
            if missing_keys:
                self.logger.record({
                    "error": f"Missing required configuration keys: {', '.join(missing_keys)}",
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section as dict.

        Args:
            section: Configuration section name

        Returns:
            Dictionary of section key-value pairs
        """
        with self.lock:
            return self.store.get_section(section)

    def update(self, key: str, value: Any) -> bool:
        """
        Update a configuration value with validation.

        Args:
            key: Dot-separated configuration key
            value: New value

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            with self.lock:
                if self._frozen:
                    self.logger.record({
                        "error": f"Cannot update {key}: configuration is frozen",
                        "timestamp": time.time(),
                        "conversation_id": "init"
                    })
                    return False

                is_valid, corrected_value = self.validator.validate(key, value)
                if not is_valid:
                    return False

                old_hash = self._last_config_hash
                self.store.set_value(key, value)
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "config_update",
                    "key": key,
                    "value": value,
                    "old_hash": old_hash,
                    "new_hash": self._last_config_hash,
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update config key {key}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            return False

    def update_batch(self, updates: Dict[str, Any], rollback_on_failure: bool = True) -> bool:
        """
        Update multiple configuration values atomically with rollback option.

        Args:
            updates: Dictionary of key-value pairs to update
            rollback_on_failure: Revert changes if any update fails

        Returns:
            True if all updates succeeded, False otherwise
        """
        try:
            with self.lock:
                if self._frozen:
                    self.logger.record({
                        "error": "Cannot update batch: configuration is frozen",
                        "timestamp": time.time(),
                        "conversation_id": "init"
                    })
                    return False

                original_config = self.store.flat_config.copy()
                for key, value in updates.items():
                    if not self.update(key, value):
                        if rollback_on_failure:
                            self.store.flat_config = original_config
                            self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                            self.store.update_cache(self.DEFAULT_SCHEMA)
                            self._last_config_hash = self._compute_config_hash()
                            self.logger.record({
                                "event": "config_rollback",
                                "reason": f"Failed to update {key}",
                                "timestamp": time.time(),
                                "conversation_id": "init"
                            })
                        return False
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update batch config: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            if rollback_on_failure:
                self.store.flat_config = original_config
                self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                self.store.update_cache(self.DEFAULT_SCHEMA)
                self._last_config_hash = self._compute_config_hash()
            return False

    def save_config(self, file_path: Optional[str] = None, compress: bool = False, max_retries: int = 3) -> bool:
        """
        Save current configuration to file atomically.

        Args:
            file_path: Optional path to save config (defaults to config_file)
            compress: Save as gzip-compressed file
            max_retries: Number of retry attempts for I/O operations

        Returns:
            True if save succeeded, False otherwise
        """
        with self.lock:
            return self.file_handler.save(self.store.flat_config, file_path, compress, max_retries)

    def diff_config(self, old_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Compare current config with an old config and return differences.

        Args:
            old_config: Previous configuration dictionary

        Returns:
            Dictionary of changed keys with old and new values
        """
        try:
            with self.lock:
                diff = {}
                for key in self.store.flat_config:
                    old_value = old_config.get(key)
                    new_value = self.store.flat_config.get(key)
                    if old_value != new_value:
                        diff[key] = {"old": old_value, "new": new_value}
                for key in old_config:
                    if key not in self.store.flat_config:
                        diff[key] = {"old": old_config[key], "new": None}
                self.logger.record({
                    "event": "config_diff",
                    "changed_keys": list(diff.keys()),
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                return diff
        except Exception as e:
            self.logger.record({
                "error": f"Config diff failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            return {}

    def register_schema(self, schemas: List[ConfigSchema]):
        """
        Dynamically extend the configuration schema.

        Args:
            schemas: List of ConfigSchema objects to add
        """
        try:
            with self.lock:
                if self._frozen:
                    self.logger.record({
                        "error": "Cannot register schema: configuration is frozen",
                        "timestamp": time.time(),
                        "conversation_id": "init"
                    })
                    return
                self.validator.register(schemas)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA + schemas)
                self.store.update_cache(self.DEFAULT_SCHEMA + schemas)
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "schema_registered",
                    "new_fields": [s.field for s in schemas],
                    "config_hash": self._last_config_hash,
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to register schema: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })

    def get_state(self) -> Dict[str, Any]:
        """
        Export current configuration state.

        Returns:
            Dictionary containing config state
        """
        with self.lock:
            return {
                "config_file": self.config_file,
                "config": self.store.flat_config,
                "frozen": self._frozen,
                "config_hash": self._last_config_hash
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load configuration state.

        Args:
            state: Dictionary containing config state
        """
        try:
            with self.lock:
                self.config_file = state.get("config_file", self.config_file)
                self.store.flat_config = state.get("config", {})
                self._frozen = state.get("frozen", False)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                self.store.update_cache(self.DEFAULT_SCHEMA)
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "config_load_state",
                    "config_file": self.config_file,
                    "config_hash": self._last_config_hash,
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load config state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            raise

    def tune(self, **kwargs) -> bool:
        """
        Dynamically tune configuration parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            True if tuning succeeded, False otherwise
        """
        return self.update_batch(kwargs)

    def load_profile(self, profile: str) -> bool:
        """
        Load a configuration profile from a file.

        Args:
            profile: Profile name (e.g., 'development', 'production')

        Returns:
            True if profile loaded successfully, False otherwise
        """
        profile_file = f"{os.path.splitext(self.config_file)[0]}_{profile}.json"
        try:
            with self.lock:
                config = self.file_handler.load()
                if not config:
                    self.logger.record({
                        "error": f"Profile file {profile_file} not found",
                        "timestamp": time.time(),
                        "conversation_id": "init"
                    })
                    return False
                self.store.flat_config = config
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                self.store.update_cache(self.DEFAULT_SCHEMA)
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "profile_load",
                    "profile": profile,
                    "config_file": profile_file,
                    "config_hash": self._last_config_hash,
                    "timestamp": time.time(),
                    "conversation_id": "init"
                })
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load profile {profile}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "init"
            })
            return False

    def set_global_blend(self, weight_cap: Optional[float] = None, base_temp: Optional[float] = None) -> bool:
        """
        Set global blend parameters for the system.

        Args:
            weight_cap: Scaffold weight cap (0.5 to 1.0)
            base_temp: Base temperature (0.5 to 1.5)

        Returns:
            True if update succeeded, False otherwise
        """
        updates = {}
        prefix = "controls_config."
        
        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            updates[f"{prefix}scaffold_weight_cap"] = weight_cap
            
        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            updates[f"{prefix}base_temperature"] = base_temp
            
        if updates:
            return self.update_batch(updates)
        return True

    def validate_section(self, section: str, required_keys: List[str]) -> bool:
        """
        Validate a configuration section and its required keys.

        Args:
            section: Configuration section name
            required_keys: List of required keys in the section

        Returns:
            True if section is valid, False otherwise
        """
        try:
            with self.lock:
                # Check if section exists
                if section not in self.store.structured_config:
                    self.logger.record({
                        "error": f"Configuration section '{section}' not found",
                        "timestamp": time.time(),
                        "conversation_id": "validate"
                    })
                    return False

                # Check for required keys
                missing_keys = [key for key in required_keys if key not in self.store.structured_config[section]]
                if missing_keys:
                    self.logger.record({
                        "error": f"Missing required keys in section '{section}': {', '.join(missing_keys)}",
                        "timestamp": time.time(),
                        "conversation_id": "validate"
                    })
                    return False

                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to validate section '{section}': {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "validate"
            })
            return False

    def tune_parameter(self, section: str, key: str, value: Any, min_value: Any = None, max_value: Any = None) -> bool:
        """
        Tune a configuration parameter with validation.

        Args:
            section: Configuration section name
            key: Parameter key
            value: New parameter value
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            True if parameter was updated successfully, False otherwise
        """
        try:
            with self.lock:
                # Validate value range if min/max provided
                if min_value is not None and value < min_value:
                    self.logger.record({
                        "error": f"Value {value} below minimum {min_value} for {section}.{key}",
                        "timestamp": time.time(),
                        "conversation_id": "tune"
                    })
                    return False
                    
                if max_value is not None and value > max_value:
                    self.logger.record({
                        "error": f"Value {value} above maximum {max_value} for {section}.{key}",
                        "timestamp": time.time(),
                        "conversation_id": "tune"
                    })
                    return False

                # Update the parameter
                success = self.update(section, key, value)
                if success:
                    self.logger.record({
                        "info": f"Tuned {section}.{key} to {value}",
                        "timestamp": time.time(),
                        "conversation_id": "tune"
                    })
                return success
        except Exception as e:
            self.logger.record({
                "error": f"Failed to tune {section}.{key}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "tune"
            })
            return False

    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update a configuration section with new values.

        Args:
            section: Configuration section name
            updates: Dictionary of key-value pairs to update

        Returns:
            True if update successful, False otherwise
        """
        try:
            with self.lock:
                # Check if section exists
                if section not in self.store.structured_config:
                    self.logger.record({
                        "error": f"Configuration section '{section}' not found",
                        "timestamp": time.time(),
                        "conversation_id": "update"
                    })
                    return False

                # Update values
                for key, value in updates.items():
                    if key in self.store.structured_config[section]:
                        self.store.structured_config[section][key] = value
                        self.logger.record({
                            "action": f"Updated {section}.{key}",
                            "new_value": str(value),
                            "timestamp": time.time(),
                            "conversation_id": "update"
                        })

                # Save changes
                self.save_config()
                return True

        except Exception as e:
            self.logger.record({
                "error": f"Failed to update section '{section}': {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "update"
            })
            return False

    def validate_or_raise(self, model_config: Optional[Any] = None) -> None:
        """
        Validate the entire configuration and raise a ValueError with detailed error messages if validation fails.
        
        Args:
            model_config: Optional model configuration for layer validation
            
        Raises:
            ValueError: If configuration validation fails, with detailed error messages
        """
        try:
            # Validate required keys
            self.validate_keys([
                "core_config.base_model_name",
                "core_config.scaffold_model_name",
                "training_config.learning_rate",
                "curiosity_config.enable_curiosity",
                "cross_attn_config.memory_weight"
            ])

            # Validate cross-attention layers
            cross_attn_layers = self.get("core_config.cross_attn_layers", [5, 7])
            if not isinstance(cross_attn_layers, list):
                raise ValueError("core_config.cross_attn_layers must be a list")

            # Validate layer indices if not using dynamic layers
            if not self.get("core_config.use_dynamic_layers", False) and model_config is not None:
                base_model_name = self.get("core_config.base_model_name", "gpt2")
                try:
                    base_config = model_config or AutoConfig.from_pretrained(base_model_name)
                    invalid_layers = [l for l in cross_attn_layers if not (0 <= l < base_config.num_hidden_layers)]
                    if invalid_layers:
                        raise ValueError(f"Invalid cross_attn_layers: {invalid_layers} for {base_config.num_hidden_layers} layers")
                except Exception as e:
                    raise ValueError(f"Failed to validate cross-attention layers: {str(e)}")

            # Validate custom layers if using custom layer selection
            if self.get("core_config.layer_selection_mode", "balanced") == "custom":
                custom_layers = self.get("core_config.custom_layers", [])
                if not isinstance(custom_layers, list):
                    raise ValueError("core_config.custom_layers must be a list")

                if model_config is not None:
                    try:
                        base_model_name = self.get("core_config.base_model_name", "gpt2")
                        base_config = model_config or AutoConfig.from_pretrained(base_model_name)
                        invalid_custom = [l for l in custom_layers if not (0 <= l < base_config.num_hidden_layers)]
                        if invalid_custom:
                            raise ValueError(f"Invalid custom_layers: {invalid_custom} for {base_model_name}")
                    except Exception as e:
                        raise ValueError(f"Failed to validate custom layers: {str(e)}")

            # Log successful validation
            self.logger.record({
                "event": "config_validation",
                "status": "success",
                "timestamp": time.time(),
                "conversation_id": "validate",
                "config_snapshot": self.get_state()["config"]
            })

        except Exception as e:
            self.logger.record({
                "error": f"Configuration validation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "validate"
            })
            raise ValueError(f"Configuration validation failed: {str(e)}")

if __name__ == "__main__":
    from sovl_logger import LoggerConfig
    logger = Logger(LoggerConfig())
    config_manager = ConfigManager("sovl_config.json", logger)
    try:
        config_manager.validate_keys(["core_config.base_model_name", "curiosity_config.attention_weight"])
    except ValueError as e:
        print(e)
