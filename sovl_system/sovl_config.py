import json
import os
import gzip
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from threading import Lock
import traceback
import re
import time
from sovl_logger import Logger

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

class ConfigManager:
    """Manages SOVLSystem configuration with validation, thread safety, and persistence."""
    
    SCHEMA = [
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
        self.config = {}  # Flat dictionary
        self._structured_config = {}  # Structured dictionary
        self._cache = {}  # Cache for frequent access
        self.lock = Lock()
        self._frozen = False
        self._last_config_hash = None
        self._load_config()

    def _load_config(self, max_retries: int = 3):
        """Load and validate configuration file with retry logic."""
        for attempt in range(max_retries):
            try:
                with self.lock:
                    if os.path.exists(self.config_file):
                        if self.config_file.endswith('.gz'):
                            with gzip.open(self.config_file, 'rt', encoding='utf-8') as f:
                                self.config = json.load(f)
                        else:
                            with open(self.config_file, 'r', encoding='utf-8') as f:
                                self.config = json.load(f)
                    else:
                        self.config = {}
                    self._validate_config()
                    self._build_structured_config()
                    self._update_cache()
                    self._last_config_hash = self._compute_config_hash()
                    self.logger.record({
                        "event": "config_load",
                        "config_file": self.config_file,
                        "config_hash": self._last_config_hash,
                        "timestamp": time.time()
                    })
                    return
            except (FileNotFoundError, json.JSONDecodeError, gzip.BadGzipFile) as e:
                self.logger.record({
                    "error": f"Attempt {attempt + 1} failed to load config {self.config_file}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                if attempt == max_retries - 1:
                    self.config = {}
                    self._validate_config()
                    self._build_structured_config()
                    self._update_cache()
                    self.logger.record({
                        "warning": f"Using empty config after {max_retries} failed attempts",
                        "timestamp": time.time()
                    })
            except Exception as e:
                self.logger.record({
                    "error": f"Fatal error loading config: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                raise
            time.sleep(0.1)

    def _compute_config_hash(self) -> str:
        """Compute a hash of the current config for change tracking."""
        try:
            config_str = json.dumps(self.config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.record({
                "error": f"Config hash computation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return ""

    def _validate_config(self):
        """Validate configuration against schema with detailed feedback."""
        for schema in self.SCHEMA:
            value = self.get(schema.field, schema.default)
            if value is None and schema.required:
                self.logger.record({
                    "error": f"Required field {schema.field} is missing",
                    "suggested": f"Set to default: {schema.default}",
                    "timestamp": time.time()
                })
                self._set_value(schema.field, schema.default)
                continue
            if value is None and schema.nullable:
                continue
            if value is not None:
                if not isinstance(value, schema.type):
                    self.logger.record({
                        "warning": f"Invalid type for {schema.field}: expected {schema.type.__name__}, got {type(value).__name__}",
                        "suggested": f"Set to default: {schema.default}",
                        "timestamp": time.time()
                    })
                    self._set_value(schema.field, schema.default)
                    continue
                if schema.validator and not schema.validator(value):
                    valid_options = getattr(schema.validator, '__doc__', '') or str(schema.validator)
                    self.logger.record({
                        "warning": f"Invalid value for {schema.field}: {value}",
                        "suggested": f"Valid options: {valid_options}, default: {schema.default}",
                        "timestamp": time.time()
                    })
                    self._set_value(schema.field, schema.default)
                    continue
                if schema.range:
                    min_val, max_val = schema.range
                    if not (min_val <= value <= max_val):
                        self.logger.record({
                            "warning": f"Value for {schema.field} out of range [{min_val}, {max_val}]: {value}",
                            "suggested": f"Set to default: {schema.default}",
                            "timestamp": time.time()
                        })
                        self._set_value(schema.field, schema.default)

    def _build_structured_config(self):
        """Build structured config from flat config."""
        self._structured_config = {
            "core_config": {},
            "lora_config": {},
            "training_config": {"dry_run_params": {}},
            "curiosity_config": {},
            "cross_attn_config": {},
            "controls_config": {},
            "logging_config": {},
        }
        for schema in self.SCHEMA:
            keys = schema.field.split('.')
            section = keys[0]
            if len(keys) == 2:
                field = keys[1]
                self._structured_config[section][field] = self.get(schema.field, schema.default)
            elif len(keys) == 3 and section == "training_config" and keys[1] == "dry_run_params":
                field = keys[2]
                self._structured_config[section]["dry_run_params"][field] = self.get(schema.field, schema.default)

    def _update_cache(self):
        """Update cache with current config values."""
        self._cache = {schema.field: self.get(schema.field, schema.default) for schema in self.SCHEMA}

    def _set_value(self, key: str, value: Any):
        """Set value in flat and structured configs."""
        keys = key.split('.')
        if len(keys) == 2:
            section, field = keys
            if section not in self.config:
                self.config[section] = {}
            self.config[section][field] = value
            if section in self._structured_config:
                self._structured_config[section][field] = value
        elif len(keys) == 3 and keys[0] == "training_config" and keys[1] == "dry_run_params":
            section, sub_section, field = keys
            if section not in self.config:
                self.config[section] = {}
            if sub_section not in self.config[section]:
                self.config[section][sub_section] = {}
            self.config[section][sub_section][field] = value
            if section in self._structured_config and sub_section in self._structured_config[section]:
                self._structured_config[section][sub_section][field] = value
        self._cache[key] = value

    def freeze(self):
        """Prevent further updates to the configuration."""
        with self.lock:
            self._frozen = True
            self.logger.record({
                "event": "config_frozen",
                "timestamp": time.time()
            })

    def unfreeze(self):
        """Allow updates to the configuration."""
        with self.lock:
            self._frozen = False
            self.logger.record({
                "event": "config_unfrozen",
                "timestamp": time.time()
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
            if key in self._cache:
                return self._cache[key]
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    return default
            if value == {} or value is None:
                self.logger.record({
                    "warning": f"Key '{key}' is empty or missing. Using default: {default}",
                    "timestamp": time.time()
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
                    "timestamp": time.time()
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
            return self._structured_config.get(section, {})

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
                        "timestamp": time.time()
                    })
                    return False
                for schema in self.SCHEMA:
                    if schema.field == key:
                        if value is None and not schema.nullable:
                            self.logger.record({
                                "error": f"Null value not allowed for {key}",
                                "timestamp": time.time()
                            })
                            return False
                        if value is not None and not isinstance(value, schema.type):
                            self.logger.record({
                                "error": f"Invalid type for {key}: expected {schema.type.__name__}, got {type(value).__name__}",
                                "timestamp": time.time()
                            })
                            return False
                        if schema.validator and value is not None and not schema.validator(value):
                            valid_options = getattr(schema.validator, '__doc__', '') or str(schema.validator)
                            self.logger.record({
                                "error": f"Invalid value for {key}: {value}",
                                "suggested": f"Valid options: {valid_options}",
                                "timestamp": time.time()
                            })
                            return False
                        if schema.range and value is not None and not (schema.range[0] <= value <= schema.range[1]):
                            self.logger.record({
                                "error": f"Value for {key} out of range {schema.range}: {value}",
                                "timestamp": time.time()
                            })
                            return False
                        break
                else:
                    self.logger.record({
                        "error": f"Unknown configuration key: {key}",
                        "timestamp": time.time()
                    })
                    return False

                self._set_value(key, value)
                new_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "config_update",
                    "key": key,
                    "value": value,
                    "old_hash": self._last_config_hash,
                    "new_hash": new_hash,
                    "timestamp": time.time()
                })
                self._last_config_hash = new_hash
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update config key {key}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
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
                        "timestamp": time.time()
                    })
                    return False
                if rollback_on_failure:
                    original_config = self.config.copy()
                for key, value in updates.items():
                    if not self.update(key, value):
                        if rollback_on_failure:
                            self.config = original_config
                            self._build_structured_config()
                            self._update_cache()
                            self._last_config_hash = self._compute_config_hash()
                            self.logger.record({
                                "event": "config_rollback",
                                "reason": f"Failed to update {key}",
                                "timestamp": time.time()
                            })
                        return False
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update batch config: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            if rollback_on_failure:
                self.config = original_config
                self._build_structured_config()
                self._update_cache()
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
        save_path = file_path or self.config_file
        temp_file = f"{save_path}.tmp"
        for attempt in range(max_retries):
            try:
                with self.lock:
                    if compress:
                        with gzip.open(temp_file, 'wt', encoding='utf-8') as f:
                            json.dump(self.config, f, indent=2)
                    else:
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(self.config, f, indent=2)
                    os.replace(temp_file, save_path)
                    self.logger.record({
                        "event": "config_save",
                        "file_path": save_path,
                        "compressed": compress,
                        "config_hash": self._last_config_hash,
                        "timestamp": time.time()
                    })
                    return True
            except Exception as e:
                self.logger.record({
                    "error": f"Attempt {attempt + 1} failed to save config to {save_path}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if attempt == max_retries - 1:
                    return False
                time.sleep(0.1)
        return False

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
                for key in self.config:
                    old_value = old_config.get(key)
                    new_value = self.config.get(key)
                    if old_value != new_value:
                        diff[key] = {"old": old_value, "new": new_value}
                for key in old_config:
                    if key not in self.config:
                        diff[key] = {"old": old_config[key], "new": None}
                self.logger.record({
                    "event": "config_diff",
                    "changed_keys": list(diff.keys()),
                    "timestamp": time.time()
                })
                return diff
        except Exception as e:
            self.logger.record({
                "error": f"Config diff failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
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
                        "timestamp": time.time()
                    })
                    return
                self.SCHEMA.extend(schemas)
                self._validate_config()
                self._build_structured_config()
                self._update_cache()
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "schema_registered",
                    "new_fields": [s.field for s in schemas],
                    "config_hash": self._last_config_hash,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to register schema: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
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
                "config": self.config,
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
                self.config = state.get("config", {})
                self._frozen = state.get("frozen", False)
                self._validate_config()
                self._build_structured_config()
                self._update_cache()
                self._last_config_hash = self._compute_config_hash()
                self.logger.record({
                    "event": "config_load_state",
                    "config_file": self.config_file,
                    "config_hash": self._last_config_hash,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load config state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
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
                if os.path.exists(profile_file):
                    if profile_file.endswith('.gz'):
                        with gzip.open(profile_file, 'rt', encoding='utf-8') as f:
                            self.config = json.load(f)
                    else:
                        with open(profile_file, 'r', encoding='utf-8') as f:
                            self.config = json.load(f)
                    self._validate_config()
                    self._build_structured_config()
                    self._update_cache()
                    self._last_config_hash = self._compute_config_hash()
                    self.logger.record({
                        "event": "profile_load",
                        "profile": profile,
                        "config_file": profile_file,
                        "config_hash": self._last_config_hash,
                        "timestamp": time.time()
                    })
                    return True
                else:
                    self.logger.record({
                        "error": f"Profile file {profile_file} not found",
                        "timestamp": time.time()
                    })
                    return False
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load profile {profile}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    if __name__ == "__main__":
        from sovl_logger import LoggerConfig
        logger = Logger(LoggerConfig())
        config_manager = ConfigManager("sovl_config.json", logger)
        try:
            config_manager.validate_keys(["core_config.base_model_name", "curiosity_config.attention_weight"])
        except ValueError as e:
            print(e)
