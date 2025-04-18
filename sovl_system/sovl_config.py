import json
import os
import gzip
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable, Tuple, NamedTuple
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

class SchemaValidator:
    """Handles configuration schema validation logic."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.schemas: Dict[str, ConfigSchema] = {}

    def register(self, schemas: List[ConfigSchema]) -> None:
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

class ConfigStore:
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

    def set_value(self, key: str, value: Any) -> None:
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

    def rebuild_structured(self, schemas: List[ConfigSchema]) -> None:
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

    def update_cache(self, schemas: List[ConfigSchema]) -> None:
        """Update cache with current config values."""
        self.cache = {schema.field: self.get_value(schema.field, schema.default) for schema in schemas}

class FileHandler:
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

class ConfigKey(NamedTuple):
    """Represents a configuration key with section and field."""
    section: str
    field: str

    def __str__(self) -> str:
        return f"{self.section}.{self.field}"

class ConfigKeys:
    """Type-safe configuration keys."""
    # Processor Config
    PROCESSOR_MIN_REP_LENGTH = ConfigKey("processor_config", "min_rep_length")
    
    # Controls Config
    CONTROLS_MEMORY_THRESHOLD = ConfigKey("controls_config", "memory_threshold")
    
    # Add more keys as needed...

class ConfigManager:
    """Manages SOVLSystem configuration with validation, thread safety, and persistence.
    
    Example usage:
        # Old way (string literals)
        min_rep_length = config_manager.get("processor_config.min_rep_length")
        
        # New way (type-safe)
        from sovl_config import ConfigKeys
        min_rep_length = config_manager.get(ConfigKeys.PROCESSOR_MIN_REP_LENGTH)
    """

    DEFAULT_SCHEMA = [
        # core_config
        ConfigSchema("core_config.base_model_name", str, "SmolLM2-360M", required=True),
        ConfigSchema("core_config.scaffold_model_name", str, "SmolLM2-135M", required=True),
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
        ConfigSchema("training_config.weight_decay", float, 0.01, range=(0.0, 0.1)),
        ConfigSchema("training_config.total_steps", int, 1000, range=(100, 10000)),
        ConfigSchema("training_config.max_grad_norm", float, 1.0, range=(0.1, 10.0)),
        ConfigSchema("training_config.use_amp", bool, True),
        ConfigSchema("training_config.checkpoint_interval", int, 1000, range=(100, 10000)),
        ConfigSchema("training_config.scheduler_type", str, "linear", lambda x: x in ["linear", "cosine", "constant"]),
        ConfigSchema("training_config.cosine_min_lr", float, 1e-6, range=(1e-7, 1e-3)),
        ConfigSchema("training_config.warmup_ratio", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("training_config.metrics_to_track", list, ["loss", "accuracy", "confidence"], lambda x: all(isinstance(i, str) for i in x)),
        ConfigSchema("training_config.repetition_n", int, 3, range=(1, 10)),
        ConfigSchema("training_config.checkpoint_path", str, "checkpoints/sovl_trainer"),
        ConfigSchema("training_config.validate_every_n_steps", int, 100, range=(10, 1000)),
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
        ConfigSchema("controls_config.enable_scaffold", bool, True),
        ConfigSchema("controls_config.injection_strategy", str, "sequential", lambda x: x in ["sequential", "parallel", "replace"]),
        # logging_config
        ConfigSchema("logging_config.log_dir", str, "logs"),
        ConfigSchema("logging_config.log_file", str, "sovl_logs.jsonl"),
        ConfigSchema("logging_config.debug_log_file", str, "sovl_debug.log"),
        ConfigSchema("logging_config.max_size_mb", int, 10, range=(0, 100)),
        ConfigSchema("logging_config.compress_old", bool, False),
        ConfigSchema("logging_config.max_in_memory_logs", int, 1000, range=(100, 10000)),
        ConfigSchema("logging_config.schema_version", str, "1.1"),
        # Dynamic Weighting
        ConfigSchema("dynamic_weighting.min_weight", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("dynamic_weighting.max_weight", float, 1.0, range=(0.0, 1.0)),
        ConfigSchema("dynamic_weighting.weight_decay", float, 0.01, range=(0.0, 1.0)),
        ConfigSchema("dynamic_weighting.momentum", float, 0.9, range=(0.0, 1.0)),
        ConfigSchema("dynamic_weighting.history_size", int, 10, range=(1, 100)),
        ConfigSchema("dynamic_weighting.enable_dynamic_scaling", bool, True),

        # Preprocessing
        ConfigSchema("preprocessing.remove_special_chars", bool, True),
        ConfigSchema("preprocessing.lowercase", bool, True),
        ConfigSchema("preprocessing.remove_extra_spaces", bool, True),
        ConfigSchema("preprocessing.max_length", int, 512, range=(1, 2048)),

        # Augmentation
        ConfigSchema("augmentation.synonym_replacement_prob", float, 0.3, range=(0.0, 1.0)),
        ConfigSchema("augmentation.word_dropout_prob", float, 0.1, range=(0.0, 1.0)),
        ConfigSchema("augmentation.max_augmentations", int, 3, range=(1, 10)),

        # Hardware
        ConfigSchema("hardware.enable_cuda", bool, True),
        ConfigSchema("hardware.memory_query_interval", float, 0.1, range=(0.0, 10.0)),
        ConfigSchema("hardware.mock_memory_total_mb", float, 8192.0, range=(0.0, 16384.0)),
    ]

    def __init__(self, config_file: str, logger: Logger):
        self.config_file = os.getenv("SOVL_CONFIG_FILE", config_file)
        self.logger = logger
        self.store = ConfigStore()
        self.validator = SchemaValidator(logger)
        self.file_handler = FileHandler(self.config_file, logger)
        self.lock = Lock()
        self._frozen = False
        self._last_config_hash = ""
        self._subscribers: set[Callable[[], None]] = set()
        self.validator.register(self.DEFAULT_SCHEMA)
        self._initialize_config()

    def _initialize_config(self) -> None:
        with self.lock:
            self.store.flat_config = self.file_handler.load()
            self._validate_and_set_defaults()
            self.store.rebuild_structured(self.DEFAULT_SCHEMA)
            self.store.update_cache(self.DEFAULT_SCHEMA)
            self._last_config_hash = self._compute_config_hash()
            self._log_event("config_load", "Configuration loaded successfully", "info", {
                "config_file": self.config_file,
                "config_hash": self._last_config_hash,
                "schema_version": self.DEFAULT_SCHEMA[-1].default
            })

    def _compute_config_hash(self) -> str:
        try:
            config_str = json.dumps(self.store.flat_config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            self._log_error("Config hash computation failed", {"error": str(e)})
            return ""

    def _validate_and_set_defaults(self) -> None:
        for schema in self.DEFAULT_SCHEMA:
            value = self.store.get_value(schema.field, schema.default)
            is_valid, corrected_value = self.validator.validate(schema.field, value)
            if not is_valid:
                self.store.set_value(schema.field, corrected_value)
                self._log_event("config_validation", f"Set default value for {schema.field}", "warning", {
                    "field": schema.field,
                    "default_value": corrected_value
                })

    def _log_event(self, event_type: str, message: str, level: str, additional_info: Dict[str, Any] = None) -> None:
        self.logger.record_event(event_type=event_type, message=message, level=level, additional_info=additional_info or {})

    def _log_error(self, message: str, additional_info: Dict[str, Any] = None) -> None:
        self._log_event("config_error", message, "error", {
            **(additional_info or {}),
            "stack_trace": traceback.format_exc()
        })

    def freeze(self) -> None:
        with self.lock:
            self._frozen = True
            self._log_event("config_frozen", "Configuration frozen", "info", {"timestamp": time.time()})

    def unfreeze(self) -> None:
        with self.lock:
            self._frozen = False
            self._log_event("config_unfrozen", "Configuration unfrozen", "info", {"timestamp": time.time()})

    def get(self, key: Union[str, ConfigKey], default: Any = None) -> Any:
        """Get a configuration value with type-safe key support."""
        with self.lock:
            if isinstance(key, ConfigKey):
                key = str(key)
            value = self.store.get_value(key, default)
            if value == {} or value is None:
                self._log_event("config_warning", f"Key '{key}' is empty or missing. Using default: {default}", "warning", {
                    "key": key,
                    "default_value": default
                })
                return default
            return value

    def validate_keys(self, required_keys: List[str]) -> None:
        with self.lock:
            missing_keys = [key for key in required_keys if self.get(key, None) is None]
            if missing_keys:
                self._log_error(f"Missing required configuration keys: {', '.join(missing_keys)}", {"keys": missing_keys})
                raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    def get_section(self, section: str) -> Dict[str, Any]:
        with self.lock:
            return self.store.get_section(section)

    def update(self, key: str, value: Any) -> bool:
        try:
            with self.lock:
                if self._frozen:
                    self._log_error("Cannot update: configuration is frozen", {"key": key})
                    return False

                is_valid, corrected_value = self.validator.validate(key, value)
                if not is_valid:
                    return False

                old_hash = self._last_config_hash
                self.store.set_value(key, corrected_value)
                self._last_config_hash = self._compute_config_hash()
                self._notify_subscribers()
                self._log_event("config_update", f"Updated configuration key: {key}", "info", {
                    "key": key,
                    "value": corrected_value,
                    "old_hash": old_hash,
                    "new_hash": self._last_config_hash
                })
                return True
        except Exception as e:
            self._log_error(f"Failed to update config key {key}", {"key": key, "error": str(e)})
            return False

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self.lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        with self.lock:
            self._subscribers.discard(callback)

    def _notify_subscribers(self) -> None:
        with self.lock:
            for callback in self._subscribers:
                try:
                    callback()
                except Exception as e:
                    self._log_error("Failed to notify subscriber of config change", {"error": str(e)})

    def update_batch(self, updates: Dict[str, Any], rollback_on_failure: bool = True) -> bool:
        if not updates:
            return True

        with self.lock:
            if self._frozen:
                self._log_error("Cannot update batch: configuration is frozen", {"updates": list(updates.keys())})
                return False

            backup = self.store.flat_config.copy()
            successful_updates = {}
            try:
                for key, value in updates.items():
                    is_valid, corrected_value = self.validator.validate(key, value)
                    if not is_valid:
                        raise ValueError(f"Invalid value for {key}: {value}")
                    self.store.set_value(key, corrected_value)
                    successful_updates[key] = corrected_value

                if not self.file_handler.save(self.store.flat_config):
                    raise RuntimeError("Failed to save configuration")

                self._last_config_hash = self._compute_config_hash()
                self._notify_subscribers()
                return True
            except Exception as e:
                if rollback_on_failure:
                    self.store.flat_config = backup
                    self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                    self.store.update_cache(self.DEFAULT_SCHEMA)
                    self._last_config_hash = self._compute_config_hash()
                    self._log_event("config_rollback", f"Configuration rollback triggered: {str(e)}", "error", {
                        "failed_updates": list(updates.keys()),
                        "successful_updates": list(successful_updates.keys())
                    })
                else:
                    self._log_error("Configuration update failed", {
                        "failed_updates": list(updates.keys()),
                        "successful_updates": list(successful_updates.keys()),
                        "error": str(e)
                    })
                return False

    def save_config(self, file_path: Optional[str] = None, compress: bool = False, max_retries: int = 3) -> bool:
        with self.lock:
            return self.file_handler.save(self.store.flat_config, file_path, compress, max_retries)

    def diff_config(self, old_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        try:
            with self.lock:
                diff = {}
                for key in set(self.store.flat_config) | set(old_config):
                    old_value = old_config.get(key)
                    new_value = self.store.flat_config.get(key)
                    if old_value != new_value:
                        diff[key] = {"old": old_value, "new": new_value}
                self._log_event("config_diff", "Configuration differences computed", "info", {
                    "changed_keys": list(diff.keys())
                })
                return diff
        except Exception as e:
            self._log_error("Config diff failed", {"error": str(e)})
            return {}

    def register_schema(self, schemas: List[ConfigSchema]) -> None:
        try:
            with self.lock:
                if self._frozen:
                    self._log_error("Cannot register schema: configuration is frozen")
                    return
                self.validator.register(schemas)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA + schemas)
                self.store.update_cache(self.DEFAULT_SCHEMA + schemas)
                self._last_config_hash = self._compute_config_hash()
                self._log_event("schema_registered", f"New fields registered", "info", {
                    "new_fields": [s.field for s in schemas],
                    "config_hash": self._last_config_hash
                })
        except Exception as e:
            self._log_error("Failed to register schema", {"error": str(e)})

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "config_file": self.config_file,
                "config": self.store.flat_config,
                "frozen": self._frozen,
                "config_hash": self._last_config_hash
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        try:
            with self.lock:
                self.config_file = state.get("config_file", self.config_file)
                self.store.flat_config = state.get("config", {})
                self._frozen = state.get("frozen", False)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                self.store.update_cache(self.DEFAULT_SCHEMA)
                self._last_config_hash = self._compute_config_hash()
                self._log_event("config_load_state", "Configuration state loaded", "info", {
                    "config_file": self.config_file,
                    "config_hash": self._last_config_hash
                })
        except Exception as e:
            self._log_error("Failed to load config state", {"error": str(e)})
            raise

    def tune(self, **kwargs) -> bool:
        return self.update_batch(kwargs)

    def load_profile(self, profile: str) -> bool:
        profile_file = f"{os.path.splitext(self.config_file)[0]}_{profile}.json"
        try:
            with self.lock:
                config = self.file_handler.load()
                if not config:
                    self._log_error(f"Profile file {profile_file} not found", {"profile_file": profile_file})
                    return False
                self.store.flat_config = config
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self.DEFAULT_SCHEMA)
                self.store.update_cache(self.DEFAULT_SCHEMA)
                self._last_config_hash = self._compute_config_hash()
                self._log_event("profile_load", f"Profile {profile} loaded", "info", {
                    "profile": profile,
                    "config_file": profile_file,
                    "config_hash": self._last_config_hash
                })
                return True
        except Exception as e:
            self._log_error(f"Failed to load profile {profile}", {"error": str(e)})
            return False

    def set_global_blend(self, weight_cap: Optional[float] = None, base_temp: Optional[float] = None) -> bool:
        updates = {}
        prefix = "controls_config."

        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            updates[f"{prefix}scaffold_weight_cap"] = weight_cap

        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            updates[f"{prefix}base_temperature"] = base_temp

        return self.update_batch(updates) if updates else True

    def validate_section(self, section: str, required_keys: List[str]) -> bool:
        try:
            with self.lock:
                if section not in self.store.structured_config:
                    self._log_error(f"Configuration section '{section}' not found", {"section": section})
                    return False

                missing_keys = [key for key in required_keys if key not in self.store.structured_config[section]]
                if missing_keys:
                    self._log_error(f"Missing required keys in section '{section}': {', '.join(missing_keys)}", {
                        "section": section,
                        "missing_keys": missing_keys
                    })
                    return False

                return True
        except Exception as e:
            self._log_error(f"Failed to validate section '{section}'", {"section": section, "error": str(e)})
            return False

    def tune_parameter(self, section: str, key: str, value: Any, min_value: Any = None, max_value: Any = None) -> bool:
        full_key = f"{section}.{key}"
        try:
            with self.lock:
                if min_value is not None and value < min_value:
                    self._log_error(f"Value {value} below minimum {min_value} for {full_key}", {
                        "section": section,
                        "key": key,
                        "value": value,
                        "min_value": min_value
                    })
                    return False

                if max_value is not None and value > max_value:
                    self._log_error(f"Value {value} above maximum {max_value} for {full_key}", {
                        "section": section,
                        "key": key,
                        "value": value,
                        "max_value": max_value
                    })
                    return False

                success = self.update(full_key, value)
                if success:
                    self._log_event("config_info", f"Tuned {full_key} to {value}", "info", {
                        "section": section,
                        "key": key,
                        "value": value
                    })
                return success
        except Exception as e:
            self._log_error(f"Failed to tune {full_key}", {"section": section, "key": key, "error": str(e)})
            return False

    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                if section not in self.store.structured_config:
                    self._log_error(f"Configuration section '{section}' not found", {"section": section})
                    return False

                batch_updates = {f"{section}.{key}": value for key, value in updates.items()}
                return self.update_batch(batch_updates)
        except Exception as e:
            self._log_error(f"Failed to update section '{section}'", {"section": section, "error": str(e)})
            return False

    def validate_or_raise(self, model_config: Optional[Any] = None) -> None:
        try:
            self.validate_keys([
                "core_config.base_model_name",
                "core_config.scaffold_model_name",
                "training_config.learning_rate",
                "curiosity_config.enable_curiosity",
                "cross_attn_config.memory_weight"
            ])

            cross_attn_layers = self.get("core_config.cross_attn_layers", [5, 7])
            if not isinstance(cross_attn_layers, list):
                raise ValueError("core_config.cross_attn_layers must be a list")

            if not self.get("core_config.use_dynamic_layers", False) and model_config is not None:
                base_model_name = self.get("core_config.base_model_name", "gpt2")
                try:
                    base_config = model_config or AutoConfig.from_pretrained(base_model_name)
                    invalid_layers = [l for l in cross_attn_layers if not (0 <= l < base_config.num_hidden_layers)]
                    if invalid_layers:
                        raise ValueError(f"Invalid cross_attn_layers: {invalid_layers} for {base_config.num_hidden_layers} layers")
                except Exception as e:
                    raise ValueError(f"Failed to validate cross-attention layers: {str(e)}")

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

            self._log_event("config_validation", "Configuration validation successful", "info", {
                "config_snapshot": self.get_state()["config"]
            })
        except Exception as e:
            self._log_error("Configuration validation failed", {"error": str(e)})
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def validate_value(self, key: str, value: Any) -> bool:
        try:
            with self.lock:
                schema = next((s for s in self.DEFAULT_SCHEMA if s.field == key), None)
                if not schema:
                    self._log_error(f"Unknown configuration key: {key}", {"key": key})
                    return False

                is_valid, _ = self.validator.validate(key, value)
                if not is_valid:
                    self._log_error(f"Invalid value for {key}: {value}", {"key": key, "value": value})
                return is_valid
        except Exception as e:
            self._log_error(f"Failed to validate value for {key}", {"key": key, "error": str(e)})
            return False

    def validate_with_model(self, model_config: Any) -> bool:
        try:
            self.validate_or_raise(model_config)
            return True
        except Exception as e:
            self._log_error("Configuration validation failed", {"error": str(e), "model_config": str(model_config)})
            return False

if __name__ == "__main__":
    from sovl_logger import LoggerConfig
    logger = Logger(LoggerConfig())
    config_manager = ConfigManager("sovl_config.json", logger)
    try:
        config_manager.validate_keys(["core_config.base_model_name", "curiosity_config.attention_weight"])
    except ValueError as e:
        print(e)
