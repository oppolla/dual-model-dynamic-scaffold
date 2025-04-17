import json
import os
import gzip
import uuid
import time
import logging
from datetime import datetime
from threading import Lock
from typing import List, Dict, Union, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import torch
from logging.handlers import RotatingFileHandler
import traceback
import psutil

@dataclass
class LoggerConfig:
    """Configuration for Logger with validation."""
    log_file: str = "sovl_logs.jsonl"
    max_size_mb: int = 10
    compress_old: bool = False
    max_in_memory_logs: int = 1000
    rotation_count: int = 5
    max_log_age_days: int = 30  # Maximum age of logs to keep
    prune_interval_hours: int = 24  # How often to prune old logs
    memory_threshold_mb: int = 100  # Memory threshold to trigger aggressive pruning

    _RANGES = {
        "max_size_mb": (0, 100),
        "max_in_memory_logs": (100, 10000),
        "max_log_age_days": (1, 365),
        "prune_interval_hours": (1, 168),
        "memory_threshold_mb": (10, 1000)
    }

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.log_file, str) or not self.log_file.endswith(".jsonl"):
            raise ValueError("log_file must be a .jsonl file path")
        if not isinstance(self.compress_old, bool):
            raise ValueError("compress_old must be a boolean")
        for key, (min_val, max_val) in self._RANGES.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key == "log_file":
                if not isinstance(value, str) or not value.endswith(".jsonl"):
                    raise ValueError("log_file must be a .jsonl file path")
            elif key == "compress_old":
                if not isinstance(value, bool):
                    raise ValueError("compress_old must be a boolean")
            elif key in self._RANGES:
                min_val, max_val = self._RANGES[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            elif key == "rotation_count":
                if not isinstance(value, int) or value < 0:
                    raise ValueError("rotation_count must be a non-negative integer")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)

class _LogValidator:
    """Handles log entry validation logic."""
    
    REQUIRED_FIELDS = {'timestamp', 'conversation_id'}
    OPTIONAL_FIELDS = {'prompt', 'response', 'confidence_score', 'error', 'warning', 'mood', 'variance', 'logits_shape'}
    FIELD_VALIDATORS = {
        'timestamp': lambda x: isinstance(x, (str, float, int)),
        'conversation_id': lambda x: isinstance(x, str),
        'confidence_score': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
        'is_error_prompt': lambda x: isinstance(x, bool),
        'mood': lambda x: x in {'melancholic', 'restless', 'calm', 'curious'},
        'variance': lambda x: isinstance(x, (int, float)) and x >= 0.0,
        'logits_shape': lambda x: isinstance(x, (tuple, list, str))
    }

    def __init__(self, fallback_logger: logging.Logger):
        self.fallback_logger = fallback_logger

    def validate_entry(self, entry: Dict) -> bool:
        """Validate log entry structure and types."""
        if not isinstance(entry, dict):
            self.fallback_logger.warning("Log entry is not a dictionary")
            return False

        try:
            # Ensure required fields
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
            if 'conversation_id' not in entry:
                entry['conversation_id'] = str(uuid.uuid4())

            # Validate field types
            for field, validator in self.FIELD_VALIDATORS.items():
                if field in entry and not validator(entry[field]):
                    self.fallback_logger.warning(f"Invalid value for field {field}: {entry[field]}")
                    return False

            return True
        except Exception as e:
            self.fallback_logger.error(f"Validation failed: {str(e)}")
            return False

class _FileHandler:
    """Manages file operations for logging."""
    
    def __init__(self, config: LoggerConfig, fallback_logger: logging.Logger):
        self.config = config
        self.fallback_logger = fallback_logger

    def safe_file_op(self, operation: Callable, *args, **kwargs):
        """Execute file operation with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    self.fallback_logger.error(f"File operation failed after {max_retries} retries: {str(e)}")
                    raise
                time.sleep(0.1 * (attempt + 1))

    def atomic_write(self, filename: str, content: str) -> None:
        """Perform atomic file write using temporary file."""
        temp_file = f"{filename}.tmp"
        try:
            with self.safe_file_op(open, temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.safe_file_op(os.replace, temp_file, filename)
        except Exception as e:
            self.fallback_logger.error(f"Atomic write failed: {str(e)}")
            if os.path.exists(temp_file):
                self.safe_file_op(os.remove, temp_file)
            raise

    def rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if self.config.max_size_mb <= 0 or not os.path.exists(self.config.log_file):
            return

        file_size = os.path.getsize(self.config.log_file)
        if file_size < self.config.max_size_mb * 1024 * 1024:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = f"{self.config.log_file}.{timestamp}"

            if self.config.compress_old:
                rotated_file += ".gz"
                with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                    with self.safe_file_op(gzip.open, rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                self.safe_file_op(os.rename, self.config.log_file, rotated_file)

            self.fallback_logger.info(f"Rotated logs to {rotated_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to rotate log file: {str(e)}")

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file."""
        if not os.path.exists(self.config.log_file):
            return None

        compressed_file = f"{self.config.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                with self.safe_file_op(gzip.open, compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            if not keep_original:
                self.safe_file_op(os.remove, self.config.log_file)

            self.fallback_logger.info(f"Compressed logs to {compressed_file}")
            return compressed_file
        except Exception as e:
            self.fallback_logger.error(f"Failed to compress logs: {str(e)}")
            return None

    def manage_rotation(self, max_files: int = 5) -> None:
        """Manage rotated log files, keeping only max_files most recent."""
        if not os.path.exists(self.config.log_file):
            return

        try:
            base_name = os.path.basename(self.config.log_file)
            log_dir = os.path.dirname(self.config.log_file) or '.'

            rotated_files = [
                os.path.join(log_dir, f) for f in os.listdir(log_dir)
                if f.startswith(base_name) and f != base_name
            ]

            rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            for old_file in rotated_files[max_files:]:
                try:
                    self.safe_file_op(os.remove, old_file)
                    self.fallback_logger.info(f"Removed old log file {old_file}")
                except OSError:
                    self.fallback_logger.error(f"Failed to remove old log file {old_file}")
        except Exception as e:
            self.fallback_logger.error(f"Error managing log rotation: {str(e)}")

    def write_batch(self, entries: List[Dict]) -> None:
        """Optimized batch writing with validation and atomic write."""
        if not entries:
            return

        valid_entries = []
        for entry in entries:
            if self.validate_entry(entry):
                if "error" in entry or "warning" in entry:
                    entry["is_error_prompt"] = True
                valid_entries.append(entry)
            else:
                self.fallback_logger.warning(f"Invalid log entry skipped: {entry}")

        if not valid_entries:
            return

        try:
            with self.safe_file_op(open, self.config.log_file, 'a') as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.fallback_logger.error(f"Error writing batch: {str(e)}")

    def cleanup(self) -> None:
        """Clean up logging resources for the file handler."""
        try:
            self.manage_rotation()
            self.compress_logs()
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to clean up file handler: {str(e)}",
                    error_type="file_handler_cleanup_error",
                    stack_trace=traceback.format_exc()
                )

class Logger:
    """Main logger class for the SOVL system."""
    
    def __init__(self, config: LoggerConfig, fallback_logger: Logger = None):
        """Initialize logger with configuration."""
        self.config = config
        self.fallback_logger = fallback_logger
        self._log_cache = {}
        self._log_buffer = []
        self._last_prune_time = time.time()
        self._validator = _LogValidator(fallback_logger)
        self._file_handler = _FileHandler(config, fallback_logger)
        self._min_level = "info"
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing logs from file."""
        try:
            if os.path.exists(self.config.log_file):
                with open(self.config.log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if self._validator.validate_entry(entry):
                                self._log_buffer.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to load existing logs: {str(e)}",
                    error_type="log_loading_error",
                    stack_trace=traceback.format_exc()
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def _should_prune(self) -> bool:
        """Check if logs should be pruned."""
        current_time = time.time()
        if current_time - self._last_prune_time < self.config.prune_interval_hours * 3600:
            return False
            
        memory_usage = self._get_memory_usage()
        if memory_usage > self.config.memory_threshold_mb:
            return True
            
        return False

    def _prune_old_logs(self) -> None:
        """Prune old logs based on age and memory usage."""
        try:
            current_time = time.time()
            self._log_buffer = [
                entry for entry in self._log_buffer
                if current_time - entry.get('timestamp', 0) < self.config.max_log_age_days * 86400
            ]
            
            if self._get_memory_usage() > self.config.memory_threshold_mb:
                self._log_buffer = self._log_buffer[-self.config.max_in_memory_logs:]
                
            self._last_prune_time = current_time
            
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to prune logs: {str(e)}",
                    error_type="log_pruning_error",
                    stack_trace=traceback.format_exc()
                )

    def set_min_level(self, level: str) -> None:
        """Set minimum logging level."""
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if level.lower() not in valid_levels:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Invalid logging level: {level}",
                    error_type="invalid_log_level",
                    stack_trace=None,
                    additional_info={"valid_levels": list(valid_levels)}
                )
            return
        self._min_level = level.lower()

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        level_priority = {
            'debug': 0,
            'info': 1,
            'warning': 2,
            'error': 3,
            'critical': 4
        }
        return level_priority.get(level.lower(), 0) >= level_priority.get(self._min_level, 0)

    def _get_cached_log(self, key: str) -> Optional[Dict]:
        """Get cached log entry."""
        if key in self._log_cache:
            entry = self._log_cache[key]
            if time.time() - entry.get('timestamp', 0) < 3600:  # Cache for 1 hour
                return entry
            del self._log_cache[key]
        return None

    def _cache_log(self, key: str, entry: Dict) -> None:
        """Cache log entry."""
        self._log_cache[key] = entry
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._log_cache.items()
            if current_time - entry.get('timestamp', 0) >= 3600
        ]
        for key in expired_keys:
            del self._log_cache[key]

    def _write_batch(self) -> None:
        """Write buffered logs to file."""
        try:
            if not self._log_buffer:
                return
                
            entries = self._log_buffer.copy()
            self._log_buffer.clear()
            
            self._file_handler.write_batch(entries)
            
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to write log batch: {str(e)}",
                    error_type="log_write_error",
                    stack_trace=traceback.format_exc()
                )

    def log_error(
        self,
        error_msg: str,
        error_type: str = None,
        stack_trace: str = None,
        conversation_id: str = None,
        state_hash: str = None,
        **kwargs
    ) -> None:
        """Log an error event."""
        if not self._should_log('error'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': 'error',
            'level': 'error',
            'error_msg': error_msg,
            'error_type': error_type or 'unknown_error',
            'stack_trace': stack_trace,
            'conversation_id': conversation_id,
            'state_hash': state_hash,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_memory_usage(
        self,
        phase: str,
        device: torch.device,
        **kwargs
    ) -> None:
        """Log memory usage statistics."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': 'memory_usage',
            'level': 'info',
            'phase': phase,
            'device': str(device),
            'memory_stats': memory_usage(device),
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_memory_health(
        self,
        model_size: int,
        trainer: Optional[SOVLTrainer] = None,
        **kwargs
    ) -> None:
        """Log memory health status."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': 'memory_health',
            'level': 'info',
            'model_size': model_size,
            'trainer_state': trainer.state if trainer else None,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_training_event(
        self,
        event_type: str,
        epoch: int = None,
        loss: float = None,
        batch_size: int = None,
        data_exposure: float = None,
        conversation_id: str = None,
        state_hash: str = None,
        **kwargs
    ) -> None:
        """Log a training-related event."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'level': 'info',
            'epoch': epoch,
            'loss': loss,
            'batch_size': batch_size,
            'data_exposure': data_exposure,
            'conversation_id': conversation_id,
            'state_hash': state_hash,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_generation_event(
        self,
        prompt: str,
        response: str,
        confidence_score: float,
        generation_params: dict = None,
        conversation_id: str = None,
        state_hash: str = None,
        **kwargs
    ) -> None:
        """Log a generation event."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': 'generation',
            'level': 'info',
            'prompt': prompt,
            'response': response,
            'confidence_score': confidence_score,
            'generation_params': generation_params,
            'conversation_id': conversation_id,
            'state_hash': state_hash,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_cleanup_event(
        self,
        phase: str,
        success: bool,
        error: str = None,
        conversation_id: str = None,
        state_hash: str = None,
        **kwargs
    ) -> None:
        """Log a cleanup event."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': 'cleanup',
            'level': 'info',
            'phase': phase,
            'success': success,
            'error': error,
            'conversation_id': conversation_id,
            'state_hash': state_hash,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def log_curiosity_event(
        self,
        event_type: str,
        question: str = None,
        score: float = None,
        spontaneous: bool = False,
        answered: bool = False,
        conversation_id: str = None,
        state_hash: str = None,
        **kwargs
    ) -> None:
        """Log a curiosity-related event."""
        if not self._should_log('info'):
            return
            
        entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'level': 'info',
            'question': question,
            'score': score,
            'spontaneous': spontaneous,
            'answered': answered,
            'conversation_id': conversation_id,
            'state_hash': state_hash,
            **kwargs
        }
        
        if self._validator.validate_entry(entry):
            self._log_buffer.append(entry)
            self._write_batch()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'log_buffer_size': len(self._log_buffer),
            'cache_size': len(self._log_cache),
            'memory_usage_mb': self._get_memory_usage(),
            'last_prune_time': self._last_prune_time
        }

    def cleanup(self) -> None:
        """Clean up logger resources."""
        try:
            self._write_batch()
            self._log_buffer.clear()
            self._log_cache.clear()
            
            if self._file_handler:
                self._file_handler.cleanup()
                
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to cleanup logger: {str(e)}",
                    error_type="logger_cleanup_error",
                    stack_trace=traceback.format_exc()
                )

class LoggingManager:
    """Manages logging setup and configuration for the SOVL system."""

    _DEFAULT_CONFIG = {
        "logging.max_size_mb": 10,
        "logging.compress_old": True,
        "logging.rotation_count": 5,
        "logging.level": "INFO"
    }

    def __init__(
        self,
        config_manager: ConfigManager,
        log_dir: str = "logs",
        system_log_file: str = "sovl_system.log",
        debug_log_file: str = "sovl_debug.log"
    ):
        """
        Initialize the logging manager with configuration and file paths.

        Args:
            config_manager: ConfigManager instance for accessing logging settings.
            log_dir: Directory to store log files.
            system_log_file: Name of the system log file.
            debug_log_file: Name of the debug log file.
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be an instance of ConfigManager")
            
        self.config_manager = config_manager
        self.log_dir = log_dir
        self.system_log_file = os.path.join(log_dir, system_log_file)
        self.debug_log_file = os.path.join(log_dir, debug_log_file)
        self.loggers = {}
        
        # Configure fallback logger first
        self._configure_fallback_logger()
        
        # Ensure logging configuration exists
        self._ensure_logging_config()
        self._validate_config()
        self._setup_logging()

    def _configure_fallback_logger(self) -> None:
        """Configure the fallback logger with proper handlers and formatting."""
        # Get the fallback logger
        fallback_logger = logging.getLogger(__name__)
        
        # Set log level from config or default to INFO
        log_level = self.config_manager.get(
            "logging.level",
            self._DEFAULT_CONFIG["logging.level"]
        )
        fallback_logger.setLevel(getattr(logging, log_level))
        
        # Remove any existing handlers to avoid duplicates
        for handler in fallback_logger.handlers[:]:
            fallback_logger.removeHandler(handler)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Create detailed formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        fallback_logger.addHandler(console_handler)
        
        # Log successful configuration
        fallback_logger.info(
            f"Configured fallback logger with level {log_level}"
        )

    def _ensure_logging_config(self) -> None:
        """Ensure logging configuration exists in config manager."""
        try:
            # Get or create logging section
            logging_config = self.config_manager.get_section("logging", {})
            
            # Set default values if not present
            for key, default_value in self._DEFAULT_CONFIG.items():
                if key not in logging_config:
                    self.config_manager.set(key, default_value)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to ensure logging configuration: {str(e)}")

    def _validate_config(self) -> None:
        """Validate logging configuration."""
        try:
            max_size = self.config_manager.get(
                "logging.max_size_mb",
                self._DEFAULT_CONFIG["logging.max_size_mb"]
            )
            if max_size <= 0:
                raise ValueError("Log file size must be positive")
        except Exception as e:
            print(f"Logging configuration validation failed: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """Set up logging configuration and handlers."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate and get configuration values
        try:
            max_size_mb = self.config_manager.get(
                "logging.max_size_mb",
                self._DEFAULT_CONFIG["logging.max_size_mb"]
            )
            if not isinstance(max_size_mb, (int, float)) or max_size_mb <= 0:
                raise ValueError(
                    f"Invalid logging.max_size_mb: {max_size_mb}. Must be a positive number."
                )

            compress_old = self.config_manager.get(
                "logging.compress_old",
                self._DEFAULT_CONFIG["logging.compress_old"]
            )
            if not isinstance(compress_old, bool):
                raise ValueError(
                    f"Invalid logging.compress_old: {compress_old}. Must be a boolean."
                )

            rotation_count = self.config_manager.get(
                "logging.rotation_count",
                self._DEFAULT_CONFIG["logging.rotation_count"]
            )
            if not isinstance(rotation_count, int) or rotation_count < 0:
                raise ValueError(
                    f"Invalid logging.rotation_count: {rotation_count}. Must be a non-negative integer."
                )

            # Configure system logger with validated settings
            system_config = LoggerConfig(
                log_file=self.system_log_file,
                max_size_mb=max_size_mb,
                compress_old=compress_old,
                max_in_memory_logs=1000,
                rotation_count=rotation_count
            )
            self.loggers["system"] = Logger(system_config)

            # Configure debug logger with same validated settings
            debug_config = LoggerConfig(
                log_file=self.debug_log_file,
                max_size_mb=max_size_mb,
                compress_old=compress_old,
                max_in_memory_logs=1000,
                rotation_count=rotation_count
            )
            self.loggers["debug"] = Logger(debug_config)

            # Log successful setup
            self.loggers["system"].log_event(
                event_type="logging_initialized",
                message="Logging system initialized successfully",
                level="info",
                additional_info={
                    "max_size_mb": max_size_mb,
                    "compress_old": compress_old,
                    "rotation_count": rotation_count
                }
            )

        except ValueError as e:
            # Log configuration error and re-raise
            logging.getLogger(__name__).error(
                f"Invalid logging configuration: {str(e)}"
            )
            raise
        except Exception as e:
            # Log unexpected error and re-raise
            logging.getLogger(__name__).error(
                f"Failed to setup logging: {str(e)}",
                exc_info=True
            )
            raise

    def setup_logging(self) -> Logger:
        """
        Set up and return the main logger.

        Returns:
            The main logger instance
        """
        if not self.loggers:
            self._setup_logging()
        return self.loggers["system"]

    def get_logger(self, name: str) -> Logger:
        """
        Get a logger instance by name.

        Args:
            name: The name of the logger to retrieve.

        Returns:
            The requested logger instance.

        Raises:
            KeyError: If the logger name is not found.
        """
        if name not in self.loggers:
            raise KeyError(f"Logger '{name}' not found")
        return self.loggers[name]

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update logging configuration and reconfigure loggers.

        Args:
            new_config: Dictionary containing new configuration values.
        """
        # Update configuration
        for key, value in new_config.items():
            if key.startswith("logging."):
                self.config_manager.set(key, value)

        # Revalidate and reconfigure
        self._validate_config()
        self._setup_logging()

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, 
                  conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log an error with standardized format."""
        self.get_logger("system").log_error(
            error_msg=error_msg,
            error_type=error_type,
            stack_trace=stack_trace,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_memory_usage(self, phase: str, device: torch.device, **kwargs) -> None:
        """
        Log memory usage statistics.

        Args:
            phase: The phase or operation being logged (e.g., "training", "generation")
            device: The torch device to get memory stats from
            **kwargs: Additional memory-related information to log
        """
        self.get_logger("system").log_memory_usage(
            phase=phase,
            device=device,
            **kwargs
        )

    def log_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None, **kwargs) -> None:
        """
        Log memory health check results.

        Args:
            model_size: Size of the model in bytes
            trainer: Optional trainer instance for additional memory stats
            **kwargs: Additional health-related information to log
        """
        self.get_logger("system").log_memory_health(
            model_size=model_size,
            trainer=trainer,
            **kwargs
        )

    def log_training_event(self, event_type: str, epoch: int = None, loss: float = None,
                          batch_size: int = None, data_exposure: float = None,
                          conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log training-related events."""
        self.get_logger("system").log_training_event(
            event_type=event_type,
            epoch=epoch,
            loss=loss,
            batch_size=batch_size,
            data_exposure=data_exposure,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_generation_event(self, prompt: str, response: str, confidence_score: float,
                            generation_params: dict = None, conversation_id: str = None,
                            state_hash: str = None, **kwargs) -> None:
        """Log generation-related events."""
        self.get_logger("system").log_generation_event(
            prompt=prompt,
            response=response,
            confidence_score=confidence_score,
            generation_params=generation_params,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_cleanup_event(self, phase: str, success: bool, error: str = None,
                         conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log cleanup-related events."""
        self.get_logger("system").log_cleanup_event(
            phase=phase,
            success=success,
            error=error,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def record(self, entry: Dict) -> None:
        """Write a validated log entry."""
        self.get_logger("system").record(entry)

    def cleanup(self) -> None:
        """Clean up logging resources for all loggers."""
        for name, logger in self.loggers.items():
            try:
                logger.cleanup()
                self.get_logger("system").log_event(
                    event_type="logger_cleanup",
                    message=f"Successfully cleaned up logger {name}",
                    level="info"
                )
            except Exception as e:
                # Use system logger to log the error
                self.get_logger("system").log_error(
                    error_msg=f"Failed to clean up logger {name}: {str(e)}",
                    error_type="cleanup_error",
                    stack_trace=traceback.format_exc()
                )
                # Also log to debug logger for more detailed information
                self.get_logger("debug").log_event(
                    event_type="logger_cleanup_error",
                    message=f"Detailed error during cleanup of logger {name}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "logger_name": name,
                        "stack_trace": traceback.format_exc()
                    }
                )

class LoggingError(Exception):
    """Raised for logging-related errors."""
    pass
