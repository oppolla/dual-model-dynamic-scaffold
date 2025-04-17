import time
import traceback
from typing import Optional, Dict, Any, TYPE_CHECKING, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import os
from threading import Lock
from sovl_config import ConfigManager
from sovl_cli import run_cli
from sovl_logger import LoggingManager
from sovl_state import SOVLState
from sovl_error import ErrorHandler
from sovl_utils import calculate_confidence, detect_repetitions
from sovl_plugin import PluginManager
from collections import deque
from sovl_state_manager import StateManager

if TYPE_CHECKING:
    from sovl_main import SOVLSystem

class SOVLOrchestrator:
    """
    Orchestrates the initialization, execution, and shutdown of the SOVL system.

    Responsible for setting up core components (ConfigManager, SOVLSystem),
    selecting execution modes (e.g., CLI), and ensuring clean shutdown with
    state saving and resource cleanup.
    """
    # Constants for configuration
    DEFAULT_CONFIG_PATH: str = "sovl_config.json"
    DEFAULT_LOG_FILE: str = "sovl_orchestrator_logs.jsonl"
    LOG_MAX_SIZE_MB: int = 10
    SAVE_PATH_SUFFIX: str = "_final.json"

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, log_file: str = DEFAULT_LOG_FILE) -> None:
        """
        Initialize the orchestrator with configuration and logging.

        Args:
            config_path: Path to the configuration file.
            log_file: Path to the orchestrator's log file.

        Raises:
            RuntimeError: If initialization of ConfigManager or SOVLSystem fails.
        """
        self._initialize_logger(log_file)
        self._log_event("orchestrator_init_start", {"config_path": config_path})

        try:
            # Initialize ConfigManager with validation
            self.config_manager = self._create_config_manager(config_path)
            
            # Validate configuration sections
            self._validate_config_sections()
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log_event("device_initialized", {"device": str(self.device)})
            
            # Initialize state manager
            self.state_manager = StateManager(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            
            # Load state from file if exists, otherwise initialize new state
            self.state = self.state_manager.load_state()
            
            # Initialize system early to ensure state consistency
            self._system = None
            self.initialize_system()
            
            # Initialize error handler and plugin manager after system is ready
            self.error_handler = ErrorHandler(self.logger)
            self.plugin_manager = PluginManager(
                config_manager=self.config_manager,
                logger=self.logger,
                state=self.state
            )
            
            self._lock = Lock()
            self._log_event("orchestrator_init_success", {
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "system_state_hash": self._system.state_tracker.state.state_hash if self._system else None
            })
        except Exception as e:
            self._log_error("Orchestrator initialization failed", e)
            self.error_handler.handle_generic_error(
                error=e,
                context="orchestrator_initialization",
                fallback_action=lambda: self._cleanup_resources()
            )
            raise RuntimeError(f"Failed to initialize orchestrator: {str(e)}") from e

    def _create_config_manager(self, config_path: str) -> ConfigManager:
        """Create and initialize the configuration manager with validation."""
        try:
            config_manager = ConfigManager(config_path, self.logger)
            
            # Validate required configuration sections
            required_sections = [
                "core_config",
                "training_config",
                "curiosity_config",
                "cross_attn_config",
                "controls_config",
                "lora_config"
            ]
            
            missing_sections = [section for section in required_sections 
                              if not config_manager.has_section(section)]
            
            if missing_sections:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Missing required configuration sections",
                        "missing_sections": missing_sections
                    },
                    level="warning"
                )
                # Create missing sections with default values
                for section in missing_sections:
                    config_manager.add_section(section, {})
            
            return config_manager
        except Exception as e:
            self._log_error("Config manager creation failed", e)
            raise

    def _validate_config_sections(self) -> None:
        """Validate configuration sections for consistency."""
        try:
            # Get configuration sections
            curiosity_config = self.config_manager.get_section("curiosity_config")
            controls_config = self.config_manager.get_section("controls_config")
            
            # Define required keys and their default values
            required_keys = {
                "queue_maxlen": 10,
                "weight_ignorance": 0.7,
                "weight_novelty": 0.3,
                "metrics_maxlen": 1000,
                "novelty_threshold_spontaneous": 0.9,
                "novelty_threshold_response": 0.8,
                "pressure_threshold": 0.7,
                "pressure_drop": 0.3,
                "silence_threshold": 20.0,
                "question_cooldown": 60.0,
                "max_new_tokens": 8,
                "base_temperature": 1.1,
                "temperament_influence": 0.4,
                "top_k": 30
            }
            
            # Check for missing keys in curiosity_config
            missing_keys = [key for key in required_keys if key not in curiosity_config]
            if missing_keys:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Missing keys in curiosity_config",
                        "missing_keys": missing_keys,
                        "default_values": {k: required_keys[k] for k in missing_keys}
                    },
                    level="warning"
                )
                # Add missing keys with default values
                for key in missing_keys:
                    curiosity_config[key] = required_keys[key]
            
            # Check for mismatches between curiosity_config and controls_config
            controls_mapping = {
                "curiosity_queue_maxlen": "queue_maxlen",
                "curiosity_weight_ignorance": "weight_ignorance",
                "curiosity_weight_novelty": "weight_novelty",
                "curiosity_metrics_maxlen": "metrics_maxlen",
                "curiosity_novelty_threshold_spontaneous": "novelty_threshold_spontaneous",
                "curiosity_novelty_threshold_response": "novelty_threshold_response",
                "curiosity_pressure_threshold": "pressure_threshold",
                "curiosity_pressure_drop": "pressure_drop",
                "curiosity_silence_threshold": "silence_threshold",
                "curiosity_question_cooldown": "question_cooldown",
                "curiosity_max_new_tokens": "max_new_tokens",
                "curiosity_base_temperature": "base_temperature",
                "curiosity_temperament_influence": "temperament_influence",
                "curiosity_top_k": "top_k"
            }
            
            mismatches = []
            for controls_key, curiosity_key in controls_mapping.items():
                if controls_key in controls_config and curiosity_key in curiosity_config:
                    if controls_config[controls_key] != curiosity_config[curiosity_key]:
                        mismatches.append({
                            "controls_key": controls_key,
                            "curiosity_key": curiosity_key,
                            "controls_value": controls_config[controls_key],
                            "curiosity_value": curiosity_config[curiosity_key]
                        })
            
            if mismatches:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Configuration mismatches between controls_config and curiosity_config",
                        "mismatches": mismatches
                    },
                    level="warning"
                )
                # Align controls_config with curiosity_config
                for mismatch in mismatches:
                    controls_config[mismatch["controls_key"]] = curiosity_config[mismatch["curiosity_key"]]
            
            # Log final configuration state
            self._log_event(
                "config_validation",
                {
                    "message": "Configuration validation complete",
                    "curiosity_config": curiosity_config,
                    "controls_config": {k: v for k, v in controls_config.items() 
                                     if k.startswith("curiosity_")}
                },
                level="info"
            )
            
        except Exception as e:
            self._log_error("Configuration validation failed", e)
            raise

    def _initialize_logger(self, log_file: str) -> None:
        """Initialize the logger with LoggingManager."""
        try:
            self.logger = LoggingManager(
                log_file=log_file,
                max_size_mb=10,
                rotation_interval="1d"
            )
            self._log_event("logger_initialized", {"log_file": log_file})
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to initialize logger",
                error_type="logger_init_error",
                stack_trace=traceback.format_exc(),
                additional_info={"log_file": log_file}
            )
            raise

    def _log_event(self, event_type: str, additional_info: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        """Log an event with standardized metadata."""
        try:
            metadata = {
                "conversation_id": getattr(self.state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(self.state, 'state_hash', None),
                "device": str(self.device) if hasattr(self, 'device') else None
            }
            if additional_info:
                metadata.update(additional_info)
            
            self.logger.record_event(
                event_type=event_type,
                message=f"Orchestrator event: {event_type}",
                level=level,
                additional_info=metadata
            )
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to log event",
                error_type="logging_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "event_type": event_type,
                    "original_error": str(e)
                }
            )

    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with standardized metadata and stack trace."""
        try:
            metadata = {
                "conversation_id": getattr(self.state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(self.state, 'state_hash', None),
                "device": str(self.device) if hasattr(self, 'device') else None,
                "error": str(error)
            }
            
            self.logger.log_error(
                error_msg=message,
                error_type="orchestrator_error",
                stack_trace=traceback.format_exc(),
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")  # Fallback to print if logger fails

    def set_system(self, system: 'SOVLSystem') -> None:
        """Set the SOVL system reference and sync state."""
        with self._lock:
            self._system = system
            self._sync_state_to_system()
            self.plugin_manager.set_system(system)
            self._log_event("system_set", {
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "system_state_hash": system.state_tracker.state.state_hash
            })

    def _sync_state_to_system(self) -> None:
        """Sync orchestrator state to system state."""
        if self._system is None:
            return
            
        try:
            # Copy orchestrator state to system state
            self._system.state_tracker.state.from_dict(
                self.state.to_dict(),
                self._system.device
            )
            
            # Validate state synchronization
            if self.state.state_hash != self._system.state_tracker.state.state_hash:
                self._log_event("state_sync_mismatch", {
                    "orchestrator_hash": self.state.state_hash,
                    "system_hash": self._system.state_tracker.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                }, level="warning")
                
        except Exception as e:
            self._log_error("State synchronization failed", e)
            raise

    def initialize_system(self) -> None:
        """Initialize the SOVL system with the current configuration."""
        try:
            if self._system is None:
                from sovl_main import SOVLSystem  # Import here to break circular dependency
                self._system = SOVLSystem(
                    self.config_manager.config_path,
                    device=self.device
                )
                
                # Sync state between orchestrator and system
                self._sync_state_to_system()
                
                # Validate state in system components
                self._validate_system_state()
                
                self._log_event("system_initialized", {
                    "device": str(self.device),
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash,
                    "system_state_hash": self._system.state_tracker.state.state_hash
                })
        except Exception as e:
            self._log_error("System initialization failed", e)
            raise
            
    def _validate_system_state(self) -> None:
        """Validate state in system components."""
        try:
            if not hasattr(self._system, 'state_tracker') or self._system.state_tracker.state is None:
                raise StateError("System state not initialized")
                
            state = self._system.state_tracker.state
            
            # Validate required state attributes
            required_attributes = [
                'history', 'dream_memory', 'seen_prompts', 'token_map',
                'temperament_score', 'confidence_history', 'curiosity'
            ]
            
            missing_attributes = [attr for attr in required_attributes 
                                if not hasattr(state, attr)]
            if missing_attributes:
                raise StateError(f"Missing required state attributes: {missing_attributes}")
                
            # Validate conversation ID consistency
            if state.history.conversation_id != self.state.history.conversation_id:
                self._log_event("state_sync_mismatch", {
                    "orchestrator_hash": self.state.state_hash,
                    "system_hash": self._system.state_tracker.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                }, level="warning")
                # Align conversation IDs
                state.history.conversation_id = self.state.history.conversation_id
                
            # Validate state data structures
            if not hasattr(state.history, 'conversation_id'):
                raise StateError("State history missing conversation_id")
                
            if not isinstance(state.dream_memory, deque):
                raise StateError("State dream_memory must be a deque")
                
            if not isinstance(state.seen_prompts, deque):
                raise StateError("State seen_prompts must be a deque")
                
            if not isinstance(state.confidence_history, deque):
                raise StateError("State confidence_history must be a deque")
                
            # Validate device consistency
            if state.device != self.device:
                raise StateError(f"State device {state.device} mismatches orchestrator device {self.device}")
                
            self._log_event("state_validated", {
                "state_hash": state.state_hash,
                "conversation_id": state.history.conversation_id,
                "dream_memory_length": len(state.dream_memory),
                "seen_prompts_count": len(state.seen_prompts),
                "confidence_history_length": len(state.confidence_history),
                "device": str(state.device)
            })
            
        except Exception as e:
            self._log_error("State validation failed", e)
            raise

    def run(self) -> None:
        """Run the SOVL system in the appropriate mode."""
        try:
            # Validate state consistency before running
            if self._system is None:
                raise RuntimeError("System not initialized")
                
            if self.state.state_hash != self._system.state_tracker.state.state_hash:
                self._log_event("state_mismatch_before_run", {
                    "orchestrator_hash": self.state.state_hash,
                    "system_hash": self._system.state_tracker.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                }, level="warning")
                # Attempt to sync states
                self._sync_state_to_system()
            
            # Run in CLI mode by default
            run_cli(self.config_manager)
            
        except Exception as e:
            self._log_error("System execution failed", e)
            self.error_handler.handle_generic_error(
                error=e,
                context="system_execution",
                fallback_action=lambda: self._handle_execution_failure()
            )
            raise

    def shutdown(self) -> None:
        """Shutdown the system and save state."""
        try:
            if self._system is not None:
                # Save final state
                self._system.state_tracker.state.save_state()
                self._log_event("system_shutdown")
        except Exception as e:
            self._log_error("System shutdown failed", e)
            self.error_handler.handle_generic_error(
                error=e,
                context="system_shutdown",
                fallback_action=lambda: self._emergency_shutdown()
            )
            raise

    def _handle_execution_failure(self) -> None:
        """Handle system execution failure with recovery actions."""
        try:
            # Attempt to save state
            if self._system and hasattr(self._system, 'state_tracker'):
                self._system.state_tracker.state.save_state()
            
            # Log failure details
            self._log_event("execution_failure_handled", {
                "state_saved": self._system is not None,
                "timestamp": time.time()
            })
        except Exception as e:
            self._log_error("Failed to handle execution failure", e)

    def _emergency_shutdown(self) -> None:
        """Perform emergency shutdown procedures."""
        try:
            # Force cleanup of resources
            self._cleanup_resources()
            
            # Log emergency shutdown
            self._log_event("emergency_shutdown", {
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"Emergency shutdown failed: {str(e)}")

    def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            # Release GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Close any open file handles
            if hasattr(self, 'logger'):
                self.logger.close()
        except Exception as e:
            print(f"Resource cleanup failed: {str(e)}")

    def _validate_configs(self):
        """Validate all configurations."""
        try:
            # Get validated configs from ConfigHandler
            self.curiosity_config = self.config_handler.curiosity_config
            self.controls_config = self.config_handler.controls_config
            
            # Log final configuration state
            self.logger.record_event(
                event_type="config_validation",
                message="Using validated configurations from ConfigHandler",
                level="info",
                additional_info={
                    "curiosity_config": self.curiosity_config,
                    "controls_config": {k: v for k, v in self.controls_config.items() 
                                     if k.startswith(("curiosity_", "enable_curiosity"))}
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to get validated configurations",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "error": str(e)
                }
            )
            raise

# Main block
if __name__ == "__main__":
    orchestrator = SOVLOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        orchestrator.logger.log_error(
            error_msg="Error running SOVL system",
            error_type="system_execution_error",
            stack_trace=traceback.format_exc(),
            additional_info={"error": str(e)}
        )
        raise
    finally:
        orchestrator.shutdown()
