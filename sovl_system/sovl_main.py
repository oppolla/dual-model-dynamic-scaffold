from typing import Optional, Any, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import time
import random
import bitsandbytes as bnb
import json
import contextlib
from collections import deque
import traceback
import os
from threading import Lock
from sovl_curiosity import CuriosityManager, CuriosityState
from sovl_logger import Logger
from sovl_io import load_training_data, validate_quantization_mode, InsufficientDataError
from sovl_state import SOVLState, ConversationHistory
from sovl_trainer import TrainingConfig, SOVLTrainer
from sovl_config import ConfigManager
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager, CrossAttentionLayer, ScaffoldTokenMapper
from sovl_processor import LogitsProcessor
from sovl_utils import (
    calculate_confidence,
    detect_repetitions,
)
from sovl_temperament import TemperamentConfig, TemperamentSystem
from sovl_memory import MemoryManager
from sovl_manager import ModelManager
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner
from sovl_error import ErrorHandler
from sovl_state_manager import StateManager
from sovl_logging import LoggingManager
import logging
from sovl_training_cycle import TrainingCycleManager
from sovl_plugin import PluginManager
import sys

# Remove sovl_conductor import and use TYPE_CHECKING for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

def calculate_confidence_score(logits, generated_ids) -> float:
    """Calculate confidence score for generated tokens."""
    try:
        processor = LogitsProcessor(logits)
        return processor.calculate_confidence(generated_ids)
    except Exception as e:
        print(f"Confidence score error: {str(e)} - Using default 0.5")
        return 0.5

class SystemContext:
    """Holds shared resources like logger, device, and config manager."""
    
    def __init__(self, config_path: str):
        """
        Initialize system context with configuration and core components.
        
        Initialization order:
        1. Initialize device
        2. Create temporary logger for ConfigManager initialization
        3. Initialize ConfigManager with temporary logger
        4. Initialize LoggingManager with ConfigManager
        5. Update ConfigManager's logger with LoggingManager's logger
        6. Initialize remaining components
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            SystemInitializationError: If any critical component fails to initialize
        """
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Create temporary logger with basic configuration
            temp_logger = Logger(LoggerConfig(
                log_file="sovl_init.log",
                max_size_mb=1,
                compress_old=False,
                max_in_memory_logs=100
            ))
            
            # Initialize ConfigManager with temporary logger
            self.config_manager = ConfigManager(config_path, temp_logger)
            
            # Get logging configuration from ConfigManager
            log_dir = self.config_manager.get("logging_config.log_dir", "logs")
            system_log_file = self.config_manager.get("logging_config.log_file", "sovl_logs.jsonl")
            debug_log_file = self.config_manager.get("logging_config.debug_log_file", "sovl_debug.log")
            
            # Initialize LoggingManager with ConfigManager and aligned configuration
            self.logging_manager = LoggingManager(
                config_manager=self.config_manager,
                log_dir=log_dir,
                system_log_file=system_log_file,
                debug_log_file=debug_log_file
            )
            
            # Get both system and debug logger instances
            self.logger = self.logging_manager.get_logger("system")
            self.debug_logger = self.logging_manager.get_logger("debug")
            
            # Update ConfigManager's logger with the proper logger
            self.config_manager.logger = self.logger
            
            # Log successful initialization
            self.logger.record_event(
                event_type="system_initialization",
                message="System context initialized successfully",
                level="info",
                additional_info={
                    "device": str(self.device),
                    "config_path": config_path,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "log_dir": log_dir,
                    "system_log_file": system_log_file,
                    "debug_log_file": debug_log_file
                }
            )
            
            # Log debug information
            self.debug_logger.record_event(
                event_type="system_initialization_debug",
                message="System context initialized with debug logging enabled",
                level="debug",
                additional_info={
                    "device": str(self.device),
                    "config_path": config_path,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "config_hash": self.config_manager._last_config_hash,
                    "log_dir": log_dir,
                    "system_log_file": system_log_file,
                    "debug_log_file": debug_log_file
                }
            )
            
        except Exception as e:
            # Log initialization failure to both temporary and stderr
            error_msg = f"Failed to initialize system context: {str(e)}"
            stack_trace = traceback.format_exc()
            
            # Try to log to temporary logger if it exists
            if 'temp_logger' in locals():
                temp_logger.record_event(
                    event_type="system_initialization_error",
                    message=error_msg,
                    level="error",
                    additional_info={
                        "stack_trace": stack_trace,
                        "config_path": config_path
                    }
                )
            
            # Always print to stderr
            print(f"CRITICAL: {error_msg}", file=sys.stderr)
            print(f"Stack trace:\n{stack_trace}", file=sys.stderr)
            
            # Raise a custom exception with full context
            raise SystemInitializationError(
                message=error_msg,
                config_path=config_path,
                stack_trace=stack_trace
            ) from e

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class ConfigHandler:
    """Manages configuration validation and access."""
    def __init__(self, context: SystemContext):
        self.context = context
        self.core_config = context.config_manager.get_section("core_config")
        self.training_config = context.config_manager.get_section("training_config")
        self.curiosity_config = context.config_manager.get_section("curiosity_config")
        self.cross_attn_config = context.config_manager.get_section("cross_attn_config")
        self.controls_config = context.config_manager.get_section("controls_config")
        self.lora_config = context.config_manager.get_section("lora_config")
        
        # Validate curiosity-related configurations
        self._validate_curiosity_configs()

    def _validate_curiosity_configs(self):
        """Validate and synchronize curiosity-related configurations."""
        try:
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
            missing_keys = [key for key in required_keys if key not in self.curiosity_config]
            if missing_keys:
                self.context.logger.record_event(
                    event_type="config_validation",
                    message="Missing keys in curiosity_config",
                    level="warning",
                    additional_info={
                        "missing_keys": missing_keys,
                        "default_values": {k: required_keys[k] for k in missing_keys}
                    }
                )
                # Add missing keys with default values
                for key in missing_keys:
                    self.curiosity_config[key] = required_keys[key]
            
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
                if controls_key in self.controls_config and curiosity_key in self.curiosity_config:
                    if self.controls_config[controls_key] != self.curiosity_config[curiosity_key]:
                        mismatches.append({
                            "controls_key": controls_key,
                            "curiosity_key": curiosity_key,
                            "controls_value": self.controls_config[controls_key],
                            "curiosity_value": self.curiosity_config[curiosity_key]
                        })
            
            if mismatches:
                self.context.logger.record_event(
                    event_type="config_validation",
                    message="Configuration mismatches between controls_config and curiosity_config",
                    level="warning",
                    additional_info={"mismatches": mismatches}
                )
                # Align controls_config with curiosity_config
                for mismatch in mismatches:
                    self.controls_config[mismatch["controls_key"]] = self.curiosity_config[mismatch["curiosity_key"]]
            
            # Log final configuration state
            self.context.logger.record_event(
                event_type="config_validation",
                message="Curiosity configuration validation complete",
                level="info",
                additional_info={
                    "curiosity_config": self.curiosity_config,
                    "controls_config": {k: v for k, v in self.controls_config.items() if k.startswith("curiosity_")}
                }
            )
            
        except Exception as e:
            self.context.logger.record_event(
                event_type="config_validation_error",
                message="Failed to validate curiosity configurations",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def validate(self, model_config: Any = None) -> bool:
        """
        Validate the configuration, propagating any validation errors.
        
        Args:
            model_config: Optional model configuration for layer validation
            
        Returns:
            bool: True if validation succeeds
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            # First validate basic configuration without model context
            self.context.config_manager.validate_or_raise(None)
            return True
        except ValueError as e:
            # Log the error and re-raise to prevent silent failures
            self.context.logger.record_event(
                event_type="config_validation_error",
                message="Configuration validation failed",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def validate_with_model(self, model_config: Any) -> bool:
        """
        Validate configuration with model-specific checks.
        
        Args:
            model_config: Model configuration for layer validation
            
        Returns:
            bool: True if validation succeeds
            
        Raises:
            ValueError: If configuration validation fails
        """
        try:
            # Validate with model context for layer-specific checks
            self.context.config_manager.validate_or_raise(model_config)
            return True
        except ValueError as e:
            # Log the error and re-raise to prevent silent failures
            self.context.logger.record_event(
                event_type="config_validation_error",
                message="Model-specific configuration validation failed",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

class ModelLoader:
    """Handles model loading and token mapping."""
    
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        self.context = context
        self.config_handler = config_handler
        self.base_model = None
        self.scaffold_model = None
        self.base_tokenizer = None
        self.scaffold_tokenizer = None
        self.token_mapper = None
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize model components and token mapper."""
        try:
            # Load base model and tokenizer
            self.base_model, self.base_tokenizer = self._load_base_model()
            
            # Initialize token mapper
            self.token_mapper = ScaffoldTokenMapper(
                base_tokenizer=self.base_tokenizer,
                scaffold_tokenizer=self.scaffold_tokenizer,
                logger=self.context.logger
            )
            
            # Log initialization
            self.context.logger.record_event(
                event_type="model_loader_initialized",
                message="Model loader initialized successfully",
                level="info",
                additional_info={
                    "base_model_device": str(self.base_model.device),
                    "scaffold_model_device": str(self.scaffold_model.device) if self.scaffold_model else None,
                    "base_vocab_size": len(self.base_tokenizer),
                    "scaffold_vocab_size": len(self.scaffold_tokenizer) if self.scaffold_tokenizer else None
                }
            )
        except Exception as e:
            self.context.logger.record_event(
                event_type="model_loader_init_failed",
                message="Failed to initialize model loader",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def sync_token_map(self, state: SOVLState) -> None:
        """
        Synchronize token maps between ScaffoldTokenMapper and SOVLState.
        
        Args:
            state: Current SOVLState instance
            
        Raises:
            ValueError: If token map validation fails
        """
        try:
            # Validate state token map
            self._validate_token_map(state.token_map)
            
            # Update token mapper with state's token map
            self.token_mapper.update_token_map(state.token_map)
            
            # Log synchronization
            self.context.logger.record_event(
                event_type="token_map_synced",
                message="Token maps synchronized successfully",
                level="info",
                additional_info={
                    "token_map_size": len(state.token_map),
                    "base_vocab_size": len(self.base_tokenizer),
                    "scaffold_vocab_size": len(self.scaffold_tokenizer) if self.scaffold_tokenizer else None,
                    "state_hash": state.state_hash
                }
            )
        except Exception as e:
            self.context.logger.record_event(
                event_type="token_map_sync_failed",
                message="Failed to synchronize token maps",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                    "state_hash": state.state_hash
                }
            )
            raise ValueError(f"Token map synchronization failed: {str(e)}")

    def _validate_token_map(self, token_map: Dict[int, List[int]]) -> None:
        """
        Validate token map against tokenizer vocabularies.
        
        Args:
            token_map: Token mapping dictionary to validate
            
        Raises:
            ValueError: If validation fails
        """
        try:
            base_vocab_size = len(self.base_tokenizer)
            scaffold_vocab_size = len(self.scaffold_tokenizer) if self.scaffold_tokenizer else None
            
            # Validate base token IDs
            for base_id in token_map.keys():
                if not 0 <= base_id < base_vocab_size:
                    raise ValueError(f"Invalid base token ID {base_id} exceeds vocab size {base_vocab_size}")
            
            # Validate scaffold token IDs
            if scaffold_vocab_size is not None:
                for scaffold_ids in token_map.values():
                    for scaffold_id in scaffold_ids:
                        if not 0 <= scaffold_id < scaffold_vocab_size:
                            raise ValueError(f"Invalid scaffold token ID {scaffold_id} exceeds vocab size {scaffold_vocab_size}")
            
            # Log validation
            self.context.logger.record_event(
                event_type="token_map_validated",
                message="Token map validation successful",
                level="info",
                additional_info={
                    "token_map_size": len(token_map),
                    "base_vocab_size": base_vocab_size,
                    "scaffold_vocab_size": scaffold_vocab_size
                }
            )
        except Exception as e:
            self.context.logger.record_event(
                event_type="token_map_validation_failed",
                message="Token map validation failed",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise ValueError(f"Token map validation failed: {str(e)}")

    def inject_cross_attention(self) -> None:
        """Inject cross-attention layers into the scaffold model."""
        try:
            # Ensure models are on correct device
            self.base_model = self.base_model.to(self.context.device)
            self.scaffold_model = self.scaffold_model.to(self.context.device)
            
            # Sync token maps before injection
            if hasattr(self.context, 'state_tracker'):
                self.sync_token_map(self.context.state_tracker.state)
            
            # Inject cross-attention layers
            self._inject_cross_attention_layers()
            
            # Log successful injection
            self.context.logger.record_event(
                event_type="cross_attention_injected",
                message="Cross-attention layers injected successfully",
                level="info",
                additional_info={
                    "base_model_device": str(self.base_model.device),
                    "scaffold_model_device": str(self.scaffold_model.device),
                    "token_map_size": len(self.token_mapper.token_map) if self.token_mapper else None
                }
            )
        except Exception as e:
            self.context.logger.record_event(
                event_type="cross_attention_injection_failed",
                message="Failed to inject cross-attention layers",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

class StateTracker:
    """Tracks and manages the system's state."""
    
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        """Initialize the state tracker with context and configuration."""
        self.context = context
        self.config_handler = config_handler
        self.state_manager = StateManager(
            config_manager=context.config_manager,
            logger=context.logger,
            device=context.device
        )
        # Auto-initialize state
        self.state = self.state_manager.initialize_state()
        self._log_event("state_tracker_initialized", {
            "state_hash": self.state.state_hash,
            "conversation_id": self.state.history.conversation_id
        })
        
    def _log_event(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.context.logger.record_event(
            event_type=f"state_{event}",
            message=f"State event: {event}",
            level="info",
            additional_info=data
        )
        
    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with stack trace."""
        self.context.logger.log_error(
            error_msg=message,
            error_type="state_error",
            stack_trace=traceback.format_exc(),
            additional_info={"error": str(error)}
        )
        
    def load_state(self) -> None:
        """Load state from file or initialize new state if not found."""
        try:
            # Only load if state path exists, otherwise keep initialized state
            state_path = self.config_handler.config_manager.get("state_config.state_path")
            if state_path and os.path.exists(state_path):
                self.state = self.state_manager.load_state(state_path)
                self._log_event("state_loaded", {
                    "state_path": state_path,
                    "state_hash": self.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                })
            else:
                self._log_event("state_initialized", {
                    "state_hash": self.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                })
        except Exception as e:
            self._log_error("State loading failed", e)
            raise StateError(f"Failed to load state: {str(e)}")
            
    def save_state(self) -> None:
        """Save current state to file."""
        try:
            state_path = self.config_handler.config_manager.get("state_config.state_path")
            if state_path:
                self.state_manager.save_state(state_path)
                self._log_event("state_saved", {
                    "state_path": state_path,
                    "state_hash": self.state.state_hash
                })
        except Exception as e:
            self._log_error("State saving failed", e)
            raise StateError(f"Failed to save state: {str(e)}")
            
    def get_state(self) -> SOVLState:
        """Get the current state instance."""
        if self.state is None:
            raise StateError("State not initialized")
        return self.state

class ErrorManager:
    """Handles errors and recovery across components."""
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        self.context = context
        self.state_tracker = state_tracker
        self.error_handler = ErrorHandler(
            config_manager=context.config_manager,
            logger=context.logger,
            error_log_file="sovl_errors.jsonl",
            max_error_log_size_mb=10,
            compress_old=True,
            state=state_tracker.state
        )

    def handle_generation_error(self, error: Exception, prompt: str) -> str:
        self.context.logger.log_error(
            error_msg=f"Generation failed: {str(error)}",
            error_type="generation_error",
            stack_trace=traceback.format_exc(),
            conversation_id=self.state_tracker.state.conversation_id,
            state_hash=self.state_tracker.state.get_state_hash(),
            additional_info={"prompt": prompt}
        )
        return self.error_handler.handle_generation_error(error, prompt)

    def handle_curiosity_error(self, error: Exception, context: str) -> Optional[str]:
        self.context.logger.log_error(
            error_msg=f"Curiosity error: {str(error)}",
            error_type="curiosity_error",
            stack_trace=traceback.format_exc(),
            conversation_id=self.state_tracker.state.conversation_id,
            state_hash=self.state_tracker.state.get_state_hash(),
            additional_info={"context": context}
        )
        return self.error_handler.handle_curiosity_error(error, context)

class MemoryMonitor:
    """Monitors system memory health."""
    def __init__(self, context: SystemContext):
        self.context = context
        self.memory_manager = MemoryManager(
            config_manager=context.config_manager,
            device=context.device,
            logger=context.logger
        )

    def check_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None) -> bool:
        """
        Check memory health and log the results.

        Args:
            model_size: Size of the model in bytes
            trainer: Optional trainer instance for additional memory stats

        Returns:
            bool: True if memory health is good, False otherwise
        """
        try:
            # Get memory health status
            is_healthy = self.memory_manager.check_memory_health(model_size, trainer)
            
            # Log memory health check
            self.context.logger.log_memory_health(
                model_size=model_size,
                trainer=trainer,
                health_status="healthy" if is_healthy else "unhealthy",
                device=self.context.device
            )
            
            return is_healthy
        except Exception as e:
            # Log error if memory check fails
            self.context.logger.log_error(
                error_msg=f"Memory health check failed: {str(e)}",
                error_type="memory_error",
                stack_trace=traceback.format_exc()
            )
            return False

class TemperamentAdjuster:
    """Modulates model behavior based on temperament."""
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        self.context = context
        self.state_tracker = state_tracker
        self.temperament_system = TemperamentSystem.create_from_config(
            config_manager=context.config_manager,
            logger=context.logger,
            device=context.device
        )
        self.context.logger.record({
            "event": "temperament_initialized",
            "timestamp": time.time(),
            "conversation_id": self.state_tracker.state.conversation_id
        })

    def update_temperament(self, curiosity_manager: Optional[CuriosityManager] = None):
        """Update temperament based on current state and optional curiosity manager."""
        try:
            if curiosity_manager is None:
                self.context.logger.record_event(
                    event_type="temperament_update_skipped",
                    message="No curiosity manager provided, updating temperament without curiosity influence",
                    level="warning",
                    additional_info={
                        "conversation_id": self.state_tracker.state.conversation_id,
                        "state_hash": self.state_tracker.state.get_state_hash()
                    }
                )
                # Update temperament without curiosity influence
                self.temperament_system.update_from_state(
                    state=self.state_tracker.state,
                    curiosity_manager=None
                )
            else:
                # Update with curiosity influence
                self.temperament_system.update_from_state(
                    state=self.state_tracker.state,
                    curiosity_manager=curiosity_manager
                )
        except Exception as e:
            self.context.logger.record({
                "error": f"Failed to update temperament: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "has_curiosity_manager": curiosity_manager is not None
            })
            raise

class CuriosityEngine:
    """Generates curiosity-driven questions."""
    def __init__(self, context: SystemContext, model_loader: ModelLoader, 
                 state_tracker: StateTracker, error_manager: ErrorManager):
        self.context = context
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        
        # Check if curiosity is enabled
        self.enable_curiosity = self.context.config_manager.get("curiosity_config.enable_curiosity", True)
        
        # Initialize curiosity manager if enabled
        if self.enable_curiosity:
            self.curiosity_manager = CuriosityManager(
                config=self.context.config_manager.get_section("curiosity_config"),
                logger=context.logger,
                device=context.device,
                state=state_tracker.state
            )
            self.context.logger.record_event(
                event_type="curiosity_initialized",
                message="Curiosity manager initialized successfully",
                level="info",
                additional_info={
                    "config": {
                        "weight_ignorance": self.context.config_manager.get("curiosity_config.weight_ignorance", 0.7),
                        "weight_novelty": self.context.config_manager.get("curiosity_config.weight_novelty", 0.3),
                        "base_temperature": self.context.config_manager.get("curiosity_config.base_temperature", 1.1)
                    }
                }
            )
        else:
            self.curiosity_manager = None
            self.context.logger.record_event(
                event_type="curiosity_disabled",
                message="Curiosity is disabled in configuration",
                level="info",
                additional_info={
                    "reason": "enable_curiosity is False in curiosity_config"
                }
            )

    def generate_question(self, context_str: str = "", spontaneous: bool = False) -> Optional[str]:
        """Generate a curiosity-driven question if enabled."""
        if not self.enable_curiosity or not self.curiosity_manager:
            self.context.logger.record_event(
                event_type="curiosity_skipped",
                message="Question generation skipped - curiosity is disabled",
                level="info",
                additional_info={
                    "context": context_str,
                    "spontaneous": spontaneous
                }
            )
            return None

        try:
            question = self.curiosity_manager.generate_question(
                context=context_str,
                spontaneous=spontaneous,
                model=self.model_loader.base_model,
                tokenizer=self.model_loader.base_tokenizer
            )
            if question:
                self.context.logger.record_event(
                    event_type="question_generated",
                    message="Curiosity question generated successfully",
                    level="info",
                    additional_info={
                        "question": question,
                        "spontaneous": spontaneous,
                        "context_length": len(context_str)
                    }
                )
            return question
        except Exception as e:
            return self.error_manager.handle_curiosity_error(e, "question_generation")

class CycleTrainer:
    """Manages regular training cycles."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler, 
                 model_loader: ModelLoader, state_tracker: StateTracker):
        self.context = context
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        
        # Ensure models are on the correct device
        self.model_loader.base_model = self.model_loader.base_model.to(self.context.device)
        self.model_loader.scaffolds[0] = self.model_loader.scaffolds[0].to(self.context.device)
        
        # Initialize trainer with models on correct device
        self.trainer = self._initialize_trainer()
        self.training_cycle_manager = TrainingCycleManager(
            trainer=self.trainer,
            config_manager=context.config_manager,
            logger=context.logger
        )
        
        # Log device information
        self.context.logger.record_event(
            event_type="cycle_trainer_initialized",
            message="Cycle trainer initialized with models on correct device",
            level="info",
            additional_info={
                "base_model_device": next(self.model_loader.base_model.parameters()).device,
                "scaffold_model_device": next(self.model_loader.scaffolds[0].parameters()).device,
                "target_device": str(self.context.device)
            }
        )

    def _initialize_trainer(self) -> SOVLTrainer:
        # Get base training config
        training_config = TrainingConfig(
            learning_rate=self.config_handler.training_config.get("learning_rate", 0.0003),
            grad_accum_steps=self.config_handler.training_config.get("accumulation_steps", 4),
            weight_decay=0.01,
            total_steps=1000,
            max_grad_norm=1.0,
            use_amp=(self.context.device.type == "cuda"),
            max_patience=self.config_handler.training_config.get("max_patience", 2),
            batch_size=self.config_handler.training_config.get("batch_size", 1),
            max_epochs=self.config_handler.training_config.get("train_epochs", 3),
            validate_every_n_steps=100,
            checkpoint_interval=1000,
            checkpoint_path="checkpoints/sovl_trainer",
            scheduler_type="linear",
            cosine_min_lr=1e-6,
            warmup_ratio=0.1,
            dropout_rate=self.config_handler.lora_config.get("lora_dropout", 0.1),
            max_seq_length=self.config_handler.training_config.get("max_seq_length", 128),
            metrics_to_track=["loss", "accuracy", "confidence"],
            enable_gestation=self.config_handler.controls_config.get("enable_gestation", True),
            enable_sleep_training=self.config_handler.controls_config.get("enable_sleep_training", True),
            enable_lifecycle_weighting=self.config_handler.controls_config.get("enable_lifecycle_weighting", True),
            lifecycle_capacity_factor=self.config_handler.training_config.get("lifecycle_capacity_factor", 0.01),
            lifecycle_curve=self.config_handler.training_config.get("lifecycle_curve", "sigmoid_linear"),
            sleep_conf_threshold=self.config_handler.controls_config.get("sleep_conf_threshold", 0.7),
            sleep_log_min=self.config_handler.controls_config.get("sleep_log_min", 10),
            accumulation_steps=self.config_handler.training_config.get("accumulation_steps", 4),
            exposure_gain_eager=self.config_handler.training_config.get("exposure_gain_eager", 3),
            exposure_gain_default=self.config_handler.training_config.get("exposure_gain_default", 2),
            dream_memory_weight=self.config_handler.controls_config.get("dream_memory_weight", 0.1),
            enable_dreaming=self.config_handler.controls_config.get("enable_dreaming", True),
            repetition_n=3,
            sigmoid_scale=self.config_handler.training_config.get("sigmoid_scale", 0.5),
            sigmoid_shift=self.config_handler.training_config.get("sigmoid_shift", 5.0)
        )

        # Handle curiosity configuration based on enable_curiosity
        if self.config_handler.curiosity_config.get("enable_curiosity", True):
            training_config.curiosity_weight_ignorance = self.config_handler.curiosity_config.get("weight_ignorance", 0.7)
            training_config.curiosity_weight_novelty = self.config_handler.curiosity_config.get("weight_novelty", 0.3)
            training_config.curiosity_pressure_threshold = self.config_handler.curiosity_config.get("pressure_threshold", 0.7)
            training_config.curiosity_pressure_drop = self.config_handler.curiosity_config.get("pressure_drop", 0.3)
            training_config.curiosity_novelty_threshold_spontaneous = self.config_handler.curiosity_config.get("novelty_threshold_spontaneous", 0.9)
            training_config.curiosity_novelty_threshold_response = self.config_handler.curiosity_config.get("novelty_threshold_response", 0.8)
            training_config.curiosity_silence_threshold = self.config_handler.curiosity_config.get("silence_threshold", 20.0)
            training_config.curiosity_question_cooldown = self.config_handler.curiosity_config.get("question_cooldown", 60.0)
            training_config.curiosity_queue_maxlen = self.config_handler.curiosity_config.get("queue_maxlen", 10)
            training_config.curiosity_max_new_tokens = self.config_handler.curiosity_config.get("max_new_tokens", 8)
            training_config.curiosity_base_temperature = self.config_handler.curiosity_config.get("base_temperature", 1.1)
            training_config.curiosity_temperament_influence = self.config_handler.curiosity_config.get("temperament_influence", 0.4)
            training_config.curiosity_top_k = self.config_handler.curiosity_config.get("top_k", 30)
        else:
            # Set all curiosity-related parameters to 0 or disabled values
            training_config.curiosity_weight_ignorance = 0.0
            training_config.curiosity_weight_novelty = 0.0
            training_config.curiosity_pressure_threshold = 0.0
            training_config.curiosity_pressure_drop = 0.0
            training_config.curiosity_novelty_threshold_spontaneous = 0.0
            training_config.curiosity_novelty_threshold_response = 0.0
            training_config.curiosity_silence_threshold = 0.0
            training_config.curiosity_question_cooldown = 0.0
            training_config.curiosity_queue_maxlen = 0
            training_config.curiosity_max_new_tokens = 0
            training_config.curiosity_base_temperature = 0.0
            training_config.curiosity_temperament_influence = 0.0
            training_config.curiosity_top_k = 0

        def loss_fn(logits, labels):
            # Ensure inputs are on the correct device
            logits = logits.to(self.context.device)
            labels = labels.to(self.context.device)
            mask = labels != -100
            return F.cross_entropy(
                logits.view(-1, logits.size(-1))[mask.view(-1)],
                labels.view(-1)[mask.view(-1)],
                ignore_index=-100
            )

        # Initialize trainer with models on correct device
        trainer = SOVLTrainer(
            model=self.model_loader.scaffolds[0],
            config=training_config,
            device=self.context.device,
            loss_fn=loss_fn,
            logger=self.context.logger,
            memory_lock=Lock(),
            tokenizer=self.model_loader.base_tokenizer,
            state=self.state_tracker.state
        )
        
        # Log trainer initialization with device information
        self.context.logger.record_event(
            event_type="trainer_initialized",
            message="Trainer initialized with models on correct device",
            level="info",
            additional_info={
                "model_device": next(trainer.model.parameters()).device,
                "target_device": str(self.context.device),
                "use_amp": training_config.use_amp
            }
        )
        
        return trainer

class GestationTrainer:
    """Manages gestation-specific training."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler, 
                 model_loader: ModelLoader, state_tracker: StateTracker):
        self.context = context
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        # Gestation-specific logic can be added here

    def handle_gestation_complete(self, batch_size: int, avg_loss: float):
        self.state_tracker.update_gestation_metrics(batch_size, avg_loss)
        self.context.logger.record({
            "event": "gestation_complete_handled",
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time(),
            "conversation_id": self.state_tracker.state.conversation_id,
            "state_hash": self.state_tracker.state.get_state_hash()
        })

class SleepTrainer:
    """Manages sleep training and dream cycles."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler, 
                 model_loader: ModelLoader, state_tracker: StateTracker):
        self.context = context
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.trainer = CycleTrainer(context, config_handler, model_loader, state_tracker).trainer

    def sleep_train(self):
        try:
            log_entries = self.context.logger.read()
            self.trainer.training_cycle_manager.run_sleep_training(log_entries)
            self.context.logger.clear()
        except Exception as e:
            self.context.logger.record({
                "error": f"Sleep training failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "stack_trace": traceback.format_exc()
            })
            raise

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int):
        self.state_tracker.update_dream_metrics(dream_prompt, is_novel, memory_count)
        self.context.logger.record({
            "event": "dream_complete_handled",
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time(),
            "conversation_id": self.state_tracker.state.conversation_id,
            "state_hash": self.state_tracker.state.get_state_hash()
        })

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float):
        self.state_tracker.update_sleep_metrics(batch_size, data_exposure)
        self.context.logger.record({
            "event": "sleep_train_complete_handled",
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": self.state_tracker.state.conversation_id,
            "state_hash": self.state_tracker.state.get_state_hash()
        })

class TrainingManager:
    """Coordinates all training activities."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler, 
                 model_loader: ModelLoader, state_tracker: StateTracker, 
                 error_manager: ErrorManager):
        self.context = context
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        
        # Initialize training data
        self.train_data, self.valid_data = self._load_training_data()
        
        # Initialize PluginManager
        self.plugin_manager = PluginManager(
            config_manager=context.config_manager,
            logger=context.logger,
            state=state_tracker.state
        )
        
        # Initialize components
        self.cycle_trainer = CycleTrainer(context, config_handler, model_loader, state_tracker)
        self.gestation_trainer = GestationTrainer(context, config_handler, model_loader, state_tracker)
        self.sleep_trainer = SleepTrainer(context, config_handler, model_loader, state_tracker)
        self.tuner = SOVLTuner(
            config_manager=context.config_manager,
            logger=context.logger,
            curiosity_manager=None,  # Set after CuriosityEngine
            trainer=self.cycle_trainer.trainer,
            cross_attention_injector=CrossAttentionInjector(context.config_manager, context.logger)
        )

    def _load_training_data(self) -> Tuple[List, List]:
        """
        Load and validate training data from configuration.
        
        Returns:
            Tuple[List, List]: Training and validation data
            
        Raises:
            ValueError: If data loading fails
            InsufficientDataError: If insufficient data is available
        """
        try:
            data_path = self.context.config_manager.get("training_config.data_path", "data/train.json")
            valid_split_ratio = self.context.config_manager.get("core_config.valid_split_ratio", 0.2)
            
            self.context.logger.record_event(
                event_type="data_loading",
                message="Loading training data",
                level="info",
                additional_info={
                    "data_path": data_path,
                    "valid_split_ratio": valid_split_ratio
                }
            )
            
            data = load_training_data(
                path=data_path,
                valid_split_ratio=valid_split_ratio
            )
            
            train_data = data.get("train", [])
            valid_data = data.get("valid", [])
            
            if not train_data:
                raise InsufficientDataError("No training data available")
            if not valid_data:
                raise InsufficientDataError("No validation data available")
            
            self.context.logger.record_event(
                event_type="data_loaded",
                message="Training data loaded successfully",
                level="info",
                additional_info={
                    "train_samples": len(train_data),
                    "valid_samples": len(valid_data)
                }
            )
            
            return train_data, valid_data
            
        except InsufficientDataError as e:
            self.context.logger.record_event(
                event_type="data_error",
                message="Insufficient training data",
                level="error",
                additional_info={
                    "error": str(e)
                }
            )
            raise
        except Exception as e:
            self.context.logger.record_event(
                event_type="data_error",
                message="Failed to load training data",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise ValueError(f"Failed to load training data: {str(e)}")

    def run_training_cycle(self, train_data: Optional[List] = None, valid_data: Optional[List] = None, 
                          epochs: Optional[int] = None, batch_size: Optional[int] = None):
        """
        Run a training cycle with the provided or default data.

        Args:
            train_data: Optional training data to use instead of default
            valid_data: Optional validation data to use instead of default
            epochs: Optional number of epochs to train for
            batch_size: Optional batch size for training

        Raises:
            ValueError: If no training data is available
        """
        # Use provided data or fall back to initialized data
        train_data = train_data or self.train_data
        valid_data = valid_data or self.valid_data

        # Validate data presence
        if not train_data:
            self.context.logger.record_event(
                event_type="training_error",
                message="No training data available for training cycle",
                level="error",
                additional_info={
                    "using_provided_data": train_data is not self.train_data,
                    "valid_data_available": bool(valid_data)
                }
            )
            raise ValueError("No training data available for training cycle")

        # Log training cycle start
        self.context.logger.record_event(
            event_type="training_cycle_start",
            message="Starting training cycle",
            level="info",
            additional_info={
                "train_samples": len(train_data),
                "valid_samples": len(valid_data),
                "epochs": epochs,
                "batch_size": batch_size
            }
        )

        # Execute pre-training hooks with full context
        self.plugin_manager.execute_hook("on_training_step", {
            "batch_size": len(train_data),
            "dry_run": False,
            "train_samples": len(train_data),
            "valid_samples": len(valid_data),
            "epochs": epochs,
            "batch_size": batch_size,
            "state": self.state_tracker.state.to_dict()
        })
        
        try:
            # Run the training cycle
            self.cycle_trainer.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Execute post-training hooks with full context
            self.plugin_manager.execute_hook("on_training_step_complete", {
                "batch_size": len(train_data),
                "result": "success",
                "train_samples": len(train_data),
                "valid_samples": len(valid_data),
                "epochs": epochs,
                "batch_size": batch_size,
                "state": self.state_tracker.state.to_dict()
            })
            
        except Exception as e:
            # Execute error hook if training fails
            self.plugin_manager.execute_hook("on_training_error", {
                "error": str(e),
                "stack_trace": traceback.format_exc(),
                "train_samples": len(train_data),
                "valid_samples": len(valid_data),
                "epochs": epochs,
                "batch_size": batch_size,
                "state": self.state_tracker.state.to_dict()
            })
            
            # Log training error
            self.context.logger.record_event(
                event_type="training_error",
                message="Training cycle failed",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        """Handle completion of a training cycle."""
        self.state_tracker.update_data_exposure(data_exposure)
        
        # Execute training complete hook
        self.plugin_manager.execute_hook("on_training_complete", {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "data_exposure": data_exposure,
            "state": self.state_tracker.state.to_dict()
        })
        
        self.context.logger.record_event(
            event_type="training_complete",
            message="Training cycle completed",
            level="info",
            additional_info={
                "epoch": epoch,
                "avg_loss": avg_loss,
                "data_exposure": data_exposure
            }
        )
