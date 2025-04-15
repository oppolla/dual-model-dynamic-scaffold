from typing import Optional, Any, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager, CrossAttentionLayer
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

        Args:
            config_path: Path to the configuration file
        """
        # Initialize device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize ConfigManager first
            self.config_manager = ConfigManager(config_path)
            
            # Initialize LoggingManager with ConfigManager
            self.logging_manager = LoggingManager(
                config_manager=self.config_manager,
                log_dir="logs",
                system_log_file="sovl_system.log",
                debug_log_file="sovl_debug.log"
            )
            
            # Get both system and debug logger instances
            self.logger = self.logging_manager.get_logger("system")
            self.debug_logger = self.logging_manager.get_logger("debug")
            
            # Log initialization
            self.logger.record_event(
                event_type="system_initialization",
                message="System context initialized",
                level="info",
                additional_info={
                    "device": str(self.device),
                    "config_path": config_path
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
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            )
            
        except Exception as e:
            # If logging isn't available, print error
            print(f"Failed to initialize system context: {str(e)}")
            raise

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

    def validate(self, model_config: Any = None) -> bool:
        try:
            self.context.config_manager.validate_or_raise(model_config)
            return True
        except ValueError as e:
            self.context.logger.record({
                "error": str(e),
                "timestamp": time.time(),
                "conversation_id": "validate"
            })
            return False
        except Exception as e:
            self.context.logger.record({
                "error": f"Unexpected error during config validation: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "validate"
            })
            return False

class ModelLoader:
    """Loads and manages models, tokenizers, and scaffold integration."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        self.context = context
        self.config_handler = config_handler
        self.model_manager = ModelManager(
            config_manager=context.config_manager,
            logger=context.logger,
            device=context.device
        )
        self.base_model = self.model_manager.get_base_model()
        self.scaffolds = [self.model_manager.get_scaffold_model()]
        self.base_tokenizer = self.model_manager.get_base_tokenizer()
        self.scaffold_tokenizer = self.model_manager.get_scaffold_tokenizer()
        self.scaffold_unk_id = self.model_manager.get_scaffold_unk_id()
        self.scaffold_manager = ScaffoldManager(context.config_manager, context.logger)
        self.scaffold_token_mapper = None

    def inject_cross_attention(self):
        try:
            injector = CrossAttentionInjector(
                config_manager=self.context.config_manager,
                logger=self.context.logger
            )
            injector.inject_cross_attention(
                model=self.base_model,
                scaffold_model=self.scaffolds[0],
                core_config=self.config_handler.core_config,
                cross_attn_config=self.config_handler.cross_attn_config,
                lora_config=self.config_handler.lora_config,
                token_map=self.scaffold_token_mapper,
                device=self.context.device
            )
        except Exception as e:
            self.context.logger.record({
                "error": f"Cross-attention injection failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": "init"
            })
            raise

class StateTracker:
    """Centralizes state management and conversation history."""
    def __init__(self, context: SystemContext, config_handler: ConfigHandler):
        self.context = context
        self.state_manager = StateManager(
            config_manager=config_handler.config_manager,
            logger=context.logger,
            device=context.device
        )
        self.state = self.state_manager.load_state()

    def update_conversation(self, prompt: str, response: str):
        self.state.conversation_history.append({"prompt": prompt, "response": response})

    def update_data_exposure(self, data_exposure: float):
        self.state.update_data_exposure(data_exposure)

    def update_gestation_metrics(self, batch_size: int, avg_loss: float):
        self.state.update_gestation_metrics(batch_size, avg_loss)

    def update_dream_metrics(self, dream_prompt: str, is_novel: bool, memory_count: int):
        self.state.update_dream_metrics(dream_prompt, is_novel, memory_count)

    def update_sleep_metrics(self, batch_size: int, data_exposure: float):
        self.state.update_sleep_metrics(batch_size, data_exposure)

    def update_curiosity_metrics(self, question: str, score: float, spontaneous: bool, answered: bool):
        if self.state.curiosity:
            self.state.curiosity.update_metrics(
                question=question,
                score=score,
                spontaneous=spontaneous,
                answered=answered,
                conversation_id=self.state.conversation_id,
                state_hash=self.state.get_state_hash()
            )

    def load_state(self):
        try:
            self.state = self.state_manager.load_state()
            self.context.logger.record({
                "event": "state_loaded",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id,
                "state_hash": self.state.get_state_hash()
            })
        except Exception as e:
            self.context.logger.record({
                "error": f"State loading failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            raise

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
        try:
            self.temperament_system.update_from_state(
                state=self.state_tracker.state,
                curiosity_manager=curiosity_manager
            )
        except Exception as e:
            self.context.logger.record({
                "error": f"Failed to update temperament: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state_tracker.state.conversation_id
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
        self.curiosity_manager = (
            CuriosityManager(
                config=self.context.config_manager.get_section("curiosity_config"),
                logger=context.logger,
                device=context.device,
                state=state_tracker.state
            ) if self.context.config_manager.get_section("curiosity_config").get("enable_curiosity", True)
            else None
        )

    def generate_question(self, context_str: str = "", spontaneous: bool = False) -> Optional[str]:
        if not self.curiosity_manager:
            self.context.logger.record({
                "warning": "Curiosity manager not initialized",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id
            })
            return None
        try:
            question = self.curiosity_manager.generate_question(
                context=context_str,
                spontaneous=spontaneous,
                model=self.model_loader.base_model,
                tokenizer=self.model_loader.base_tokenizer
            )
            if question:
                self.context.logger.log_curiosity_event(
                    event_type="question_generated",
                    question=question,
                    spontaneous=spontaneous,
                    conversation_id=self.state_tracker.state.conversation_id,
                    state_hash=self.state_tracker.state.get_state_hash()
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
        self.trainer = self._initialize_trainer()
        self.training_cycle_manager = TrainingCycleManager(
            trainer=self.trainer,
            config_manager=context.config_manager,
            logger=context.logger
        )

    def _initialize_trainer(self) -> SOVLTrainer:
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
            sigmoid_shift=self.config_handler.training_config.get("sigmoid_shift", 5.0),
            curiosity_weight_ignorance=self.config_handler.curiosity_config.get("weight_ignorance", 0.7),
            curiosity_weight_novelty=self.config_handler.curiosity_config.get("weight_novelty", 0.3),
            curiosity_pressure_threshold=self.config_handler.curiosity_config.get("pressure_threshold", 0.7),
            curiosity_pressure_drop=self.config_handler.curiosity_config.get("pressure_drop", 0.3),
            curiosity_novelty_threshold_spontaneous=self.config_handler.curiosity_config.get("novelty_threshold_spontaneous", 0.9),
            curiosity_novelty_threshold_response=self.config_handler.curiosity_config.get("novelty_threshold_response", 0.8),
            curiosity_silence_threshold=self.config_handler.curiosity_config.get("silence_threshold", 20.0),
            curiosity_question_cooldown=self.config_handler.curiosity_config.get("question_cooldown", 60.0),
            curiosity_queue_maxlen=self.config_handler.curiosity_config.get("queue_maxlen", 10),
            curiosity_max_new_tokens=self.config_handler.curiosity_config.get("max_new_tokens", 8),
            curiosity_base_temperature=self.config_handler.curiosity_config.get("base_temperature", 1.1),
            curiosity_temperament_influence=self.config_handler.curiosity_config.get("temperament_influence", 0.4),
            curiosity_top_k=self.config_handler.curiosity_config.get("top_k", 30)
        )
        def loss_fn(logits, labels):
            mask = labels != -100
            return F.cross_entropy(
                logits.view(-1, logits.size(-1))[mask.view(-1)],
                labels.view(-1)[mask.view(-1)],
                ignore_index=-100
            )
        return SOVLTrainer(
            model=self.model_loader.scaffolds[0],
            config=training_config,
            device=self.context.device,
            loss_fn=loss_fn,
            logger=self.context.logger,
            memory_lock=Lock(),
            tokenizer=self.model_loader.base_tokenizer,
            state=self.state_tracker.state
        )

    def train_step(self, batch: List[dict], dry_run: bool = False, 
                   dry_run_params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Perform a single training step on a batch of data.

        Args:
            batch: List of training examples
            dry_run: Whether to perform a dry run
            dry_run_params: Optional parameters for dry run

        Returns:
            Optional[float]: Loss value if not dry run, None otherwise

        Raises:
            ValueError: If batch format is invalid
        """
        # Validate batch format
        for i, item in enumerate(batch):
            if "prompt" not in item or "completion" not in item:
                self.context.logger.log_error(
                    error_msg=f"Invalid batch item at index {i}: missing prompt or completion",
                    error_type="data_format_error",
                    additional_info={
                        "item": item,
                        "batch_size": len(batch),
                        "required_fields": ["prompt", "completion"]
                    }
                )
                raise ValueError(f"Invalid batch item at index {i}: missing prompt or completion")

        try:
            scaffold_provider = self.model_loader.scaffold_manager.get_scaffold_context
            return self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=scaffold_provider,
                dry_run=dry_run,
                dry_run_params=dry_run_params
            )
        except Exception as e:
            self.context.logger.record({
                "event": "training_error",
                "error": str(e),
                "batch_size": len(batch),
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "state_hash": self.state_tracker.state.get_state_hash()
            })
            raise

    def run_training_cycle(self, train_data: List, valid_data: List, 
                          epochs: Optional[int] = None, batch_size: Optional[int] = None):
        try:
            def scaffold_provider(batch):
                prompts = batch.get("prompt", [])
                scaffold_inputs = self.model_loader.generation_manager.tokenize_and_map(prompts)
                return self.model_loader.get_scaffold_hidden_states(scaffold_inputs)
            self.training_cycle_manager.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                scaffold_provider=scaffold_provider,
                epochs=epochs,
                batch_size=batch_size
            )
        except Exception as e:
            self.context.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "state_hash": self.state_tracker.state.get_state_hash()
            })
            raise

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

    def train_step(self, batch: List[dict], dry_run: bool = False, 
                   dry_run_params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        return self.cycle_trainer.train_step(batch, dry_run, dry_run_params)

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
            self.context.logger.log_error(
                error_msg="No training data available for training cycle",
                error_type="training_error",
                conversation_id=self.state_tracker.state.conversation_id,
                state_hash=self.state_tracker.state.get_state_hash(),
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

        # Execute pre-training hooks
        self.plugin_manager.execute_hook("on_training_step", {
            "batch_size": len(train_data),
            "dry_run": False
        })
        
        try:
            # Run the training cycle
            self.cycle_trainer.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Execute post-training hooks
            self.plugin_manager.execute_hook("on_training_step_complete", {
                "batch_size": len(train_data),
                "result": "success"
            })
            
        except Exception as e:
            # Log training error
            self.context.logger.log_error(
                error_msg=f"Training cycle failed: {str(e)}",
                error_type="training_error",
                stack_trace=traceback.format_exc(),
                conversation_id=self.state_tracker.state.conversation_id,
                state_hash=self.state_tracker.state.get_state_hash()
            )
            raise

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        self.state_tracker.update_data_exposure(data_exposure)
        self.context.logger.record({
            "event": "training_complete_handled",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": self.state_tracker.state.conversation_id,
            "state_hash": self.state_tracker.state.get_state_hash()
        })

class ResponseGenerator:
    """Generates responses using models and scaffolds."""
    def __init__(self, context: SystemContext, model_loader: ModelLoader, 
                 state_tracker: StateTracker, error_manager: ErrorManager):
        self.context = context
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self.generation_manager = GenerationManager(
            config_manager=context.config_manager,
            base_model=model_loader.base_model,
            scaffolds=model_loader.scaffolds,
            base_tokenizer=model_loader.base_tokenizer,
            scaffold_tokenizer=model_loader.scaffold_tokenizer,
            state=state_tracker.state,
            logger=context.logger,
            error_logger=context.logger,
            cross_attention_injector=CrossAttentionInjector(context.config_manager, context.logger),
            scaffold_manager=model_loader.scaffold_manager,
            temperament=None,  # Set after TemperamentAdjuster
            curiosity_manager=None  # Set after CuriosityEngine
        )

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 scaffold_weight: Optional[float] = None, **kwargs) -> str:
        try:
            response = self.generation_manager.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                scaffold_weight=scaffold_weight,
                **kwargs
            )
            self.state_tracker.update_conversation(prompt, response)
            self.context.logger.record({
                "event": "generation_complete",
                "prompt": prompt,
                "response_length": len(response),
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id
            })
            return response
        except Exception as e:
            return self.error_manager.handle_generation_error(e, prompt)

    def has_repetition(self, output_ids: List[int], n: int = 3) -> bool:
        return self.generation_manager.has_repetition(output_ids, n)

    def tokenize_and_map(self, prompts: List[str], max_length: Optional[int] = None) -> Dict:
        try:
            return self.generation_manager.tokenize_and_map(prompts, max_length)
        except Exception as e:
            self.context.logger.record({
                "error": f"Tokenization failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id
            })
            raise

    def update_token_map_memory(self, prompt: str, confidence: float):
        try:
            self.generation_manager._update_token_map_memory(prompt, confidence)
        except Exception as e:
            self.context.logger.record({
                "error": f"Token map memory update failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id
            })
            raise

    def clear_scaffold_cache(self):
        try:
            self.generation_manager._clear_scaffold_cache()
        except Exception as e:
            self.context.logger.record({
                "error": f"Scaffold cache clear failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id
            })
            raise

class ComponentManager:
    """Manages initialization and lifecycle of SOVL system components."""
    def __init__(self, context: SystemContext, state_tracker: StateTracker, error_manager: ErrorManager):
        self.context = context
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self._initialized = False
        self._components = {}

    def initialize(self) -> None:
        """Initialize all system components in the correct order."""
        if self._initialized:
            return

        try:
            # Initialize core components first
            self._initialize_core_components()
            
            # Initialize specialized components
            self._initialize_specialized_components()
            
            # Initialize plugin manager last
            self._initialize_plugin_manager()
            
            self._initialized = True
            
            self.context.logger.record_event(
                event_type="components_initialized",
                message="All system components initialized successfully",
                level="info"
            )
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Component initialization failed: {str(e)}",
                error_type="initialization_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        # Initialize temperament system
        self._components['temperament_system'] = TemperamentSystem.create_from_config(
            self.context.config_manager,
            self.context.logger,
            self.context.device
        )
        
        # Initialize generation manager
        self._components['generation_manager'] = GenerationManager(
            config_manager=self.context.config_manager,
            logger=self.context.logger,
            state=self.state_tracker.state
        )

    def _initialize_specialized_components(self) -> None:
        """Initialize specialized system components."""
        # Initialize training manager
        self._components['training_manager'] = TrainingManager(
            context=self.context,
            config_handler=self.context.config_handler,
            model_loader=self.context.model_loader,
            state_tracker=self.state_tracker,
            error_manager=self.error_manager
        )
        
        # Initialize curiosity engine
        self._components['curiosity_engine'] = CuriosityEngine(
            context=self.context,
            model_loader=self.context.model_loader,
            state_tracker=self.state_tracker,
            error_manager=self.error_manager
        )

    def _initialize_plugin_manager(self) -> None:
        """Initialize plugin manager."""
        self._components['plugin_manager'] = PluginManager(
            config_manager=self.context.config_manager,
            logger=self.context.logger,
            state=self.state_tracker.state
        )

    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        if not self._initialized:
            raise RuntimeError("Components not initialized")
        if name not in self._components:
            raise KeyError(f"Component {name} not found")
        return self._components[name]

class SOVLSystem:
    """Main system class for SOVL."""
    def __init__(self, config_path: str):
        """Initialize the SOVL system with configuration and core components."""
        self.config_path = config_path
        self.context = SystemContext(config_path)
        self.state_tracker = StateTracker(self.context)
        self.error_manager = ErrorManager(self.context, self.state_tracker)
        
        # Initialize training data with error handling
        try:
            self.train_data, self.valid_data = load_training_data(
                self.context.config_manager,
                self.context.logger
            )
            
            # Log successful data initialization
            self.context.logger.record_event(
                event_type="data_initialization",
                message=f"Loaded {len(self.train_data)} training and {len(self.valid_data)} validation samples",
                level="info",
                additional_info={
                    "train_samples": len(self.train_data),
                    "valid_samples": len(self.valid_data),
                    "config_path": config_path
                }
            )
            
        except InsufficientDataError as e:
            # Log error and raise with clear message
            self.context.logger.log_error(
                error_msg=str(e),
                error_type="insufficient_data_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": config_path,
                    "min_entries": self.context.config_manager.get("core_config.min_training_entries", 10)
                }
            )
            raise RuntimeError("Cannot initialize system without sufficient training data") from e
            
        except Exception as e:
            # Log error and raise with clear message
            self.context.logger.log_error(
                error_msg=f"Failed to load training data: {str(e)}",
                error_type="data_initialization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": config_path
                }
            )
            raise RuntimeError("Failed to initialize system due to data loading error") from e
        
        # Initialize component manager
        self.component_manager = ComponentManager(
            context=self.context,
            state_tracker=self.state_tracker,
            error_manager=self.error_manager
        )
        
        # Initialize all components
        self.component_manager.initialize()
        
        # Set system reference in plugin manager
        self.plugin_manager = self.component_manager.get_component('plugin_manager')
        self.plugin_manager.set_system(self)
        
        self.context.logger.record_event(
            event_type="system_initialization",
            message="SOVL system initialized",
            level="info",
            additional_info={
                "config_path": config_path,
                "device": str(self.context.device),
                "training_data_loaded": bool(self.train_data),
                "validation_data_loaded": bool(self.valid_data)
            }
        )

    @property
    def generation_manager(self) -> GenerationManager:
        """Get generation manager instance."""
        return self.component_manager.get_component('generation_manager')

    @property
    def training_manager(self) -> TrainingManager:
        """Get training manager instance."""
        return self.component_manager.get_component('training_manager')

    @property
    def curiosity_engine(self) -> CuriosityEngine:
        """Get curiosity engine instance."""
        return self.component_manager.get_component('curiosity_engine')

    @property
    def temperament_system(self) -> TemperamentSystem:
        """Get temperament system instance."""
        return self.component_manager.get_component('temperament_system')

    def _update_temperament(self) -> None:
        try:
            self.temperament_system.update_from_state(
                self.state_tracker.state,
                self.curiosity_engine.curiosity_manager if self.curiosity_engine else None
            )
            self.context.logger.record_event(
                event_type="temperament_update",
                message="Temperament updated successfully",
                level="info",
                additional_info={
                    "temperament_score": self.state_tracker.state.temperament_score,
                    "mood_label": self.state_tracker.state.mood_label
                }
            )
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to update temperament: {str(e)}",
                error_type="temperament_error",
                stack_trace=traceback.format_exc(),
                conversation_id=self.state_tracker.state.conversation_id,
                state_hash=self.state_tracker.state.get_state_hash()
            )

    def generate_response(self, prompt: str) -> str:
        try:
            # Execute pre-generate hooks
            self.plugin_manager.execute_hook("pre_generate", {"prompt": prompt})
            
            response = self.generation_manager.generate(prompt)
            
            # Execute post-generate hooks
            self.plugin_manager.execute_hook("post_generate", {
                "prompt": prompt,
                "response": response
            })
            
            self.context.logger.record_event(
                event_type="response_generation",
                message="Response generated successfully",
                level="info",
                additional_info={
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                }
            )
            return response
        except Exception as e:
            # Execute error hooks
            self.plugin_manager.execute_hook("on_error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "stack_trace": traceback.format_exc()
            })
            return self.error_manager.handle_generation_error(e, prompt)

    def train_step(self, batch: List[dict], dry_run: bool = False, 
                   dry_run_params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        # Execute pre-training hooks
        self.plugin_manager.execute_hook("on_training_step", {
            "batch_size": len(batch),
            "dry_run": dry_run
        })
        
        result = self.training_manager.train_step(batch, dry_run, dry_run_params)
        
        # Execute post-training hooks
        self.plugin_manager.execute_hook("on_training_step_complete", {
            "batch_size": len(batch),
            "result": result
        })
        
        return result

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
            self.context.logger.log_error(
                error_msg="No training data available for training cycle",
                error_type="training_error",
                conversation_id=self.state_tracker.state.conversation_id,
                state_hash=self.state_tracker.state.get_state_hash(),
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

        # Execute pre-training hooks
        self.plugin_manager.execute_hook("on_training_step", {
            "batch_size": len(train_data),
            "dry_run": False
        })
        
        try:
            # Run the training cycle
            self.training_manager.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Execute post-training hooks
            self.plugin_manager.execute_hook("on_training_step_complete", {
                "batch_size": len(train_data),
                "result": "success"
            })
            
        except Exception as e:
            # Log training error
            self.context.logger.log_error(
                error_msg=f"Training cycle failed: {str(e)}",
                error_type="training_error",
                stack_trace=traceback.format_exc(),
                conversation_id=self.state_tracker.state.conversation_id,
                state_hash=self.state_tracker.state.get_state_hash()
            )
            raise

    def generate_curiosity_question(self, context: str = "", spontaneous: bool = False) -> Optional[str]:
        return self.curiosity_engine.generate_question(context, spontaneous)

    def update_metrics(self, question: str, score: float, spontaneous: bool = False, 
                       answered: bool = False):
        self.state_tracker.update_curiosity_metrics(question, score, spontaneous, answered)

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        self.training_manager.handle_training_complete(epoch, avg_loss, data_exposure)

    def handle_gestation_complete(self, batch_size: int, avg_loss: float):
        self.training_manager.gestation_trainer.handle_gestation_complete(batch_size, avg_loss)

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int):
        self.training_manager.sleep_trainer.handle_dream_complete(dream_prompt, is_novel, memory_count)

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float):
        self.training_manager.sleep_trainer.handle_sleep_train_complete(batch_size, data_exposure)

    def update_temperament(self):
        self._update_temperament()

    def check_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None) -> bool:
        return self.memory_monitor.check_memory_health(model_size, trainer)

    def set_scaffold_influence(self, weight: float, curve: str = 'linear'):
        """Set scaffold influence weight with dynamic adjustment.
        
        Args:
            weight: Base weight value between 0 and 1
            curve: Weight adjustment curve type ('linear' or 'sigmoid_linear')
        """
        try:
            # Validate weight range
            weight = max(0.0, min(1.0, weight))
            
            # Update scaffold influence in all components
            if hasattr(self, 'generation_manager'):
                self.generation_manager.set_scaffold_influence(weight, curve)
                
            if hasattr(self, 'training_manager'):
                self.training_manager.set_scaffold_influence(weight, curve)
                
            if hasattr(self, 'scaffold_manager'):
                self.scaffold_manager.set_scaffold_influence(weight, curve)
                
            # Log the update
            self.context.logger.record({
                "event": "scaffold_influence_updated",
                "weight": weight,
                "curve": curve,
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "state_hash": self.state_tracker.state.get_state_hash()
            })
            
        except Exception as e:
            self.context.logger.record({
                "error": f"Failed to set scaffold influence: {str(e)}",
                "weight": weight,
                "curve": curve,
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "stack_trace": traceback.format_exc()
            })
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict) -> torch.Tensor:
        """Get hidden states from scaffold model with optional toggle.
        
        Args:
            scaffold_inputs: Dictionary containing input tensors for scaffold model
            
        Returns:
            torch.Tensor: Hidden states from scaffold model or zeros if disabled
        """
        try:
            # Check if scaffold is enabled in config
            if not self.config_manager.get("controls_config.enable_scaffold", True):
                return torch.zeros_like(scaffold_inputs["input_ids"], dtype=torch.float, device=self.context.device)
            
            # Use mixed precision if available
            with torch.autocast(
                device_type=self.context.device.type,
                dtype=torch.float16 if self.context.device.type == 'cuda' else torch.bfloat16
            ):
                # Get scaffold outputs with hidden states
                scaffold_outputs = self.scaffolds[0](
                    **{k: v for k, v in scaffold_inputs.items() if k in ['input_ids', 'attention_mask']},
                    output_hidden_states=True,
                )
                
                # Extract last layer hidden states
                hidden_states = (
                    scaffold_outputs.hidden_states[-1]
                    if hasattr(scaffold_outputs, 'hidden_states')
                    else scaffold_outputs.base_model_output.hidden_states[-1]
                )
                
                # Detach to prevent gradient flow
                return hidden_states.detach()
                
        except Exception as e:
            self.context.logger.record({
                "error": f"Failed to get scaffold hidden states: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state_tracker.state.conversation_id,
                "stack_trace": traceback.format_exc()
            })
            # Fallback to zeros on error
            return torch.zeros_like(scaffold_inputs["input_ids"], dtype=torch.float, device=self.context.device)

    def cleanup(self):
        """Cleanup system resources."""
        try:
            # Execute cleanup hooks
            self.plugin_manager.execute_hook("on_cleanup", {})
            
            # Save state
            self.state_tracker.state.save_state()
            
            # Cleanup other components
            self.generation_manager.cleanup()
            self.training_manager.cleanup()
            
            self.context.logger.record_event(
                event_type="system_cleanup",
                message="System cleanup completed",
                level="info"
            )
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Cleanup failed: {str(e)}",
                error_type="cleanup_error",
                stack_trace=traceback.format_exc()
            )

    def set_orchestrator(self, orchestrator: 'SOVLOrchestrator') -> None:
        """Set the orchestrator reference."""
        with self._lock:
            self._orchestrator = orchestrator

    def get_orchestrator(self) -> Optional['SOVLOrchestrator']:
        """Get the orchestrator reference."""
        with self._lock:
            return self._orchestrator

if __name__ == "__main__":
    from sovl_conductor import SOVLOrchestrator
    orchestrator = SOVLOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        print(f"Error running SOVL system: {str(e)}")
        raise
    finally:
        orchestrator.shutdown()
