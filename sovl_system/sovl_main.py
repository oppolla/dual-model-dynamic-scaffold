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
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager
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

def calculate_confidence_score(logits, generated_ids) -> float:
    """Calculate confidence score for generated tokens."""
    try:
        processor = LogitsProcessor(logits)
        return processor.calculate_confidence(generated_ids)
    except Exception as e:
        print(f"Confidence score error: {str(e)} - Using default 0.5")
        return 0.5

class SOVLSystem:
    """Main orchestrator for the SOVL system, managing model interactions and training."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize logging
        self._setup_logging()

        # Initialize core components
        self._initialize_components()

        # Load and validate training data
        self._load_training_data()

        # Initialize state manager and load state
        self.state_manager = StateManager(self.config_manager, self.logger, self.device)
        self.state = self.state_manager.load_state()

        # Post-initialization setup
        self._post_initialize()

    def _setup_logging(self) -> None:
        """Configure logging for system and error events."""
        self.logger = Logger(
            log_file=self.config_manager.get("logging_config.log_file", "sovl_system_logs.jsonl"),
            max_size_mb=self.config_manager.get("logging_config.max_size_mb", 20),
            compress_old=self.config_manager.get("logging_config.compress_old", True)
        )
        self.logger.manage_rotation(max_files=7)
        self.error_logger = Logger(
            log_file="sovl_errors.jsonl",
            max_size_mb=10,
            compress_old=True
        )

    def _initialize_components(self) -> None:
        """Initialize core system components."""
        # Cache configuration sections
        self.core_config = self.config_manager.get_section("core_config")
        self.training_config = self.config_manager.get_section("training_config")
        self.curiosity_config = self.config_manager.get_section("curiosity_config")
        self.cross_attn_config = self.config_manager.get_section("cross_attn_config")
        self.controls_config = self.config_manager.get_section("controls_config")
        self.lora_config = self.config_manager.get_section("lora_config")

        # Validate configuration early
        if not self._validate_config():
            raise ValueError("Configuration validation failed")

        # Initialize error handler
        self.error_handler = ErrorHandler(
            config_manager=self.config_manager,
            logger=self.logger,
            error_log_file="sovl_errors.jsonl",
            max_error_log_size_mb=10,
            compress_old=True,
            state=None  # Set after state initialization
        )

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            config_manager=self.config_manager,
            device=self.device,
            logger=self.logger
        )

        # Initialize temperament system
        self._initialize_temperament()

        # Initialize model manager
        self.model_manager = ModelManager(
            config_manager=self.config_manager,
            logger=self.logger,
            device=self.device
        )

        # Get models and tokenizers
        self.base_model = self.model_manager.get_base_model()
        self.scaffolds = [self.model_manager.get_scaffold_model()]
        self.base_tokenizer = self.model_manager.get_base_tokenizer()
        self.scaffold_tokenizer = self.model_manager.get_scaffold_tokenizer()
        self.scaffold_unk_id = self.model_manager.get_scaffold_unk_id()

        # Initialize scaffold manager
        self.scaffold_manager = ScaffoldManager(self.config_manager, self.logger)
        self.scaffold_token_mapper = None  # Initialized when needed

        # Initialize cross-attention injector
        self.cross_attention_injector = CrossAttentionInjector(
            config_manager=self.config_manager,
            logger=self.logger
        )

        # Inject cross-attention
        self._insert_cross_attention()

        # Initialize trainer
        self._initialize_trainer()

        # Initialize generation manager
        self.generation_manager = GenerationManager(
            config_manager=self.config_manager,
            base_model=self.base_model,
            scaffolds=self.scaffolds,
            base_tokenizer=self.base_tokenizer,
            scaffold_tokenizer=self.scaffold_tokenizer,
            state=None,  # Set after state initialization
            logger=self.logger,
            error_logger=self.error_logger,
            cross_attention_injector=self.cross_attention_injector,
            scaffold_manager=self.scaffold_manager,
            temperament=self.temperament_system,
            curiosity_manager=None  # Set after curiosity initialization
        )

        # Initialize curiosity
        self.curiosity_manager = (
            CuriosityManager(
                config=self.curiosity_config,
                logger=self.logger,
                device=self.device,
                state=None  # Set after state initialization
            ) if self.curiosity_config.get("enable_curiosity", True) else None
        )

        # Initialize tuner
        self.tuner = SOVLTuner(
            config_manager=self.config_manager,
            logger=self.logger,
            curiosity_manager=self.curiosity_manager,
            trainer=self.trainer,
            cross_attention_injector=self.cross_attention_injector
        )

    def _initialize_temperament(self) -> None:
        """Initialize the temperament system."""
        try:
            self.temperament_system = TemperamentSystem.create_from_config(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            self.logger.record({
                "event": "temperament_initialized",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to initialize temperament system: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.conversation_id
            })
            raise

    def _initialize_trainer(self) -> None:
        """Set up the trainer with training configuration."""
        training_config = TrainingConfig(
            learning_rate=self.training_config.get("learning_rate", 0.0003),
            grad_accum_steps=self.training_config.get("accumulation_steps", 4),
            weight_decay=0.01,
            total_steps=1000,
            max_grad_norm=1.0,
            use_amp=(self.device.type == "cuda"),
            max_patience=self.training_config.get("max_patience", 2),
            batch_size=self.training_config.get("batch_size", 1),
            max_epochs=self.training_config.get("train_epochs", 3),
            validate_every_n_steps=100,
            checkpoint_interval=1000,
            checkpoint_path="checkpoints/sovl_trainer",
            scheduler_type="linear",
            cosine_min_lr=1e-6,
            warmup_ratio=0.1,
            dropout_rate=self.lora_config.get("lora_dropout", 0.1),
            max_seq_length=self.training_config.get("max_seq_length", 128),
            metrics_to_track=["loss", "accuracy", "confidence"],
            enable_gestation=self.controls_config.get("enable_gestation", True),
            enable_sleep_training=self.controls_config.get("enable_sleep_training", True),
            enable_lifecycle_weighting=self.controls_config.get("enable_lifecycle_weighting", True),
            lifecycle_capacity_factor=self.training_config.get("lifecycle_capacity_factor", 0.01),
            lifecycle_curve=self.training_config.get("lifecycle_curve", "sigmoid_linear"),
            sleep_conf_threshold=self.controls_config.get("sleep_conf_threshold", 0.7),
            sleep_log_min=self.controls_config.get("sleep_log_min", 10),
            accumulation_steps=self.training_config.get("accumulation_steps", 4),
            exposure_gain_eager=self.training_config.get("exposure_gain_eager", 3),
            exposure_gain_default=self.training_config.get("exposure_gain_default", 2),
            dream_memory_weight=self.controls_config.get("dream_memory_weight", 0.1),
            enable_dreaming=self.controls_config.get("enable_dreaming", True),
            repetition_n=3,
            sigmoid_scale=self.training_config.get("sigmoid_scale", 0.5),
            sigmoid_shift=self.training_config.get("sigmoid_shift", 5.0),
            curiosity_weight_ignorance=self.curiosity_config.get("weight_ignorance", 0.7),
            curiosity_weight_novelty=self.curiosity_config.get("weight_novelty", 0.3),
            curiosity_pressure_threshold=self.curiosity_config.get("pressure_threshold", 0.7),
            curiosity_pressure_drop=self.curiosity_config.get("pressure_drop", 0.3),
            curiosity_novelty_threshold_spontaneous=self.curiosity_config.get("novelty_threshold_spontaneous", 0.9),
            curiosity_novelty_threshold_response=self.curiosity_config.get("novelty_threshold_response", 0.8),
            curiosity_silence_threshold=self.curiosity_config.get("silence_threshold", 20.0),
            curiosity_question_cooldown=self.curiosity_config.get("question_cooldown", 60.0),
            curiosity_queue_maxlen=self.curiosity_config.get("queue_maxlen", 10),
            curiosity_max_new_tokens=self.curiosity_config.get("max_new_tokens", 8),
            curiosity_base_temperature=self.curiosity_config.get("base_temperature", 1.1),
            curiosity_temperament_influence=self.curiosity_config.get("temperament_influence", 0.4),
            curiosity_top_k=self.curiosity_config.get("top_k", 30)
        )
        def loss_fn(logits, labels):
            mask = labels != -100
            return F.cross_entropy(
                logits.view(-1, logits.size(-1))[mask.view(-1)],
                labels.view(-1)[mask.view(-1)],
                ignore_index=-100
            )
        self.trainer = SOVLTrainer(
            model=self.scaffolds[0],
            config=training_config,
            device=self.device,
            loss_fn=loss_fn,
            logger=self.logger,
            memory_lock=Lock(),
            tokenizer=self.base_tokenizer,
            state=None  # Set after state initialization
        )
        self.trainer.memory_check = self.check_memory_health

    def _load_training_data(self) -> None:
        """Load and split training data."""
        try:
            self.train_data, self.valid_data = load_training_data(self.config_manager, self.logger)
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load training data: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": "init",
                "stack_trace": traceback.format_exc()
            })
            self.train_data = []
            self.valid_data = []

    def _post_initialize(self) -> None:
        """Perform post-initialization setup."""
        self.last_question_time = time.time()

    def _validate_config(self) -> bool:
        """
        Validate the configuration using ConfigManager.
        
        Returns:
            bool: True if validation succeeds, False otherwise
        """
        try:
            self.config_manager.validate_or_raise(self.model.config)
            return True
        except ValueError as e:
            self.logger.record({
                "error": str(e),
                "timestamp": time.time(),
                "conversation_id": "validate"
            })
            return False
        except Exception as e:
            self.logger.record({
                "error": f"Unexpected error during config validation: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "validate"
            })
            return False

    def _insert_cross_attention(self) -> None:
        """Inject cross-attention layers into the model."""
        try:
            self.cross_attention_injector.inject_cross_attention(
                model=self.base_model,
                scaffold_model=self.scaffolds[0],
                core_config=self.core_config,
                cross_attn_config=self.cross_attn_config,
                lora_config=self.lora_config,
                token_map=self.scaffold_token_mapper,
                device=self.device
            )
        except Exception as e:
            self.error_handler.handle_cross_attention_error(e)
            raise

    def check_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None) -> bool:
        """Check system memory health."""
        return self.memory_manager.check_memory_health(model_size, trainer)

    def generate_curiosity_question(self, context: str = "", spontaneous: bool = False) -> Optional[str]:
        """Generate a curiosity-driven question."""
        if not self.curiosity_manager:
            self.logger.record({
                "warning": "Curiosity manager not initialized",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            return None
        try:
            question = self.curiosity_manager.generate_question(
                context=context,
                spontaneous=spontaneous,
                model=self.base_model,
                tokenizer=self.base_tokenizer
            )
            if question:
                self.logger.log_curiosity_event(
                    event_type="question_generated",
                    question=question,
                    spontaneous=spontaneous,
                    conversation_id=self.state.conversation_id,
                    state_hash=self.state.get_state_hash()
                )
            return question
        except Exception as e:
            self.error_handler.handle_curiosity_error(e, "question_generation")
            return None

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float) -> None:
        """Handle completion of a training cycle."""
        self.state.update_data_exposure(data_exposure)
        self.logger.record({
            "event": "training_complete_handled",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def handle_gestation_complete(self, batch_size: int, avg_loss: float) -> None:
        """Handle completion of a gestation cycle."""
        self.state.update_gestation_metrics(batch_size, avg_loss)
        self.logger.record({
            "event": "gestation_complete_handled",
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Handle completion of a dream cycle."""
        self.state.update_dream_metrics(dream_prompt, is_novel, memory_count)
        self.logger.record({
            "event": "dream_complete_handled",
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float) -> None:
        """Handle completion of a sleep training cycle."""
        self.state.update_sleep_metrics(batch_size, data_exposure)
        self.logger.record({
            "event": "sleep_train_complete_handled",
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def update_metrics(self, question: str, score: float, spontaneous: bool = False, answered: bool = False) -> None:
        """Update curiosity metrics."""
        if self.state.curiosity:
            self.state.curiosity.update_metrics(
                question=question,
                score=score,
                spontaneous=spontaneous,
                answered=answered,
                conversation_id=self.history.conversation_id,
                state_hash=self.state.state_hash()
            )

    def _update_temperament(self) -> None:
        """Update the temperament system based on current state."""
        try:
            self.temperament_system.update_from_state(
                state=self.state,
                curiosity_manager=self.curiosity_manager
            )
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update temperament: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.conversation_id
            })
            raise

    def train_step(self, batch: List[dict], dry_run: bool = False, dry_run_params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Execute a single training step."""
        try:
            scaffold_provider = self.scaffold_manager.get_scaffold_context
            return self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=scaffold_provider,
                dry_run=dry_run,
                dry_run_params=dry_run_params
            )
        except Exception as e:
            self.logger.record({
                "event": "training_error",
                "error": str(e),
                "batch_size": len(batch),
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id,
                "state_hash": self.state.state_hash()
            })
            raise

    def run_training_cycle(self, train_data: Optional[List] = None, valid_data: Optional[List] = None, 
                         epochs: Optional[int] = None, batch_size: Optional[int] = None) -> None:
        """Run a full training cycle."""
        train_data = train_data or self.train_data
        valid_data = valid_data or self.valid_data
        epochs = epochs or self.training_config.get("train_epochs", 3)
        batch_size = batch_size or self.training_config.get("batch_size", 1)

        if len(train_data) < batch_size or not valid_data:
            self.logger.record({
                "warning": "Insufficient data for training",
                "train_data_size": len(train_data),
                "valid_data_size": len(valid_data),
                "batch_size": batch_size,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            print("Not enough data for training.")
            return

        influence_weight = (
            self.trainer.get_life_curve_weight()
            if self.controls_config.get("enable_lifecycle_weighting", True)
            else self.last_weight
        )
        self.set_scaffold_influence(influence_weight)
        self.logger.record({
            "event": "training_cycle_start",
            "epochs": epochs,
            "batch_size": batch_size,
            "data_exposure": self.trainer.data_exposure,
            "scaffold_influence": influence_weight,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"Data exposure: {self.trainer.data_exposure} | Scaffold influence: {influence_weight:.3f}")

        dry_run = self.training_config.get("dry_run", False)
        dry_run_params = self.training_config.get("dry_run_params", {})
        if dry_run and dry_run_params.get("skip_training", True):
            print("\n=== DRY RUN TRAINING ===")
            dry_batch = train_data[:dry_run_params.get("max_samples", 2)]
            loss = self.train_step(dry_batch, dry_run=True, dry_run_params=dry_run_params)
            self.logger.record({
                "event": "dry_run_training_complete",
                "loss": loss,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            print(f"Dry run training complete: Loss = {loss}")
            return

        print(f"\n--- Training ({epochs} epochs) ---")
        start_time = time.time()

        def scaffold_provider(batch):
            prompts = batch.get("prompt", [])
            scaffold_inputs = self.tokenize_and_map(prompts)
            return self.get_scaffold_hidden_states(scaffold_inputs)

        try:
            training_results = self.trainer.run_training_cycle(
                train_data=train_data,
                validation_data=valid_data,
                scaffold_provider=scaffold_provider,
                max_epochs=epochs,
                early_stopping_patience=self.training_config.get("max_patience", 3)
            )
        except Exception as e:
            self.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            raise

        self.last_weight = self.trainer.get_life_curve_weight()
        self.set_scaffold_influence(self.last_weight)
        self.logger.record({
            "event": "training_cycle_complete",
            "duration": time.time() - start_time,
            "last_weight": self.last_weight,
            "training_history": training_results.get("training_history", []),
            "best_val_loss": training_results.get("best_val_loss", float("inf")),
            "final_epoch": training_results.get("final_epoch", 0),
            "early_stopped": training_results.get("early_stopped", False),
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"--- Training Finished ({time.time() - start_time:.2f}s) ---")

    def _sleep_train(self) -> None:
        """Train on dream-generated content."""
        if not self.controls_config.get("enable_sleep_training", True):
            self.logger.record({
                "event": "sleep_training_skipped",
                "reason": "Sleep training disabled",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id
            })
            return

        print("\n--- Sleep Training Initiated ---")
        try:
            log_entries = self.logger.read()
            self.trainer.sleep_train(log_entries)
            self.last_trained = time.time()
            self.logger.clear()
            self.last_weight = self.trainer.get_life_curve_weight()

            if self.controls_config.get("enable_temperament", True):
                self._update_temperament()
                self.last_temperament_score = self.temperament_system.score

            self.logger.record({
                "event": "sleep_training_complete",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id
            })
        except Exception as e:
            self.logger.record({
                "error": f"Sleep training failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "stack_trace": traceback.format_exc()
            })
            raise
        print("--- Sleep Training Complete ---")

    def has_repetition(self, output_ids: List[int], n: int = 3) -> bool:
        """Check for repetition in generated output."""
        return self.generation_manager.has_repetition(output_ids, n)

    def _handle_error_prompt(self, error_msg: str) -> str:
        """Generate a response to a system error."""
        return self.generation_manager._handle_error_prompt(error_msg)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, scaffold_weight: Optional[float] = None, **kwargs) -> str:
        """Generate a response for the given prompt."""
        try:
            response = self.generation_manager.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                scaffold_weight=scaffold_weight,
                **kwargs
            )
            self.logger.record({
                "event": "generation_complete",
                "prompt": prompt,
                "response_length": len(response),
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            return response
        except Exception as e:
            self.logger.record({
                "error": f"Generation failed: {str(e)}",
                "prompt": prompt,
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            return self.error_handler.handle_generation_error(e, prompt)

    def tokenize_and_map(self, prompts: List[str], max_length: Optional[int] = None) -> Dict:
        """Tokenize prompts and map to scaffold token space."""
        try:
            return self.generation_manager.tokenize_and_map(prompts, max_length)
        except Exception as e:
            self.logger.record({
                "error": f"Tokenization failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            raise

    def _update_token_map_memory(self, prompt: str, confidence: float) -> None:
        """Update token map memory based on prompt confidence."""
        try:
            self.generation_manager._update_token_map_memory(prompt, confidence)
        except Exception as e:
            self.logger.record({
                "error": f"Token map memory update failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            raise

    def _clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches."""
        try:
            self.generation_manager._clear_scaffold_cache()
        except Exception as e:
            self.logger.record({
                "error": f"Scaffold cache clear failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            raise

    def set_scaffold_influence(self, weight: float) -> None:
        """Set the influence weight for scaffold integration."""
        self.last_weight = weight
        self.logger.record({
            "event": "scaffold_influence_updated",
            "weight": weight,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def load_state(self) -> None:
        """Load saved system state."""
        try:
            self.state = self.state_manager.load_state()
            self.logger.record({
                "event": "state_loaded",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
        except Exception as e:
            self.logger.record({
                "error": f"State loading failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id
            })
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict) -> torch.Tensor:
        """Get hidden states from scaffold model."""
        try:
            # Placeholder; actual implementation depends on scaffold model
            return torch.zeros_like(scaffold_inputs["input_ids"], dtype=torch.float, device=self.device)
        except Exception as e:
            self.logger.record({
                "error": f"Failed to get scaffold hidden states: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.conversation_id
            })
            raise

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
