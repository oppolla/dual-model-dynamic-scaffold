from typing import Optional
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
from collections import deque, defaultdict, OrderedDict
import traceback
import os
from threading import Lock
from sovl_curiosity import CuriosityManager
from sovl_logger import Logger
from sovl_io import (
    load_training_data,
    validate_quantization_mode,
    InsufficientDataError
)
from sovl_state import SOVLState, ConversationHistory
from sovl_trainer import TrainingConfig, SOVLTrainer, collate_batch
from sovl_config import ConfigManager
from sovl_scaffold import inject_cross_attention, CrossAttentionInjector, ScaffoldManager, ScaffoldTokenMapper, build_scaffold_token_mapping
from sovl_processor import LogitsProcessor
from sovl_utils import (
    NumericalGuard,
    safe_divide,
    memory_usage,
    log_memory_usage,
    cosine_similarity_matrix,
    float_lt,
    get_parameter_count,
    adjust_temperature,
    calculate_confidence,
    detect_repetitions,
    validate_layer_indices,
)
from sovl_temperament import TemperamentConfig, TemperamentSystem
from sovl_memory import MemoryManager

def calculate_confidence_score(logits, generated_ids) -> float:
    """Wrapper for backward compatibility"""
    try:
        processor = LogitsProcessor(logits)
        return processor.calculate_confidence(generated_ids)
    except Exception as e:
        print(f"Confidence score error: {str(e)} - Using default 0.5")
        return 0.5

# Initialize logger for data loading
logger = Logger(
    log_file="sovl_logs.jsonl",
    max_size_mb=50,  # Rotate logs after 50MB
    compress_old=True  # Compress rotated logs
)

# Load training data
try:
    TRAIN_DATA = load_training_data("sovl_seed.jsonl", min_entries=0)
    if not TRAIN_DATA:
        logger.write({
            "warning": "No data loaded from sovl_seed.jsonl",
            "timestamp": time.time(),
            "conversation_id": "init"
        })
        print("Warning: No data loaded from sovl_seed.jsonl!")
except InsufficientDataError as e:
    logger.write({
        "error": str(e),
        "timestamp": time.time(),
        "conversation_id": "init",
        "is_error_prompt": True  # Flag for error listening
    })
    print(f"Data loading error: {e}")
    TRAIN_DATA = []
except Exception as e:
    logger.write({
        "error": f"Unexpected error during data loading: {e}",
        "timestamp": time.time(),
        "conversation_id": "init",
        "is_error_prompt": True,
        "stack_trace": traceback.format_exc()  # Include full traceback
    })
    print(f"Unexpected error during data loading: {e}")
    TRAIN_DATA = []

if not TRAIN_DATA:
    logger.write({"warning": "No data loaded from sovl_seed.jsonl", "timestamp": time.time(), "conversation_id": "init"})
    print("Warning: No data loaded from sovl_seed.jsonl!")

# Split training data
if TRAIN_DATA:
    TRAIN_DATA, VALID_DATA = load_training_data(config_manager, logger, TRAIN_DATA, VALID_SPLIT_RATIO)
else:
    VALID_DATA = []
    logger.write({
        "warning": "No training data available",
        "timestamp": time.time()
    })

# Initialize ConfigManager
config_manager = ConfigManager("sovl_config.json")

# Core Model Config
BASE_MODEL_NAME = config_manager.get("core_config.base_model_name", "gpt2", expected_type=str)
SCAFFOLD_MODEL_NAME = config_manager.get("core_config.scaffold_model_name", "gpt2", expected_type=str)
CROSS_ATTN_LAYERS = config_manager.get("core_config.cross_attn_layers", [0, 1, 2], expected_type=list)
USE_DYNAMIC_LAYERS = config_manager.get("core_config.use_dynamic_layers", False, expected_type=bool)
LAYER_SELECTION_MODE = config_manager.get("core_config.layer_selection_mode", "balanced", expected_type=str)
CUSTOM_LAYERS = config_manager.get("core_config.custom_layers", [], expected_type=list)
VALID_SPLIT_RATIO = config_manager.get("core_config.valid_split_ratio", 0.2, expected_type=float)
RANDOM_SEED = config_manager.get("core_config.random_seed", 42, expected_type=int)
UANTIZATION_MODE = validate_quantization_mode(
    config_manager.get("core_config.quantization", "fp16", expected_type=str),
    logger
)
    # Curiosity Config
ENABLE_CURIOSITY = config_manager.get("controls_config.enable_curiosity", True, expected_type=bool)
CURIOSITY_WEIGHT_IGNORANCE = config_manager.get("controls_config.curiosity_weight_ignorance", 0.5, expected_type=float)
CURIOSITY_WEIGHT_NOVELTY = config_manager.get("controls_config.curiosity_weight_novelty", 0.5, expected_type=float)
CURIOSITY_PRESSURE_THRESHOLD = config_manager.get("controls_config.curiosity_pressure_threshold", 0.7, expected_type=float)
CURIOSITY_PRESSURE_DROP = config_manager.get("controls_config.curiosity_pressure_drop", 0.2, expected_type=float)
CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS = config_manager.get("controls_config.curiosity_novelty_threshold_spontaneous", 0.8, expected_type=float)
CURIOSITY_NOVELTY_THRESHOLD_RESPONSE = config_manager.get("controls_config.curiosity_novelty_threshold_response", 0.6, expected_type=float)
CURIOSITY_SILENCE_THRESHOLD = config_manager.get("controls_config.curiosity_silence_threshold", 30.0, expected_type=float)
CURIOSITY_QUESTION_COOLDOWN = config_manager.get("controls_config.curiosity_question_cooldown", 60.0, expected_type=float)
CURIOSITY_QUEUE_MAXLEN = config_manager.get("controls_config.curiosity_queue_maxlen", 10, expected_type=int)
CURIOSITY_MAX_NEW_TOKENS = config_manager.get("controls_config.curiosity_max_new_tokens", 10, expected_type=int)
CURIOSITY_BASE_TEMPERATURE = config_manager.get("controls_config.curiosity_base_temperature", 0.9, expected_type=float)
CURIOSITY_TEMPERAMENT_INFLUENCE = config_manager.get("controls_config.curiosity_temperament_influence", 0.3, expected_type=float)
CURIOSITY_TOP_K = config_manager.get("controls_config.curiosity_top_k", 40, expected_type=int)

# LoRA Config
LORA_RANK = config_manager.get("lora_config.lora_rank", 8, expected_type=int)
LORA_ALPHA = config_manager.get("lora_config.lora_alpha", 16, expected_type=int)
LORA_DROPOUT = config_manager.get("lora_config.lora_dropout", 0.1, expected_type=float)
LORA_TARGET_MODULES = config_manager.get("lora_config.lora_target_modules", ["q_proj", "v_proj"], expected_type=list)

# Training Config
LEARNING_RATE = config_manager.get("training_config.learning_rate", 2e-5, expected_type=float)
TRAIN_EPOCHS = config_manager.get("training_config.train_epochs", 3, expected_type=int)
BATCH_SIZE = config_manager.get("training_config.batch_size", 2, expected_type=int)
MAX_SEQ_LENGTH = config_manager.get("training_config.max_seq_length", 512, expected_type=float)
SIGMOID_SCALE = config_manager.get("training_config.sigmoid_scale", 0.5, expected_type=float)
SIGMOID_SHIFT = config_manager.get("training_config.sigmoid_shift", 5.0, expected_type=float)

# Exposed Controls
SLEEP_CONF_THRESHOLD = config_manager.get("controls_config.sleep_conf_threshold", 0.7, expected_type=float)
SLEEP_TIME_FACTOR = config_manager.get("controls_config.sleep_time_factor", 1.0, expected_type=float)
SLEEP_LOG_MIN = config_manager.get("controls_config.sleep_log_min", 10, expected_type=int)
DREAM_SWING_VAR = config_manager.get("controls_config.dream_swing_var", 0.1, expected_type=float)
DREAM_LIFECYCLE_DELTA = config_manager.get("controls_config.dream_lifecycle_delta", 0.1, expected_type=float)
DREAM_TEMPERAMENT_ON = config_manager.get("controls_config.dream_temperament_on", True, expected_type=bool)
DREAM_NOISE_SCALE = config_manager.get("controls_config.dream_noise_scale", 0.05, expected_type=float)
TEMP_EAGER_THRESHOLD = config_manager.get("controls_config.temp_eager_threshold", 0.8, expected_type=float)
TEMP_SLUGGISH_THRESHOLD = config_manager.get("controls_config.temp_sluggish_threshold", 0.6, expected_type=float)
MEMORY_THRESHOLD = config_manager.get("controls_config.memory_threshold", 0.85, expected_type=float)
if not isinstance(MEMORY_THRESHOLD, (int, float)) or MEMORY_THRESHOLD <= 0 or MEMORY_THRESHOLD > 1:
    logger.write({"warning": f"Invalid MEMORY_THRESHOLD '{MEMORY_THRESHOLD}'. Defaulting to 0.85.", "timestamp": time.time(), "conversation_id": "init"})
    MEMORY_THRESHOLD = 0.85
TEMP_MOOD_INFLUENCE = config_manager.get("controls_config.temp_mood_influence", 0.0, expected_type=float)
SCAFFOLD_WEIGHT_CAP = config_manager.get("controls_config.scaffold_weight_cap", 1.0, expected_type=float)
BASE_TEMPERATURE = config_manager.get("controls_config.base_temperature", 0.7, expected_type=float)
SAVE_PATH_PREFIX = config_manager.get("controls_config.save_path_prefix", "state", expected_type=str)
DREAM_MEMORY_WEIGHT = config_manager.get("controls_config.dream_memory_weight", 0.1, expected_type=float)
DREAM_MEMORY_MAXLEN = config_manager.get("controls_config.dream_memory_maxlen", 10, expected_type=int)
DREAM_PROMPT_WEIGHT = config_manager.get("controls_config.dream_prompt_weight", 0.5, expected_type=float)
DREAM_NOVELTY_BOOST = config_manager.get("controls_config.dream_novelty_boost", 0.03, expected_type=float)
TEMP_CURIOSITY_BOOST = config_manager.get("controls_config.temp_curiosity_boost", 0.5, expected_type=float)
TEMP_RESTLESS_DROP = config_manager.get("controls_config.temp_restless_drop", 0.1, expected_type=float)
TEMP_MELANCHOLY_NOISE = config_manager.get("controls_config.temp_melancholy_noise", 0.02, expected_type=float)
CONF_FEEDBACK_STRENGTH = config_manager.get("controls_config.conf_feedback_strength", 0.5, expected_type=float)
TEMP_SMOOTHING_FACTOR = config_manager.get("controls_config.temp_smoothing_factor", 0.0, expected_type=float)
LIFECYCLE_CAPACITY_FACTOR = config_manager.get("training_config.lifecycle_capacity_factor", 0.01, expected_type=float)
LIFECYCLE_CURVE = config_manager.get("training_config.lifecycle_curve", "sigmoid_linear", expected_type=str)
DREAM_MEMORY_DECAY = config_manager.get("controls_config.dream_memory_decay", 0.95, expected_type=float)
DREAM_PRUNE_THRESHOLD = config_manager.get("controls_config.dream_prune_threshold", 0.1, expected_type=float)
USE_SCAFFOLD_MEMORY = config_manager.get("controls_config.use_scaffold_memory", True, expected_type=bool)
USE_TOKEN_MAP_MEMORY = config_manager.get("controls_config.use_token_map_memory", True, expected_type=bool)
DYNAMIC_CROSS_ATTN_MODE = config_manager.get("controls_config.dynamic_cross_attn_mode", None)
DRY_RUN = config_manager.get("training_config.dry_run", False, expected_type=bool)
DRY_RUN_MAX_SAMPLES = config_manager.get("training_config.dry_run_params.max_samples", 2, expected_type=int)
DRY_RUN_MAX_LENGTH = config_manager.get("training_config.dry_run_params.max_length", 128, expected_type=int)
DRY_RUN_VALIDATE_ARCH = config_manager.get("training_config.dry_run_params.validate_architecture", True, expected_type=bool)
DRY_RUN_SKIP_TRAINING = config_manager.get("training_config.dry_run_params.skip_training", True, expected_type=bool)
MEMORY_DECAY_RATE = config_manager.get("controls_config.memory_decay_rate", 0.95, expected_type=float)
HAS_WOKEN = config_manager.get("controls_config.has_woken", False, expected_type=bool)
IS_SLEEPING = config_manager.get("controls_config.is_sleeping", False, expected_type=bool)
ACCUMULATION_STEPS = config_manager.get("training_config.accumulation_steps", 4, expected_type=int)
EXPOSURE_GAIN_EAGER = config_manager.get("training_config.exposure_gain_eager", 3, expected_type=int)
EXPOSURE_GAIN_DEFAULT = config_manager.get("training_config.exposure_gain_default", 2, expected_type=int)
MAX_PATIENCE = config_manager.get("training_config.max_patience", 2, expected_type=int)
CONFIDENCE_HISTORY_MAXLEN = config_manager.get("controls_config.confidence_history_maxlen", 5, expected_type=int)
TEMPERAMENT_HISTORY_MAXLEN = config_manager.get("controls_config.temperament_history_maxlen", 5, expected_type=int)
ENABLE_DREAMING = config_manager.get("controls_config.enable_dreaming", True, expected_type=bool)
ENABLE_TEMPERAMENT = config_manager.get("controls_config.enable_temperament", True, expected_type=bool)
ENABLE_CONFIDENCE_TRACKING = config_manager.get("controls_config.enable_confidence_tracking", True, expected_type=bool)
ENABLE_GESTATION = config_manager.get("controls_config.enable_gestation", True, expected_type=bool)
ENABLE_SLEEP_TRAINING = config_manager.get("controls_config.enable_sleep_training", True, expected_type=bool)
ENABLE_CROSS_ATTENTION = config_manager.get("controls_config.enable_cross_attention", True, expected_type=bool)
ENABLE_DYNAMIC_CROSS_ATTENTION = config_manager.get("controls_config.enable_dynamic_cross_attention", True, expected_type=bool)
ENABLE_LORA_ADAPTERS = config_manager.get("controls_config.enable_lora_adapters", True, expected_type=bool)
ENABLE_REPETITION_CHECK = config_manager.get("controls_config.enable_repetition_check", True, expected_type=bool)
ENABLE_PROMPT_DRIVEN_DREAMS = config_manager.get("controls_config.enable_prompt_driven_dreams", True, expected_type=bool)
ENABLE_LIFECYCLE_WEIGHTING = config_manager.get("controls_config.enable_lifecycle_weighting", True, expected_type=bool)
ENABLE_ERROR_LISTENING = config_manager.get("controls_config.enable_error_listening", True, expected_type=bool)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")       

class SOVLSystem:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = Logger(
            log_file=config_manager.get("logging_config.log_file", "sovl_system_logs.jsonl"),
            max_size_mb=config_manager.get("logging_config.max_size_mb", 20),
            compress_old=config_manager.get("logging_config.compress_old", True)
        )
        self.logger.manage_rotation(max_files=7)
        self.error_logger = Logger(
            log_file="sovl_errors.jsonl",
            max_size_mb=10,
            compress_old=True
        )

        # Initialize memory manager
        self.memory_manager = MemoryManager(
            config_manager=config_manager,
            device=DEVICE,
            logger=self.logger
        )

        # Cache configuration sections
        self.core_config = config_manager.get_section("core_config")
        self.training_config = config_manager.get_section("training_config")
        self.curiosity_config = config_manager.get_section("curiosity_config")
        self.cross_attn_config = config_manager.get_section("cross_attn_config")
        self.controls_config = config_manager.get_section("controls_config")
        self.lora_config = config_manager.get_section("lora_config")

        # Initialize system parameters
        self.quantization_mode = self.core_config.get("quantization", "fp16")
        self.base_temperature = self.controls_config.get("base_temperature", 0.7)
        self.use_scaffold_memory = self.controls_config.get("use_scaffold_memory", True)
        self.use_token_map_memory = self.controls_config.get("use_token_map_memory", True)
        self.scaffold_weight = self.controls_config.get("scaffold_weight_cap", 1.0)
        self.memory_threshold = self.controls_config.get("memory_threshold", 0.85)
        self.memory_decay_rate = self.controls_config.get("memory_decay_rate", 0.95)
        self.dynamic_cross_attn_mode = self.controls_config.get("dynamic_cross_attn_mode", None)
        self.has_woken = self.controls_config.get("has_woken", False)
        self.is_sleeping = self.controls_config.get("is_sleeping", False)

        # Initialize temperament system
        temperament_config = TemperamentConfig(
            eager_threshold=self.controls_config.get("temp_eager_threshold", 0.7),
            sluggish_threshold=self.controls_config.get("temp_sluggish_threshold", 0.3),
            mood_influence=self.controls_config.get("temp_mood_influence", 0.5),
            curiosity_boost=self.controls_config.get("temp_curiosity_boost", 0.2),
            restless_drop=self.controls_config.get("temp_restless_drop", 0.1),
            melancholy_noise=self.controls_config.get("temp_melancholy_noise", 0.1),
            confidence_feedback_strength=self.controls_config.get("temp_conf_feedback_strength", 0.3),
            temp_smoothing_factor=self.controls_config.get("temp_smoothing_factor", 0.1),
            history_maxlen=self.controls_config.get("temperament_history_maxlen", 5),
            lifecycle_params={
                "gestation": self.controls_config.get("temp_gestation_params", {"bias": 0.2, "decay": 0.1}),
                "awakening": self.controls_config.get("temp_awakening_params", {"bias": 0.1, "decay": 0.05}),
                "maturity": self.controls_config.get("temp_maturity_params", {"bias": 0.0, "decay": 0.0}),
                "decline": self.controls_config.get("temp_decline_params", {"bias": -0.1, "decay": 0.05})
            }
        )
        self.temperament_system = TemperamentSystem(
            config=temperament_config,
            logger=self.logger,
            device=DEVICE
        )

        # Validate configuration
        self._validate_config()

        # Initialize models and tokenizers
        self.base_config = AutoConfig.from_pretrained(self.core_config["base_model_name"])
        self.scaffold_config = AutoConfig.from_pretrained(self.core_config["scaffold_model_name"])
        self.dry_run = self.training_config.get("dry_run", False)
        self.dry_run_params = self.training_config.get("dry_run_params", {})

        print(f"Loading base model: {self.core_config['base_model_name']}")
        try:
            quantization_config = (
                {"load_in_8bit": True} if self.quantization_mode == "int8" else
                {"load_in_4bit": True} if self.quantization_mode == "int4" else {}
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.core_config["base_model_name"],
                config=self.base_config,
                **quantization_config
            ).to(DEVICE)
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.logger.record({
                "event": "model_loaded",
                "model_type": "base",
                "model_name": self.core_config["base_model_name"],
                "quantization": self.quantization_mode,
                "timestamp": time.time()
            })
            print(f"Base model '{self.core_config['base_model_name']}' loaded and frozen.")
        except Exception as e:
            self.error_logger.record({
                "error": f"Failed to load base model: {str(e)}",
                "model_name": self.core_config["base_model_name"],
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

        print(f"Loading scaffold model: {self.core_config['scaffold_model_name']}")
        try:
            scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
                self.core_config["scaffold_model_name"],
                config=self.scaffold_config,
                **quantization_config
            )
            if self.controls_config.get("enable_lora_adapters", True):
                lora_config = LoraConfig(
                    r=self.lora_config.get("lora_rank", 8),
                    lora_alpha=self.lora_config.get("lora_alpha", 16),
                    target_modules=self.lora_config.get("lora_target_modules", ["c_attn", "c_proj", "c_fc"]),
                    lora_dropout=self.lora_config.get("lora_dropout", 0.1),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.scaffolds = [get_peft_model(scaffold_model_raw, lora_config).to(DEVICE)]
                print("LoRA adapters applied to scaffold[0].")
            else:
                self.scaffolds = [scaffold_model_raw.to(DEVICE)]
                print("Scaffold loaded without LoRA adapters.")
        except Exception as e:
            self.error_logger.record({
                "error": f"Failed to load scaffold model: {str(e)}",
                "model_name": self.core_config["scaffold_model_name"],
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

        print(f"Loading tokenizers from: {self.core_config['base_model_name']} and {self.core_config['scaffold_model_name']}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.core_config["base_model_name"])
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(self.core_config["scaffold_model_name"])
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        self.scaffolds[0].config.pad_token_id = self.scaffold_tokenizer.pad_token_id

        # Initialize trainer
        training_config = TrainingConfig(
            learning_rate=self.training_config.get("learning_rate", 0.0003),
            grad_accum_steps=self.training_config.get("accumulation_steps", 4),
            weight_decay=0.01,
            total_steps=1000,  # Placeholder, computed dynamically
            max_grad_norm=1.0,
            use_amp=(DEVICE.type == "cuda"),
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
            device=DEVICE,
            loss_fn=loss_fn,
            logger=self.logger,
            memory_lock=Lock(),
            tokenizer=self.base_tokenizer,
            state=None
        )
        self.trainer.memory_check = self.check_memory_health

        # Register callbacks
        def log_curiosity_event(event_name: str, details: dict):
            log_entry = {
                "event": event_name,
                "details": details,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            }
            self.logger.record(log_entry)

        self.trainer.register_callback("on_training_complete", lambda epoch, loss, exposure: log_curiosity_event(
            "training_complete", {"epoch": epoch, "avg_loss": loss, "data_exposure": exposure}
        ))
        self.trainer.register_callback("on_gestation_complete", lambda batch_size, loss: log_curiosity_event(
            "gestation_complete", {"batch_size": batch_size, "avg_loss": loss}
        ))
        self.trainer.register_callback("on_dream_complete", lambda prompt, novel, count: log_curiosity_event(
            "dream_complete", {"dream_prompt": prompt, "is_novel": novel, "memory_count": count}
        ))
        self.trainer.register_callback("on_sleep_train_complete", lambda batch_size, exposure: log_curiosity_event(
            "sleep_train_complete", {"batch_size": batch_size, "data_exposure": exposure}
        ))

        # Initialize token map using ScaffoldManager
        self.scaffold_manager = ScaffoldManager(config_manager, self.logger)
        self.scaffold_token_mapper = None  # Will be initialized when needed
        self.scaffold_unk_id = self.controls_config.get("scaffold_unk_id", self.scaffold_tokenizer.unk_token_id)
     
        # Initialize cross-attention injector
        self.cross_attention_injector = CrossAttentionInjector(
            config_manager=config_manager,
            logger=self.logger
        )
        
        # Inject cross-attention
        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Cross-attention injection complete.")

        # Initialize state
        self.memory_lock = Lock()
        self.mem_usage_history = deque(maxlen=10)
        self.dynamic_threshold_base = self.memory_threshold
        self.max_patience = self.training_config.get("max_patience", 2)
        self.history = ConversationHistory(maxlen=self.controls_config.get("conversation_history_maxlen", 10))
        self.last_trained = 0
        self.last_weight = 0.0

        self.state = SOVLState(
            config_manager,
            dream_memory_maxlen=self.controls_config.get("dream_memory_maxlen", 10),
            confidence_history_maxlen=self.controls_config.get("confidence_history_maxlen", 5),
            temperament_history_maxlen=self.controls_config.get("temperament_history_maxlen", 5),
            conversation_history_maxlen=self.controls_config.get("conversation_history_maxlen", 10),
            max_seen_prompts=self.controls_config.get("max_seen_prompts", 1000),
            prompt_timeout=self.controls_config.get("prompt_timeout", 86400.0),
            temperament_decay_rate=self.controls_config.get("temperament_decay_rate", 0.95),
            curiosity=CuriosityState(config_manager, self.logger)
        )
        self.state.set_scaffold_unk_id(self.scaffold_unk_id)
        self.trainer.state = self.state

        # Initialize curiosity
        self.curiosity_manager = (
            CuriosityManager(
                config=self.curiosity_config,
                logger=self.logger,
                device=DEVICE,
                state=self.state.curiosity
            ) if self.curiosity_config.get("enable_curiosity", True) else None
        )
        self.last_question_time = time.time()

        self.load_state()

        # After initializing the processor and token maps, add:
        self.processor.set_token_map(self.state.token_map, self.scaffold_unk_id)

    def _validate_config(self):
        """Validate required configuration keys and layer settings."""
        config_snapshot = OrderedDict(sorted(self.config_manager.get_state()["config"].items()))
        try:
            self.config_manager.validate_keys([
                "core_config.base_model_name",
                "core_config.scaffold_model_name",
                "training_config.learning_rate",
                "curiosity_config.enable_curiosity",
                "cross_attn_config.memory_weight"
            ])
            cross_attn_layers = self.core_config.get("cross_attn_layers", [5, 7])
            if not isinstance(cross_attn_layers, list):
                raise ValueError("core_config.cross_attn_layers must be a list!")
            if not self.core_config.get("use_dynamic_layers", False):
                base_config = AutoConfig.from_pretrained(self.core_config["base_model_name"])
                invalid_layers = [l for l in cross_attn_layers if not (0 <= l < base_config.num_hidden_layers)]
                if invalid_layers:
                    raise ValueError(f"Invalid cross_attn_layers: {invalid_layers} for {base_config.num_hidden_layers} layers.")
            if self.core_config.get("layer_selection_mode", "balanced") == "custom":
                custom_layers = self.core_config.get("custom_layers", [])
                base_config = AutoConfig.from_pretrained(self.core_config["base_model_name"])
                invalid_custom = [l for l in custom_layers if not (0 <= l < base_config.num_hidden_layers)]
                if invalid_custom:
                    raise ValueError(f"Invalid custom_layers: {invalid_custom} for {self.core_config['base_model_name']}")
            self.logger.record({
                "event": "config_validation",
                "status": "success",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "config_snapshot": config_snapshot,
                "state_hash": self.state.state_hash() if hasattr(self, "state") else None
            })
        except Exception as e:
            self.error_logger.record({
                "error": f"Config validation failed: {str(e)}",
                "type": type(e).__name__,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "stack_trace": traceback.format_exc(),
                "config_snapshot": config_snapshot,
                "validation_stage": "runtime",
                "state_hash": self.state.state_hash() if hasattr(self, "state") else None
            })
            raise

    def _verify_cross_attention_injection(self) -> bool:
        """Verify that cross-attention layers were properly injected."""
        try:
            # Check if expected layers exist
            cross_attn_layers = self.core_config.get("cross_attn_layers", [])
            expected_layers = set(cross_attn_layers)
            found_layers = set()

            # Scan model for cross-attention layers
            for name, module in self.base_model.named_modules():
                if "cross_attention" in name.lower():
                    try:
                        # Extract layer index from module name
                        parts = name.split('.')
                        if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
                            layer_idx = int(parts[2])
                            found_layers.add(layer_idx)
                    except (ValueError, IndexError):
                        continue
                    
            # Log verification results
            self.logger.record({
                "event": "cross_attention_verification",
                "expected_layers": list(expected_layers),
                "found_layers": list(found_layers),
                "timestamp": time.time()
            })

            # Check if all expected layers were found
            if not expected_layers.issubset(found_layers):
                missing_layers = expected_layers - found_layers
                self.logger.record({
                    "warning": f"Missing cross-attention layers: {missing_layers}",
                    "timestamp": time.time()
                })
                return False

            # Verify layer dimensions and structure
            for layer_idx in expected_layers:
                try:
                    layer = self.base_model.transformer.h[layer_idx]
                    if not hasattr(layer, 'cross_attention'):
                        self.logger.record({
                            "warning": f"Layer {layer_idx} missing cross_attention attribute",
                            "timestamp": time.time()
                        })
                        return False

                    # Verify dimensions match
                    if layer.cross_attention.hidden_size != self.base_config.hidden_size:
                        self.logger.record({
                            "warning": f"Layer {layer_idx} dimension mismatch",
                            "timestamp": time.time()
                        })
                        return False

                    # Verify attention heads match
                    if layer.cross_attention.num_attention_heads != self.base_config.num_attention_heads:
                        self.logger.record({
                            "warning": f"Layer {layer_idx} attention heads mismatch",
                            "timestamp": time.time()
                        })
                        return False
                except Exception as e:
                    self.logger.record({
                        "warning": f"Error verifying layer {layer_idx}: {str(e)}",
                        "timestamp": time.time()
                    })
                    return False

            return True
        except Exception as e:
            self.logger.record({
                "error": f"Cross-attention verification failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _insert_cross_attention(self):
        """Inject cross-attention layers using the scaffold injector."""
        if not self.cross_attn_config.get("enable_cross_attention", True):
            self.logger.record({
                "event": "cross_attention",
                "status": "disabled",
                "timestamp": time.time()
            })
            return
            
        try:
            # Use the injector from sovl_scaffold
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
            self.error_logger.record({
                "error": f"Cross-attention injection failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def check_memory_health(self, model_size: int, trainer: Optional[Trainer] = None):
        """Autonomically reduce GPU memory usage if approaching capacity."""
        try:
            if torch.cuda.is_available():
                # Get memory stats
                memory_stats = torch.cuda.memory_stats()
                current_memory = memory_stats["allocated_bytes.all.current"]
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_ratio = current_memory / total_memory
                
                # Log memory usage
                self.logger.log_memory_usage(
                    phase="health_check",
                    device=self.device,
                    memory_ratio=memory_ratio,
                    model_size=model_size
                )
                
                # ... rest of the method implementation ...
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Memory health check failed: {str(e)}",
                error_type="memory_error",
                stack_trace=traceback.format_exc()
            )

    def generate_curiosity_question(self, state: SOVLState, tokenizer: PreTrainedTokenizer,
                                  model: PreTrainedModel, context: str = "",
                                  spontaneous: bool = False) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            # ... question generation implementation ...
            
            if question:
                self.logger.log_curiosity_event(
                    event_type="question_generated",
                    question=question,
                    spontaneous=spontaneous,
                    conversation_id=state.conversation_id,
                    state_hash=state.get_state_hash()
                )
                
            return question
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Question generation failed: {str(e)}",
                error_type="generation_error",
                stack_trace=traceback.format_exc()
            )
            return None

    def tune_curiosity(self, enable=None, spontaneous_threshold=None, response_threshold=None, pressure_threshold=None,
                       pressure_drop=None, silence_threshold=None, question_cooldown=None, queue_maxlen=None,
                       weight_ignorance=None, weight_novelty=None, max_new_tokens=None, base_temperature=None,
                       temperament_influence=None, top_k=None, attention_weight=None, question_timeout=None):
        """Tune curiosity parameters using the state's curiosity manager."""
        self.state.curiosity.tune_curiosity(
            pressure=pressure_threshold,
            decay_rate=pressure_drop,
            question_timeout=question_timeout
        )
        updates = {
            'enable': enable,
            'spontaneous_threshold': spontaneous_threshold,
            'response_threshold': response_threshold,
            'pressure_threshold': pressure_threshold,
            'pressure_drop': pressure_drop,
            'silence_threshold': silence_threshold,
            'question_cooldown': question_cooldown,
            'queue_maxlen': queue_maxlen,
            'weight_ignorance': weight_ignorance,
            'weight_novelty': weight_novelty,
            'max_new_tokens': max_new_tokens,
            'base_temperature': base_temperature,
            'temperament_influence': temperament_influence,
            'top_k': top_k,
            'attention_weight': attention_weight,
            'question_timeout': question_timeout
        }
        self.curiosity_manager.tune_curiosity(**{k: v for k, v in updates.items() if v is not None})

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
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

    def handle_gestation_complete(self, batch_size: int, avg_loss: float):
        self.state.update_gestation_metrics(batch_size, avg_loss)
        self.logger.record({
            "event": "gestation_complete_handled",
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int):
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

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float):
        self.state.update_sleep_metrics(batch_size, data_exposure)
        self.logger.record({
            "event": "sleep_train_complete_handled",
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })

    def update_metrics(self, question, score, spontaneous=False, answered=False):
        """Update curiosity metrics using the state's curiosity manager."""
        if self.state.curiosity:
            self.state.curiosity.update_metrics(
                question=question,
                score=score,
                spontaneous=spontaneous,
                answered=answered,
                conversation_id=self.history.conversation_id,
                state_hash=self.state.state_hash()
            )

    def tune_dream(self, swing_var=None, lifecycle_delta=None, temperament_on=None, noise_scale=None,
                   memory_weight=None, memory_maxlen=None, prompt_weight=None, novelty_boost=None,
                   memory_decay=None, prune_threshold=None):
        updates = {}
        prefix = "controls_config."
        if swing_var is not None and 0.05 <= swing_var <= 0.2:
            updates[f"{prefix}dream_swing_var"] = swing_var
            self.controls_config["dream_swing_var"] = swing_var
        if lifecycle_delta is not None and 0.05 <= lifecycle_delta <= 0.2:
            updates[f"{prefix}dream_lifecycle_delta"] = lifecycle_delta
            self.controls_config["dream_lifecycle_delta"] = lifecycle_delta
        if temperament_on is not None:
            updates[f"{prefix}dream_temperament_on"] = bool(temperament_on)
            self.controls_config["dream_temperament_on"] = bool(temperament_on)
        if noise_scale is not None and 0.01 <= noise_scale <= 0.1:
            updates[f"{prefix}dream_noise_scale"] = noise_scale
            self.controls_config["dream_noise_scale"] = noise_scale
        if memory_weight is not None and 0 <= memory_weight <= 0.5:
            updates[f"{prefix}dream_memory_weight"] = memory_weight
            self.controls_config["dream_memory_weight"] = memory_weight
            self.trainer.config.dream_memory_weight = memory_weight
        if memory_maxlen is not None and 5 <= memory_maxlen <= 20:
            updates[f"{prefix}dream_memory_maxlen"] = memory_maxlen
            self.controls_config["dream_memory_maxlen"] = memory_maxlen
            self.state.dream_memory_maxlen = memory_maxlen
            self.state.dream_memory = deque(self.state.dream_memory, maxlen=memory_maxlen)
        if prompt_weight is not None and 0 <= prompt_weight <= 1:
            updates[f"{prefix}dream_prompt_weight"] = prompt_weight
            self.controls_config["dream_prompt_weight"] = prompt_weight
        if novelty_boost is not None and 0 <= novelty_boost <= 0.05:
            updates[f"{prefix}dream_novelty_boost"] = novelty_boost
            self.controls_config["dream_novelty_boost"] = novelty_boost
        if memory_decay is not None and 0 <= memory_decay <= 1:
            updates[f"{prefix}dream_memory_decay"] = memory_decay
            self.controls_config["dream_memory_decay"] = memory_decay
        if prune_threshold is not None and 0 <= prune_threshold <= 1:
            updates[f"{prefix}dream_prune_threshold"] = prune_threshold
            self.controls_config["dream_prune_threshold"] = prune_threshold

        if updates:
            self.config_manager.update_batch(updates)
            self.config_manager.save_config()
            self.logger.record({
                "event": "tune_dream",
                "params": updates,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            print(f"Dream params updated: {updates}")

    def adjust_temperament(self, eager_threshold=None, sluggish_threshold=None, mood_influence=None,
                          curiosity_boost=None, restless_drop=None, melancholy_noise=None,
                          conf_feedback_strength=None, temp_smoothing_factor=None, decay_rate=None):
        """Adjust temperament parameters using the temperament system."""
        self.temperament_system.adjust_temperament(
            eager_threshold=eager_threshold,
            sluggish_threshold=sluggish_threshold,
            mood_influence=mood_influence,
            curiosity_boost=curiosity_boost,
            restless_drop=restless_drop,
            melancholy_noise=melancholy_noise,
            conf_feedback_strength=conf_feedback_strength,
            temp_smoothing_factor=temp_smoothing_factor,
            decay_rate=decay_rate
        )

    def _update_temperament(self):
        """Update temperament using the temperament system."""
        avg_confidence = safe_divide(
            self.state.sleep_confidence_sum,
            self.state.sleep_confidence_count,
            default=0.5
        )
        lifecycle_stage = safe_divide(
            self.trainer.data_exposure,
            self.trainer.lora_capacity,
            default=0.0
        )

        self.temperament_system.update_temperament(
            confidence=avg_confidence,
            lifecycle_stage=lifecycle_stage,
            time_since_last=None,  # TODO: Add time tracking
            curiosity_pressure=self.curiosity_manager.pressure if self.curiosity_manager else None
        )

        # Update state with new temperament score
        self.state.temperament_score = self.temperament_system.score
        self.state.temperament_history = deque(
            list(self.temperament_system._history),
            maxlen=self.controls_config.get("temperament_history_maxlen", 5)
        )

    def set_global_blend(self, weight_cap=None, base_temp=None):
        updates = {}
        prefix = "controls_config."
        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            updates[f"{prefix}scaffold_weight_cap"] = weight_cap
            self.controls_config["scaffold_weight_cap"] = weight_cap
            self.scaffold_weight = weight_cap
        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            updates[f"{prefix}base_temperature"] = base_temp
        self.state.temperament_history.append(self.state.temperament_score)
    
        if self.curiosity_config.get("enable_curiosity", True) and self.curiosity_manager:
            self.curiosity_manager.update_pressure(
                temperament=self.state.temperament_score,
                confidence=avg_confidence,
                silence_duration=0.0,
                context_vector=self.state.curiosity.context_vector
            )
    
        label = (
            "melancholic" if float_lt(self.state.temperament_score, -0.5) else
            "restless" if float_lt(self.state.temperament_score, 0.0) else
            "calm" if float_lt(self.state.temperament_score, 0.5) else "curious"
        )
        self.logger.record({
            "event": "temperament_updated",
            "score": self.state.temperament_score,
            "label": label,
            "lifecycle_stage": lifecycle_stage,
            "avg_confidence": avg_confidence,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"Temperament score: {self.state.temperament_score:.3f} ({label}, lifecycle: {lifecycle_stage:.2f}), confidence feedback: {avg_confidence:.2f}")
    
    def train_step(self, batch):
        """Execute a single training step with scaffold context."""
        try:
            max_seq_length = self.training_config.get("max_seq_length", 128)
            if self.dry_run:
                print("Dry run train step")
                dry_batch = [
                    {
                        'prompt': item['prompt'][:self.dry_run_params['max_length']],
                        'completion': item['completion'][:self.dry_run_params['max_length']]
                    }
                    for item in batch[:self.dry_run_params['max_samples']]
                ]
                formatted_batch = collate_batch(
                    dry_batch,
                    self.base_tokenizer.pad_token_id,
                    max_seq_length,
                    self.base_tokenizer
                )
                prompts = formatted_batch['prompt']
                scaffold_inputs = self.tokenize_and_map(prompts, max_length=max_seq_length)
                scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
                loss, confidence = self.trainer.train_step(
                    batch=formatted_batch,
                    scaffold_context=scaffold_hidden_states,
                    dry_run=True
                )
                self.logger.record({
                    "event": "dry_run_train_step",
                    "loss": float(loss) if loss is not None else None,
                    "confidence": float(confidence) if confidence is not None else None,
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "state_hash": self.state.state_hash()
                })
                print(f"Dry run loss: {loss}")
                return None
    
            prompts = [item['prompt'] for item in batch]
            scaffold_inputs = self.tokenize_and_map(prompts, max_length=max_seq_length)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
    
            formatted_batch = collate_batch(
                batch,
                self.base_tokenizer.pad_token_id,
                max_seq_length,
                self.base_tokenizer
            )
            loss, confidence = self.trainer.train_step(
                batch=formatted_batch,
                scaffold_context=scaffold_hidden_states,
                grad_clip=True,
                dry_run=False,
                memory_check=self.check_memory_health
            )
    
            if loss is not None and self.use_token_map_memory and confidence is not None:
                self._update_token_map_memory(prompts[0], confidence)
    
            self.logger.record({
                "event": "training_step",
                "loss": float(loss) if loss is not None else None,
                "confidence": float(confidence) if confidence is not None else None,
                "batch_size": len(batch),
                "timestamp": time.time(),
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
    
            return loss
    
        except Exception as e:
            self.error_logger.record({
                "error": str(e),
                "type": type(e).__name__,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "batch_size": len(batch),
                "phase": "training",
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            raise
        
    def run_training_cycle(self, train_data, valid_data, epochs=None, batch_size=None):
        """Run a full training cycle."""
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
    
        if self.dry_run and self.dry_run_params['skip_training']:
            print("\n=== DRY RUN TRAINING ===")
            dry_batch = train_data[:self.dry_run_params['max_samples']]
            loss = self.train_step(dry_batch)
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
            prompts = batch['prompt']
            scaffold_inputs = self.tokenize_and_map(prompts)
            return self.get_scaffold_hidden_states(scaffold_inputs)
    
        self.trainer.train(
            train_data=train_data,
            valid_data=valid_data,
            scaffold_provider=scaffold_provider
        )
    
        self.last_weight = self.trainer.get_life_curve_weight()
        self.set_scaffold_influence(self.last_weight)
        self.logger.record({
            "event": "training_cycle_complete",
            "duration": time.time() - start_time,
            "last_weight": self.last_weight,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"--- Training Finished ({time.time() - start_time:.2f}s) ---")
    
    def has_repetition(self, output_ids, n=3):
        """Check for repetition in generated output."""
        ids = output_ids.tolist()
        special_ids = {
            self.base_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id,
            self.base_tokenizer.bos_token_id,
            self.base_tokenizer.unk_token_id
        }
        filtered = [i for i in ids if i not in special_ids]
        for i in range(len(filtered) - 2 * n):
            if filtered[i:i + n] == filtered[i + n:i + 2 * n]:
                return True
        return False
    
    def _handle_error_prompt(self, error_msg):
        """Generate a response to a system error."""
        temp_history = self.history
        self.history = ConversationHistory(maxlen=self.controls_config.get("conversation_history_maxlen", 10))
        response = self.generate(
            f"System error detected: {error_msg} What happened?",
            max_new_tokens=self.curiosity_config.get("max_new_tokens", 60),
            temperature=self.controls_config.get("base_temperature", 0.7) + 0.2,
            top_k=self.curiosity_config.get("top_k", 50),
            do_sample=True
        )
        self.logger.record({
            "prompt": f"System error detected: {error_msg} What happened?",
            "response": response,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "is_error_prompt": True,
            "confidence_score": 0.5,
            "state_hash": self.state.state_hash()
        })
        self.history = temp_history
        return response
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        """Generate a response for the given prompt."""
        generated_ids = []
        try:
            max_new_tokens = max_new_tokens or self.curiosity_config.get("max_new_tokens", 50)
            generation_params = {
                "prompt_length": len(prompt),
                "max_new_tokens": max_new_tokens,
                "scaffold_weight": scaffold_weight,
                "temperature": kwargs.get("temperature", self.controls_config.get("base_temperature", 0.7)),
                "top_k": kwargs.get("top_k", self.curiosity_config.get("top_k", 30)),
                "do_sample": kwargs.get("do_sample", False)
            }
            self.logger.record({
                "event": "generation_initiated",
                "params": generation_params,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            print(f"Generation initiated: prompt='{prompt[:30]}...', max_new_tokens={max_new_tokens}, scaffold_weight={scaffold_weight}")
    
            self.check_memory_health()
            if self.is_sleeping:
                print("\rGestation Interrupted", end="", flush=True)
                time.sleep(0.5)
                print("\r                   ", end="")
                self._reset_sleep_state()
    
            start_time = time.time()
            base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
            input_ids = base_inputs['input_ids']
            input_length = input_ids.shape[1]
    
            scaffold_inputs = self.tokenize_and_map(prompt)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
    
            temp = adjust_temperature(
                self.controls_config.get("base_temperature", 0.7),
                self.state.temperament_score,
                self.controls_config.get("temp_mood_influence", 0.5)
            )
            generation_params["adjusted_temperature"] = temp
    
            dynamic_factor = None
            if self.controls_config.get("enable_dynamic_cross_attention", False) and self.dynamic_cross_attn_mode:
                try:
                    last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
                    if self.dynamic_cross_attn_mode == 'confidence':
                        dynamic_factor = torch.tensor(last_conf, device=DEVICE, dtype=torch.float)
                    elif self.dynamic_cross_attn_mode == 'temperament':
                        dynamic_factor = torch.tensor(self.state.temperament_score, device=DEVICE, dtype=torch.float)
                    else:
                        self.logger.record({
                            "warning": f"Invalid dynamic factor for mode {self.dynamic_cross_attn_mode}",
                            "timestamp": time.time(),
                            "conversation_id": self.history.conversation_id,
                            "state_hash": self.state.state_hash()
                        })
                except Exception as e:
                    self.logger.record({
                        "warning": f"Failed to compute dynamic factor: {str(e)}",
                        "timestamp": time.time(),
                        "conversation_id": self.history.conversation_id,
                        "state_hash": self.state.state_hash()
                    })
    
            self._clear_scaffold_cache()
            chunk_size = self.training_config.get("generation_chunk_size", 512)
            memory_tensors = None
            dream_memory_info = {"used": False, "tensor_count": 0, "shapes": []}
            dream_memory_weight = self.controls_config.get("dream_memory_weight", 0.1)
            if self.state.dream_memory and dream_memory_weight > 0:
                try:
                    with self.state.memory_lock:
                        dream_tensors, dream_weights = zip(*self.state.dream_memory)
                        dream_memory_info["tensor_count"] = len(dream_tensors)
                        dream_memory_info["shapes"] = [list(t.shape) for t in dream_tensors]
                        for tensor in dream_tensors:
                            if tensor.shape[-1] != self.state.hidden_size:
                                raise ValueError(f"Dream tensor shape {tensor.shape} mismatches hidden_size {self.state.hidden_size}")
                        dream_tensors = torch.stack([t.detach().to(DEVICE) for t in dream_tensors])
                        dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=DEVICE)
                        memory_tensors = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / dream_weights.sum()
                        dream_memory_info["used"] = True
                except Exception as e:
                    self.logger.record({
                        "warning": f"Dream memory preparation failed: {str(e)}",
                        "timestamp": time.time(),
                        "conversation_id": self.history.conversation_id,
                        "dream_memory_len": len(self.state.dream_memory),
                        "dream_tensor_shapes": [tuple(t.shape) for t, _ in self.state.dream_memory] if self.state.dream_memory else [],
                        "state_hash": self.state.state_hash()
                    })
                    dream_memory_info["error"] = str(e)
    
            with self._scaffold_context(scaffold_hidden_states):
                self.set_scaffold_influence(weight=scaffold_weight)
                for chunk_start in range(0, input_ids.size(1), chunk_size):
                    chunk_end = chunk_start + chunk_size
                    input_chunk = input_ids[:, chunk_start:chunk_end]
                    outputs = self.base_model.generate(
                        input_chunk,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.base_tokenizer.pad_token_id,
                        eos_token_id=self.base_tokenizer.eos_token_id,
                        temperature=temp,
                        return_dict_in_generate=True,
                        output_scores=True,
                        scaffold_context=scaffold_hidden_states,
                        memory_tensors=memory_tensors,
                        memory_weight=dream_memory_weight,
                        dynamic_factor=dynamic_factor,
                        **kwargs
                    )
                    generated_ids.extend(outputs.sequences[0][input_length:].tolist())
    
            print(f"Generation completed in {time.time() - start_time:.2f}s.")
            confidence_score = 0.5
            if self.controls_config.get("enable_confidence_tracking", True):
                confidence_score = calculate_confidence_score(outputs.scores, generated_ids)
                with self.memory_lock:
                    self.state.sleep_confidence_sum += confidence_score
                    self.state.sleep_confidence_count += 1
                    self.state.confidence_history.append(confidence_score)
    
            special_ids = {
                self.base_tokenizer.pad_token_id,
                self.base_tokenizer.eos_token_id,
                self.base_tokenizer.bos_token_id,
                self.base_tokenizer.unk_token_id
            }
            if self.controls_config.get("enable_repetition_check", True) and self.has_repetition(torch.tensor(generated_ids)):
                print("Warning: Repetition detected. Truncating.")
                original_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                for i in range(len(generated_ids) - 6):
                    if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                        generated_ids = generated_ids[:i + 3]
                        break
                self.logger.record({
                    "warning": "Repetition detected",
                    "original_text": original_text,
                    "truncated_at": i + 3,
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "state_hash": self.state.state_hash()
                })
            response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
            last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
            if self.curiosity_config.get("enable_curiosity", True) and self.curiosity_manager:
                self.state.curiosity.prune_old_questions(self.curiosity_config.get("question_timeout", 3600.0))
                self.curiosity_manager.update_pressure(
                    self.state.temperament_score,
                    last_conf,
                    0.0,
                    self.state.curiosity.context_vector
                )
                if self.curiosity_manager.should_erupt():
                    q = self.generate_curiosity_question(prompt)
                    if q and isinstance(q, str) and q.strip():
                        response += f" {q}"
                        self.state.curiosity.update_question_history(q, time.time())
                        self.logger.record({
                            "prompt": q,
                            "response": "",
                            "timestamp": time.time(),
                            "conversation_id": self.history.conversation_id,
                            "confidence_score": 0.0,
                            "is_system_question": True,
                            "state_hash": self.state.state_hash()
                        })
    
            log_entry = {
                "prompt": prompt,
                "response": response,
                "timestamp": start_time,
                "conversation_id": self.history.conversation_id,
                "confidence_score": confidence_score,
                "is_system_question": False,
                "generation_params": generation_params,
                "dream_memory_info": dream_memory_info,
                "state_hash": self.state.state_hash()
            }
            self.logger.record(log_entry)
            self.history.add_message(prompt, response)
            if self.use_token_map_memory:
                self._update_token_map_memory(prompt, confidence_score)
    
            if self.controls_config.get("enable_gestation", True) and self._should_gestate():
                self._gestate()
    
            print(f"Generation took {time.time() - start_time:.2f} seconds.")
            return response
    
        except torch.cuda.OutOfMemoryError as oom:
            error_details = {
                "error": "CUDA out of memory",
                "type": "OOM",
                "prompt": prompt[:200],
                "timestamp": time.time(),
                "memory_stats": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                } if torch.cuda.is_available() else None,
                "generation_params": generation_params,
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            }
            self.error_logger.record(error_details)
            torch.cuda.empty_cache()
    
            if self.controls_config.get("enable_error_listening", True):
                try:
                    return self._handle_error_prompt("GPU memory error occurred")
                except Exception as e:
                    self.error_logger.record({
                        "error": f"Failed to handle OOM error: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.history.conversation_id,
                        "state_hash": self.state.state_hash()
                    })
                    return "System is low on memory - please try a shorter prompt"
            return "System is low on memory - please try a shorter prompt"
    
        except Exception as e:
            error_details = {
                "error": str(e),
                "type": type(e).__name__,
                "prompt": prompt[:200],
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "generation_params": generation_params,
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            }
            self.error_logger.record(error_details)
    
            if self.controls_config.get("enable_error_listening", True):
                try:
                    return self._handle_error_prompt(f"Generation error: {str(e)}")
                except Exception as inner_e:
                    self.error_logger.record({
                        "error": f"Failed to handle generation error: {str(inner_e)}",
                        "original_error": str(e),
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.history.conversation_id,
                        "state_hash": self.state.state_hash()
                    })
            return "An error occurred during generation"
    
        finally:
            if generated_ids:
                del generated_ids
            if 'outputs' in locals():
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.record({
                "event": "generate_cleanup",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
    
    @torch.no_grad()
    def validate_epoch(self, valid_data):
        """Validate the model on the provided data."""
        if self.dry_run:
            print("\n=== DRY RUN VALIDATION ===")
            loss = random.random()
            self.logger.record({
                "event": "dry_run_validation",
                "loss": loss,
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "state_hash": self.state.state_hash()
            })
            return loss
    
        def scaffold_provider(batch):
            prompts = batch['prompt']
            scaffold_inputs = self.tokenize_and_map(prompts)
            return self.get_scaffold_hidden_states(scaffold_inputs)
    
        valid_loss, metrics = self.trainer.validate(valid_data, scaffold_provider)
        self.logger.record({
            "event": "validation_complete",
            "loss": valid_loss,
            "metrics": metrics,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"Validation Loss: {valid_loss:.4f}, Metrics: {metrics}")
        return valid_loss
    
    def cleanup(self, model_size: int, trainer: Optional[Trainer] = None):
        """Free GPU memory."""
        try:
            # ... cleanup implementation ...
            
            self.logger.log_cleanup_event(
                phase="cleanup",
                success=True,
                model_size=model_size
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Cleanup failed: {str(e)}",
                error_type="cleanup_error",
                stack_trace=traceback.format_exc()
            )
    
    def _should_gestate(self):
        """Check if gestation should occur based on log size and time."""
        if not self.controls_config.get("enable_gestation", True):
            return False
        log_entries = self.logger.read()
        time_since_last = time.time() - self.last_trained
        sleep_log_min = self.controls_config.get("sleep_log_min", 10)
        gestation_cooldown = self.controls_config.get("gestation_cooldown", 60.0)
        should_gestate = len(log_entries) >= sleep_log_min and time_since_last > gestation_cooldown
        self.logger.record({
            "event": "check_gestation",
            "should_gestate": should_gestate,
            "log_entries_count": len(log_entries),
            "time_since_last": time_since_last,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        return should_gestate
    
    def new_conversation(self):
        """Start a new conversation."""
        self.memory_manager.new_conversation()
        self.logger.record({
            "event": "new_conversation",
            "new_id": self.history.conversation_id,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"New conversation: {self.history.conversation_id}")
    
    def _reset_sleep_state(self):
        """Reset sleep-related state."""
        self.is_sleeping = False
        self.trainer._reset_sleep_state()
        self.logger.record({
            "event": "reset_sleep_state",
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
    
    def tokenize_and_map(self, prompts, max_length=None):
        """Tokenize prompts and map to scaffold token space."""
        if self.scaffold_token_mapper is None:
            self.scaffold_token_mapper = build_scaffold_token_mapping(
                self.base_tokenizer,
                self.scaffold_tokenizer
            )
        return self.scaffold_token_mapper.tokenize_and_map(prompts, max_length)

    def _update_token_map_memory(self, prompt, confidence):
        """Update token map memory based on prompt confidence."""
        if self.scaffold_token_mapper is None:
            self.scaffold_token_mapper = build_scaffold_token_mapping(
                self.base_tokenizer,
                self.scaffold_tokenizer
            )
        self.scaffold_token_mapper.update_token_map_memory(prompt, confidence)

    # Main block
    if __name__ == "__main__":
        from sovl_config import ConfigManager
        from sovl_cli import run_cli
        config_manager = ConfigManager("sovl_config.json")
        system = SOVLSystem(config_manager)
        run_cli(config_manager=config_manager)    
