from typing import Optional, Any, List, Dict
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
from sovl_manager import ModelManager
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner
from sovl_error import ErrorHandler

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

        # Initialize error handler
        self.error_handler = ErrorHandler(
            config_manager=config_manager,
            logger=self.logger,
            error_log_file="sovl_errors.jsonl",
            max_error_log_size_mb=10,
            compress_old=True,
            state=None  # Will be set after state initialization
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

        # Initialize model manager
        self.model_manager = ModelManager(
            config_manager=config_manager,
            logger=self.logger,
            device=DEVICE
        )

        # Get models and tokenizers from model manager
        self.base_model = self.model_manager.get_base_model()
        self.scaffolds = [self.model_manager.get_scaffold_model()]
        self.base_tokenizer = self.model_manager.get_base_tokenizer()
        self.scaffold_tokenizer = self.model_manager.get_scaffold_tokenizer()
        self.scaffold_unk_id = self.model_manager.get_scaffold_unk_id()

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

        # Initialize token map using ScaffoldManager
        self.scaffold_manager = ScaffoldManager(config_manager, self.logger)
        self.scaffold_token_mapper = None  # Will be initialized when needed

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

        # Initialize generation manager
        self.generation_manager = GenerationManager(
            config_manager=config_manager,
            base_model=self.base_model,
            scaffolds=self.scaffolds,
            base_tokenizer=self.base_tokenizer,
            scaffold_tokenizer=self.scaffold_tokenizer,
            state=self.state,
            logger=self.logger,
            error_logger=self.error_logger,
            cross_attention_injector=self.cross_attention_injector,
            scaffold_manager=self.scaffold_manager,
            temperament=self.temperament_system,
            curiosity_manager=self.curiosity_manager
        )

        self.tuner = SOVLTuner(
            config_manager=self.config_manager,
            logger=self.logger,
            curiosity_manager=self.curiosity_manager,
            trainer=self.trainer,
            cross_attention_injector=self.cross_attention_injector
        )

    def _validate_config(self, model_config: Optional[Any] = None) -> bool:
        """
        Validate system configuration.

        Args:
            model_config: Optional model configuration for layer validation

        Returns:
            bool: True if validation succeeds, False otherwise
        """
        try:
            # Delegate validation to ConfigManager
            if not self.config_manager.validate_config(model_config):
                return False

            # Additional system-specific validation can be added here if needed
            return True

        except Exception as e:
            self.logger.record({
                "error": f"Configuration validation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": "validate"
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

            # Verify the injection
            expected_layers = self.core_config.get("cross_attn_layers", [])
            if not self.cross_attention_injector.verify_injection(
                self.base_model,
                expected_layers,
                self.base_config
            ):
                raise ValueError("Cross-attention layer verification failed")

        except Exception as e:
            self.error_handler.handle_cross_attention_error(e)

    def check_memory_health(self, model_size: int, trainer: Optional[Trainer] = None):
        """Delegate memory health check to MemoryManager."""
        return self.memory_manager.check_memory_health(model_size, trainer)

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
            self.error_handler.handle_curiosity_error(e, "question_generation")
            return None

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

    def train_step(self, batch: List[dict], dry_run: bool = False, dry_run_params: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Execute a single training step.
        
        Args:
            batch: List of training examples
            dry_run: Whether to perform a dry run
            dry_run_params: Parameters for dry run if enabled
            
        Returns:
            Optional[float]: Loss value if training was performed, None if dry run
        """
        try:
            # Get scaffold context from scaffold manager
            scaffold_provider = self.scaffold_manager.get_scaffold_context if hasattr(self, 'scaffold_manager') else None
            
            # Delegate to trainer
            return self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=scaffold_provider,
                dry_run=dry_run,
                dry_run_params=dry_run_params
            )
            
        except Exception as e:
            self.logger({
                "event": "training_error",
                "error": str(e),
                "batch_size": len(batch),
                "timestamp": time.time(),
                "conversation_id": getattr(self.state, "conversation_id", "training"),
                "state_hash": getattr(self.state, "state_hash", None)
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

        training_results = self.trainer.run_training_cycle(
            train_data=train_data,
            validation_data=valid_data,
            scaffold_provider=scaffold_provider,
            max_epochs=epochs,
            early_stopping_patience=self.training_config.get("max_patience", 3)
        )

        self.last_weight = self.trainer.get_life_curve_weight()
        self.set_scaffold_influence(self.last_weight)
        self.logger.record({
            "event": "training_cycle_complete",
            "duration": time.time() - start_time,
            "last_weight": self.last_weight,
            "training_history": training_results.get("training_history", []),
            "best_val_loss": training_results.get("best_val_loss", float('inf')),
            "final_epoch": training_results.get("final_epoch", 0),
            "early_stopped": training_results.get("early_stopped", False),
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "state_hash": self.state.state_hash()
        })
        print(f"--- Training Finished ({time.time() - start_time:.2f}s) ---")

    def _sleep_train(self):
        """Train on dream-generated content."""
        if not self.controls_config.get("enable_sleep_training", True):
            return
            
        print("\n--- Sleep Training Initiated ---")
        log_entries = self.logger.read()
        
        # Delegate sleep training to trainer
        self.trainer.sleep_train(log_entries)
        
        # Update system state
        self.last_trained = time.time()
        self.logger.clear()
        self.last_weight = self.trainer.get_life_curve_weight()
        
        # Update temperament if enabled
        if self.enable_temperament:
            self._update_temperament()
            self.last_temperament_score = self.temperament_score
            
        print("--- Sleep Training Complete ---")

    def has_repetition(self, output_ids, n=3):
        """Check for repetition in generated output."""
        return self.generation_manager.has_repetition(output_ids, n)
    
    def _handle_error_prompt(self, error_msg):
        """Generate a response to a system error."""
        return self.generation_manager._handle_error_prompt(error_msg)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        """Generate a response for the given prompt using the generation manager."""
        try:
            return self.generation_manager.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                scaffold_weight=scaffold_weight,
                **kwargs
            )
        except Exception as e:
            return self.error_handler.handle_generation_error(e, prompt)

    def tokenize_and_map(self, prompts, max_length=None):
        """Tokenize prompts and map to scaffold token space using the generation manager."""
        return self.generation_manager.tokenize_and_map(prompts, max_length)

    def _update_token_map_memory(self, prompt, confidence):
        """Update token map memory based on prompt confidence using the generation manager."""
        self.generation_manager._update_token_map_memory(prompt, confidence)

    def _clear_scaffold_cache(self):
        """Clear scaffold-related caches using the generation manager."""
        self.generation_manager._clear_scaffold_cache()

    # Main block
    if __name__ == "__main__":
        from sovl_config import ConfigManager
        from sovl_cli import run_cli
        config_manager = ConfigManager("sovl_config.json")
        system = SOVLSystem(config_manager)
        run_cli(config_manager=config_manager)    
