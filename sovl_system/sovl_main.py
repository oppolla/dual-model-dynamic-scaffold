from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import time
import random
import bitsandbytes as bnb
import json
import sys
import contextlib
from collections import deque, defaultdict
import uuid
import os
from threading import Lock
from sovl_logger import Logger
from sovl_io import load_config, get_config_value, load_jsonl
from sovl_state import SOVLState, ConversationHistory
from sovl_trainer import TrainingConfig, SOVLTrainer, collate_batch

class InsufficientDataError(Exception):
    pass

from sovl_processor import LogitsProcessor

def calculate_confidence_score(logits, generated_ids) -> float:
    """Wrapper for backward compatibility"""
    try:
        processor = LogitsProcessor(logits)
        return processor.calculate_confidence(generated_ids)
    except Exception as e:
        print(f"Confidence score error: {str(e)} - Using default 0.5")
        return 0.5

# Load training data
try:
    TRAIN_DATA = load_jsonl("sovl_seed.jsonl")
except InsufficientDataError as e:
    print(e)
    TRAIN_DATA = []  # Fallback to empty list
except Exception as e:
    print(f"Unexpected error during data loading: {e}")
    TRAIN_DATA = []  # Fallback to empty list

if not TRAIN_DATA:
    print("Error: No data loaded from sovl_seed.jsonl!")

# Load config and set global variables
with open("sovl_config.json", "r") as f:
    config = json.load(f)

def get_config_value(config, key, default=None):
    if isinstance(config, dict):
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
            if not isinstance(value, dict) and k != keys[-1]:
                return default
        if isinstance(value, dict) and not value:
            print(f"Warning: '{key}' missing or empty, using {default}")
            return default
        return value if value != {} else default
    print(f"Warning: '{key}' not found, using {default}")
    return default

# Core Model Config
core_config = config.get("core_config", {})
BASE_MODEL_NAME = get_config_value(core_config, "base_model_name", "gpt2")
SCAFFOLD_MODEL_NAME = get_config_value(core_config, "scaffold_model_name", "gpt2")
CROSS_ATTN_LAYERS = get_config_value(core_config, "cross_attn_layers", [0, 1, 2])
USE_DYNAMIC_LAYERS = get_config_value(core_config, "use_dynamic_layers", False)
LAYER_SELECTION_MODE = get_config_value(core_config, "layer_selection_mode", "balanced")
CUSTOM_LAYERS = get_config_value(core_config, "custom_layers", [])
VALID_SPLIT_RATIO = get_config_value(core_config, "valid_split_ratio", 0.2)
RANDOM_SEED = get_config_value(core_config, "random_seed", 42)
QUANTIZATION_MODE = get_config_value(core_config, "quantization", "fp16")
if QUANTIZATION_MODE not in ["fp16", "int8", "int4"]:
    print(f"Warning: Invalid quantization '{QUANTIZATION_MODE}'. Defaulting to 'fp16'.")
    QUANTIZATION_MODE = "fp16"

# LoRA Config
lora_config = get_config_value(config, "lora_config", {})
LORA_RANK = get_config_value(lora_config, "lora_rank", 8)
LORA_ALPHA = get_config_value(lora_config, "lora_alpha", 16)
LORA_DROPOUT = get_config_value(lora_config, "lora_dropout", 0.1)
LORA_TARGET_MODULES = get_config_value(lora_config, "lora_target_modules", ["q_proj", "v_proj"])

# Training Config
training_config = get_config_value(config, "training_config", {})
LEARNING_RATE = get_config_value(training_config, "learning_rate", 2e-5)
TRAIN_EPOCHS = get_config_value(training_config, "train_epochs", 3)
BATCH_SIZE = get_config_value(training_config, "batch_size", 2)
MAX_SEQ_LENGTH = get_config_value(training_config, "max_seq_length", 512)
SIGMOID_SCALE = get_config_value(training_config, "sigmoid_scale", 0.5)
SIGMOID_SHIFT = get_config_value(training_config, "sigmoid_shift", 5.0)

# Exposed Controls
controls_config = get_config_value(config, "controls_config", {})
SLEEP_CONF_THRESHOLD = get_config_value(controls_config, "sleep_conf_threshold", 0.7)
SLEEP_TIME_FACTOR = get_config_value(controls_config, "sleep_time_factor", 1.0)
SLEEP_LOG_MIN = get_config_value(controls_config, "sleep_log_min", 10)
DREAM_SWING_VAR = get_config_value(controls_config, "dream_swing_var", 0.1)
DREAM_LIFECYCLE_DELTA = get_config_value(controls_config, "dream_lifecycle_delta", 0.1)
DREAM_TEMPERAMENT_ON = get_config_value(controls_config, "dream_temperament_on", True)
DREAM_NOISE_SCALE = get_config_value(controls_config, "dream_noise_scale", 0.05)
TEMP_EAGER_THRESHOLD = get_config_value(controls_config, "temp_eager_threshold", 0.8)
TEMP_SLUGGISH_THRESHOLD = get_config_value(controls_config, "temp_sluggish_threshold", 0.6)
MEMORY_THRESHOLD = get_config_value(controls_config, "memory_threshold", 0.85)
if not isinstance(MEMORY_THRESHOLD, (int, float)) or MEMORY_THRESHOLD <= 0 or MEMORY_THRESHOLD > 1:
    MEMORY_THRESHOLD = 0.85  # Reset to default if invalid
TEMP_MOOD_INFLUENCE = get_config_value(controls_config, "temp_mood_influence", 0.0)
SCAFFOLD_WEIGHT_CAP = get_config_value(controls_config, "scaffold_weight_cap", 1.0)
BASE_TEMPERATURE = get_config_value(controls_config, "base_temperature", 0.7)
SAVE_PATH_PREFIX = get_config_value(controls_config, "save_path_prefix", "state")
DREAM_MEMORY_WEIGHT = get_config_value(controls_config, "dream_memory_weight", 0.1)
DREAM_MEMORY_MAXLEN = get_config_value(controls_config, "dream_memory_maxlen", 10)
DREAM_PROMPT_WEIGHT = get_config_value(controls_config, "dream_prompt_weight", 0.5)
DREAM_NOVELTY_BOOST = get_config_value(controls_config, "dream_novelty_boost", 0.03)
TEMP_CURIOSITY_BOOST = get_config_value(controls_config, "temp_curiosity_boost", 0.5)
TEMP_RESTLESS_DROP = get_config_value(controls_config, "temp_restless_drop", 0.1)
TEMP_MELANCHOLY_NOISE = get_config_value(controls_config, "temp_melancholy_noise", 0.02)
CONF_FEEDBACK_STRENGTH = get_config_value(controls_config, "conf_feedback_strength", 0.5)
TEMP_SMOOTHING_FACTOR = get_config_value(controls_config, "temp_smoothing_factor", 0.0)
LIFECYCLE_CAPACITY_FACTOR = get_config_value(training_config, "lifecycle_capacity_factor", 0.01)
LIFECYCLE_CURVE = get_config_value(training_config, "lifecycle_curve", "sigmoid_linear")
DREAM_MEMORY_DECAY = get_config_value(controls_config, "dream_memory_decay", 0.95)
DREAM_PRUNE_THRESHOLD = get_config_value(controls_config, "dream_prune_threshold", 0.1)
USE_SCAFFOLD_MEMORY = get_config_value(controls_config, "use_scaffold_memory", True)
USE_TOKEN_MAP_MEMORY = get_config_value(controls_config, "use_token_map_memory", True)
DYNAMIC_CROSS_ATTN_MODE = get_config_value(controls_config, "dynamic_cross_attn_mode", None)
DRY_RUN = get_config_value(training_config, "dry_run", False)
dry_run_config = get_config_value(training_config, "dry_run_params", {})
DRY_RUN_MAX_SAMPLES = get_config_value(dry_run_config, "max_samples", 2)
DRY_RUN_MAX_LENGTH = get_config_value(dry_run_config, "max_length", 128)
DRY_RUN_VALIDATE_ARCH = get_config_value(dry_run_config, "validate_architecture", True)
DRY_RUN_SKIP_TRAINING = get_config_value(dry_run_config, "skip_training", True)
MEMORY_DECAY_RATE = get_config_value(controls_config, "memory_decay_rate", 0.95)
HAS_WOKEN = get_config_value(controls_config, "has_woken", False)
IS_SLEEPING = get_config_value(controls_config, "is_sleeping", False)
ACCUMULATION_STEPS = get_config_value(training_config, "accumulation_steps", 4)
EXPOSURE_GAIN_EAGER = get_config_value(training_config, "exposure_gain_eager", 3)
EXPOSURE_GAIN_DEFAULT = get_config_value(training_config, "exposure_gain_default", 2)
MAX_PATIENCE = get_config_value(training_config, "max_patience", 2)
CONFIDENCE_HISTORY_MAXLEN = get_config_value(controls_config, "confidence_history_maxlen", 5)
TEMPERAMENT_HISTORY_MAXLEN = get_config_value(controls_config, "temperament_history_maxlen", 5)
ENABLE_DREAMING = get_config_value(controls_config, "enable_dreaming", True)
ENABLE_TEMPERAMENT = get_config_value(controls_config, "enable_temperament", True)
ENABLE_CONFIDENCE_TRACKING = get_config_value(controls_config, "enable_confidence_tracking", True)
ENABLE_GESTATION = get_config_value(controls_config, "enable_gestation", True)
ENABLE_SLEEP_TRAINING = get_config_value(controls_config, "enable_sleep_training", True)
ENABLE_CROSS_ATTENTION = get_config_value(controls_config, "enable_cross_attention", True)
ENABLE_DYNAMIC_CROSS_ATTENTION = get_config_value(controls_config, "enable_dynamic_cross_attention", True)
ENABLE_LORA_ADAPTERS = get_config_value(controls_config, "enable_lora_adapters", True)
ENABLE_REPETITION_CHECK = get_config_value(controls_config, "enable_repetition_check", True)
ENABLE_PROMPT_DRIVEN_DREAMS = get_config_value(controls_config, "enable_prompt_driven_dreams", True)
ENABLE_LIFECYCLE_WEIGHTING = get_config_value(controls_config, "enable_lifecycle_weighting", True)
ENABLE_ERROR_LISTENING = get_config_value(controls_config, "enable_error_listening", True)
# TCQS Controls
ENABLE_CURIOSITY = get_config_value(controls_config, "enable_curiosity", True)
CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS = get_config_value(controls_config, "curiosity_novelty_threshold_spontaneous", 0.9)
CURIOSITY_NOVELTY_THRESHOLD_RESPONSE = get_config_value(controls_config, "curiosity_novelty_threshold_response", 0.8)
CURIOSITY_PRESSURE_THRESHOLD = get_config_value(controls_config, "curiosity_pressure_threshold", 0.7)
CURIOSITY_PRESSURE_DROP = get_config_value(controls_config, "curiosity_pressure_drop", 0.3)
CURIOSITY_SILENCE_THRESHOLD = get_config_value(controls_config, "curiosity_silence_threshold", 20.0)
CURIOSITY_QUESTION_COOLDOWN = get_config_value(controls_config, "curiosity_question_cooldown", 60.0)
CURIOSITY_QUEUE_MAXLEN = get_config_value(controls_config, "curiosity_queue_maxlen", 10)
CURIOSITY_WEIGHT_IGNORANCE = get_config_value(controls_config, "curiosity_weight_ignorance", 0.7)
CURIOSITY_WEIGHT_NOVELTY = get_config_value(controls_config, "curiosity_weight_novelty", 0.3)
# Curiosity Generation Controls
CURIOSITY_MAX_NEW_TOKENS = get_config_value(controls_config, "curiosity_max_new_tokens", 8)
CURIOSITY_BASE_TEMPERATURE = get_config_value(controls_config, "curiosity_base_temperature", 1.1)
CURIOSITY_TEMPERAMENT_INFLUENCE = get_config_value(controls_config, "curiosity_temperament_influence", 0.4)
CURIOSITY_TOP_K = get_config_value(controls_config, "curiosity_top_k", 30)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def _validate_config():
    required_keys = ["core_config.base_model_name", "training_config.learning_rate"]
    missing_keys = []
    for key in required_keys:
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
        if not value:
            missing_keys.append(key)
    
    if missing_keys:
        error_message = f"Missing required configuration keys: {', '.join(missing_keys)}"
        print(f"Configuration Validation Error: {error_message}")
        raise ValueError(error_message)

    assert isinstance(CROSS_ATTN_LAYERS, list), "CROSS_ATTN_LAYERS must be a list!"
    if not USE_DYNAMIC_LAYERS:
        base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        invalid_layers = [l for l in CROSS_ATTN_LAYERS if not (0 <= l < base_config.num_hidden_layers)]
        if invalid_layers:
            raise ValueError(f"Invalid CROSS_ATTN_LAYERS: {invalid_layers} for {base_config.num_hidden_layers} layers.")
    if LAYER_SELECTION_MODE == "custom":
        base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        invalid_custom = [l for l in CUSTOM_LAYERS if not (0 <= l < base_config.num_hidden_layers)]
        if invalid_custom:
            raise ValueError(f"Invalid CUSTOM_LAYERS: {invalid_custom} for {BASE_MODEL_NAME}")

_validate_config()

# Split training data
random.seed(RANDOM_SEED)
random.shuffle(TRAIN_DATA)
split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")
if not TRAIN_DATA or not VALID_DATA:
    print("Warning: TRAIN_DATA or VALID_DATA empty. Training may fail.")

def get_cross_attention_layers(model):
    total_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.layers)
    if total_layers == 0:
        return []
    if LAYER_SELECTION_MODE == "early":
        return list(range(0, total_layers//3))
    elif LAYER_SELECTION_MODE == "late":
        return list(range(2*total_layers//3, total_layers))
    elif LAYER_SELECTION_MODE == "custom" and CUSTOM_LAYERS:
        return [l for l in CUSTOM_LAYERS if 0 <= l < total_layers]
    else:
        return list(range(total_layers//3, 2*total_layers//3))

class SimpleCrossAttentionFuser(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.influence_weight = 1.0
        self.blend_strength = 0.5

    def set_influence_weight(self, weight):
        self.influence_weight = max(0.0, weight)

    def set_blend_strength(self, strength):
        self.blend_strength = max(0.0, min(1.0, strength))

    def forward(self, base_hidden_state, scaffold_context):
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        attn_output, _ = self.cross_attention(query=base_hidden_state, key=pooled_scaffold_context, value=pooled_scaffold_context)
        gate_values = self.gate(base_hidden_state)
        scaffold_effect = gate_values * (attn_output * self.influence_weight)
        fused_state = (1 - self.blend_strength) * base_hidden_state + self.blend_strength * scaffold_effect
        return self.layer_norm(fused_state)

class TrueCuriosity:
    def __init__(self, sovl_system):
        self.sovl = sovl_system
        self.novelty_cache = deque(maxlen=100)

    def calculate_metric(self, question: str) -> float:
        q_inputs = self.sovl.base_tokenizer(question, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
        with torch.no_grad():
            q_emb = self.sovl.scaffolds[0](**q_inputs, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        base_conf = calculate_confidence_score(self.sovl.base_model(**q_inputs).logits, q_inputs.input_ids)
        scaf_conf = calculate_confidence_score(self.sovl.scaffolds[0](**q_inputs).logits, q_inputs.input_ids)
        mem_sim = 0
        if self.sovl.dream_memory:
            dream_embs, _ = zip(*self.sovl.dream_memory)
            mem_sim = max([F.cosine_similarity(q_emb, emb).item() for emb in dream_embs], default=0)
        ignorance = 1 - max(base_conf, scaf_conf)
        novelty = 1 - mem_sim
        return ignorance * self.sovl.curiosity_weight_ignorance + novelty * self.sovl.curiosity_weight_novelty

class CuriosityPressure:
    def __init__(self):
        self.value = 0.0

    def update(self, temperament: float, confidence: float, silence: float):
        self.value += (temperament * 0.1 + (1 - confidence) * 0.05 + silence * 0.02)
        self.value = max(0.0, min(1.0, self.value))

    def should_erupt(self, threshold):
        return self.value > threshold and random.random() < 0.3

class SOVLSystem:
    def __init__(self):
        self.logger = Logger("sovl_logs.jsonl")
        config_defaults = {"quantization_mode": QUANTIZATION_MODE, "enable_error_listening": ENABLE_ERROR_LISTENING}
        config = load_config("sovl_config.json", config_defaults)
        self.quantization_mode = config.get("quantization_mode", QUANTIZATION_MODE)
        self.enable_error_listening = get_config_value(config, "controls_config.enable_error_listening", ENABLE_ERROR_LISTENING)

        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        self.dry_run = DRY_RUN
        self.dry_run_params = {
            'max_samples': DRY_RUN_MAX_SAMPLES,
            'max_length': DRY_RUN_MAX_LENGTH,
            'validate_architecture': DRY_RUN_VALIDATE_ARCH,
            'skip_training': DRY_RUN_SKIP_TRAINING
        }

        print(f"Loading base model: {BASE_MODEL_NAME}")
        quantization_config = {"load_in_8bit": True} if self.quantization_mode == "int8" else {"load_in_4bit": True} if self.quantization_mode == "int4" else {}
        self.base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, config=self.base_config, **quantization_config).to(DEVICE)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"Base model '{BASE_MODEL_NAME}' loaded and frozen.")

        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(SCAFFOLD_MODEL_NAME, config=self.scaffold_config, **quantization_config)
        if ENABLE_LORA_ADAPTERS:
            lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM)
            self.scaffolds = [get_peft_model(scaffold_model_raw, lora_config).to(DEVICE)]
            print("LoRA adapters applied to scaffold[0].")
        else:
            self.scaffolds = [scaffold_model_raw.to(DEVICE)]
            print("Scaffold loaded without LoRA adapters.")

        print(f"Loading tokenizers from: {BASE_MODEL_NAME} and {SCAFFOLD_MODEL_NAME}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        self.scaffolds[0].config.pad_token_id = self.scaffold_tokenizer.pad_token_id

        # Initialize SOVLTrainer with extended config
        training_config = TrainingConfig(
            learning_rate=LEARNING_RATE,
            grad_accum_steps=ACCUMULATION_STEPS,
            weight_decay=0.01,
            total_steps=(len(TRAIN_DATA) // BATCH_SIZE) * TRAIN_EPOCHS,
            max_grad_norm=1.0,
            use_amp=(DEVICE.type == "cuda"),
            max_patience=MAX_PATIENCE,
            batch_size=BATCH_SIZE,
            max_epochs=TRAIN_EPOCHS,
            validate_every_n_steps=len(TRAIN_DATA) // BATCH_SIZE,
            checkpoint_interval=1000,
            checkpoint_path="checkpoints/sovl_trainer",
            scheduler_type="linear",
            cosine_min_lr=1e-6,
            warmup_ratio=0.1,
            dropout_rate=LORA_DROPOUT,
            max_seq_length=MAX_SEQ_LENGTH,
            metrics_to_track=["loss", "accuracy", "confidence"],
            enable_gestation=ENABLE_GESTATION,
            enable_sleep_training=ENABLE_SLEEP_TRAINING,
            enable_lifecycle_weighting=ENABLE_LIFECYCLE_WEIGHTING,
            lifecycle_capacity_factor=LIFECYCLE_CAPACITY_FACTOR,
            lifecycle_curve=LIFECYCLE_CURVE,
            sleep_conf_threshold=SLEEP_CONF_THRESHOLD,
            sleep_log_min=SLEEP_LOG_MIN,
            accumulation_steps=ACCUMULATION_STEPS,
            exposure_gain_eager=EXPOSURE_GAIN_EAGER,
            exposure_gain_default=EXPOSURE_GAIN_DEFAULT,
            dream_memory_weight=DREAM_MEMORY_WEIGHT,
            enable_dreaming=ENABLE_DREAMING,
            repetition_n=3,
            sigmoid_scale=SIGMOID_SCALE,
            sigmoid_shift=SIGMOID_SHIFT
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
            logger=self.logger.record,
            memory_lock=Lock(),
            tokenizer=self.base_tokenizer,
            state=None  # Set after state initialization
        )
        self.trainer.memory_check = self.check_memory_health

        # Register callbacks for curiosity
        self.trainer.register_callback("on_training_complete", self.handle_training_complete)
        self.trainer.register_callback("on_gestation_complete", self.handle_gestation_complete)
        self.trainer.register_callback("on_dream_complete", self.handle_dream_complete)
        self.trainer.register_callback("on_sleep_train_complete", self.handle_sleep_train_complete)

        def build_token_map(base_tokenizer, scaffold_tokenizer):
            token_map = defaultdict(lambda: [scaffold_tokenizer.unk_token_id])
            for base_token, base_id in base_tokenizer.get_vocab().items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(normalized, add_special_tokens=False, max_length=3, truncation=True) or [scaffold_tokenizer.unk_token_id]
                token_map[base_id] = {'ids': scaffold_ids, 'weight': 1.0}
            return token_map

        self.token_map = build_token_map(self.base_tokenizer, self.scaffold_tokenizer)
        self.special_token_map = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id or self.scaffold_tokenizer.sep_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        self.scaffold_unk_id = self.scaffold_tokenizer.unk_token_id

        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Cross-attention injection complete.")

        self._temp_scaffold_context = None
        self.memory_lock = Lock()
        self.mem_usage_history = deque(maxlen=10)
        self.dynamic_threshold_base = MEMORY_THRESHOLD
        self.max_patience = MAX_PATIENCE
        self.has_woken = HAS_WOKEN
        self.use_scaffold_memory = USE_SCAFFOLD_MEMORY
        self.use_token_map_memory = USE_TOKEN_MAP_MEMORY
        self.memory_decay_rate = MEMORY_DECAY_RATE
        self.history = ConversationHistory()
        self.last_trained = 0
        self.dynamic_cross_attn_mode = DYNAMIC_CROSS_ATTN_MODE
        self.last_weight = 0.0
        self.is_sleeping = False

        # Initialize state from config
        self.state = SOVLState(config)
        self.state.set_scaffold_unk_id(self.scaffold_tokenizer.unk_token_id)
        self.trainer.state = self.state  # Share state with trainer
       
        # Curiosity system
        self.curiosity = TrueCuriosity(self) if ENABLE_CURIOSITY else None
        self.pressure = CuriosityPressure() if ENABLE_CURIOSITY else None
        self.last_question_time = time.time()

        self.load_state()

    def check_memory_health(self):
        """Autonomically reduce GPU memory usage if nearing capacity with dynamic adjustments."""
        if not torch.cuda.is_available():
            return
    
        with self.memory_lock:
            current_mem = torch.cuda.memory_allocated()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            mem_ratio = current_mem / total_mem
            self.mem_usage_history.append(mem_ratio)
            
            # Calculate model size (approximate)
            model_size = sum(p.numel() * p.element_size() for p in self.base_model.parameters())
            model_size += sum(p.numel() * p.element_size() for scaffold in self.scaffolds for p in scaffold.parameters())
            
            # Calculate average memory usage
            avg_mem_usage = sum(self.mem_usage_history) / len(self.mem_usage_history) if self.mem_usage_history else mem_ratio
    
            dynamic_threshold = min(
                0.95,
                max(
                    0.7,
                    self.dynamic_threshold_base * (1 + (model_size/total_mem) * 0.1 - avg_mem_usage * 0.2)
                )
            )
            lifecycle_stage = self.trainer.data_exposure / self.trainer.lora_capacity if self.trainer.lora_capacity > 0 else 0.0
            from sovl_utils import float_lt
            if float_lt(lifecycle_stage, 0.25):
                memory_pruned = False
                quantization_changed = False
                cache_cleared = False
                batch_size_reduced = False
    
                torch.cuda.empty_cache()
                cache_cleared = True
    
                if hasattr(self, 'dream_memory') and len(self.dream_memory) > 0:
                    original_len = len(self.dream_memory)
                    sorted_mem = sorted(self.dream_memory, key=lambda x: x[1], reverse=True)
                    keep_len = max(1, original_len // 2)
                    self.dream_memory = deque(
                        [(t.detach().cpu(), w) for t, w in sorted_mem if w > 0.5][:keep_len],
                        maxlen=self.state.dream_memory_maxlen
                    )
                    if len(self.dream_memory) < original_len:
                        memory_pruned = True
    
                if not hasattr(self, '_original_batch_size'):
                    self._original_batch_size = BATCH_SIZE
                if BATCH_SIZE > 1:
                    global BATCH_SIZE
                    BATCH_SIZE = max(1, BATCH_SIZE // 2)
                    batch_size_reduced = True
    
                if self.quantization_mode != "int8":
                    self.set_quantization_mode("int8")
                    quantization_changed = True
    
                self.logger.record({
                    "error": "memory_threshold_exceeded",
                    "details": {
                        "current_memory": current_mem,
                        "total_memory": total_mem,
                        "memory_pruned": memory_pruned,
                        "quantization_changed": quantization_changed,
                        "cache_cleared": cache_cleared,
                        "batch_size_reduced": batch_size_reduced,
                        "new_batch_size": BATCH_SIZE if batch_size_reduced else None,
                        "dynamic_threshold": dynamic_threshold,
                        "threshold": MEMORY_THRESHOLD
                    },
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "is_error_prompt": self.enable_error_listening
                })
                print(f"Attention: Memory adjusted (GPU: {mem_ratio:.0%}, Threshold: {dynamic_threshold:.2f}) - "
                      f"Cache Cleared: {cache_cleared}, Pruned: {memory_pruned}, "
                      f"Batch Reduced: {batch_size_reduced}, Quantized: {quantization_changed}")
    
            elif mem_ratio < dynamic_threshold * 0.8:
                if hasattr(self, '_original_batch_size') and BATCH_SIZE < self._original_batch_size:
                    global BATCH_SIZE
                    BATCH_SIZE = self._original_batch_size
                    print(f"Restored batch size to {BATCH_SIZE}")
                    delattr(self, '_original_batch_size')

    def generate_curiosity_question(self, context: str = None, spontaneous: bool = False) -> Optional[str]:
        if not ENABLE_CURIOSITY:
            return None
        
        if not context and self.state.dream_memory:
            dream_embs, _ = zip(*self.state.dream_memory)
            seed = self.generate("", max_new_tokens=5, temperature=1.3, do_sample=True)
            context = " ".join(seed.split()[:3])
        elif not context:
            context = ""

        temp = CURIOSITY_BASE_TEMPERATURE + (self.state.temperament_score * CURIOSITY_TEMPERAMENT_INFLUENCE)
        temp = max(0.7, min(1.7, temp))

        output = self.generate(context, max_new_tokens=CURIOSITY_MAX_NEW_TOKENS, 
                             temperature=temp, top_k=CURIOSITY_TOP_K, do_sample=True)

        if not output.endswith("?"):
            output += "?"

        score = self.curiosity.calculate_metric(output)
        threshold = CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS if spontaneous else CURIOSITY_NOVELTY_THRESHOLD_RESPONSE
        return output if score > threshold else None
    
    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        """Handle training completion callback."""
        if not ENABLE_CURIOSITY or not self.curiosity:
            return
        context = f"Training completed at epoch {epoch} with loss {avg_loss:.4f}"
        q = self.generate_curiosity_question(context, spontaneous=False)
        if q:
            score = self.curiosity.calculate_metric(q)
            self.state.unanswered_q.append((q, score))
            self.logger.record({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, score, spontaneous=False)
            self.pressure.value -= CURIOSITY_PRESSURE_DROP
            print(f"Pressure after training question: {self.pressure.value:.2f}")
            self.last_question_time = time.time()
            print(f"Curiosity question after training: {q}")

    def handle_gestation_complete(self, batch_size: int, avg_loss: float):
        """Handle gestation completion callback."""
        if not ENABLE_CURIOSITY or not self.curiosity:
            return
        context = f"Gestation processed {batch_size} entries with loss {avg_loss:.4f}"
        q = self.generate_curiosity_question(context, spontaneous=False)
        if q:
            score = self.curiosity.calculate_metric(q)
            self.state.unanswered_q.append((q, score))
            self.logger.record({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, score, spontaneous=False)
            self.pressure.value -= CURIOSITY_PRESSURE_DROP
            print(f"Pressure after gestation question: {self.pressure.value:.2f}")
            self.last_question_time = time.time()
            print(f"Curiosity question after gestation: {q}")

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int):
        """Handle dream completion callback."""
        if not ENABLE_CURIOSITY or not self.curiosity:
            return
        context = f"Dreamed about '{dream_prompt[:30]}...' (novel: {is_novel}, memories: {memory_count})"
        q = self.generate_curiosity_question(context, spontaneous=False)
        if q:
            score = self.curiosity.calculate_metric(q)
            self.state.unanswered_q.append((q, score))
            self.logger.record({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, score, spontaneous=False)
            self.pressure.value -= CURIOSITY_PRESSURE_DROP
            print(f"Pressure after dream question: {self.pressure.value:.2f}")
            self.last_question_time = time.time()
            print(f"Curiosity question after dreaming: {q}")

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float):
        """Handle sleep training completion callback."""
        if not ENABLE_CURIOSITY or not self.curiosity:
            return
        context = f"Sleep training processed {batch_size} entries, exposure {data_exposure}"
        q = self.generate_curiosity_question(context, spontaneous=False)
        if q:
            score = self.curiosity.calculate_metric(q)
            self.state.unanswered_q.append((q, score))
            self.logger.record({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, score, spontaneous=False)
            self.pressure.value -= CURIOSITY_PRESSURE_DROP
            print(f"Pressure after sleep training question: {self.pressure.value:.2f}")
            self.last_question_time = time.time()
            print(f"Curiosity question after sleep training: {q}")
    
    def update_metrics(self, question, score, spontaneous=False, answered=False):
        self.metrics["curiosity_eruptions"] += 1
        if spontaneous:
            self.metrics["spontaneous_questions"] += 1
        if answered:
            self.metrics["answered_questions"] += 1
        self.metrics["avg_novelty"] = (self.metrics["avg_novelty"] * self.metrics["eruption_count"] + score) / (self.metrics["eruption_count"] + 1)
        self.metrics["eruption_count"] += 1

    def check_silence(self, elapsed: float):
        if not ENABLE_CURIOSITY or not self.pressure:
            return
        if (elapsed > CURIOSITY_SILENCE_THRESHOLD and 
            self.pressure.value > CURIOSITY_PRESSURE_THRESHOLD and 
            (time.time() - self.last_question_time) > CURIOSITY_QUESTION_COOLDOWN):
            q = self.generate_curiosity_question(spontaneous=True)
            if q:
                print(f"{q}")
                self.logger.record({
                    "prompt": q,
                    "response": "",
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "confidence_score": 0.0,
                    "is_system_question": True
                })
                self.update_metrics(q, self.curiosity.calculate_metric(q), spontaneous=True)
                self.pressure.value -= CURIOSITY_PRESSURE_DROP
                self.last_question_time = time.time()
            elif self.state.unanswered_q:
                q, score = self.state.unanswered_q.popleft()
                print(f"{q}")
                self.logger.record({
                    "prompt": q,
                    "response": "",
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "confidence_score": 0.0,
                    "is_system_question": True
                })
                self.update_metrics(q, score, spontaneous=True)
                self.pressure.value -= CURIOSITY_PRESSURE_DROP * 0.7
                self.last_question_time = time.time()

    def tune_curiosity(self, enable=None, spontaneous_threshold=None, response_threshold=None, pressure_threshold=None, 
                       pressure_drop=None, silence_threshold=None, question_cooldown=None, queue_maxlen=None, 
                       weight_ignorance=None, weight_novelty=None, max_new_tokens=None, base_temperature=None, 
                       temperament_influence=None, top_k=None):
        if enable is not None:
            global ENABLE_CURIOSITY
            ENABLE_CURIOSITY = bool(enable)
            if ENABLE_CURIOSITY and not self.curiosity:
                self.curiosity = TrueCuriosity(self)
                self.pressure = CuriosityPressure()
            elif not ENABLE_CURIOSITY:
                self.curiosity = None
                self.pressure = None
        if spontaneous_threshold is not None and 0.5 <= spontaneous_threshold <= 1.0:
            global CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS
            CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS = spontaneous_threshold
        if response_threshold is not None and 0.5 <= response_threshold <= 1.0:
            global CURIOSITY_NOVELTY_THRESHOLD_RESPONSE
            CURIOSITY_NOVELTY_THRESHOLD_RESPONSE = response_threshold
        if pressure_threshold is not None and 0.5 <= pressure_threshold <= 0.9:
            global CURIOSITY_PRESSURE_THRESHOLD
            CURIOSITY_PRESSURE_THRESHOLD = pressure_threshold
        if pressure_drop is not None and 0.1 <= pressure_drop <= 0.5:
            global CURIOSITY_PRESSURE_DROP
            CURIOSITY_PRESSURE_DROP = pressure_drop
        if silence_threshold is not None and 5.0 <= silence_threshold <= 60.0:
            global CURIOSITY_SILENCE_THRESHOLD
            CURIOSITY_SILENCE_THRESHOLD = silence_threshold
        if question_cooldown is not None and 30.0 <= question_cooldown <= 120.0:
            global CURIOSITY_QUESTION_COOLDOWN
            CURIOSITY_QUESTION_COOLDOWN = question_cooldown
        if queue_maxlen is not None and 5 <= queue_maxlen <= 20:
            self.state.curiosity_queue_maxlen = queue_maxlen
            self.state.unanswered_q = deque(self.state.unanswered_q, maxlen=queue_maxlen)
        if weight_ignorance is not None and 0.0 <= weight_ignorance <= 1.0:
            global CURIOSITY_WEIGHT_IGNORANCE
            CURIOSITY_WEIGHT_IGNORANCE = weight_ignorance
        if weight_novelty is not None and 0.0 <= weight_novelty <= 1.0:
            global CURIOSITY_WEIGHT_NOVELTY
            CURIOSITY_WEIGHT_NOVELTY = weight_novelty
        if max_new_tokens is not None and 5 <= max_new_tokens <= 12:
            global CURIOSITY_MAX_NEW_TOKENS
            CURIOSITY_MAX_NEW_TOKENS = max_new_tokens
        if base_temperature is not None and 0.5 <= base_temperature <= 1.5:
            global CURIOSITY_BASE_TEMPERATURE
            CURIOSITY_BASE_TEMPERATURE = base_temperature
        if temperament_influence is not None and 0.1 <= temperament_influence <= 0.6:
            global CURIOSITY_TEMPERAMENT_INFLUENCE
            CURIOSITY_TEMPERAMENT_INFLUENCE = temperament_influence
        if top_k is not None and 10 <= top_k <= 50:
            global CURIOSITY_TOP_K
            CURIOSITY_TOP_K = top_k
        print(f"Curiosity params: enable={ENABLE_CURIOSITY}, spontaneous_threshold={CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS}, "
              f"response_threshold={CURIOSITY_NOVELTY_THRESHOLD_RESPONSE}, pressure_threshold={CURIOSITY_PRESSURE_THRESHOLD}, "
              f"pressure_drop={CURIOSITY_PRESSURE_DROP}, silence_threshold={CURIOSITY_SILENCE_THRESHOLD}, "
              f"question_cooldown={CURIOSITY_QUESTION_COOLDOWN}, queue_maxlen={self.state.curiosity_queue_maxlen}, "
              f"weight_ignorance={CURIOSITY_WEIGHT_IGNORANCE}, weight_novelty={CURIOSITY_WEIGHT_NOVELTY}, "
              f"max_new_tokens={CURIOSITY_MAX_NEW_TOKENS}, base_temperature={CURIOSITY_BASE_TEMPERATURE}, "
              f"temperament_influence={CURIOSITY_TEMPERAMENT_INFLUENCE}, top_k={CURIOSITY_TOP_K}")

    def toggle_memory(self, mode):
        modes = {
            'scaffold_mem': (True, False),
            'token_mem': (False, True),
            'both_mem': (True, True),
            'no_mem': (False, False)
        }
        if mode in modes:
            self.use_scaffold_memory, self.use_token_map_memory = modes[mode]
            print(f"Memory toggled: Scaffold={self.use_scaffold_memory}, Token Map={self.use_token_map_memory}")
        else:
            print("Invalid memory mode. Use: 'scaffold_mem', 'token_mem', 'both_mem', 'no_mem'")
            
    def wake_up(self):
        if self.has_woken:
            return None
        wake_seed = (int(time.time() * 1000) + random.randint(0, 100)) % 10000
        torch.manual_seed(wake_seed)
        random.seed(wake_seed)
        prompt = " "
        with torch.no_grad():
            response = self.generate(prompt, max_new_tokens=15, temperature=1.7, top_k=30, do_sample=True)
        self.is_sleeping = IS_SLEEPING
        self.has_woken = True
        print(f"\n{response}")

        q = self.generate_curiosity_question() if not self.state.unanswered_q else self.state.unanswered_q.popleft()[0]
        if q:
            print(f"{q}")
            self.logger.record({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, self.curiosity.calculate_metric(q))

        self.logger.record({"event": "wake_up", "response": response, "timestamp": time.time(), "conversation_id": self.history.conversation_id})
        return response

    def print_memory_stats(self, label="", verbose=False):
        if verbose and torch.cuda.is_available():
            print(f"\n--- Memory Stats ({label}) ---")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def set_scaffold_influence(self, weight=None, blend_strength=None, layer_weights=None):
        base_layers = self._get_model_layers(self.base_model)
        layers = get_cross_attention_layers(self.base_model) if USE_DYNAMIC_LAYERS else CROSS_ATTN_LAYERS
        
        if weight is not None:
            self.last_weight = weight
        
        if layer_weights is not None and len(layer_weights) == len(layers):
            for idx, layer_idx in enumerate(layers):
                if layer_idx < len(base_layers) and hasattr(base_layers[layer_idx], 'cross_attn'):
                    base_layers[layer_idx].cross_attn.set_influence_weight(layer_weights[idx])
            weight_display = "per-layer"
        else:
            for layer_idx in layers:
                if layer_idx < len(base_layers) and hasattr(base_layers[layer_idx], 'cross_attn'):
                    base_layers[layer_idx].cross_attn.set_influence_weight(self.last_weight)
            weight_display = f"{self.last_weight:.2f}"

        if blend_strength is not None:
            for layer_idx in layers:
                if layer_idx < len(base_layers) and hasattr(base_layers[layer_idx], 'cross_attn'):
                    base_layers[layer_idx].cross_attn.set_blend_strength(blend_strength)

        print(f"Scaffold influence: weight={weight_display}, blend_strength={blend_strength if blend_strength is not None else 'unchanged'}")

    def tune_cross_attention(self, weight=None, blend_strength=None, layer_weights=None, dynamic_mode=None):
        if layer_weights is not None:
            layers = get_cross_attention_layers(self.base_model) if USE_DYNAMIC_LAYERS else CROSS_ATTN_LAYERS
            if len(layer_weights) != len(layers):
                print(f"Error: layer_weights length ({len(layer_weights)}) must match cross-attn layers ({len(layers)})")
                return

        self.set_scaffold_influence(weight, blend_strength, layer_weights)
        
        if dynamic_mode in ['confidence', 'temperament']:
            self.dynamic_cross_attn_mode = dynamic_mode
            print(f"Dynamic cross-attention enabled: {dynamic_mode}")
        elif dynamic_mode == 'off':
            self.dynamic_cross_attn_mode = None
            print("Dynamic cross-attention disabled")
        elif dynamic_mode is not None:
            print("Invalid dynamic_mode. Use: 'confidence', 'temperament', or 'off'")    

    def set_quantization_mode(self, mode):
        if mode in ["fp16", "int8", "int4"] and mode != self.quantization_mode:
            self.quantization_mode = mode
            print(f"Quantization mode set to '{mode}'. Restart system to apply.")
        else:
            print(f"Invalid mode '{mode}' or no change.")

    def toggle_dynamic_layers(self, enable):
        global USE_DYNAMIC_LAYERS
        if enable != USE_DYNAMIC_LAYERS:
            USE_DYNAMIC_LAYERS = enable
            print(f"Dynamic layers {'enabled' if enable else 'disabled'}. Restart to apply.")

    def set_sleep_params(self, conf_threshold=None, time_factor=None, log_min=None):
        if conf_threshold is not None and 0.5 <= conf_threshold <= 0.9:
            global SLEEP_CONF_THRESHOLD
            SLEEP_CONF_THRESHOLD = conf_threshold
            self.trainer.config.sleep_conf_threshold = SLEEP_CONF_THRESHOLD
        if time_factor is not None and 0.5 <= time_factor <= 5.0:
            global SLEEP_TIME_FACTOR
            SLEEP_TIME_FACTOR = time_factor
        if log_min is not None and 5 <= log_min <= 20:
            global SLEEP_LOG_MIN
            SLEEP_LOG_MIN = log_min
            self.trainer.config.sleep_log_min = SLEEP_LOG_MIN
        print(f"Sleep params: conf={SLEEP_CONF_THRESHOLD}, time_factor={SLEEP_TIME_FACTOR}, log_min={SLEEP_LOG_MIN}")

    def tune_dream(self, swing_var=None, lifecycle_delta=None, temperament_on=None, noise_scale=None, memory_weight=None, memory_maxlen=None, prompt_weight=None, novelty_boost=None, memory_decay=None, prune_threshold=None):
        if swing_var is not None and 0.05 <= swing_var <= 0.2:
            global DREAM_SWING_VAR
            DREAM_SWING_VAR = swing_var
        if lifecycle_delta is not None and 0.05 <= lifecycle_delta <= 0.2:
            global DREAM_LIFECYCLE_DELTA
            DREAM_LIFECYCLE_DELTA = lifecycle_delta
        if temperament_on is not None:
            global DREAM_TEMPERAMENT_ON
            DREAM_TEMPERAMENT_ON = bool(temperament_on)
        if noise_scale is not None and 0.01 <= noise_scale <= 0.1:
            global DREAM_NOISE_SCALE
            DREAM_NOISE_SCALE = noise_scale
        if memory_weight is not None and 0 <= memory_weight <= 0.5:
            global DREAM_MEMORY_WEIGHT
            DREAM_MEMORY_WEIGHT = memory_weight
            self.trainer.config.dream_memory_weight = DREAM_MEMORY_WEIGHT
        if memory_maxlen is not None and 5 <= memory_maxlen <= 20:
            self.state.dream_memory_maxlen = memory_maxlen
            self.state.dream_memory = deque(self.state.dream_memory, maxlen=memory_maxlen)
        if prompt_weight is not None and 0 <= prompt_weight <= 1:
            global DREAM_PROMPT_WEIGHT
            DREAM_PROMPT_WEIGHT = prompt_weight
        if novelty_boost is not None and 0 <= novelty_boost <= 0.05:
            global DREAM_NOVELTY_BOOST
            DREAM_NOVELTY_BOOST = novelty_boost
        if memory_decay is not None and 0 <= memory_decay <= 1:
            global DREAM_MEMORY_DECAY
            DREAM_MEMORY_DECAY = memory_decay
        if prune_threshold is not None and 0 <= prune_threshold <= 1:
            global DREAM_PRUNE_THRESHOLD
            DREAM_PRUNE_THRESHOLD = prune_threshold
        print(f"Dream params: swing_var={DREAM_SWING_VAR}, lifecycle_delta={DREAM_LIFECYCLE_DELTA}, temperament_on={DREAM_TEMPERAMENT_ON}, noise_scale={DREAM_NOISE_SCALE}, memory_weight={DREAM_MEMORY_WEIGHT}, memory_maxlen={self.state.dream_memory_maxlen}, prompt_weight={DREAM_PROMPT_WEIGHT}, novelty_boost={DREAM_NOVELTY_BOOST}, memory_decay={DREAM_MEMORY_DECAY}, prune_threshold={DREAM_PRUNE_THRESHOLD}")

    def adjust_temperament(self, eager_threshold=None, sluggish_threshold=None, mood_influence=None, curiosity_boost=None, restless_drop=None, melancholy_noise=None, conf_feedback_strength=None, temp_smoothing_factor=None):
        if eager_threshold is not None and 0.7 <= eager_threshold <= 0.9:
            global TEMP_EAGER_THRESHOLD
            TEMP_EAGER_THRESHOLD = eager_threshold
        if sluggish_threshold is not None and 0.4 <= sluggish_threshold <= 0.6:
            global TEMP_SLUGGISH_THRESHOLD
            TEMP_SLUGGISH_THRESHOLD = sluggish_threshold
        if mood_influence is not None and 0 <= mood_influence <= 1:
            global TEMP_MOOD_INFLUENCE
            TEMP_MOOD_INFLUENCE = mood_influence
        if curiosity_boost is not None and 0 <= curiosity_boost <= 0.5:
            global TEMP_CURIOSITY_BOOST
            TEMP_CURIOSITY_BOOST = curiosity_boost
        if restless_drop is not None and 0 <= restless_drop <= 0.5:
            global TEMP_RESTLESS_DROP
            TEMP_RESTLESS_DROP = restless_drop
        if melancholy_noise is not None and 0 <= melancholy_noise <= 0.05:
            global TEMP_MELANCHOLY_NOISE
            TEMP_MELANCHOLY_NOISE = melancholy_noise
        if conf_feedback_strength is not None and 0 <= conf_feedback_strength <= 1:
            global CONF_FEEDBACK_STRENGTH
            CONF_FEEDBACK_STRENGTH = conf_feedback_strength
        if temp_smoothing_factor is not None and 0 <= temp_smoothing_factor <= 1:
            global TEMP_SMOOTHING_FACTOR
            TEMP_SMOOTHING_FACTOR = temp_smoothing_factor
        print(f"Temperament params: eager={TEMP_EAGER_THRESHOLD}, sluggish={TEMP_SLUGGISH_THRESHOLD}, mood_influence={TEMP_MOOD_INFLUENCE}, curiosity_boost={TEMP_CURIOSITY_BOOST}, restless_drop={TEMP_RESTLESS_DROP}, melancholy_noise={TEMP_MELANCHOLY_NOISE}, conf_feedback_strength={CONF_FEEDBACK_STRENGTH}, smoothing_factor={TEMP_SMOOTHING_FACTOR}")

    def set_global_blend(self, weight_cap=None, base_temp=None):
        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            global SCAFFOLD_WEIGHT_CAP
            SCAFFOLD_WEIGHT_CAP = weight_cap
        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            global BASE_TEMPERATURE
            BASE_TEMPERATURE = base_temp
        print(f"Global blend: weight_cap={SCAFFOLD_WEIGHT_CAP}, base_temp={BASE_TEMPERATURE}")

    def tune_lifecycle(self, capacity_factor=None, curve=None):
        if capacity_factor is not None and 0.001 <= capacity_factor <= 0.1:
            global LIFECYCLE_CAPACITY_FACTOR
            LIFECYCLE_CAPACITY_FACTOR = capacity_factor
            self.trainer.config.lifecycle_capacity_factor = LIFECYCLE_CAPACITY_FACTOR
            self.trainer.lora_capacity = sum(p.numel() for p in self.scaffolds[0].parameters() if p.requires_grad) * LIFECYCLE_CAPACITY_FACTOR
        if curve in ["sigmoid_linear", "exponential"]:
            global LIFECYCLE_CURVE
            LIFECYCLE_CURVE = curve
            self.trainer.config.lifecycle_curve = LIFECYCLE_CURVE
        print(f"Lifecycle params: capacity_factor={LIFECYCLE_CAPACITY_FACTOR}, curve={LIFECYCLE_CURVE}")

    def save_state(self, path_prefix=None):
        """Save all system state with proper serialization"""
        if path_prefix is None:
            path_prefix = SAVE_PATH_PREFIX
        try:
            # Save scaffold and cross-attention
            torch.save(self.scaffolds[0].state_dict(), f"{path_prefix}_scaffold.pth")
            cross_attn_dict = {k: v for k, v in self.base_model.state_dict().items() 
                              if 'cross_attn' in k}
            torch.save(cross_attn_dict, f"{path_prefix}_cross_attn.pth")

            # Save token map
            with open(f"{path_prefix}_token_map.json", "w") as f:
                json.dump({str(k): v for k, v in self.token_map.items()}, f)

            # Save state using module's serialization
            with open(f"{path_prefix}_state.json", "w") as f:
                json.dump(self.state.to_dict(), f)

            print(f"State saved to {path_prefix}_*.pth/json")
        except Exception as e:
            print(f"Save failed: {e}")
            # Attempt partial save if possible
            try:
                with open(f"{path_prefix}_state.json", "w") as f:
                    json.dump(self.state.to_dict(), f)
                print("At least saved core state")
            except:
                print("Complete save failure!")

    def load_state(self, path_prefix=None):
        """Load all system state with proper device handling"""
        if path_prefix is None:
            path_prefix = SAVE_PATH_PREFIX
        try:
            # Load scaffold and cross-attention first
            if os.path.exists(f"{path_prefix}_scaffold.pth"):
                self.scaffolds[0].load_state_dict(torch.load(f"{path_prefix}_scaffold.pth"))
                print("Scaffold state loaded.")

            if os.path.exists(f"{path_prefix}_cross_attn.pth"):
                state_dict = self.base_model.state_dict()
                state_dict.update(torch.load(f"{path_prefix}_cross_attn.pth"))
                self.base_model.load_state_dict(state_dict)
                print("Cross-attention state loaded.")

            # Load token map and state with device awareness
            if os.path.exists(f"{path_prefix}_token_map.json"):
                with open(f"{path_prefix}_token_map.json", "r") as f:
                    loaded_map = json.load(f)
                    self.token_map = defaultdict(lambda: [self.scaffold_unk_id], 
                                               {int(k): v for k, v in loaded_map.items()})
                print("Token map loaded.")

            if os.path.exists(f"{path_prefix}_state.json"):
                with open(f"{path_prefix}_state.json", "r") as f:
                    state_data = json.load(f)
                self.state.from_dict(state_data, device=DEVICE)
                print("System state loaded.")

        except Exception as e:
            print(f"Load failed: {e}. Starting fresh.")
            self.state = SOVLState(config)
            self.state.set_scaffold_unk_id(self.scaffold_tokenizer.unk_token_id)

    def _clear_scaffold_cache(self):
        with self.memory_lock:
            if hasattr(self, '_temp_scaffold_context') and isinstance(self._temp_scaffold_context, torch.Tensor):
                self._temp_scaffold_context = self._temp_scaffold_context.detach().cpu()
                del self._temp_scaffold_context
            self._temp_scaffold_context = None
            if self.state.last_prompt_embedding is not None:
                self.state.last_prompt_embedding = self.state.last_prompt_embedding.detach().cpu()
                self.state.last_prompt_embedding = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @contextlib.contextmanager
    def _scaffold_context(self, scaffold_hidden_states):
        try:
            self._temp_scaffold_context = scaffold_hidden_states
            yield
        finally:
            self._clear_scaffold_cache()

    def tokenize_and_map(self, prompts, max_length=MAX_SEQ_LENGTH, padding='max_length'):
        if isinstance(prompts, str):
            prompts = [prompts]
        inputs = self.base_tokenizer(prompts, return_tensors='pt', padding=padding, truncation=True, max_length=max_length).to(DEVICE)
        scaffold_input_ids = self.map_sequence(inputs.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()
        return {'input_ids': scaffold_input_ids, 'attention_mask': scaffold_attention_mask}

    def get_scaffold_hidden_states(self, scaffold_inputs):
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_outputs = self.scaffolds[0](**scaffold_inputs, output_hidden_states=True)
            return scaffold_outputs.hidden_states[-1] if hasattr(scaffold_outputs, 'hidden_states') else scaffold_outputs.base_model_output.hidden_states[-1]

    def _get_model_layers(self, model):
        actual_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return actual_model.transformer.h
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return actual_model.model.layers
        elif hasattr(actual_model, 'layers'):
            return actual_model.layers
        elif hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
            return actual_model.decoder.layers
        raise ValueError(f"Cannot determine layer structure for {actual_model.__class__.__name__}")

    def _insert_cross_attention(self):
        if not ENABLE_CROSS_ATTENTION:
            print("Cross-attention disabled.")
            return
        base_layers = self._get_model_layers(self.base_model)
        num_base_layers = len(base_layers)
        hidden_dim = self.base_config.hidden_size
        num_heads = self.base_config.num_attention_heads

        if self.scaffold_config.hidden_size != hidden_dim:
            print(f"Adding projection: {self.scaffold_config.hidden_size} -> {hidden_dim}")
            self.scaffold_proj = nn.Linear(self.scaffold_config.hidden_size, hidden_dim).to(DEVICE)
            self.scaffold_proj.weight.requires_grad_(True)
            self.scaffold_proj.bias.requires_grad_(True)
        else:
            self.scaffold_proj = None

        layers = get_cross_attention_layers(self.base_model) if USE_DYNAMIC_LAYERS else CROSS_ATTN_LAYERS
        print(f"Using layers: {layers}")

        for layer_idx in layers:
            if layer_idx >= num_base_layers:
                print(f"Warning: Layer {layer_idx} out of bounds ({num_base_layers} layers). Skipping.")
                continue
            original_layer = base_layers[layer_idx]
            cross_attn_fuser = SimpleCrossAttentionFuser(hidden_dim=hidden_dim, num_heads=num_heads).to(DEVICE)

            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module, parent_system):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                    self._parent_system = parent_system

                def forward(self, hidden_states, **kwargs):
                    outputs = self.orig_layer(hidden_states, **kwargs)
                    base_hidden_state_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    scaffold_context = self._parent_system._temp_scaffold_context
                    if scaffold_context is not None:
                        scaffold_context = scaffold_context.to(base_hidden_state_output.device)
                        if self._parent_system.scaffold_proj is not None:
                            scaffold_context = self._parent_system.scaffold_proj(scaffold_context)
                        if self._parent_system.state.dream_memory and DREAM_MEMORY_WEIGHT > 0:
                            dream_tensors, dream_weights = zip(*self._parent_system.state.dream_memory)
                            dream_tensors = torch.stack(dream_tensors).to(base_hidden_state_output.device)
                            dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=base_hidden_state_output.device)
                            from sovl_utils import safe_divide
                            dream_avg = safe_divide(
                                (dream_tensors * dream_weights.unsqueeze(-1)).sum(dim=0),
                                dream_weights.sum()
                            )
                            scaffold_context = (1 - DREAM_MEMORY_WEIGHT) * scaffold_context + DREAM_MEMORY_WEIGHT * dream_avg
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)
                        return (fused_hidden_state,) + outputs[1:] if isinstance(outputs, tuple) else fused_hidden_state
                    return outputs

            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Injected cross-attention into layer {layer_idx}")

    def enable_dry_run(self, max_samples=2, max_length=128, validate_architecture=True, skip_training=True):
        self.dry_run = True
        self.dry_run_params = {'max_samples': max_samples, 'max_length': max_length, 'validate_architecture': validate_architecture, 'skip_training': skip_training}
        print(f"Dry run activated (max_samples={max_samples}, max_length={max_length})")

    def map_sequence(self, base_input_ids):
        batch_size = base_input_ids.size(0)
        seq_len = base_input_ids.size(1)

        available_memory_limit = torch.cuda.max_memory_allocated(device=DEVICE) - torch.cuda.memory_allocated(device=DEVICE)
        token_size = 4
        max_expanded_len = min(seq_len * 3, MAX_SEQ_LENGTH, available_memory_limit // token_size)

        mapped_ids = torch.full(
            (batch_size, max_expanded_len),
            self.scaffold_tokenizer.pad_token_id,
            dtype=torch.long,
            device=DEVICE
        )
        truncated = False

        for batch_idx in range(batch_size):
            position = 0
            for base_id in base_input_ids[batch_idx]:
                base_id_item = base_id.item()
                if base_id_item in self.special_token_map:
                    mapped_tokens = [self.special_token_map[base_id_item]]
                else:
                    try:
                        mapped_entry = self.state.token_map[base_id_item]
                        mapped_tokens = mapped_entry['ids'] if isinstance(mapped_entry, dict) else mapped_entry
                    except Exception as e:
                        print(f"Token mapping error for ID {base_id_item}: {e}")
                        mapped_tokens = [self.scaffold_unk_id]

                if position + len(mapped_tokens) > max_expanded_len:
                    truncated = True
                    break

                for token in mapped_tokens:
                    if position >= max_expanded_len:
                        truncated = True
                        break
                    mapped_ids[batch_idx, position] = token
                    position += 1

                if truncated:
                    break

        if truncated:
            print(f"Warning: Token mapping truncated to {max_expanded_len}. Consider adjusting limits or input size.")
            self.logger.record(
                {
                    "warning": f"Token mapping truncated to {max_expanded_len}",
                    "original_length": seq_len,
                    "allowed_length": max_expanded_len,
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id
                }
            )

        return mapped_ids[:, :min(max_expanded_len, MAX_SEQ_LENGTH)]

    def _update_token_map_memory(self, prompt, confidence):
        if not self.use_token_map_memory:
            return
        with self.memory_lock:
            tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            for token_id in tokens:
                if token_id in self.token_map:
                    self.token_map[token_id]['weight'] = min(self.token_map[token_id]['weight'] + confidence * 0.1, 2.0)
            for token_id in self.token_map:
                self.token_map[token_id]['weight'] *= MEMORY_DECAY_RATE

    def _gestate(self, resume=False):
        """Perform gestation by delegating to the trainer."""
        log_entries = self.logger.read()
        if not log_entries:
            print("No log data to gestate.")
            return False

        # Delegate to trainer
        result = self.trainer.gestate(log_entries, resume)
        if not result:
            self.last_trained = time.time()
            self.logger.clear()
            self.last_weight = self.trainer.get_life_curve_weight()
            self.set_scaffold_influence(self.last_weight)
            print(f"Growth stage: {self.last_weight:.2f}, Exposure: {self.trainer.data_exposure}")
        return result

    def _sleep_train(self):
        """Perform sleep training by delegating to the trainer."""
        if not ENABLE_SLEEP_TRAINING:
            return
        print("\n--- Sleep Training Initiated ---")
        log_entries = self.logger.read()
        if not log_entries:
            print("No log data to train on.")
            return

        # Delegate to trainer
        success = self.trainer.sleep_train(log_entries)
        if success:
            self.last_trained = time.time()
            self.logger.clear()
            self.last_weight = self.trainer.get_life_curve_weight()
            print("--- Sleep Training Complete ---")

    def _update_temperament(self):
        avg_confidence = self.state.sleep_confidence_sum / self.state.sleep_confidence_count if self.state.sleep_confidence_count > 0 else 0.5
        lifecycle_stage = self.trainer.data_exposure / self.trainer.lora_capacity if self.trainer.lora_capacity > 0 else 0.0
    
        base_score = 2.0 * (avg_confidence - 0.5)
    
        if lifecycle_stage < 0.25:
            bias = TEMP_CURIOSITY_BOOST * (1 - lifecycle_stage / 0.25)
        elif lifecycle_stage < 0.75:
            bias = 0.0
            if len(self.state.temperament_history) >= 5:
                variance = torch.var(torch.tensor(list(self.state.temperament_history))).item()
                bias -= 0.2 * variance
        else:
            bias = -TEMP_CURIOSITY_BOOST * (lifecycle_stage - 0.75) / 0.25
    
        target_score = base_score + bias + (CONF_FEEDBACK_STRENGTH * (avg_confidence - 0.5))
        target_score = max(-1.0, min(1.0, target_score))
        alpha = 0.1 * (1 - TEMP_SMOOTHING_FACTOR)
        self.state.temperament_score = (1 - alpha) * self.state.temperament_score + alpha * target_score
        self.state.temperament_score = max(-1.0, min(1.0, self.state.temperament_score))
        self.state.temperament_history = deque(self.state.temperament_history, maxlen=TEMPERAMENT_HISTORY_MAXLEN)
        self.state.temperament_history.append(self.state.temperament_score)
    
        label = "melancholic" if self.state.temperament_score <= -0.5 else "restless" if self.state.temperament_score <= 0.0 else "calm" if self.state.temperament_score <= 0.5 else "curious"
        print(f"Temperament score: {self.state.temperament_score:.3f} ({label}, lifecycle: {lifecycle_stage:.2f}), confidence feedback: {avg_confidence:.2f}")

    def train_step(self, batch):
        """Execute a single training step with scaffold context."""
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
                self.trainer.config.max_seq_length,
                self.base_tokenizer
            )
            prompts = formatted_batch['prompt']
            scaffold_inputs = self.tokenize_and_map(prompts)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
            loss, confidence = self.trainer.train_step(
                batch=formatted_batch,
                scaffold_context=scaffold_hidden_states,
                dry_run=True
            )
            print(f"Dry run loss: {loss}")
            return None

        # Prepare scaffold context
        prompts = [item['prompt'] for item in batch]
        scaffold_inputs = self.tokenize_and_map(prompts)
        scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

        # Train step via trainer
        formatted_batch = collate_batch(
            batch,
            self.base_tokenizer.pad_token_id,
            self.trainer.config.max_seq_length,
            self.base_tokenizer
        )
        loss, confidence = self.trainer.train_step(
            batch=formatted_batch,
            scaffold_context=scaffold_hidden_states,
            grad_clip=True,
            dry_run=False,
            memory_check=self.check_memory_health
        )

        # Update token map
        if loss is not None and self.use_token_map_memory and confidence is not None:
            self._update_token_map_memory(prompts[0], confidence)

        return loss

    def run_training_cycle(self, train_data, valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        if len(train_data) < batch_size or not valid_data:
            print("Not enough data for training.")
            return

        if ENABLE_LIFECYCLE_WEIGHTING:
            influence_weight = self.trainer.get_life_curve_weight()
        else:
            influence_weight = self.last_weight
        self.set_scaffold_influence(influence_weight)
        print(f"Data exposure: {self.trainer.data_exposure} | Scaffold influence: {influence_weight:.3f}")

        if self.dry_run and self.dry_run_params['skip_training']:
            print("\n=== DRY RUN TRAINING ===")
            dry_batch = train_data[:self.dry_run_params['max_samples']]
            loss = self.train_step(dry_batch)
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
        print(f"--- Training Finished ({time.time() - start_time:.2f}s) ---")

    def has_repetition(self, output_ids, n=3):
        ids = output_ids.tolist()
        special_ids = {self.base_tokenizer.pad_token_id, self.base_tokenizer.eos_token_id, self.base_tokenizer.bos_token_id, self.base_tokenizer.unk_token_id}
        filtered = [i for i in ids if i not in special_ids]
        for i in range(len(filtered) - 2*n):
            if filtered[i:i+n] == filtered[i+n:i+2*n]:
                return True
        return False
    
    def _handle_error_prompt(self, error_msg):
        """Generate a response to a system error."""
        temp_history = self.history
        self.history = ConversationHistory()
        response = self.generate(
            f"System error detected: {error_msg} What happened?",
            max_new_tokens=60,
            temperature=BASE_TEMPERATURE + 0.2,
            top_k=50,
            do_sample=True
        )
        self.logger.record({
            "prompt": f"System error detected: {error_msg} What happened?",
            "response": response,
            "timestamp": time.time(),
            "conversation_id": self.history.conversation_id,
            "is_error_prompt": True,
            "confidence_score": 0.5
        })
        self.history = temp_history
        return response

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        """Generate a response for the given prompt."""
        try:
            print(f"Generation initiated: prompt='{prompt[:30]}...', max_new_tokens={max_new_tokens}, scaffold_weight={scaffold_weight}")
            self.check_memory_health()
            if self.is_sleeping:
                print("\rGestation Interrupted", end="", flush=True)
                time.sleep(0.5)
                print("\r                   ", end="")
                self.trainer._reset_sleep_state()

            start_time = time.time()
            base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
            input_ids = base_inputs['input_ids']
            input_length = input_ids.shape[1]

            scaffold_inputs = self.tokenize_and_map(prompt)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

            temp = BASE_TEMPERATURE
            if ENABLE_TEMPERAMENT and TEMP_MOOD_INFLUENCE > 0:
                temp_adjustment = TEMP_MOOD_INFLUENCE * 0.3 * self.state.temperament_score
                temp += temp_adjustment
                temp = max(0.5, min(1.5, temp))

            self._clear_scaffold_cache()
            generated_ids = []
            chunk_size = 512
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
                        **kwargs
                    )
                    generated_ids.extend(outputs.sequences[0][input_length:].tolist())

            print(f"Generation completed in {time.time() - start_time:.2f}s.")
            confidence_score = 0.5
            if ENABLE_CONFIDENCE_TRACKING:
                confidence_score = calculate_confidence_score(outputs.scores, generated_ids)
                with self.memory_lock:
                    self.state.sleep_confidence_sum += confidence_score
                    self.state.sleep_confidence_count += 1
                    self.state.confidence_history.append(confidence_score)

            special_ids = {self.base_tokenizer.pad_token_id, self.base_tokenizer.eos_token_id, 
                          self.base_tokenizer.bos_token_id, self.base_tokenizer.unk_token_id}
            if ENABLE_REPETITION_CHECK and self.trainer.has_repetition(torch.tensor(generated_ids), special_ids):
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
                    "conversation_id": self.history.conversation_id
                })
            response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

            last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
            if ENABLE_CURIOSITY:
                self.pressure.update(self.state.temperament_score, last_conf, 0.0)
                if self.pressure.should_erupt(CURIOSITY_PRESSURE_THRESHOLD):
                    q = self.generate_curiosity_question(prompt)
                    if q:
                        response += f" {q}"
                        self.logger.record({
                            "prompt": q,
                            "response": "",
                            "timestamp": time.time(),
                            "conversation_id": self.history.conversation_id,
                            "confidence_score": 0.0,
                            "is_system_question": True
                        })
                        self.update_metrics(q, self.curiosity.calculate_metric(q))
                        self.pressure.value -= CURIOSITY_PRESSURE_DROP

            self.logger.record({
                "prompt": prompt,
                "response": response,
                "timestamp": start_time,
                "conversation_id": self.history.conversation_id,
                "confidence_score": confidence_score,
                "is_system_question": False
            })
            self.history.add_message(prompt, response)
            if self.use_token_map_memory:
                with self.memory_lock:
                    self._update_token_map_memory(prompt, confidence_score)

            if ENABLE_GESTATION and self._should_gestate():
                self._gestate()
            print(f"Generation took {time.time() - start_time:.2f} seconds.")
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            return response

        except torch.cuda.OutOfMemoryError:
            error_msg = "Memory error—GPU ran out of space."
            self.logger.record({
                "error": "CUDA out of memory",
                "prompt": prompt,
                "timestamp": time.time(),
                "conversation_id": str(uuid.uuid4()),
                "is_error_prompt": True
            })
            if self.enable_error_listening:
                error_response = self._handle_error_prompt(error_msg)
                print(f"Self-reflection: {error_response}")
            return error_msg
        except (ValueError, RuntimeError) as e:
            error_msg = f"Error: Generation failed ({str(e)})."
            self.logger.record({
                "error": str(e),
                "prompt": prompt,
                "timestamp": time.time(),
                "conversation_id": str(uuid.uuid4()),
                "is_error_prompt": True
            })
            if self.enable_error_listening:
                error_response = self._handle_error_prompt(error_msg)
                print(f"Self-reflection: {error_response}")
            return "Something broke—check logs!"
        except Exception:
            raise

    @torch.no_grad()
    def validate_epoch(self, valid_data):
        """Validate the model on the provided data."""
        if self.dry_run:
            print("\n=== DRY RUN VALIDATION ===")
            return random.random()

        def scaffold_provider(batch):
            prompts = batch['prompt']
            scaffold_inputs = self.tokenize_and_map(prompts)
            return self.get_scaffold_hidden_states(scaffold_inputs)

        valid_loss, metrics = self.trainer.validate(valid_data, scaffold_provider)
        print(f"Validation Loss: {valid_loss:.4f}, Metrics: {metrics}")
        return valid_loss

    def cleanup(self):
        """Comprehensive cleanup with state preservation."""
        try:
            self.trainer.cleanup()
            self.save_state()

            if hasattr(self, '_temp_scaffold_context'):
                self._temp_scaffold_context = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("System cleanup completed with state preservation")
        except Exception as e:
            print(f"System cleanup failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _should_gestate(self):
        """Check if gestation should occur based on log size and time."""
        if not ENABLE_GESTATION:
            return False
        log_entries = self.logger.read()
        time_since_last = time.time() - self.last_trained
        return len(log_entries) >= SLEEP_LOG_MIN and time_since_last > 60.0           

    def new_conversation(self):
        """Start a new conversation."""
        old_id = self.history.conversation_id
        self.history = ConversationHistory()
        self._clear_scaffold_cache()
        print(f"New conversation: {self.history.conversation_id} (Previous: {old_id})")

    def _reset_sleep_state(self):
        """Reset sleep-related state (delegated to trainer)."""
        self.is_sleeping = False
        self.trainer._reset_sleep_state()

if not TRAIN_DATA or not VALID_DATA:
    print("Warning: TRAIN_DATA or VALID_DATA empty. Training may fail.")

print("Quantization mode:", QUANTIZATION_MODE)

if __name__ == "__main__":
    from sovl_cli import run_cli
    run_cli()
