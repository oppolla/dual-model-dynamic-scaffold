from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
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
import gc

class InsufficientDataError(Exception):
    """Raised when the loaded JSONL data has fewer than the minimum required entries."""
    pass

def load_jsonl(file_path, min_entries=10):
    data = []
    error_log = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    entry = json.loads(line.strip())
                    # Validate required fields and their types
                    if not isinstance(entry.get("prompt"), str) or not isinstance(entry.get("response"), str):
                        error_log.append(f"Line {line_number}: Missing or invalid 'prompt' or 'response'. Skipping.")
                        print(f"Validation Error: {error_log[-1]}")  # Added log for debugging
                        continue
                    data.append({"prompt": entry["prompt"], "completion": entry["response"]})
                except json.JSONDecodeError:
                    error_log.append(f"Line {line_number}: Failed to decode JSON. Skipping.")
                    print(f"Decode Error: {error_log[-1]}")  # Added log for debugging
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {file_path} not found. Please provide a valid file path.")
    except IOError as e:
        raise IOError(f"Error: I/O error({e.errno}): {e.strerror}.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
    
    # Log errors if any
    if error_log:
        with open("data_load_errors.log", "w") as log_file:
            log_file.write("\n".join(error_log))
        print(f"Warnings encountered during data loading. See 'data_load_errors.log' for details.")
    
    # Check minimum threshold
    print(f"Data Validation: {len(data)} entries loaded out of {min_entries} required.")  # Added monitoring log
    if len(data) < min_entries:
        raise InsufficientDataError(f"Loaded only {len(data)} valid entries from {file_path}. Need at least {min_entries} for training.")
    
    return data

def calculate_confidence_score(logits, generated_ids):
    if not logits or not isinstance(logits, (list, tuple)) or len(logits) == 0 or len(logits) != len(generated_ids):
        print(f"Warning: Logits length {len(logits) if logits else 'N/A'} != generated_ids length {len(generated_ids)}. Defaulting confidence to 0.5.")
        return 0.5
    try:
        stacked_logits = torch.stack(logits)
        probs = torch.softmax(stacked_logits, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        if max_probs.var().item() < 1e-5:  # Flat probs = low confidence
            return 0.2
        return max_probs.mean().item()
    except (RuntimeError, TypeError) as e:
        print(f"Confidence score failed: {e}")
        return 0.5

# Load training data
try:
    TRAIN_DATA = load_jsonl("sample_log.jsonl")
except InsufficientDataError as e:
    print(e)
    TRAIN_DATA = []  # Fallback to empty list
except Exception as e:
    print(f"Unexpected error during data loading: {e}")
    TRAIN_DATA = []  # Fallback to empty list

if not TRAIN_DATA:
    print("Error: No data loaded from sample_log.jsonl!")

# Load config and set global variables
with open("config.json", "r") as f:
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
    for key in required_keys:
        keys = key.split('.')
        value = config
        for k in keys:
            value = value.get(k, {})
        if not value:
            print(f"Error: Required config key '{key}' missing!")
            sys.exit(1)
    assert isinstance(CROSS_ATTN_LAYERS, list), "CROSS_ATTN_LAYERS must be a list!"
    if not USE_DYNAMIC_LAYERS:
        base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        invalid_layers = [l for l in CROSS_ATTN_LAYERS if not (0 <= l < base_config.num_hidden_layers)]
        assert not invalid_layers, f"Invalid CROSS_ATTN_LAYERS: {invalid_layers} for {base_config.num_hidden_layers} layers."
    if LAYER_SELECTION_MODE == "custom":
        base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        invalid_custom = [l for l in CUSTOM_LAYERS if not (0 <= l < base_config.num_hidden_layers)]
        assert not invalid_custom, f"Invalid CUSTOM_LAYERS: {invalid_custom} for {BASE_MODEL_NAME}"

_validate_config()

# Split training data
random.seed(RANDOM_SEED)
random.shuffle(TRAIN_DATA)
split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")
if not TRAIN_DATA or not VALID_DATA:
    print("Warning: TRAIN_DATA or VALID_DATA empty. Training may fail.")
    # Don't exit here; let the system handle it gracefully

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

class ConversationHistory:
    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = deque(maxlen=10)

    def add_message(self, prompt, response):
        self.messages.append({"prompt": prompt, "response": response})

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

class ThreadSafeLogger:
    def __init__(self, filename="log.jsonl"):
        self.filename = filename
        self.lock = Lock()

    def write(self, data):
        try:
            with self.lock:
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
        except (IOError, TypeError, ValueError) as e:
            print(f"Logging failed: {e}")
            raise

    def read(self):
        data = []
        try:
            with self.lock:
                with open(self.filename, "r", encoding="utf-8") as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            return data
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Log read failed: {e}")
            raise

    def clear(self):
        try:
            with self.lock:
                open(self.filename, "w").close()
        except Exception as e:
            print(f"Log clear failed: {e}")
            raise

class SOVLSystem:
    def __init__(self):
        self.quantization_mode = QUANTIZATION_MODE
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        self.dry_run = DRY_RUN
        self.dry_run_params = {
            'max_samples': DRY_RUN_MAX_SAMPLES,
            'max_length': DRY_RUN_MAX_LENGTH,
            'validate_architecture': DRY_RUN_VALIDATE_ARCH,
            'skip_training': DRY_RUN_SKIP_TRAINING
        }
        self.dream_memory_maxlen = DREAM_MEMORY_MAXLEN

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
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.data_exposure = 0
        self.seen_prompts = set()
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.max_patience = MAX_PATIENCE
        self.has_woken = HAS_WOKEN
        self.use_scaffold_memory = USE_SCAFFOLD_MEMORY
        self.use_token_map_memory = USE_TOKEN_MAP_MEMORY
        self.memory_decay_rate = MEMORY_DECAY_RATE
        self.logger = ThreadSafeLogger("log.jsonl")
        self.history = ConversationHistory()
        self.last_trained = 0
        self.dynamic_cross_attn_mode = DYNAMIC_CROSS_ATTN_MODE
        self.sleep_confidence_sum = 0.0
        self.sleep_confidence_count = 0
        self.confidence_history = deque(maxlen=CONFIDENCE_HISTORY_MAXLEN)
        self.temperament_history = deque(maxlen=TEMPERAMENT_HISTORY_MAXLEN)
        self.lora_capacity = sum(p.numel() for p in self.scaffolds[0].parameters() if p.requires_grad) * LIFECYCLE_CAPACITY_FACTOR
        self.last_weight = 0.0
        self.is_sleeping = False
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_optimizer = None
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

        self.temperament_score = 0.0
        self.last_temperament_score = 0.0
        self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
        self.last_prompt_embedding = None

        self.sleep_conf_threshold = SLEEP_CONF_THRESHOLD
        self.sleep_time_factor = SLEEP_TIME_FACTOR
        self.sleep_log_min = SLEEP_LOG_MIN
        self.dream_swing_var = DREAM_SWING_VAR
        self.dream_lifecycle_delta = DREAM_LIFECYCLE_DELTA
        self.dream_temperament_on = DREAM_TEMPERAMENT_ON
        self.dream_noise_scale = DREAM_NOISE_SCALE
        self.temp_eager_threshold = TEMP_EAGER_THRESHOLD
        self.temp_sluggish_threshold = TEMP_SLUGGISH_THRESHOLD
        self.temp_mood_influence = TEMP_MOOD_INFLUENCE
        self.scaffold_weight_cap = SCAFFOLD_WEIGHT_CAP
        self.base_temperature = BASE_TEMPERATURE
        self.save_path_prefix = SAVE_PATH_PREFIX
        self.dream_memory_weight = DREAM_MEMORY_WEIGHT
        self.dream_memory_decay = DREAM_MEMORY_DECAY
        self.dream_prune_threshold = DREAM_PRUNE_THRESHOLD
        self.dream_prompt_weight = DREAM_PROMPT_WEIGHT
        self.dream_novelty_boost = DREAM_NOVELTY_BOOST
        self.temp_curiosity_boost = TEMP_CURIOSITY_BOOST
        self.temp_restless_drop = TEMP_RESTLESS_DROP
        self.temp_melancholy_noise = TEMP_MELANCHOLY_NOISE
        self.conf_feedback_strength = CONF_FEEDBACK_STRENGTH
        self.temp_smoothing_factor = TEMP_SMOOTHING_FACTOR
        self.lifecycle_capacity_factor = LIFECYCLE_CAPACITY_FACTOR
        self.lifecycle_curve = LIFECYCLE_CURVE
        self.enable_dreaming = ENABLE_DREAMING
        self.enable_temperament = ENABLE_TEMPERAMENT
        self.enable_confidence_tracking = ENABLE_CONFIDENCE_TRACKING
        self.enable_gestation = ENABLE_GESTATION
        self.enable_sleep_training = ENABLE_SLEEP_TRAINING
        self.enable_cross_attention = ENABLE_CROSS_ATTENTION
        self.enable_dynamic_cross_attention = ENABLE_DYNAMIC_CROSS_ATTENTION
        self.enable_lora_adapters = ENABLE_LORA_ADAPTERS
        self.enable_repetition_check = ENABLE_REPETITION_CHECK
        self.enable_prompt_driven_dreams = ENABLE_PROMPT_DRIVEN_DREAMS
        self.enable_lifecycle_weighting = ENABLE_LIFECYCLE_WEIGHTING
        self.enable_curiosity = ENABLE_CURIOSITY
        self.curiosity_novelty_threshold_spontaneous = CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS
        self.curiosity_novelty_threshold_response = CURIOSITY_NOVELTY_THRESHOLD_RESPONSE
        self.curiosity_pressure_threshold = CURIOSITY_PRESSURE_THRESHOLD
        self.curiosity_pressure_drop = CURIOSITY_PRESSURE_DROP
        self.curiosity_silence_threshold = CURIOSITY_SILENCE_THRESHOLD
        self.curiosity_question_cooldown = CURIOSITY_QUESTION_COOLDOWN
        self.curiosity_queue_maxlen = CURIOSITY_QUEUE_MAXLEN
        self.curiosity_weight_ignorance = CURIOSITY_WEIGHT_IGNORANCE
        self.curiosity_weight_novelty = CURIOSITY_WEIGHT_NOVELTY

        self.curiosity = TrueCuriosity(self) if self.enable_curiosity else None
        self.unanswered_q = deque(maxlen=self.curiosity_queue_maxlen) if self.enable_curiosity else deque(maxlen=10)
        self.pressure = CuriosityPressure() if self.enable_curiosity else None
        self.last_question_time = time.time()
        self.metrics = {
            "curiosity_eruptions": 0,
            "spontaneous_questions": 0,
            "answered_questions": 0,
            "avg_novelty": 0.0,
            "eruption_count": 0
        }
        self.curiosity_max_new_tokens = CURIOSITY_MAX_NEW_TOKENS
        self.curiosity_base_temperature = CURIOSITY_BASE_TEMPERATURE
        self.curiosity_temperament_influence = CURIOSITY_TEMPERAMENT_INFLUENCE
        self.curiosity_top_k = CURIOSITY_TOP_K

        self.load_state()

    def generate_curiosity_question(self, context: str = None, spontaneous: bool = False) -> Optional[str]:
        if not self.enable_curiosity:
            return None
        
        if not context and self.dream_memory:
            dream_embs, _ = zip(*self.dream_memory)
            seed = self.generate("", max_new_tokens=5, temperature=1.3, do_sample=True)
            context = " ".join(seed.split()[:3])
        elif not context:
            context = ""

        temp = self.curiosity_base_temperature + (self.temperament_score * self.curiosity_temperament_influence)
        temp = max(0.7, min(1.7, temp))

        output = self.generate(context, max_new_tokens=self.curiosity_max_new_tokens, 
                             temperature=temp, top_k=self.curiosity_top_k, do_sample=True)

        if not output.endswith("?"):
            output += "?"

        score = self.curiosity.calculate_metric(output)
        threshold = self.curiosity_novelty_threshold_spontaneous if spontaneous else self.curiosity_novelty_threshold_response
        return output if score > threshold else None
    
    def update_metrics(self, question, score, spontaneous=False, answered=False):
        self.metrics["curiosity_eruptions"] += 1
        if spontaneous:
            self.metrics["spontaneous_questions"] += 1
        if answered:
            self.metrics["answered_questions"] += 1
        self.metrics["avg_novelty"] = (self.metrics["avg_novelty"] * self.metrics["eruption_count"] + score) / (self.metrics["eruption_count"] + 1)
        self.metrics["eruption_count"] += 1

    def check_silence(self, elapsed: float):
        if not self.enable_curiosity or not self.pressure:
            return
        if (elapsed > self.curiosity_silence_threshold and 
            self.pressure.value > self.curiosity_pressure_threshold and 
            (time.time() - self.last_question_time) > self.curiosity_question_cooldown):
            q = self.generate_curiosity_question(spontaneous=True)
            if q:
                print(f"{q}")
                self.logger.write({
                    "prompt": q,
                    "response": "",
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "confidence_score": 0.0,
                    "is_system_question": True
                })
                self.update_metrics(q, self.curiosity.calculate_metric(q), spontaneous=True)
                self.pressure.value -= self.curiosity_pressure_drop
                self.last_question_time = time.time()
            elif self.unanswered_q:
                q, score = self.unanswered_q.popleft()
                print(f"{q}")
                self.logger.write({
                    "prompt": q,
                    "response": "",
                    "timestamp": time.time(),
                    "conversation_id": self.history.conversation_id,
                    "confidence_score": 0.0,
                    "is_system_question": True
                })
                self.update_metrics(q, score, spontaneous=True)
                self.pressure.value -= self.curiosity_pressure_drop * 0.7
                self.last_question_time = time.time()

    def tune_curiosity(self, enable=None, spontaneous_threshold=None, response_threshold=None, pressure_threshold=None, 
                       pressure_drop=None, silence_threshold=None, question_cooldown=None, queue_maxlen=None, 
                       weight_ignorance=None, weight_novelty=None, max_new_tokens=None, base_temperature=None, 
                       temperament_influence=None, top_k=None):
        if enable is not None:
            self.enable_curiosity = bool(enable)
            if self.enable_curiosity and not self.curiosity:
                self.curiosity = TrueCuriosity(self)
                self.pressure = CuriosityPressure()
            elif not self.enable_curiosity:
                self.curiosity = None
                self.pressure = None
        if spontaneous_threshold is not None and 0.5 <= spontaneous_threshold <= 1.0:
            self.curiosity_novelty_threshold_spontaneous = spontaneous_threshold
        if response_threshold is not None and 0.5 <= response_threshold <= 1.0:
            self.curiosity_novelty_threshold_response = response_threshold
        if pressure_threshold is not None and 0.5 <= pressure_threshold <= 0.9:
            self.curiosity_pressure_threshold = pressure_threshold
        if pressure_drop is not None and 0.1 <= pressure_drop <= 0.5:
            self.curiosity_pressure_drop = pressure_drop
        if silence_threshold is not None and 5.0 <= silence_threshold <= 60.0:
            self.curiosity_silence_threshold = silence_threshold
        if question_cooldown is not None and 30.0 <= question_cooldown <= 120.0:
            self.curiosity_question_cooldown = question_cooldown
        if queue_maxlen is not None and 5 <= queue_maxlen <= 20:
            self.curiosity_queue_maxlen = queue_maxlen
            self.unanswered_q = deque(self.unanswered_q, maxlen=queue_maxlen)
        if weight_ignorance is not None and 0.0 <= weight_ignorance <= 1.0:
            self.curiosity_weight_ignorance = weight_ignorance
        if weight_novelty is not None and 0.0 <= weight_novelty <= 1.0:
            self.curiosity_weight_novelty = weight_novelty
        if max_new_tokens is not None and 5 <= max_new_tokens <= 12:
            self.curiosity_max_new_tokens = max_new_tokens
        if base_temperature is not None and 0.5 <= base_temperature <= 1.5:
            self.curiosity_base_temperature = base_temperature
        if temperament_influence is not None and 0.1 <= temperament_influence <= 0.6:
            self.curiosity_temperament_influence = temperament_influence
        if top_k is not None and 10 <= top_k <= 50:
            self.curiosity_top_k = top_k
        print(f"Curiosity params: enable={self.enable_curiosity}, spontaneous_threshold={self.curiosity_novelty_threshold_spontaneous}, "
              f"response_threshold={self.curiosity_novelty_threshold_response}, pressure_threshold={self.curiosity_pressure_threshold}, "
              f"pressure_drop={self.curiosity_pressure_drop}, silence_threshold={self.curiosity_silence_threshold}, "
              f"question_cooldown={self.curiosity_question_cooldown}, queue_maxlen={self.curiosity_queue_maxlen}, "
              f"weight_ignorance={self.curiosity_weight_ignorance}, weight_novelty={self.curiosity_weight_novelty}, "
              f"max_new_tokens={self.curiosity_max_new_tokens}, base_temperature={self.curiosity_base_temperature}, "
              f"temperament_influence={self.curiosity_temperament_influence}, top_k={self.curiosity_top_k}")

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

        q = self.generate_curiosity_question() if not self.unanswered_q else self.unanswered_q.popleft()[0]
        if q:
            print(f"{q}")
            self.logger.write({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, self.curiosity.calculate_metric(q))

        self.logger.write({"event": "wake_up", "response": response, "timestamp": time.time(), "conversation_id": self.history.conversation_id})
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
            self.sleep_conf_threshold = conf_threshold
        if time_factor is not None and 0.5 <= time_factor <= 5.0:
            self.sleep_time_factor = time_factor
        if log_min is not None and 5 <= log_min <= 20:
            self.sleep_log_min = log_min
        print(f"Sleep params: conf={self.sleep_conf_threshold}, time_factor={self.sleep_time_factor}, log_min={self.sleep_log_min}")

    def tune_dream(self, swing_var=None, lifecycle_delta=None, temperament_on=None, noise_scale=None, memory_weight=None, memory_maxlen=None, prompt_weight=None, novelty_boost=None, memory_decay=None, prune_threshold=None):
        if swing_var is not None and 0.05 <= swing_var <= 0.2:
            self.dream_swing_var = swing_var
        if lifecycle_delta is not None and 0.05 <= lifecycle_delta <= 0.2:
            self.dream_lifecycle_delta = lifecycle_delta
        if temperament_on is not None:
            self.dream_temperament_on = bool(temperament_on)
        if noise_scale is not None and 0.01 <= noise_scale <= 0.1:
            self.dream_noise_scale = noise_scale
        if memory_weight is not None and 0 <= memory_weight <= 0.5:
            self.dream_memory_weight = memory_weight
        if memory_maxlen is not None and 5 <= memory_maxlen <= 20:
            self.dream_memory_maxlen = memory_maxlen
            self.dream_memory = deque(self.dream_memory, maxlen=memory_maxlen)
        if prompt_weight is not None and 0 <= prompt_weight <= 1:
            self.dream_prompt_weight = prompt_weight
        if novelty_boost is not None and 0 <= novelty_boost <= 0.05:
            self.dream_novelty_boost = novelty_boost
        if memory_decay is not None and 0 <= memory_decay <= 1:
            self.dream_memory_decay = memory_decay
        if prune_threshold is not None and 0 <= prune_threshold <= 1:
            self.dream_prune_threshold = prune_threshold
        print(f"Dream params: swing_var={self.dream_swing_var}, lifecycle_delta={self.dream_lifecycle_delta}, temperament_on={self.dream_temperament_on}, noise_scale={self.dream_noise_scale}, memory_weight={self.dream_memory_weight}, memory_maxlen={self.dream_memory_maxlen}, prompt_weight={self.dream_prompt_weight}, novelty_boost={self.dream_novelty_boost}, memory_decay={self.dream_memory_decay}, prune_threshold={self.dream_prune_threshold}")

    def adjust_temperament(self, eager_threshold=None, sluggish_threshold=None, mood_influence=None, curiosity_boost=None, restless_drop=None, melancholy_noise=None, conf_feedback_strength=None, temp_smoothing_factor=None):
        if eager_threshold is not None and 0.7 <= eager_threshold <= 0.9:
            self.temp_eager_threshold = eager_threshold
        if sluggish_threshold is not None and 0.4 <= sluggish_threshold <= 0.6:
            self.temp_sluggish_threshold = sluggish_threshold
        if mood_influence is not None and 0 <= mood_influence <= 1:
            self.temp_mood_influence = mood_influence
        if curiosity_boost is not None and 0 <= curiosity_boost <= 0.5:
            self.temp_curiosity_boost = curiosity_boost
        if restless_drop is not None and 0 <= restless_drop <= 0.5:
            self.temp_restless_drop = restless_drop
        if melancholy_noise is not None and 0 <= melancholy_noise <= 0.05:
            self.temp_melancholy_noise = melancholy_noise
        if conf_feedback_strength is not None and 0 <= conf_feedback_strength <= 1:
            self.conf_feedback_strength = conf_feedback_strength
        if temp_smoothing_factor is not None and 0 <= temp_smoothing_factor <= 1:
            self.temp_smoothing_factor = temp_smoothing_factor
        print(f"Temperament params: eager={self.temp_eager_threshold}, sluggish={self.temp_sluggish_threshold}, mood_influence={self.temp_mood_influence}, curiosity_boost={self.temp_curiosity_boost}, restless_drop={self.temp_restless_drop}, melancholy_noise={self.temp_melancholy_noise}, conf_feedback_strength={self.conf_feedback_strength}, smoothing_factor={self.temp_smoothing_factor}")

    def set_global_blend(self, weight_cap=None, base_temp=None):
        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            self.scaffold_weight_cap = weight_cap
        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            self.base_temperature = base_temp
        print(f"Global blend: weight_cap={self.scaffold_weight_cap}, base_temp={self.base_temperature}")

    def tune_lifecycle(self, capacity_factor=None, curve=None):
        if capacity_factor is not None and 0.001 <= capacity_factor <= 0.1:
            self.lifecycle_capacity_factor = capacity_factor
            self.lora_capacity = sum(p.numel() for p in self.scaffolds[0].parameters() if p.requires_grad) * self.lifecycle_capacity_factor
        if curve in ["sigmoid_linear", "exponential"]:
            self.lifecycle_curve = curve
        print(f"Lifecycle params: capacity_factor={self.lifecycle_capacity_factor}, curve={self.lifecycle_curve}")

    def save_state(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = self.save_path_prefix
        try:
            torch.save(self.scaffolds[0].state_dict(), f"{path_prefix}_scaffold.pth")
            cross_attn_dict = {k: v for k, v in self.base_model.state_dict().items() if 'cross_attn' in k}
            torch.save(cross_attn_dict, f"{path_prefix}_cross_attn.pth")
            with open(f"{path_prefix}_token_map.json", "w") as f:
                json.dump({str(k): v for k, v in self.token_map.items()}, f)
            metadata = {
                "data_exposure": self.data_exposure,
                "last_trained": self.last_trained,
                "temperament_score": self.temperament_score,
                "last_temperament_score": self.last_temperament_score,
                "temperament_history": list(self.temperament_history),
                "dream_memory": [(m.tolist(), w) for m, w in self.dream_memory],
                "seen_prompts": list(self.seen_prompts),
                "confidence_history": list(self.confidence_history),
                "last_weight": self.last_weight,
                "enable_curiosity": self.enable_curiosity,
                "curiosity_novelty_threshold_spontaneous": self.curiosity_novelty_threshold_spontaneous,
                "curiosity_novelty_threshold_response": self.curiosity_novelty_threshold_response,
                "curiosity_pressure_threshold": self.curiosity_pressure_threshold,
                "curiosity_pressure_drop": self.curiosity_pressure_drop,
                "curiosity_silence_threshold": self.curiosity_silence_threshold,
                "curiosity_question_cooldown": self.curiosity_question_cooldown,
                "curiosity_queue_maxlen": self.curiosity_queue_maxlen,
                "curiosity_weight_ignorance": self.curiosity_weight_ignorance,
                "curiosity_weight_novelty": self.curiosity_weight_novelty,
                "curiosity_max_new_tokens": self.curiosity_max_new_tokens,
                "curiosity_base_temperature": self.curiosity_base_temperature,
                "curiosity_temperament_influence": self.curiosity_temperament_influence,
                "curiosity_top_k": self.curiosity_top_k,
                "pressure_value": self.pressure.value if self.pressure else 0.0,
                "metrics": self.metrics,
                "unanswered_q": [(q, s) for q, s in self.unanswered_q]
            }
            with open(f"{path_prefix}_meta.json", "w") as f:
                json.dump(metadata, f)
            print(f"State saved to {path_prefix}_*.pth/json")
        except Exception as e:
            print(f"Save failed: {e}")

    def load_state(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = self.save_path_prefix
        try:
            if os.path.exists(f"{path_prefix}_scaffold.pth"):
                self.scaffolds[0].load_state_dict(torch.load(f"{path_prefix}_scaffold.pth"))
                print("Scaffold state loaded.")
            if os.path.exists(f"{path_prefix}_cross_attn.pth"):
                state_dict = self.base_model.state_dict()
                state_dict.update(torch.load(f"{path_prefix}_cross_attn.pth"))
                self.base_model.load_state_dict(state_dict)
                print("Cross-attention state loaded.")
            if os.path.exists(f"{path_prefix}_token_map.json"):
                with open(f"{path_prefix}_token_map.json", "r") as f:
                    loaded_map = json.load(f)
                    self.token_map = defaultdict(lambda: [self.scaffold_unk_id], {int(k): v for k, v in loaded_map.items()})
                print("Token map loaded.")
            if os.path.exists(f"{path_prefix}_meta.json"):
                with open(f"{path_prefix}_meta.json", "r") as f:
                    meta = json.load(f)
                    self.data_exposure = meta.get("data_exposure", 0)
                    self.last_trained = meta.get("last_trained", 0)
                    self.temperament_score = meta.get("temperament_score", 0.0)
                    self.last_temperament_score = meta.get("last_temperament_score", 0.0)
                    self.temperament_history = deque(meta.get("temperament_history", []), maxlen=TEMPERAMENT_HISTORY_MAXLEN)
                    self.dream_memory = deque([(torch.tensor(m, dtype=torch.float32).to(DEVICE), w) for m, w in meta.get("dream_memory", [])], maxlen=self.dream_memory_maxlen)
                    self.seen_prompts = set(meta.get("seen_prompts", []))
                    self.confidence_history = deque(meta.get("confidence_history", []), maxlen=CONFIDENCE_HISTORY_MAXLEN)
                    self.last_weight = meta.get("last_weight", 0.0)
                    self.enable_curiosity = meta.get("enable_curiosity", ENABLE_CURIOSITY)
                    self.curiosity_novelty_threshold_spontaneous = meta.get("curiosity_novelty_threshold_spontaneous", CURIOSITY_NOVELTY_THRESHOLD_SPONTANEOUS)
                    self.curiosity_novelty_threshold_response = meta.get("curiosity_novelty_threshold_response", CURIOSITY_NOVELTY_THRESHOLD_RESPONSE)
                    self.curiosity_pressure_threshold = meta.get("curiosity_pressure_threshold", CURIOSITY_PRESSURE_THRESHOLD)
                    self.curiosity_pressure_drop = meta.get("curiosity_pressure_drop", CURIOSITY_PRESSURE_DROP)
                    self.curiosity_silence_threshold = meta.get("curiosity_silence_threshold", CURIOSITY_SILENCE_THRESHOLD)
                    self.curiosity_question_cooldown = meta.get("curiosity_question_cooldown", CURIOSITY_QUESTION_COOLDOWN)
                    self.curiosity_queue_maxlen = meta.get("curiosity_queue_maxlen", CURIOSITY_QUEUE_MAXLEN)
                    self.curiosity_weight_ignorance = meta.get("curiosity_weight_ignorance", CURIOSITY_WEIGHT_IGNORANCE)
                    self.curiosity_weight_novelty = meta.get("curiosity_weight_novelty", CURIOSITY_WEIGHT_NOVELTY)
                    self.curiosity_max_new_tokens = meta.get("curiosity_max_new_tokens", CURIOSITY_MAX_NEW_TOKENS)
                    self.curiosity_base_temperature = meta.get("curiosity_base_temperature", CURIOSITY_BASE_TEMPERATURE)
                    self.curiosity_temperament_influence = meta.get("curiosity_temperament_influence", CURIOSITY_TEMPERAMENT_INFLUENCE)
                    self.curiosity_top_k = meta.get("curiosity_top_k", CURIOSITY_TOP_K)
                    if self.enable_curiosity:
                        self.pressure = CuriosityPressure()
                        self.pressure.value = meta.get("pressure_value", 0.0)
                        self.metrics = meta.get("metrics", self.metrics)
                        self.unanswered_q = deque(meta.get("unanswered_q", []), maxlen=self.curiosity_queue_maxlen)
                print("Metadata loaded.")
        except Exception as e:
            print(f"Load failed: {e}. Starting fresh.")

    def _clear_scaffold_cache(self):
        if hasattr(self, '_temp_scaffold_context') and isinstance(self._temp_scaffold_context, torch.Tensor):
            self._temp_scaffold_context = self._temp_scaffold_context.detach().cpu()
            del self._temp_scaffold_context
        self._temp_scaffold_context = None
        if self.last_prompt_embedding is not None:
            self.last_prompt_embedding = self.last_prompt_embedding.detach().cpu()
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
        if not self.enable_cross_attention:
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
                        if self._parent_system.dream_memory and self._parent_system.dream_memory_weight > 0:
                            dream_tensors, dream_weights = zip(*self._parent_system.dream_memory)
                            dream_tensors = torch.stack(dream_tensors).to(base_hidden_state_output.device)
                            dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=base_hidden_state_output.device)
                            dream_avg = (dream_tensors * dream_weights.unsqueeze(-1)).sum(dim=0) / dream_weights.sum()
                            scaffold_context = (1 - self._parent_system.dream_memory_weight) * scaffold_context + self._parent_system.dream_memory_weight * dream_avg
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)
                        return (fused_hidden_state,) + outputs[1:] if isinstance(outputs, tuple) else fused_hidden_state
                    return outputs

            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Injected cross-attention into layer {layer_idx}")

    def enable_dry_run(self, max_samples=2, max_length=128, validate_architecture=True, skip_training=True):
        self.dry_run = True
        self.dry_run_params = {'max_samples': max_samples, 'max_length': max_length, 'validate_architecture': validate_architecture, 'skip_training': skip_training}
        print(f"Dry run activated (max_samples={max_samples}, max_length={max_length})")

    def setup_optimizer(self, num_training_steps):
        trainable_params = list(self.scaffolds[0].parameters()) + (list(self.scaffold_proj.parameters()) if self.scaffold_proj else [])
        print(f"Optimizer Setup: {len(trainable_params)} trainable parameters with learning rate {LEARNING_RATE}.")  # Added log
        self.optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        print("Optimizer and scheduler successfully initialized.")  # Added confirmation log

    def map_sequence(self, base_input_ids):
        batch_size = base_input_ids.size(0)
        seq_len = base_input_ids.size(1)
        max_expanded_len = max(seq_len * 3, MAX_SEQ_LENGTH)
        mapped_ids = torch.full((batch_size, max_expanded_len), self.scaffold_tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
        truncated = False
        for batch_idx in range(batch_size):
            position = 0
            for base_id in base_input_ids[batch_idx]:
                mapped_entry = self.special_token_map.get(base_id.item(), self.token_map.get(base_id.item()))
                mapped_tokens = mapped_entry['ids'] if isinstance(mapped_entry, dict) else mapped_entry
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
            print(f"Warning: Token mapping truncated to {max_expanded_len}.")
            self.logger.write({"warning": f"Token mapping truncated to {max_expanded_len}", "timestamp": time.time(), "conversation_id": self.history.conversation_id})
        return mapped_ids[:, :min(max_expanded_len, MAX_SEQ_LENGTH)]

    def _update_token_map_memory(self, prompt, confidence):
        if not self.use_token_map_memory:
            return
        tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
        for token_id in tokens:
            if token_id in self.token_map:
                self.token_map[token_id]['weight'] = min(self.token_map[token_id]['weight'] + confidence * 0.1, 2.0)
        for token_id in self.token_map:
            self.token_map[token_id]['weight'] *= self.memory_decay_rate

    def _should_gestate(self):
        log_entries = self.logger.read()
        if len(log_entries) < self.sleep_log_min:
            print(f"Gestation check: Log size {len(log_entries)} < {self.sleep_log_min}. No gestation.")
            return False
        avg_confidence = self.sleep_confidence_sum / self.sleep_confidence_count if self.sleep_confidence_count > 0 else 0.5
        should_gestate = (len(log_entries) >= self.sleep_log_min) and (self.sleep_confidence_count == 0 or avg_confidence > self.sleep_conf_threshold)
        print(f"Gestation check: Confidence {avg_confidence:.2f} > {self.sleep_conf_threshold} (or no data), Log {len(log_entries)} >= {self.sleep_log_min}, Gestate: {should_gestate}")
        return should_gestate
    
    def get_life_curve_weight(self):
        x = self.data_exposure / self.lora_capacity
        weight = 1 - math.exp(-2.0 * x)
        return min(1.0, weight)
    
    def _gestate(self, resume=False):
        if not resume and not self._should_gestate():
            return False

        log_entries = self.logger.read()
        if not log_entries:
            print("No log data to gestate.")
            self._reset_sleep_state()
            return False

        if resume and (not self.sleep_batch or not self.sleep_optimizer):
            print("Warning: Resume requested but sleep state invalid. Starting fresh.")
            resume = False

        if not resume:
            self.is_sleeping = True
            self.sleep_progress = 0
            self.sleep_batch = [(self.base_tokenizer(entry["prompt"], return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE),
                                self.base_tokenizer(entry["response"], return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).input_ids.to(DEVICE))
                                for entry in log_entries if "prompt" in entry and "response" in entry]
            lr = LEARNING_RATE * 0.5 * math.exp(-self.data_exposure / self.lora_capacity)
            self.sleep_optimizer = AdamW(self.scaffolds[0].parameters(), lr=lr)
            self.sleep_total_loss = 0.0
            self.sleep_steps = 0
            print("\nSystem Gestating...")
            if self.enable_dreaming and self._should_dream():
                self._dream()
            self.data_exposure += sum(len(entry["prompt"]) + len(entry["response"]) for entry in log_entries)

        if self.sleep_progress < len(self.sleep_batch):
            inputs, labels = self.sleep_batch[self.sleep_progress]
            self.scaffolds[0].train()
            scaffold_inputs = self.tokenize_and_map(inputs["input_ids"])
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
            with self._scaffold_context(scaffold_hidden_states):
                outputs = self.base_model(**inputs)
                weight = 1.2 if any(e["prompt"] == inputs["input_ids"] and e["is_system_question"] and e["response"] for e in log_entries) else 1.0
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)) * weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scaffolds[0].parameters(), 1.0)
            self.sleep_optimizer.step()
            self.sleep_optimizer.zero_grad()
            self.sleep_total_loss += loss.item()
            self.sleep_steps += 1
            self.sleep_progress += 1
            if self.sleep_steps % 5 == 0:
                print(f"Gestation progress: {self.sleep_progress}/{len(self.sleep_batch)}, loss: {self.sleep_total_loss / self.sleep_steps:.4f}")
            return True

        avg_loss = self.sleep_total_loss / self.sleep_steps if self.sleep_steps > 0 else 0
        print(f"\nGestation complete: {len(self.sleep_batch)}/{len(self.sleep_batch)}, loss: {avg_loss:.4f}")
        self.last_trained = time.time()
        self.logger.clear()
        self.last_weight = self.get_life_curve_weight()
        self.set_scaffold_influence(self.last_weight)
        print(f"Growth stage: {self.last_weight:.2f}, Exposure: {self.data_exposure}")

        q = self.generate_curiosity_question() if not self.unanswered_q else self.unanswered_q.popleft()[0]
        if q:
            print(f"{q}")
            self.logger.write({
                "prompt": q,
                "response": "",
                "timestamp": time.time(),
                "conversation_id": self.history.conversation_id,
                "confidence_score": 0.0,
                "is_system_question": True
            })
            self.update_metrics(q, self.curiosity.calculate_metric(q))

        self._reset_sleep_state()
        return False

    def _should_dream(self):
        swing_dream = len(self.confidence_history) >= 5 and torch.var(torch.tensor(list(self.confidence_history))).item() > self.dream_swing_var
        lifecycle_dream = abs(self.temperament_score - self.last_temperament_score) > self.dream_lifecycle_delta
        history_dream = False
        if len(self.temperament_history) >= 5:
            trend = torch.tensor(list(self.temperament_history)).mean().item() - self.temperament_history[0]
            history_dream = abs(trend) > 0.3
        return swing_dream or lifecycle_dream or (self.dream_temperament_on and history_dream)

    def _dream(self):
        print("--- Dreaming ---")
        log_entries = self.logger.read()
        if not log_entries:
            print("No memories to dream on.")
            return

        last_prompt = self.history.messages[-1]["prompt"] if self.history.messages else random.choice(log_entries)["prompt"]
        prompt_inputs = self.base_tokenizer(last_prompt, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
        with torch.no_grad():
            prompt_hidden = self.scaffolds[0](**prompt_inputs, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        self.last_prompt_embedding = prompt_hidden

        if self.enable_prompt_driven_dreams:
            weights = []
            for i, entry in enumerate(log_entries):
                if "prompt" not in entry:
                    continue
                log_inputs = self.base_tokenizer(entry["prompt"], return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
                log_hidden = self.scaffolds[0](**log_inputs, output_hidden_states=True).hidden_states[-1].mean(dim=1)
                similarity = F.cosine_similarity(prompt_hidden, log_hidden).item()
                recency = (i + 1) / len(log_entries)
                weight = self.dream_prompt_weight * similarity + (1 - self.dream_prompt_weight) * recency
                weights.append(weight)
            dream_entry = random.choices(log_entries, weights=weights, k=1)[0] if weights else random.choice(log_entries)
        else:
            dream_entry = random.choice(log_entries)
        dream_prompt = dream_entry["prompt"]

        is_novel = dream_prompt not in self.seen_prompts
        noise_scale = self.dream_noise_scale + (self.temp_melancholy_noise if self.temperament_score <= -0.5 else 0) + (self.dream_novelty_boost if is_novel else 0)
        noise_scale = min(noise_scale, 0.1)

        with torch.no_grad():
            inputs = self.tokenize_and_map(dream_prompt)
            hidden_states = self.get_scaffold_hidden_states(inputs)
            noise = torch.randn_like(hidden_states) * noise_scale
            dream_layer = (hidden_states.mean(dim=1) + noise).detach().cpu()

            for i in range(len(self.dream_memory)):
                tensor, weight = self.dream_memory[i]
                self.dream_memory[i] = (tensor, weight * self.dream_memory_decay)

            self.dream_memory = deque([(t, w) for t, w in self.dream_memory if w >= self.dream_prune_threshold], maxlen=self.dream_memory_maxlen)
            self.dream_memory.append((dream_layer, 1.0))

        for _ in range(3):
            q = self.generate_curiosity_question()
            if q:
                self.unanswered_q.append((q, self.curiosity.calculate_metric(q)))

        print(f"Dreaming from prompt similarity: {max(weights) if weights else 0:.2f}, novelty boost: {self.dream_novelty_boost if is_novel else 0:.3f}, dream count: {len(self.dream_memory)}, questions queued: {len(self.unanswered_q)}")
        print("--- Dream Concluded ---")

    def _sleep_train(self):
        if not self.enable_sleep_training or not self._should_gestate():
            return
        print("\n--- Sleep Training Initiated ---")
        log_entries = self.logger.read()
        if not log_entries:
            print("No log data to train on.")
            return

        if self.enable_dreaming and self._should_dream():
            self._dream()

        self.scaffolds[0].train()
        optimizer = AdamW(self.scaffolds[0].parameters(), lr=LEARNING_RATE * 0.5)
        batch = []
        for entry in log_entries:
            if "prompt" in entry and "response" in entry:
                inputs = self.base_tokenizer(entry["prompt"], return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
                labels = self.base_tokenizer(entry["response"], return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).input_ids.to(DEVICE)
                batch.append((inputs, labels))

        total_loss = 0
        for inputs, labels in batch:
            scaffold_inputs = self.tokenize_and_map(inputs["input_ids"])
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
            with self._scaffold_context(scaffold_hidden_states):
                outputs = self.base_model(**inputs)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scaffolds[0].parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(batch) if batch else 0
        print(f"Sleep Training Loss: {avg_loss:.4f}")
        self.last_trained = time.time()
        self.logger.clear()
        self.last_weight = self.get_life_curve_weight()
        if self.enable_temperament:
            self._update_temperament()
            self.last_temperament_score = self.temperament_score
        print("--- Sleep Training Complete ---")

    def _update_temperament(self):
        avg_confidence = self.sleep_confidence_sum / self.sleep_confidence_count if self.sleep_confidence_count > 0 else 0.5
        lifecycle_stage = self.data_exposure / self.lora_capacity
        base_score = 2.0 * (avg_confidence - 0.5)
    
        if lifecycle_stage < 0.25:
            bias = self.temp_curiosity_boost * (1 - lifecycle_stage / 0.25)
        elif lifecycle_stage < 0.75:
            bias = 0.0
            if len(self.temperament_history) >= 5:
                variance = torch.var(torch.tensor(list(self.temperament_history))).item()
                bias -= 0.2 * variance
        else:
            bias = -self.temp_curiosity_boost * (lifecycle_stage - 0.75) / 0.25
    
        target_score = base_score + bias + (self.conf_feedback_strength * (avg_confidence - 0.5))
        target_score = max(-1.0, min(1.0, target_score))
        alpha = 0.1 * (1 - self.temp_smoothing_factor)
        self.temperament_score = (1 - alpha) * self.temperament_score + alpha * target_score
        self.temperament_score = max(-1.0, min(1.0, self.temperament_score))
        self.temperament_history = deque(self.temperament_history, maxlen=TEMPERAMENT_HISTORY_MAXLEN)
        self.temperament_history.append(self.temperament_score)
    
        label = "melancholic" if self.temperament_score <= -0.5 else "restless" if self.temperament_score <= 0.0 else "calm" if self.temperament_score <= 0.5 else "curious"
        print(f"Temperament score: {self.temperament_score:.3f} ({label}, lifecycle: {lifecycle_stage:.2f}), confidence feedback: {avg_confidence:.2f}")

    def train_step(self, batch):
        if self.dry_run:
            print("Dry run train step")
            dry_batch = [{'prompt': item['prompt'][:self.dry_run_params['max_length']], 'completion': item['completion'][:self.dry_run_params['max_length']]} for item in batch[:self.dry_run_params['max_samples']]]
            with torch.no_grad():
                full_texts = [p + c for p, c in zip([item['prompt'] for item in dry_batch], [item['completion'] for item in dry_batch])]
                inputs = self.base_tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
                outputs = self.base_model(**inputs)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), inputs.input_ids.view(-1), ignore_index=-100)
                print(f"Dry run loss: {loss.item()}")
            return None

        if not self.optimizer:
            print("Optimizer not set up. Setting up optimizer now.")
            num_training_steps = (len(TRAIN_DATA) // BATCH_SIZE) * TRAIN_EPOCHS
            self.setup_optimizer(num_training_steps)

        self.scaffolds[0].train()
        self.base_model.eval()

        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        base_inputs = self.base_tokenizer(full_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
        base_input_ids = base_inputs.input_ids
        base_attention_mask = base_inputs.attention_mask

        prompts_base = self.base_tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)
        scaffold_input_ids = self.map_sequence(prompts_base.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()
        scaffold_inputs = {'input_ids': scaffold_input_ids, 'attention_mask': scaffold_attention_mask}

        labels = base_input_ids.clone()
        labels[labels == self.base_tokenizer.pad_token_id] = -100
        prompt_mask = self.base_tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH).attention_mask.to(DEVICE)
        labels = torch.where(prompt_mask.bool(), -100, base_input_ids)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_outputs = self.scaffolds[0](**scaffold_inputs, output_hidden_states=True)
            scaffold_hidden_states = scaffold_outputs.hidden_states[-1]
            with self._scaffold_context(scaffold_hidden_states):
                outputs = self.base_model(input_ids=base_input_ids, attention_mask=base_attention_mask)
                # Boost weight for answered system questions
                log_entries = self.logger.read()
                weight = 1.2 if any(e["prompt"] in prompts and e.get("is_system_question", False) and e["response"] for e in log_entries) else 1.0
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1)) * weight

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss. Skipping batch.")
            self.optimizer.zero_grad()
            return None

        accumulation_steps = ACCUMULATION_STEPS
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.scaffolds[0].parameters()) + (list(self.scaffold_proj.parameters()) if self.scaffold_proj else []), max_norm=1.0)

        if (self.global_step + 1) % accumulation_steps == 0:
            if self.use_scaffold_memory:
               confidence = calculate_confidence_score(outputs.logits, base_input_ids)
               if confidence > 0.7:
                   for param in self.scaffolds[0].parameters():
                       if param.grad is not None:
                           # Apply gradient clipping before boosting
                           torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)  # Clip gradients to a maximum norm of 1.0
                           param.data += param.grad * 0.01  # Boost parameter using scaled gradient
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        self.global_step += 1

        exposure_gain = EXPOSURE_GAIN_EAGER if self.temperament_score > 0.5 else EXPOSURE_GAIN_DEFAULT
        for prompt in prompts:
            if prompt not in self.seen_prompts:
                self.seen_prompts.add(prompt)
                self.data_exposure += exposure_gain
        if self.use_token_map_memory:
            self._update_token_map_memory(prompts[0], calculate_confidence_score(outputs.logits, base_input_ids))

        return loss.item()

    def run_training_cycle(self, train_data, valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        num_training_steps = (len(train_data) // batch_size) * epochs
        if num_training_steps == 0:
            print("Not enough data or epochs for training.")
            return

        if self.enable_lifecycle_weighting:
            influence_weight = self.get_life_curve_weight()
        else:
            influence_weight = self.last_weight
        self.set_scaffold_influence(influence_weight)
        print(f"Data exposure: {self.data_exposure} | Scaffold influence: {influence_weight:.3f}")

        if not self.dry_run or not self.dry_run_params['skip_training']:
            self.setup_optimizer(num_training_steps)
        print(f"\n--- Training ({epochs} epochs) ---")
        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            random.shuffle(train_data)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                if not batch:
                    continue
                step_loss = self.train_step(batch)
                if step_loss is not None:
                    epoch_loss += step_loss
                    steps_in_epoch += 1
                    print(f"  Step {self.global_step}/{num_training_steps} | Loss: {step_loss:.4f}")
            valid_loss = self.validate_epoch(valid_data)
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"Epoch {epoch + 1} Stats: Train Loss: {avg_epoch_loss:.4f}, Valid Loss: {valid_loss:.4f}")

            if self.dry_run and self.dry_run_params['skip_training']:
                break
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = 0
            else:
                self.patience += 1
                print(f"Patience: {self.patience}/{self.max_patience}")
                if self.patience >= self.max_patience:
                    print("Early stopping.")
                    break
        print(f"--- Training Finished ({time.time() - start_time:.2f}s) ---")

    def has_repetition(self, output_ids, n=3):
        ids = output_ids.tolist()
        special_ids = {self.base_tokenizer.pad_token_id, self.base_tokenizer.eos_token_id, self.base_tokenizer.bos_token_id, self.base_tokenizer.unk_token_id}
        filtered = [i for i in ids if i not in special_ids]
        for i in range(len(filtered) - 2*n):
            if filtered[i:i+n] == filtered[i+n:i+2*n]:
                return True
        return False

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        try:
            print(f"Generation initiated: prompt='{prompt[:30]}...', max_new_tokens={max_new_tokens}, scaffold_weight={scaffold_weight}")  # Added log
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

            temp = self.base_temperature
            if self.enable_temperament and self.temp_mood_influence > 0:
                temp_adjustment = self.temp_mood_influence * 0.3 * self.temperament_score
                temp += temp_adjustment
                temp = max(0.5, min(1.5, temp))

            if self.enable_dynamic_cross_attention and self.dynamic_cross_attn_mode:
                if self.dynamic_cross_attn_mode == 'confidence' and self.confidence_history:
                    avg_conf = sum(self.confidence_history) / len(self.confidence_history)
                    dynamic_weight = max(0.5, min(2.0, avg_conf * 2))
                    self.set_scaffold_influence(weight=dynamic_weight)
                elif self.dynamic_cross_attn_mode == 'temperament':
                    dynamic_weight = max(0.5, min(2.0, 1.0 + self.temperament_score))
                    self.set_scaffold_influence(weight=dynamic_weight)

            self._clear_scaffold_cache()
            with self._scaffold_context(scaffold_hidden_states):
                self.set_scaffold_influence(weight=scaffold_weight)
                outputs = self.base_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    temperature=temp,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )
            print(f"Generation completed in {time.time() - start_time:.2f}s.")  # Added log for timing
            generated_ids = outputs.sequences[0][input_length:]
            confidence_score = 0.5
            if self.enable_confidence_tracking:
                confidence_score = calculate_confidence_score(outputs.scores, generated_ids)
                self.sleep_confidence_sum += confidence_score
                self.sleep_confidence_count += 1
                self.confidence_history.append(confidence_score)

            if self.enable_repetition_check and self.has_repetition(generated_ids, n=3):
                print("Warning: Repetition detected. Truncating.")
                original_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                for i in range(len(generated_ids) - 6):
                    if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                        generated_ids = generated_ids[:i + 3]
                        break
                self.logger.write({"warning": "Repetition detected", "original_text": original_text, "truncated_at": i + 3, "timestamp": time.time(), "conversation_id": self.history.conversation_id})
            response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Add curiosity question if pressure builds
            last_conf = self.confidence_history[-1] if self.confidence_history else 0.5
            if self.enable_curiosity:
                self.pressure.update(self.temperament_score, last_conf, 0.0)
                if self.pressure.should_erupt(self.curiosity_pressure_threshold):
                    q = self.generate_curiosity_question(prompt)
                    if q:
                        response += f" {q}"
                        self.logger.write({
                            "prompt": q,
                            "response": "",
                            "timestamp": time.time(),
                            "conversation_id": self.history.conversation_id,
                            "confidence_score": 0.0,
                            "is_system_question": True
                        })
                        self.update_metrics(q, self.curiosity.calculate_metric(q))
                        self.pressure.value -= self.curiosity_pressure_drop

            # Log the full response
            self.logger.write({
                "prompt": prompt,
                "response": response,
                "timestamp": start_time,
                "conversation_id": self.history.conversation_id,
                "confidence_score": confidence_score,
                "is_system_question": False
            })
            self.history.add_message(prompt, response)
            if self.use_token_map_memory:
                self._update_token_map_memory(prompt, confidence_score)

            if self.enable_gestation and self._should_gestate():
                self._gestate()
            print(f"Generation took {time.time() - start_time:.2f} seconds.")
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            return response

        except torch.cuda.OutOfMemoryError:
            print("Error: GPU memory full! Try smaller prompt or 'int8' mode.")
            return "Memory error—try again with less input."
        except (ValueError, RuntimeError) as e:
            print(f"Error: Generation failed ({e}). Check logs.")
            self.logger.write({"error": str(e), "prompt": prompt, "timestamp": time.time()})
            return "Something broke—check logs!"
        except Exception:
            raise

        except torch.cuda.OutOfMemoryError:
            print("Error: GPU memory full! Try smaller prompt or 'int8' mode.")
            return "Memory error—try again with less input."
        except (ValueError, RuntimeError) as e:
            print(f"Error: Generation failed ({e}). Check logs.")
            self.logger.write({"error": str(e), "prompt": prompt, "timestamp": time.time()})
            return "Something broke—check logs!"
        except Exception:
            raise

        self.logger.write({"prompt": prompt, "response": response, "timestamp": start_time, "conversation_id": self.history.conversation_id, "confidence_score": confidence_score})
        self.history.add_message(prompt, response)
        if self.use_token_map_memory:
            self._update_token_map_memory(prompt, confidence_score) 
        if self.enable_gestation and self._should_gestate():
            self._gestate()
        print(f"Generation took {time.time() - start_time:.2f} seconds.")
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()  # Free GPU memory after generation
        return response

    @torch.no_grad()
    def validate_epoch(self, valid_data):
        if self.dry_run:
            print("\n=== DRY RUN VALIDATION ===")
            return random.random()
        self.scaffolds[0].eval()
        total_loss, batches = 0, 0

        for i in range(0, len(valid_data), BATCH_SIZE):
            batch = valid_data[i:i + BATCH_SIZE]
            if not batch:
                continue
            prompts = [item['prompt'] for item in batch]
            completions = [item['completion'] for item in batch]
            full_texts = [p + c for p, c in zip(prompts, completions)]

            scaffold_inputs = self.tokenize_and_map(prompts)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

            with self._scaffold_context(scaffold_hidden_states):
                base_inputs = self.base_tokenizer(full_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH).to(DEVICE)
                labels = base_inputs.input_ids.clone()
                labels[labels == self.base_tokenizer.pad_token_id] = -100
                outputs = self.base_model(**base_inputs)
                loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1), ignore_index=-100)

            total_loss += loss.item()
            batches += 1
        return total_loss / batches if batches > 0 else 0

    def cleanup(self):
        self.save_state()
        self._clear_scaffold_cache()
        for attr in ['base_model', 'scaffolds', 'optimizer', 'scheduler']:
            try:
                if attr == 'scaffolds':
                    for scaffold in self.scaffolds:
                        del scaffold
                else:
                    delattr(self, attr)
            except Exception as e:
                print(f"Failed to delete {attr}: {e}")
        try:
            if self.last_prompt_embedding is not None:
                self.last_prompt_embedding = None
        except Exception as e:
            print(f"Failed to clear last_prompt_embedding: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup completed.")

    def new_conversation(self):
        old_id = self.history.conversation_id
        self.history = ConversationHistory()
        self._clear_scaffold_cache()
        print(f"New conversation: {self.history.conversation_id} (Previous: {old_id})")

    def _reset_sleep_state(self):
        self.is_sleeping = False
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_optimizer = None
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

if not TRAIN_DATA or not VALID_DATA:
    print("Warning: TRAIN_DATA or VALID_DATA empty. Training may fail.")

print("Quantization mode:", QUANTIZATION_MODE)

if __name__ == "__main__":
    print("\nInitializing SOVL System...")
    sovl_system = None
    try:
        sovl_system = SOVLSystem()
        if "--dry-run" in sys.argv:
            sovl_system.enable_dry_run(max_samples=2, max_length=64)
        sovl_system.wake_up()
        print("\nSystem Ready.")
        print("Commands: 'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', 'new', 'save', 'load', "
              "'sleep <conf> <time> <log>', 'dream <swing> <delta> <temp_on> <noise> <mem_weight> <mem_maxlen> <prompt_weight> <novelty_boost>', "
              "'temp <eager> <sluggish> <influence> <curiosity> <restless> <melancholy> <conf_strength> <smoothing_factor>', "
              "'blend <weight> <temp>', 'lifecycle <capacity> <curve>', "
              "'curiosity <enable> <spontaneous> <response> <pressure> <drop> <silence> <cooldown> <queue_maxlen> <ignorance> <novelty> <max_tokens> <base_temp> <temp_influence> <top_k>', "
              "'cross weight <float> | blend <float> | layers <float...> | confidence | temperament | off', "
              "'scaffold_mem', 'token_mem', 'both_mem', 'no_mem', or a prompt.")

        valid_commands = [
            'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', 'new', 'save', 'load', 
            'sleep', 'dream', 'temp', 'blend', 'lifecycle', 'cross', 'curiosity', 'scaffold_mem', 'token_mem', 'both_mem', 'no_mem'
        ]

        last_input_time = time.time()
        while True:
            user_cmd = input("\nEnter command or prompt: ").strip().lower()
            elapsed = time.time() - last_input_time
            sovl_system.check_silence(elapsed)
            parts = user_cmd.split()
            cmd = parts[0] if parts else ""

            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'train':
                sovl_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
                if sovl_system.dry_run:
                    break
            elif cmd in ['int8', 'int4', 'fp16']:
                sovl_system.set_quantization_mode(cmd)
                print(f"Re-initializing with {cmd.upper()} quantization...")
                sovl_system = SOVLSystem()
                sovl_system.wake_up()
            elif cmd == 'dynamic':
                sovl_system.toggle_dynamic_layers(True)
                print("Re-initializing with dynamic layers...")
                sovl_system = SOVLSystem()
                sovl_system.wake_up()
            elif cmd == 'fixed':
                sovl_system.toggle_dynamic_layers(False)
                print("Re-initializing with fixed layers...")
                sovl_system = SOVLSystem()
                sovl_system.wake_up()
            elif cmd == 'new':
                sovl_system.new_conversation()
            elif cmd == 'save':
                path = parts[1] if len(parts) > 1 else None
                sovl_system.save_state(path)
            elif cmd == 'load':
                path = parts[1] if len(parts) > 1 else None
                sovl_system.load_state(path)
            elif cmd == 'sleep':
                try:
                    if len(parts) != 4:
                        raise ValueError("Usage: sleep <conf> <time> <log>")
                    sovl_system.set_sleep_params(float(parts[1]), float(parts[2]), int(parts[3]))
                except ValueError as e:
                    print(f"Error: {e}")
            elif cmd == 'dream' and len(parts) == 11:
                sovl_system.tune_dream(float(parts[1]), float(parts[2]), parts[3].lower() == 'true', float(parts[4]), float(parts[5]), int(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10]))
            elif cmd == 'temp' and len(parts) == 9:
                sovl_system.adjust_temperament(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8]))
            elif cmd == 'blend' and len(parts) == 3:
                sovl_system.set_global_blend(float(parts[1]), float(parts[2]))
            elif cmd == 'lifecycle' and len(parts) == 3:
                sovl_system.tune_lifecycle(float(parts[1]), parts[2])
            elif cmd == 'curiosity' and len(parts) == 15:
                sovl_system.tune_curiosity(
                    parts[1].lower() == 'true', float(parts[2]), float(parts[3]), float(parts[4]), 
                    float(parts[5]), float(parts[6]), float(parts[7]), int(parts[8]), float(parts[9]), 
                    float(parts[10]), int(parts[11]), float(parts[12]), float(parts[13]), int(parts[14])
                )
            elif cmd == 'curiosity' and len(parts) == 11:
                sovl_system.tune_curiosity(
                    parts[1].lower() == 'true', float(parts[2]), float(parts[3]), float(parts[4]), 
                    float(parts[5]), float(parts[6]), float(parts[7]), int(parts[8]), float(parts[9]), float(parts[10])
                )
            elif cmd == 'cross' and len(parts) >= 2:
                args = parts[1:]
                if args[0] == 'weight' and len(args) == 2:
                    sovl_system.tune_cross_attention(weight=float(args[1]))
                elif args[0] == 'blend' and len(args) == 2:
                    sovl_system.tune_cross_attention(blend_strength=float(args[1]))
                elif args[0] == 'layers' and len(args) > 1:
                    layer_weights = [float(w) for w in args[1:]]
                    sovl_system.tune_cross_attention(layer_weights=layer_weights)
                elif args[0] in ['confidence', 'temperament', 'off']:
                    sovl_system.tune_cross_attention(dynamic_mode=args[0])
                else:
                    print("Usage: cross weight <float> | blend <float> | layers <float...> | confidence | temperament | off")
            elif cmd in ['scaffold_mem', 'token_mem', 'both_mem', 'no_mem']:
                sovl_system.toggle_memory(cmd)
            elif not user_cmd:
                continue
            else:
                log_entries = sovl_system.logger.read()
                last_entry = log_entries[-1] if log_entries else None
                if last_entry and last_entry.get("is_system_question", False) and not last_entry["response"]:
                    # User answered a system question
                    last_entry["response"] = user_cmd
                    last_entry["confidence_score"] = calculate_confidence_score(
                        sovl_system.base_model(**sovl_system.base_tokenizer(user_cmd, return_tensors='pt').to(DEVICE)).logits,
                        sovl_system.base_tokenizer(user_cmd).input_ids
                    )
                    sovl_system.logger.write(last_entry)  # Update log
                    sovl_system.update_metrics(last_entry["prompt"], sovl_system.curiosity.calculate_metric(last_entry["prompt"]), answered=True)
                    print("Thanks for the insight!")
                elif cmd not in valid_commands:
                    print("Error: Invalid command. Please enter a valid command.")
                    print("Valid commands are:", ', '.join(valid_commands))
                else:
                    print("\n--- Generating Response ---")
                    response = sovl_system.generate(user_cmd, max_new_tokens=60, temperature=sovl_system.base_temperature, top_k=50, do_sample=True)
                    print("\nResponse:", response)
                    print("-" * 20)
            last_input_time = time.time()

    except FileNotFoundError as e:
        print(f"\nFile error: {e}. Check 'config.json' and 'sample_log.jsonl'.")
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try smaller BATCH_SIZE, MAX_SEQ_LENGTH, or INT8/INT4.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sovl_system is not None:
            sovl_system.cleanup()
            del sovl_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
