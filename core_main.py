from typing import Optional
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
import threading
from sklearn.model_selection import KFold

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                data.append({"prompt": entry["prompt"], "completion": entry["response"]})
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Starting with empty data!")
        data = []
    return data

TRAIN_DATA = load_jsonl("sample_log.jsonl")
if not TRAIN_DATA:
    print("Error: No data loaded from sample_log.jsonl!")
else:
    with open("config.json", "r") as f:
        config = json.load(f)

    def get_config_value(config, key, default=None):
        return config.get(key, default) or (print(f"Warning: '{key}' missing, using {default}"), default)[1]

    BASE_MODEL_NAME = get_config_value(config, "base_model_name", "gpt2")
    SCAFFOLD_MODEL_NAME = get_config_value(config, "scaffold_model_name", "gpt2")
    CROSS_ATTN_LAYERS = get_config_value(config, "cross_attn_layers", [0, 1, 2])
    USE_DYNAMIC_LAYERS = get_config_value(config, "use_dynamic_layers", False)
    LAYER_SELECTION_MODE = get_config_value(config, "layer_selection_mode", "balanced")
    CUSTOM_LAYERS = get_config_value(config, "custom_layers", [])
    VALID_SPLIT_RATIO = get_config_value(config, "valid_split_ratio", 0.2)
    RANDOM_SEED = get_config_value(config, "random_seed", 42)
    QUANTIZATION_MODE = get_config_value(config, "quantization", "fp16")
    if QUANTIZATION_MODE not in ["fp16", "int8", "int4"]:
        print(f"Warning: Invalid quantization '{QUANTIZATION_MODE}'. Defaulting to 'fp16'.")
        QUANTIZATION_MODE = "fp16"

    lora_config = get_config_value(config, "lora_config", {})
    LORA_RANK = get_config_value(lora_config, "lora_rank", 8)
    LORA_ALPHA = get_config_value(lora_config, "lora_alpha", 16)
    LORA_DROPOUT = get_config_value(lora_config, "lora_dropout", 0.1)
    LORA_TARGET_MODULES = get_config_value(lora_config, "lora_target_modules", ["q_proj", "v_proj"])

    training_config = get_config_value(config, "training_config", {})
    LEARNING_RATE = get_config_value(training_config, "learning_rate", 2e-5)
    TRAIN_EPOCHS = get_config_value(training_config, "train_epochs", 3)
    BATCH_SIZE = get_config_value(training_config, "batch_size", 2)
    MAX_SEQ_LENGTH = get_config_value(training_config, "max_seq_length", 512)
    SIGMOID_SCALE = get_config_value(training_config, "sigmoid_scale", 0.5)
    SIGMOID_SHIFT = get_config_value(training_config, "sigmoid_shift", 5.0)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    def _validate_config():
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
    random.seed(RANDOM_SEED)
    random.shuffle(TRAIN_DATA)
    split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
    TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
    print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")

def get_cross_attention_layers(model):
    total_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.layers)
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

    def set_influence_weight(self, weight):
        self.influence_weight = max(0.0, min(1.0, weight))

    def forward(self, base_hidden_state, scaffold_context):
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        attn_output, _ = self.cross_attention(query=base_hidden_state, key=pooled_scaffold_context, value=pooled_scaffold_context)
        gate_values = self.gate(base_hidden_state)
        fused_state = base_hidden_state + gate_values * (attn_output * self.influence_weight)
        return self.layer_norm(fused_state)

class ConversationHistory:
    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = deque(maxlen=10)

    def add_message(self, prompt, response):
        self.messages.append({"prompt": prompt, "response": response})

class ThreadSafeLogger:
    def __init__(self, filename="log.jsonl"):
        self.filename = filename
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            try:
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
            except (IOError, TypeError, ValueError) as e:
                print(f"Logging failed: {e}")

def calculate_confidence_score(logits, generated_ids):
    if not logits or not isinstance(logits, (list, tuple)) or len(logits) == 0:
        return 0.5
    try:
        stacked_logits = torch.stack(logits)
        probs = torch.softmax(stacked_logits, dim=-1)
        return torch.max(probs, dim=-1).values.mean().item()
    except (RuntimeError, TypeError) as e:
        print(f"Confidence score failed: {e}")
        return 0.5

class BareBonesDMAO_Learn:
    def __init__(self):
        self.quantization_mode = QUANTIZATION_MODE
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        self.dry_run = False
        self.dry_run_params = {'max_samples': 2, 'max_length': 128, 'validate_architecture': True, 'skip_training': True}

        print(f"Loading base model: {BASE_MODEL_NAME}")
        quantization_config = {"load_in_8bit": True} if self.quantization_mode == "int8" else {"load_in_4bit": True} if self.quantization_mode == "int4" else {}
        self.base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, config=self.base_config, **quantization_config).to(DEVICE)
        self.print_memory_stats("After base model load", verbose=True)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"Base model '{BASE_MODEL_NAME}' loaded and frozen.")

        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(SCAFFOLD_MODEL_NAME, config=self.scaffold_config, **quantization_config)
        lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM)
        self.scaffolds = [get_peft_model(scaffold_model_raw, lora_config).to(DEVICE)]  # List for future chorus
        print("LoRA adapters applied to scaffold[0]. Trainable parameters:")
        self.scaffolds[0].print_trainable_parameters()

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
            token_map = defaultdict(lambda: [scaffold_tokenizer.unk_token_id])  # Default to unk
            for base_token, base_id in base_tokenizer.get_vocab().items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(normalized, add_special_tokens=False, max_length=3, truncation=True) or [scaffold_tokenizer.unk_token_id]
                token_map[base_id] = {'ids': scaffold_ids, 'weight': 1.0}  # Add weight for memory
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
        self.max_patience = 2
        self.has_woken = False
        self.use_scaffold_memory = True  # Toggle for scaffold memory
        self.use_token_map_memory = True  # Toggle for token map memory
        self.memory_decay_rate = 0.95  # Decay for token map weights
        self.logger = ThreadSafeLogger("log.jsonl")
        self.history = ConversationHistory()

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

    def get_life_curve_weight(self):
        lora_params = sum(p.numel() for p in self.scaffolds[0].parameters() if p.requires_grad)
        capacity_threshold = lora_params // 100
        growth = 1 / (1 + torch.exp(-SIGMOID_SCALE * (self.data_exposure - SIGMOID_SHIFT)))
        degradation = max(0, 1 - ((self.data_exposure - capacity_threshold) / capacity_threshold)) if self.data_exposure > capacity_threshold else 1.0
        return min(max((growth * degradation).item(), 0.0), 1.0)

    def wake_up(self):
        if self.has_woken:
            return None
        wake_seed = (int(time.time() * 1000) + random.randint(0, 100)) % 10000
        torch.manual_seed(wake_seed)
        random.seed(wake_seed)
        prompt = " "
        with torch.no_grad():
            response = self.generate(prompt, max_new_tokens=15, temperature=1.7, top_k=30, do_sample=True)
        self.has_woken = True
        print(f"\n{response}")
        self.logger.write({"event": "wake_up", "response": response, "timestamp": time.time(), "conversation_id": self.history.conversation_id})
        return response

    def print_memory_stats(self, label="", verbose=False):
        if verbose and torch.cuda.is_available():
            print(f"\n--- Memory Stats ({label}) ---")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def set_scaffold_influence(self, weight):
        base_layers = self._get_model_layers(self.base_model)
        layers = get_cross_attention_layers(self.base_model) if USE_DYNAMIC_LAYERS else CROSS_ATTN_LAYERS
        for layer_idx in layers:
            if layer_idx < len(base_layers) and hasattr(base_layers[layer_idx], 'cross_attn'):
                base_layers[layer_idx].cross_attn.set_influence_weight(weight)

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

    def _clear_scaffold_cache(self):
        if hasattr(self, '_temp_scaffold_context') and isinstance(self._temp_scaffold_context, torch.Tensor):
            self._temp_scaffold_context = self._temp_scaffold_context.detach().cpu()
            del self._temp_scaffold_context
        self._temp_scaffold_context = None
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
            scaffold_outputs = self.scaffolds[0](**scaffold_inputs, output_hidden_states=True)  # First scaffold for now
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
        self.optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        print("Optimizer and scheduler set up.")

    def map_sequence(self, base_input_ids):
        batch_size = base_input_ids.size(0)
        max_expanded_len = MAX_SEQ_LENGTH * 3
        mapped_ids = torch.full((batch_size, max_expanded_len), self.scaffold_tokenizer.pad_token_id, dtype=torch.long, device=DEVICE)
        truncated = False
        for batch_idx in range(batch_size):
            position = 0
            for base_id in base_input_ids[batch_idx]:
                mapped_entry = self.special_token_map.get(base_id.item(), self.token_map.get(base_id.item()))
                mapped_tokens = mapped_entry['ids'] if isinstance(mapped_entry, dict) else mapped_entry
                if position + len(mapped_tokens) > MAX_SEQ_LENGTH:
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
            print(f"Warning: Token mapping truncated to {MAX_SEQ_LENGTH}.")
            self.logger.write({"warning": f"Token mapping truncated to {MAX_SEQ_LENGTH}", "timestamp": time.time(), "conversation_id": self.history.conversation_id})
        return mapped_ids[:, :MAX_SEQ_LENGTH]

    def _update_token_map_memory(self, prompt, confidence):
        if not self.use_token_map_memory:
            return
        tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
        for token_id in tokens:
            if token_id in self.token_map:
                self.token_map[token_id]['weight'] = min(self.token_map[token_id]['weight'] + confidence * 0.1, 2.0)  # Cap at 2.0
        for token_id in self.token_map:
            self.token_map[token_id]['weight'] *= self.memory_decay_rate  # Decay over time

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
            raise RuntimeError("Optimizer not set up. Call setup_optimizer first.")
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
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss. Skipping batch.")
            return None

        accumulation_steps = 4
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.scaffolds[0].parameters()) + (list(self.scaffold_proj.parameters()) if self.scaffold_proj else []), max_norm=1.0)

        if (self.global_step + 1) % accumulation_steps == 0:
            if self.use_scaffold_memory:  # Update scaffold memory if enabled
                confidence = calculate_confidence_score(outputs.logits, base_input_ids)
                if confidence > 0.7:  # Threshold for "memorable" outputs
                    for param in self.scaffolds[0].parameters():
                        if param.grad is not None:
                            param.data += param.grad * 0.01  # Small tweak for memory
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        self.global_step += 1

        for prompt in prompts:
            if prompt not in self.seen_prompts:
                self.seen_prompts.add(prompt)
                self.data_exposure += 1
        if self.use_token_map_memory:  # Update token map memory if enabled
            self._update_token_map_memory(prompts[0], calculate_confidence_score(outputs.logits, base_input_ids))

        return loss.item()

    def run_training_cycle(self, train_data, valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        num_training_steps = (len(train_data) // batch_size) * epochs
        if num_training_steps == 0:
            print("Not enough data or epochs for training.")
            return

        influence_weight = self.get_life_curve_weight()
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
        if self.dry_run:
            print("\n=== DRY RUN GENERATION ===")
            truncated_prompt = prompt[:self.dry_run_params['max_length']]
            print(f"Input: {truncated_prompt}{'...' if len(prompt) > self.dry_run_params['max_length'] else ''}")
            return "[DRY RUN] Pretend output"

        if scaffold_weight is not None:
            self.set_scaffold_influence(scaffold_weight)
        print(f"Using quantization mode: {self.quantization_mode}")

        start_time = time.time()
        base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        scaffold_inputs = self.tokenize_and_map(prompt)
        scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)

        self._clear_scaffold_cache()
        with self._scaffold_context(scaffold_hidden_states):
            outputs = self.base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.base_tokenizer.pad_token_id,
                eos_token_id=self.base_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )

        generated_ids = outputs.sequences[0][input_length:]
        confidence_score = calculate_confidence_score(outputs.scores, generated_ids)

        if self.has_repetition(generated_ids, n=3):
            print("Warning: Repetition detected. Truncating.")
            original_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
            for i in range(len(generated_ids) - 3):
                if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                    generated_ids = generated_ids[:i + 3]
                    break
            self.logger.write({"warning": "Repetition detected", "original_text": original_text, "truncated_at": i + 3, "timestamp": time.time(), "conversation_id": self.history.conversation_id})
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        self.logger.write({"prompt": prompt, "response": response, "timestamp": start_time, "conversation_id": self.history.conversation_id, "confidence_score": confidence_score})
        self.history.add_message(prompt, response)
        if self.use_token_map_memory:
            self._update_token_map_memory(prompt, confidence_score)

        print(f"Generation took {time.time() - start_time:.2f} seconds.")
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
        self._clear_scaffold_cache()
        for attr in ['base_model', 'scaffolds', 'optimizer', 'scheduler']:
            if hasattr(self, attr):
                try:
                    if attr == 'scaffolds':
                        for scaffold in self.scaffolds:
                            del scaffold
                    else:
                        delattr(self, attr)
                except Exception as e:
                    print(f"Failed to delete {attr}: {e}")
        self._clear_scaffold_cache()
        print("Cleanup completed.")

    def new_conversation(self):
        old_id = self.history.conversation_id
        self.history = ConversationHistory()
        self._clear_scaffold_cache()
        print(f"New conversation: {self.history.conversation_id} (Previous: {old_id})")

if not TRAIN_DATA or not VALID_DATA:
    print("Warning: TRAIN_DATA or VALID_DATA empty. Training may fail.")

print("Quantization mode:", QUANTIZATION_MODE)

if __name__ == "__main__":
    print("\nInitializing Bare Bones DMAO System...")
    dmao_system = None
    try:
        dmao_system = BareBonesDMAO_Learn()
        if "--dry-run" in sys.argv:
            dmao_system.enable_dry_run(max_samples=2, max_length=64)
        dmao_system.wake_up()
        print("\nSystem Ready.")
        print("Commands: 'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', 'new', 'scaffold_mem', 'token_mem', 'both_mem', 'no_mem', or a prompt.")

        while True:
            user_cmd = input("\nEnter command or prompt: ").strip().lower()
            if user_cmd in ['quit', 'exit']:
                break
            elif user_cmd == 'train':
                dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
                if dmao_system.dry_run:
                    break
            elif user_cmd in ['int8', 'int4', 'fp16']:
                dmao_system.set_quantization_mode(user_cmd)
                print(f"Re-initializing with {user_cmd.upper()} quantization...")
                dmao_system = BareBonesDMAO_Learn()
                dmao_system.wake_up()
            elif user_cmd == 'dynamic':
                dmao_system.toggle_dynamic_layers(True)
                print("Re-initializing with dynamic layers...")
                dmao_system = BareBonesDMAO_Learn()
                dmao_system.wake_up()
            elif user_cmd == 'fixed':
                dmao_system.toggle_dynamic_layers(False)
                print("Re-initializing with fixed layers...")
                dmao_system = BareBonesDMAO_Learn()
                dmao_system.wake_up()
            elif user_cmd == 'new':
                dmao_system.new_conversation()
            elif user_cmd in ['scaffold_mem', 'token_mem', 'both_mem', 'no_mem']:
                dmao_system.toggle_memory(user_cmd)
            elif not user_cmd:
                continue
            else:
                print("\n--- Generating Response ---")
                response = dmao_system.generate(user_cmd, max_new_tokens=60, temperature=0.7, top_k=50, do_sample=True)
                print("\nResponse:", response)
                print("-" * 20)

    except FileNotFoundError as e:
        print(f"\nFile error: {e}. Check 'config.json' and 'sample_log.jsonl'.")
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try smaller BATCH_SIZE, MAX_SEQ_LENGTH, or INT8/INT4.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dmao_system is not None:
            dmao_system.cleanup()
            del dmao_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
