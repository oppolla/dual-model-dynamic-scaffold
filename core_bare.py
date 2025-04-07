from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType  # Import PEFT components
import copy
import time
import random
from train_data import TRAIN_DATA
import bitsandbytes as bnb
import json

# --- Load Configuration from JSON ---
with open("config.json", "r") as f:
    config = json.load(f)

# Extract variables from config with defaults
def get_config_value(config, key, default=None):
    try:
        return config[key]
    except KeyError:
        print(f"Warning: '{key}' missing in config.json. Using default: {default}")
        return default

BASE_MODEL_NAME = get_config_value(config, "base_model_name", "gpt2")
SCAFFOLD_MODEL_NAME = get_config_value(config, "scaffold_model_name", "gpt2")
CROSS_ATTN_LAYERS = get_config_value(config, "cross_attn_layers", [0, 1, 2])
USE_DYNAMIC_LAYERS = get_config_value(config, "use_dynamic_layers", False)
LAYER_SELECTION_MODE = get_config_value(config, "layer_selection_mode", "balanced")
CUSTOM_LAYERS = get_config_value(config, "custom_layers", [])
VALID_SPLIT_RATIO = get_config_value(config, "valid_split_ratio", 0.2)
RANDOM_SEED = get_config_value(config, "random_seed", 42)

# LoRA Configuration
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# VRAM Monitor
def print_memory_stats(label=""):
    """Prints current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"\n--- Memory Stats ({label}) ---")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(torch.cuda.memory_summary(abbreviated=True))
    else:
        print("(CPU mode - no GPU memory stats)")

# Train Data Validation Split
random.seed(RANDOM_SEED)
random.shuffle(TRAIN_DATA)
split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")

# --- Dynamic Layer Selection ---
def get_cross_attention_layers(model):
    """Dynamic layer selection based on config"""
    total_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.layers)
    
    if LAYER_SELECTION_MODE == "early":
        return list(range(0, total_layers//3))
    elif LAYER_SELECTION_MODE == "late":
        return list(range(2*total_layers//3, total_layers))
    elif LAYER_SELECTION_MODE == "custom" and CUSTOM_LAYERS:
        return [l for l in CUSTOM_LAYERS if 0 <= l < total_layers]
    else:  # balanced (default)
        return list(range(total_layers//3, 2*total_layers//3))
    #todo: invent new modes like mix of early/late, different patterns etc

# --- Simplified Cross-Attention Module ---
class SimpleCrossAttentionFuser(nn.Module):
    """
    Minimalist Fuser: Applies gated cross-attention.
    Assumes base_dim == scaffold_dim.
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.influence_weight = 1.0

    def set_influence_weight(self, weight):
        """Set influence weight (0-1 scale)"""
        self.influence_weight = max(0.0, min(1.0, weight))

    def forward(self, base_hidden_state, scaffold_context):
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        attn_output, _ = self.cross_attention(
            query=base_hidden_state,
            key=pooled_scaffold_context,
            value=pooled_scaffold_context
        )
        gate_values = self.gate(base_hidden_state)
        fused_state = base_hidden_state + gate_values * (attn_output * self.influence_weight)
        fused_state = self.layer_norm(fused_state)
        return fused_state

# --- MAIN SYSTEM ---
class BareBonesDMAO_Learn:
    def __init__(self):
        self.quantization_mode = "fp16"  # Default: full precision (FP16/BF16), options: "fp16", "int8", "int4"
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)

        # --- LOAD BASE MODEL (frozen) ---
        print(f"Loading base model: {BASE_MODEL_NAME}")
        quantization_config = {}
        if self.quantization_mode == "int8":
            quantization_config["load_in_8bit"] = True
        elif self.quantization_mode == "int4":
            quantization_config["load_in_4bit"] = True
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            config=self.base_config,
            **quantization_config
        ).to(DEVICE)
        print_memory_stats("After base model load")
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"Base model '{BASE_MODEL_NAME}' loaded and frozen.")

        # --- LOAD SCAFFOLD MODEL (dynamic) ---
        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
            SCAFFOLD_MODEL_NAME,
            config=self.scaffold_config,
            **quantization_config
        )
        print(f"Scaffold model '{SCAFFOLD_MODEL_NAME}' loaded.")
        print_memory_stats("After scaffold model load")

        # --- APPLY LoRA TO SCAFFOLD MODEL ---
        print("Applying LoRA adapters to scaffold model...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.scaffold_model = get_peft_model(scaffold_model_raw, lora_config)
        self.scaffold_model.to(DEVICE)
        print("LoRA adapters applied. Trainable scaffold parameters:")
        self.scaffold_model.print_trainable_parameters()

        # --- LOAD TOKENIZERS ---
        print(f"Loading base tokenizer from: {BASE_MODEL_NAME}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"Base tokenizer loaded (Vocab size: {self.base_tokenizer.vocab_size}).")

        print(f"Loading scaffold tokenizer from: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)
        print(f"Scaffold tokenizer loaded (Vocab size: {self.scaffold_tokenizer.vocab_size}).")

        # --- HANDLE PADDING TOKENS ---
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            print(f"Base tokenizer pad token set to EOS token: '{self.base_tokenizer.eos_token}' (ID: {self.base_tokenizer.eos_token_id})")

        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
            print(f"Scaffold tokenizer pad token set to EOS token: '{self.scaffold_tokenizer.eos_token}' (ID: {self.scaffold_tokenizer.eos_token_id})")

        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        self.scaffold_model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id

        try:
            if hasattr(self.scaffold_model, 'base_model') and hasattr(self.scaffold_model.base_model, 'model') and hasattr(self.scaffold_model.base_model.model, 'config'):
                self.scaffold_model.base_model.model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id
            elif hasattr(self.scaffold_model, 'model') and hasattr(self.scaffold_model.model, 'config'):
                self.scaffold_model.model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id
        except AttributeError:
            print("Could not set pad_token_id on underlying scaffold model config.")

                # --- BUILD TOKEN MAPPING ---
        def build_token_map(base_tokenizer, scaffold_tokenizer):
            """Build mapping with support for multi-token sequences"""
            base_vocab = base_tokenizer.get_vocab()
            scaffold_vocab = scaffold_tokenizer.get_vocab()
            token_map = {}

            for base_token, base_id in base_vocab.items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(
                    normalized, 
                    add_special_tokens=False,
                    max_length=3,  # Limit expansion length
                    truncation=True
                ) or [scaffold_tokenizer.unk_token_id]
                token_map[base_id] = scaffold_ids

            return token_map

        self.token_map = build_token_map(self.base_tokenizer, self.scaffold_tokenizer)

                # --- SPECIAL TOKEN MAPPING ---
        self.special_token_map = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id or self.scaffold_tokenizer.sep_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        self.scaffold_unk_id = self.scaffold_tokenizer.unk_token_id

                # --- INJECT CROSS ATTENTION ---
        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Cross-attention injection complete.")

                # TEMP STORAGE FOR SCAFFOLD CONTEXT
        self._temp_scaffold_context: Optional[torch.Tensor] = None

                # --- SETUP OPTIMIZER (placeholder) ---
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.max_patience = 2
        print("Quantization mode set to:", self.quantization_mode)
        print("Initialization complete. Optimizer needs setup before training.")

    def set_scaffold_influence(self, weight):
        """Set the influence weight for all cross-attention layers (0-1 scale)"""
        base_layers = self._get_model_layers(self.base_model)
        if USE_DYNAMIC_LAYERS:
            cross_attn_layers = get_cross_attention_layers(self.base_model)
        else:
            cross_attn_layers = CROSS_ATTN_LAYERS
        for layer_idx in cross_attn_layers:
            if layer_idx < len(base_layers):
                modified_layer = base_layers[layer_idx]
                if hasattr(modified_layer, 'cross_attn'):
                    modified_layer.cross_attn.set_influence_weight(weight)

    def set_quantization_mode(self, mode: str):
        """Set quantization mode: 'fp16', 'int8', or 'int4'. Restart system for full effect."""
        valid_modes = ["fp16", "int8", "int4"]
        if mode not in valid_modes:
            print(f"Invalid mode '{mode}'. Use: {valid_modes}")
            return
        if mode != self.quantization_mode:
            self.quantization_mode = mode
            print(f"Quantization mode set to '{mode}'. Restart system to apply quantization.")

    def toggle_dynamic_layers(self, enable: bool):
        """Toggle between dynamic layer selection and fixed CROSS_ATTN_LAYERS. Restart system to apply."""
        global USE_DYNAMIC_LAYERS
        if enable != USE_DYNAMIC_LAYERS:
            USE_DYNAMIC_LAYERS = enable
            print(f"Dynamic layer selection {'enabled' if enable else 'disabled'}. Restart system to apply.")

    def _get_model_layers(self, model):
        """Helper to get the main list of transformer layers"""
        actual_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return actual_model.transformer.h
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return actual_model.model.layers
        elif hasattr(actual_model, 'layers'):
            return actual_model.layers
        elif hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
            return actual_model.decoder.layers
        else:
            raise ValueError(f"Cannot determine layer structure for model: {actual_model.__class__.__name__}")

    def _insert_cross_attention(self):
        """Injects cross-attention with toggle between dynamic and fixed layers"""
        base_layers = self._get_model_layers(self.base_model)
        num_base_layers = len(base_layers)
        hidden_dim = self.base_config.hidden_size
        num_heads = self.base_config.num_attention_heads

        if self.scaffold_config.hidden_size != hidden_dim:
            print(f"Adding TRAINABLE projection: {self.scaffold_config.hidden_size} -> {hidden_dim}")
            self.scaffold_proj = nn.Linear(self.scaffold_config.hidden_size, hidden_dim).to(DEVICE)
            self.scaffold_proj.requires_grad = True  # <-- KEY CHANGE: Make it trainable
        else:
            self.scaffold_proj = None

        # Toggle between dynamic and fixed layers
        if USE_DYNAMIC_LAYERS:
            cross_attn_layers = get_cross_attention_layers(self.base_model)
            print(f"Using dynamic layer selection mode '{LAYER_SELECTION_MODE}': {cross_attn_layers}")
        else:
            cross_attn_layers = CROSS_ATTN_LAYERS
            print(f"Using fixed layers: {cross_attn_layers}")

        for layer_idx in cross_attn_layers:
            if layer_idx >= num_base_layers:
                print(f"Warning: Layer index {layer_idx} out of bounds ({num_base_layers} layers). Skipping.")
                continue

            original_layer = base_layers[layer_idx]
            cross_attn_fuser = SimpleCrossAttentionFuser(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(DEVICE)

            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module, parent_system):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                    self._parent_system = parent_system

                def forward(self, hidden_states, **kwargs):
                    outputs = self.orig_layer(hidden_states, **kwargs)
                    base_hidden_state_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    scaffold_context = getattr(self._parent_system, '_temp_scaffold_context', None)

                    if scaffold_context is not None:
                        scaffold_context = scaffold_context.to(base_hidden_state_output.device)
                        if self._parent_system.scaffold_proj is not None:
                            scaffold_context = self._parent_system.scaffold_proj(scaffold_context)
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)
                        final_outputs = (fused_hidden_state,) + outputs[1:] if isinstance(outputs, tuple) else fused_hidden_state
                        return final_outputs
                    return outputs

            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Successfully injected wrapper into layer {layer_idx}")

    def setup_optimizer(self, num_training_steps):
        """Sets up the optimizer and scheduler for LoRA training."""
        # Collect all trainable parameters (Scaffold LoRA + Projection)
        trainable_params = list(self.scaffold_model.parameters())
        if self.scaffold_proj is not None:
            trainable_params += list(self.scaffold_proj.parameters())  # <-- Add projection layer

        self.optimizer = AdamW(
            trainable_params,  # <-- Optimize both scaffold AND projection
            lr=LEARNING_RATE
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        print("Optimizer and scheduler set up.")

    def map_sequence(self, base_input_ids):
        """Handle multi-token expansions with efficient padding, preserving complete expansions"""
        batch_size = base_input_ids.size(0)
        max_expanded_len = MAX_SEQ_LENGTH * 3  # Buffer for expansions

        # Initialize tensor with padding tokens
        mapped_ids = torch.full(
            (batch_size, max_expanded_len),
            self.scaffold_tokenizer.pad_token_id,
            dtype=torch.long,
            device=DEVICE
        )

        # Track truncation for logging
        truncated = False

        # Build sequences for each item in batch
        for batch_idx in range(batch_size):
            position = 0
            for base_id in base_input_ids[batch_idx]:
                mapped_tokens = self.special_token_map.get(
                    base_id.item(),
                    self.token_map.get(base_id.item(), [self.scaffold_unk_id])
                )

                # Check if adding this expansion fits within MAX_SEQ_LENGTH
                if position + len(mapped_tokens) > MAX_SEQ_LENGTH:
                    truncated = True
                    break
                
                # Add complete expansion
                for token in mapped_tokens:
                    if position >= max_expanded_len:  # Shouldn’t hit this with buffer
                        truncated = True
                        break
                    mapped_ids[batch_idx, position] = token
                    position += 1

        if truncated:
            print(f"Warning: Token mapping truncated expansions to fit MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}. Consider increasing MAX_SEQ_LENGTH for longer prompts.")
    
        # Return up to MAX_SEQ_LENGTH
        return mapped_ids[:, :MAX_SEQ_LENGTH]

    def train_step(self, batch):
        """Performs a single training step."""
        if not self.optimizer:
            raise RuntimeError("Optimizer not set up. Call setup_optimizer first.")
        print_memory_stats("Train step start")

        self.scaffold_model.train()
        self.base_model.eval()

        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        base_tokenizer_output = self.base_tokenizer(
            full_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        base_input_ids = base_tokenizer_output.input_ids.to(DEVICE)
        base_attention_mask = base_tokenizer_output.attention_mask.to(DEVICE)

        prompts_base = self.base_tokenizer(
            prompts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        scaffold_input_ids = self.map_sequence(prompts_base.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

        scaffold_inputs = {
            'input_ids': scaffold_input_ids,
            'attention_mask': scaffold_attention_mask
        }

        labels = base_input_ids.clone()
        labels[labels == self.base_tokenizer.pad_token_id] = -100
        prompt_lengths = [len(self.base_tokenizer(p).input_ids) for p in prompts]
        for i, prompt_len in enumerate(prompt_lengths):
            actual_prompt_len_in_batch = min(prompt_len, MAX_SEQ_LENGTH)
            labels[i, :actual_prompt_len_in_batch] = -100

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_outputs = self.scaffold_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            scaffold_hidden_states = scaffold_outputs.hidden_states[-1]
            print_memory_stats("After forward pass")

            self._temp_scaffold_context = scaffold_hidden_states

            outputs = self.base_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
            )
            base_logits = outputs.logits

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(base_logits.view(-1, base_logits.size(-1)), labels.view(-1))

        accumulation_steps = 4
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss encountered. Skipping batch.")
            self._temp_scaffold_context = None
            return None

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if (self.global_step + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        self.global_step += 1  # Increment here

        print_memory_stats("After optimizer step")
        self._temp_scaffold_context = None
        return loss.item()

    def run_training_cycle(self, train_data, valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        """Modified training loop with validation"""
        num_training_steps = (len(train_data) // batch_size) * epochs
        if num_training_steps == 0:
            print("Not enough data or epochs for training.")
            return

        self.setup_optimizer(num_training_steps)
        print(f"\n--- Starting Training ({epochs} epochs) ---")
        start_train_time = time.time()
        global_step = 0

        for epoch in range(epochs):
            print_memory_stats(f"Epoch {epoch + 1} start")
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            random.shuffle(train_data)

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i: i + batch_size]
                if not batch:
                    continue

                step_loss = self.train_step(batch)
                if step_loss is not None:
                    epoch_loss += step_loss
                    steps_in_epoch += 1
                    global_step += 1
                    if global_step % 1 == 0:
                        print(f"  Step {global_step}/{num_training_steps} | Loss: {step_loss:.4f}")
                else:
                    print(f"  Step {global_step}/{num_training_steps} | Skipped")

            valid_loss = self.validate_epoch(valid_data)
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"Epoch {epoch + 1} Stats:")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print_memory_stats(f"Epoch {epoch + 1} end")

            if not hasattr(self, 'best_valid_loss'):
                self.best_valid_loss = float('inf')
                self.patience = 0
                self.max_patience = 2
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = 0
            else:
                self.patience += 1
                print(f"Patience: {self.patience}/{self.max_patience}")
                if self.patience >= self.max_patience:
                    print("Early stopping triggered.")
                    break

            if (epoch + 1) % 1 == 0:
                self.evaluate_generation_quality(num_samples=2)

        end_train_time = time.time()
        print(f"--- Training Finished ({end_train_time - start_train_time:.2f}s) ---")

    def has_repetition(self, output_ids, n=3):
        """Check for n-gram repetition in output_ids, ignoring padding and special tokens."""
        special_ids = {
            self.base_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id,
            self.base_tokenizer.bos_token_id,
            self.base_tokenizer.unk_token_id
        }
        # Filter out special tokens
        filtered_ids = [id for id in output_ids.tolist() if id not in special_ids]
        if len(filtered_ids) < n:
            return False
        for i in range(len(filtered_ids) - n):
            if all(filtered_ids[i + j] == filtered_ids[i + j + n] for j in range(n)):
                return True
        return False

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        """Generates text with optional scaffold influence control"""
        print_memory_stats("Pre-generation")
        if scaffold_weight is not None:
            self.set_scaffold_influence(scaffold_weight)
        print(f"Using quantization mode: {self.quantization_mode}")

        start_time = time.time()
        base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        scaffold_base_inputs = self.base_tokenizer(
            prompt, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        scaffold_input_ids = self.map_sequence(scaffold_base_inputs.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

        scaffold_inputs = {
            'input_ids': scaffold_input_ids,
            'attention_mask': scaffold_attention_mask
        }

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_outputs = self.scaffold_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            actual_outputs = scaffold_outputs.hidden_states if hasattr(scaffold_outputs, 'hidden_states') else scaffold_outputs.base_model_output.hidden_states
            scaffold_hidden_states = actual_outputs[-1]

        self._temp_scaffold_context = scaffold_hidden_states

        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            outputs = self.base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.base_tokenizer.pad_token_id,
                eos_token_id=self.base_tokenizer.eos_token_id,
                **kwargs
            )

        self._temp_scaffold_context = None
        print_memory_stats("Post-generation")
        generated_ids = outputs[0][input_length:]
        if self.has_repetition(generated_ids, n=3):
            print("Warning: Repetition detected in output. Truncating at first repeat.")
            for i in range(len(generated_ids) - 3):
                if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                    generated_ids = generated_ids[:i + 3]
                    break
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return response

    @torch.no_grad()
    def validate_epoch(self, valid_data):
        """Validation loss calculation"""
        self.scaffold_model.eval()
        total_loss, batches = 0, 0

        for i in range(0, len(valid_data), BATCH_SIZE):
            batch = valid_data[i:i + BATCH_SIZE]
            if not batch:
                continue

            prompts = [item['prompt'] for item in batch]
            completions = [item['completion'] for item in batch]
            full_texts = [p + c for p, c in zip(prompts, completions)]

            prompts_base = self.base_tokenizer(
                prompts,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            scaffold_input_ids = self.map_sequence(prompts_base.input_ids)
            scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

            scaffold_inputs = {
                'input_ids': scaffold_input_ids,
                'attention_mask': scaffold_attention_mask
            }

            scaffold_outputs = self.scaffold_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            self._temp_scaffold_context = scaffold_outputs.hidden_states[-1]

            base_inputs = self.base_tokenizer(
                full_texts,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            labels = base_inputs.input_ids.clone()
            labels[labels == self.base_tokenizer.pad_token_id] = -100

            outputs = self.base_model(**base_inputs)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)

            total_loss += loss.item()
            batches += 1
            self._temp_scaffold_context = None

        return total_loss / batches if batches > 0 else 0

    @torch.no_grad()
    def evaluate_generation_quality(self, num_samples=3):
        """Generate sample responses"""
        samples = random.sample(VALID_DATA, min(num_samples, len(VALID_DATA)))
        print("\n=== Generation Evaluation ===")

        for example in samples:
            print(f"\nPrompt: {example['prompt']}")
            print(f"Expected: {example['completion']}")
            for weight in [0.0, 0.5, 1.0]:
                response = self.generate(example['prompt'], scaffold_weight=weight,
                                         max_new_tokens=60, temperature=0.7)
                print(f"w={weight}: {response}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("\nInitializing Bare Bones DMAO System with Learning...")
    dmao_system = None
    try:
        dmao_system = BareBonesDMAO_Learn()
        print("\nSystem Ready.")
        print("Commands: 'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', or enter a prompt.")

        while True:
            user_cmd = input("\nEnter command or prompt: ")
            cmd = user_cmd.lower().strip()

            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'train':
                dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
            elif cmd == 'int8':
                dmao_system.set_quantization_mode("int8")
                print("Re-initializing system with INT8 quantization...")
                dmao_system = BareBonesDMAO_Learn()
            elif cmd == 'int4':
                dmao_system.set_quantization_mode("int4")
                print("Re-initializing system with INT4 quantization...")
                dmao_system = BareBonesDMAO_Learn()
            elif cmd == 'fp16':
                dmao_system.set_quantization_mode("fp16")
                print("Re-initializing system with FP16 quantization...")
                dmao_system = BareBonesDMAO_Learn()
            elif cmd == 'dynamic':
                dmao_system.toggle_dynamic_layers(True)
                print("Re-initializing system with dynamic layers...")
                dmao_system = BareBonesDMAO_Learn()
            elif cmd == 'fixed':
                dmao_system.toggle_dynamic_layers(False)
                print("Re-initializing system with fixed layers...")
                dmao_system = BareBonesDMAO_Learn()
            elif not user_cmd:
                continue
            else:
                prompt = user_cmd
                gen_params = {
                    'temperature': 0.7,
                    'top_k': 50,
                    'do_sample': True
                }
                print("\n--- Generating Response ---")
                response = dmao_system.generate(prompt, max_new_tokens=60, **gen_params)
                print("\nResponse:")
                print(response)
                print("-" * 20)

    except FileNotFoundError as e:
        print(f"\nFile error: {e}. Ensure 'config.json' and 'train_data.py' are present and correctly formatted.")
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try reducing BATCH_SIZE, MAX_SEQ_LENGTH, or switching to INT8/INT4 quantization.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dmao_system is not None:
            del dmao_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
