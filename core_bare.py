import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
import copy
import time
import random

# --- Configuration (Bare Bones + LoRA) ---
BASE_MODEL_NAME = "gpt2"  # ~117M params (Frozen)
SCAFFOLD_MODEL_NAME = "gpt2" # ~117M params (LoRA Fine-tuned)
# Inject cross-attention into a couple of layers in the *base* model
CROSS_ATTN_LAYERS = [5, 10] # Indices for GPT-2 layers (0-11)

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 16 # Typically 2*LORA_RANK
LORA_DROPOUT = 0.1
# Common target modules for LoRA in GPT-2 like models
LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"] # Adjust based on model architecture if needed

# Training Config
LEARNING_RATE = 3e-4 # Higher LR common for LoRA
TRAIN_EPOCHS = 3 # Number of epochs to train on the mini-dataset
BATCH_SIZE = 1 # Keep batch size small due to potential memory constraints
MAX_SEQ_LENGTH = 128 # Max sequence length for training/inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Minimal Hardcoded Training Data ---
# (prompt, desired_completion) pairs
TRAIN_DATA = [
    {"prompt": "The capital of France is", "completion": " Paris."},
    {"prompt": "To be or not to be,", "completion": " that is the question."},
    {"prompt": "Photosynthesis is the process where plants use", "completion": " sunlight, water, and carbon dioxide to create their own food."},
    {"prompt": "The first president of the United States was", "completion": " George Washington."},
    {"prompt": "Write a short story about a cat:", "completion": " Mittens the cat loved chasing butterflies in the sunny garden."}
]

# --- Simplified Cross-Attention Module (Unchanged) ---
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

    def forward(self, base_hidden_state, scaffold_context):
        # Simple context pooling (average)
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        # Q=base, K=pooled_scaffold, V=pooled_scaffold
        attn_output, _ = self.cross_attention(
             query=base_hidden_state,
             key=pooled_scaffold_context,
             value=pooled_scaffold_context
        )
        gate_values = self.gate(base_hidden_state)
        fused_state = base_hidden_state + gate_values * attn_output
        fused_state = self.layer_norm(fused_state)
        return fused_state

# --- Bare Bones System with Learning ---
class BareBonesDMAO_Learn:
    def __init__(self):
        # --- Load Base Model (Frozen) ---
        print(f"Loading base model: {BASE_MODEL_NAME}")
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, config=self.base_config
        ).to(DEVICE)
        self.base_model.eval() # Keep base model in eval mode
        for param in self.base_model.parameters(): # Freeze parameters
            param.requires_grad = False

        # --- Load Scaffold Model ---
        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
             SCAFFOLD_MODEL_NAME, config=self.scaffold_config
        ) # Load initially on CPU if memory constrained

        # --- Apply LoRA to Scaffold Model ---
        print("Applying LoRA adapters to scaffold model...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM # Important for Causal LM tasks
        )
        # Apply PEFT to the scaffold model
        self.scaffold_model = get_peft_model(scaffold_model_raw, lora_config)
        self.scaffold_model.to(DEVICE) # Move scaffold model (with LoRA) to GPU
        print("Trainable scaffold parameters:")
        self.scaffold_model.print_trainable_parameters()

        # --- Load Tokenizers ---
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)

        # --- Handle Padding Tokens ---
        for tokenizer, model in [(self.base_tokenizer, self.base_model), (self.scaffold_tokenizer, self.scaffold_model)]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
                print(f"Set pad token to EOS token for {model.__class__.__name__}")

        # --- Inject Cross-Attention ---
        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Injection complete.")

        # Temporary storage for scaffold context
        self._temp_scaffold_context = None

        # --- Setup Optimizer (placeholder, setup before training) ---
        self.optimizer = None
        self.scheduler = None

    def _get_model_layers(self, model):
        """Helper to get the main list of transformer layers"""
        # PEFT models often wrap the original model
        actual_model = model.base_model if hasattr(model, 'base_model') else model
        
        if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return actual_model.transformer.h # GPT-2 structure
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return actual_model.model.layers # Llama structure
        elif hasattr(actual_model, 'layers'):
            return actual_model.layers
        elif hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
             return actual_model.decoder.layers # BART/T5 structure
        else:
            raise ValueError(f"Cannot determine layer structure for model: {actual_model.__class__.__name__}")


    def _insert_cross_attention(self):
        """Injects the simplified cross-attention fuser into specified base model layers."""
        base_layers = self._get_model_layers(self.base_model)
        num_base_layers = len(base_layers)
        hidden_dim = self.base_config.hidden_size
        num_heads = self.base_config.num_attention_heads

        if self.scaffold_config.hidden_size != hidden_dim:
            print(f"Warning: Scaffold hidden size != base hidden size. Add projection if needed.")
            # Add projection here if needed

        print(f"Injecting CrossAttentionFuser at layers: {CROSS_ATTN_LAYERS}")

        for layer_idx in CROSS_ATTN_LAYERS:
            if layer_idx >= num_base_layers:
                print(f"Warning: Layer index {layer_idx} out of bounds ({num_base_layers} layers). Skipping.")
                continue

            original_layer = base_layers[layer_idx]
            cross_attn_fuser = SimpleCrossAttentionFuser(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(DEVICE)
            # Freeze the fuser parameters as well? Or allow them to train?
            # Let's freeze them to focus learning only on LoRA adapters for simplicity.
            for param in cross_attn_fuser.parameters():
                 param.requires_grad = False

            # --- Modified Layer Wrapper (mostly unchanged) ---
            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module, parent_system):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                    self._parent_system = parent_system

                def forward(self, hidden_states, **kwargs):
                    outputs = self.orig_layer(hidden_states, **kwargs)
                    base_hidden_state_output = outputs[0] if isinstance(outputs, tuple) else outputs

                    # --- Context Access --- Check temporary context
                    scaffold_context = getattr(self._parent_system, '_temp_scaffold_context', None)

                    if scaffold_context is not None:
                        # Ensure context is on the same device as the layer
                        scaffold_context = scaffold_context.to(base_hidden_state_output.device)

                        # Apply cross-attention (will run with enabled grad during training)
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)

                        final_outputs = (fused_hidden_state,) + outputs[1:] if isinstance(outputs, tuple) else fused_hidden_state
                        return final_outputs
                    else:
                        return outputs # Return original if no context

            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Successfully injected wrapper into layer {layer_idx}")

    def setup_optimizer(self, num_training_steps):
        """Sets up the optimizer and scheduler for LoRA training."""
        # Only optimize the trainable parameters of the scaffold model (LoRA adapters)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.scaffold_model.parameters()),
            lr=LEARNING_RATE
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0, # No warmup for simplicity
            num_training_steps=num_training_steps
        )
        print("Optimizer and scheduler set up.")

    def train_step(self, batch):
        """Performs a single training step."""
        if not self.optimizer:
             raise RuntimeError("Optimizer not set up. Call setup_optimizer first.")

        # Ensure scaffold model is in training mode
        self.scaffold_model.train()
        # Base model stays in eval mode, but gradients need to flow through it
        self.base_model.eval()

        # 1. Prepare Inputs/Labels
        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Tokenize for Scaffold model (needed for context)
        scaffold_inputs = self.scaffold_tokenizer(
            prompts, # Only use prompt for scaffold context? Or full text? Let's use prompt.
            return_tensors='pt',
            padding='max_length', # Pad to max length
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)

        # Tokenize for Base model (Full text needed for loss calculation)
        base_tokenizer_output = self.base_tokenizer(
            full_texts,
            return_tensors='pt',
            padding='max_length', # Pad to max length
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        base_input_ids = base_tokenizer_output.input_ids.to(DEVICE)
        base_attention_mask = base_tokenizer_output.attention_mask.to(DEVICE)

        # Create labels: shift input_ids, mask prompt tokens and padding
        labels = base_input_ids.clone()
        labels[labels == self.base_tokenizer.pad_token_id] = -100 # Mask padding

        # Mask prompt tokens (only calculate loss on completion part)
        prompt_lengths = [len(self.base_tokenizer(p).input_ids) for p in prompts]
        for i, prompt_len in enumerate(prompt_lengths):
            # Clamp prompt_len to avoid exceeding sequence length used for tokenization
            actual_prompt_len_in_batch = min(prompt_len, MAX_SEQ_LENGTH)
            labels[i, :actual_prompt_len_in_batch] = -100

        # --- Forward Pass ---
        # Use autocast for mixed precision
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):

            # 2. Get Scaffold Context (requires gradients for LoRA)
            # No need for torch.no_grad() here
            scaffold_core_model = self.scaffold_model.base_model.transformer if hasattr(self.scaffold_model.base_model, 'transformer') else self.scaffold_model.base_model.model
            scaffold_outputs = scaffold_core_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            scaffold_hidden_states = scaffold_outputs.hidden_states[-1]

            # 3. Store context temporarily (workaround)
            self._temp_scaffold_context = scaffold_hidden_states

            # 4. Forward pass through Base Model (needs gradient flow)
            # The base model's forward will use the modified layers
            # Ensure torch.enable_grad() context if needed, though autograd should handle it
            outputs = self.base_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
                # We don't pass labels here, need logits for custom loss calculation
            )
            base_logits = outputs.logits # Shape: (batch, seq_len, vocab_size)

            # 5. Calculate Loss
            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
            loss_fct = nn.CrossEntropyLoss() # Handles ignore_index=-100
            loss = loss_fct(base_logits.view(-1, base_logits.size(-1)), labels.view(-1))

        # --- Backward Pass & Optimization ---
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
             print("Warning: Invalid loss encountered. Skipping batch.")
             self._temp_scaffold_context = None # Clear context
             return None # Skip optimization

        self.optimizer.zero_grad()
        loss.backward() # Backpropagate gradients ONLY to LoRA parameters
        
        # Optional: Gradient clipping (good practice)
        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.scaffold_model.parameters()), 1.0)
        
        self.optimizer.step() # Update LoRA weights
        self.scheduler.step() # Update learning rate schedule

        # Cleanup context
        self._temp_scaffold_context = None

        return loss.item()


    def run_training_cycle(self, train_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        """Runs a training cycle on the provided data."""
        num_training_steps = (len(train_data) // batch_size) * epochs
        if num_training_steps == 0:
             print("Not enough data or epochs for training.")
             return
             
        self.setup_optimizer(num_training_steps)
        
        print(f"\n--- Starting Training ({epochs} epochs) ---")
        start_train_time = time.time()
        global_step = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            # Shuffle data each epoch
            random.shuffle(train_data)

            # Simple batching
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                if not batch: continue

                step_loss = self.train_step(batch)

                if step_loss is not None:
                    epoch_loss += step_loss
                    steps_in_epoch += 1
                    global_step += 1
                    if global_step % 1 == 0: # Print every step for small dataset
                        print(f"  Step {global_step}/{num_training_steps} | Loss: {step_loss:.4f}")
                else:
                     print(f"  Step {global_step}/{num_training_steps} | Skipped due to invalid loss")


            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        end_train_time = time.time()
        print(f"--- Training Finished ({end_train_time - start_train_time:.2f} seconds) ---")


    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, **kwargs):
        """Generates text using the base model, influenced by the *trained* scaffold model."""
        # Ensure models are in eval mode for inference
        self.base_model.eval()
        self.scaffold_model.eval()

        start_time = time.time()
        scaffold_inputs = self.scaffold_tokenizer(
            prompt, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):
            # Use the PEFT model directly
            scaffold_outputs = self.scaffold_model(
                 **scaffold_inputs,
                 output_hidden_states=True
            )
            # Access hidden states correctly for PEFT model
            # Usually the underlying model's output holds hidden_states
            actual_outputs = scaffold_outputs.hidden_states if hasattr(scaffold_outputs, 'hidden_states') else scaffold_outputs.base_model_output.hidden_states

            scaffold_hidden_states = actual_outputs[-1]

        self._temp_scaffold_context = scaffold_hidden_states

        base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):
             outputs = self.base_model.generate(
                 input_ids,
                 max_new_tokens=max_new_tokens,
                 pad_token_id=self.base_tokenizer.pad_token_id,
                 eos_token_id=self.base_tokenizer.eos_token_id,
                 **kwargs
             )

        self._temp_scaffold_context = None
        generated_ids = outputs[0][input_length:]
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return response

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nInitializing Bare Bones DMAO System with Learning...")
    try:
        dmao_system = BareBonesDMAO_Learn()
        print("\nSystem Ready.")
        print("Commands: 'quit', 'exit', 'train', or enter a prompt.")

        while True:
            user_cmd = input("\nEnter command or prompt: ")
            cmd = user_cmd.lower().strip()

            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'train':
                # Run training on the hardcoded data
                dmao_system.run_training_cycle(TRAIN_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
            elif not user_cmd:
                continue
            else:
                # Treat as a prompt for generation
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

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del dmao_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
