from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType # Import PEFT components
import copy
import time
import random
from train_data import TRAIN_DATA

# VRAM Monitior
def print_memory_stats(label=""):
    """Prints current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"\n--- Memory Stats ({label}) ---")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(torch.cuda.memory_summary(abbreviated=True))
    else:
        print("(CPU mode - no GPU memory stats)")

# --- Configuration (Bare Bones + LoRA) ---
BASE_MODEL_NAME = "gpt2"  # ~117M params (Frozen)
SCAFFOLD_MODEL_NAME = "gpt2" # ~117M params (LoRA Fine-tuned)
# Inject cross-attention into a couple of layers in the *base* model
CROSS_ATTN_LAYERS = [5, 7] # Indices for GPT-2 layers (0-11)
VALID_SPLIT_RATIO = 0.2  # Add with other configs
RANDOM_SEED = 42         # Add with other configs

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

# Train Data Validation Split
random.seed(RANDOM_SEED)
random.shuffle(TRAIN_DATA)
split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")

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
        self.influence_weight = 1.0  # Add this line

    def set_influence_weight(self, weight):  # Add this method
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
        # Modify this line to include influence_weight:
        fused_state = base_hidden_state + gate_values * (attn_output * self.influence_weight)
        fused_state = self.layer_norm(fused_state)
        return fused_state

# --- Bare Bones System with Learning ---
class BareBonesDMAO_Learn:
    def __init__(self):
        # --- Load Base Model (Frozen) ---
        print(f"Loading base model: {BASE_MODEL_NAME}")
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        # Load the full model for generation capabilities
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, config=self.base_config
        ).to(DEVICE)
        print_memory_stats("After base model load")
        self.base_model.eval() # Set to evaluation mode
        for param in self.base_model.parameters(): # Freeze parameters
            param.requires_grad = False
        print(f"Base model '{BASE_MODEL_NAME}' loaded and frozen.")

        # --- Load Scaffold Model ---
        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
             SCAFFOLD_MODEL_NAME, config=self.scaffold_config
        ) # Load initially on CPU if memory constrained
        print(f"Scaffold model '{SCAFFOLD_MODEL_NAME}' loaded.")
        print_memory_stats("After scaffold model load")

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
        print("LoRA adapters applied. Trainable scaffold parameters:")
        self.scaffold_model.print_trainable_parameters()

        # --- Load ONE Shared Tokenizer ---
        print(f"Loading shared tokenizer from: {BASE_MODEL_NAME}")
        # Load tokenizer once, using the base model's spec for consistency
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"Shared tokenizer loaded (Vocab size: {self.tokenizer.vocab_size}).")

        # --- Handle Padding Token for the SHARED Tokenizer ---
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Shared tokenizer pad token set to EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")

            # Ensure both models' configurations recognize the pad token ID from the shared tokenizer
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                 pad_token_id = self.tokenizer.eos_token_id # Fallback if pad_token_id is still None
                 print(f"Warning: pad_token_id is None, using eos_token_id ({pad_token_id}) as fallback.")

            if pad_token_id is not None:
                 self.base_model.config.pad_token_id = pad_token_id
                 self.scaffold_model.config.pad_token_id = pad_token_id
                 # Also update the underlying model config if PEFT doesn't propagate it automatically
                 # This path might vary depending on the PEFT version and base model structure
                 try:
                     if hasattr(self.scaffold_model, 'base_model') and hasattr(self.scaffold_model.base_model, 'model') and hasattr(self.scaffold_model.base_model.model, 'config'):
                         self.scaffold_model.base_model.model.config.pad_token_id = pad_token_id
                     elif hasattr(self.scaffold_model, 'model') and hasattr(self.scaffold_model.model, 'config'):
                          self.scaffold_model.model.config.pad_token_id = pad_token_id
                 except AttributeError:
                     print("Could not set pad_token_id on underlying scaffold model config.")
                 print("Pad token ID configured for both models.")
            else:
                 print("Error: Could not determine a valid pad_token_id.")

        # --- Inject Cross-Attention ---
        print("Injecting cross-attention layers...")
        self._insert_cross_attention() # This modifies self.base_model
        print("Cross-attention injection complete.")

        # Temporary storage for scaffold context to bypass generate() limitations
        self._temp_scaffold_context: Optional[torch.Tensor] = None

        # --- Setup Optimizer (placeholder, setup before training) ---
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_valid_loss = float('inf')  
        self.patience = 0                    
        self.max_patience = 2               
        print("Initialization complete. Optimizer needs setup before training.")

    def set_scaffold_influence(self, weight):  # Add this method
        """Set the influence weight for all cross-attention layers (0-1 scale)"""
        base_layers = self._get_model_layers(self.base_model)
        for layer_idx in CROSS_ATTN_LAYERS:
            if layer_idx < len(base_layers):
                modified_layer = base_layers[layer_idx]
                if hasattr(modified_layer, 'cross_attn'):
                    modified_layer.cross_attn.set_influence_weight(weight)

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
        print_memory_stats("Train step start")

        # Ensure scaffold model is in training mode
        self.scaffold_model.train()
        # Base model stays in eval mode, but gradients need to flow through it
        self.base_model.eval()

        # 1. Prepare Inputs/Labels
        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Tokenize for Scaffold model (needed for context)
        scaffold_inputs = self.tokenizer(
            prompts, # Only use prompt for scaffold context? Or full text? Let's use prompt.
            return_tensors='pt',
            padding='max_length', # Pad to max length
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)

        # Tokenize for Base model (Full text needed for loss calculation)
        base_tokenizer_output = self.tokenizer(
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
        labels[labels == self.tokenizer.pad_token_id] = -100 # Mask padding

        # Mask prompt tokens (only calculate loss on completion part)
        prompt_lengths = [len(self.tokenizer(p).input_ids) for p in prompts]
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
            print_memory_stats("After forward pass")

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

        # --- Backward Pass & Optimization with Gradient Accumulation ---
        accumulation_steps = 4  # Define how many steps to accumulate gradients over
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss encountered. Skipping batch.")
            self._temp_scaffold_context = None  # Clear context
            return None  # Skip optimization

        # Scale loss to account for accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()  # Accumulate gradients

        # Perform optimization step only after accumulation_steps
        # Note: global_step needs to be tracked outside this method, so we'll assume it's passed or tracked elsewhere
        if hasattr(self, 'global_step'):  # Ensure global_step is accessible
            if (self.global_step + 1) % accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.scaffold_model.parameters()), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            print("Warning: global_step not defined. Running without accumulation logic.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        print_memory_stats("After optimizer step")

        # Cleanup context
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
            print_memory_stats(f"Epoch {epoch+1} start")
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            random.shuffle(train_data)

            # Training Phase
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                if not batch: continue

                step_loss = self.train_step(batch)

                if step_loss is not None:
                    epoch_loss += step_loss
                    steps_in_epoch += 1
                    global_step += 1
                    if global_step % 1 == 0:
                        print(f"  Step {global_step}/{num_training_steps} | Loss: {step_loss:.4f}")
                else:
                    print(f"  Step {global_step}/{num_training_steps} | Skipped")

             # Validation Phase
            valid_loss = self.validate_epoch(valid_data)
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"Epoch {epoch + 1} Stats:")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print_memory_stats(f"Epoch {epoch+1} end")

            # Early Stopping Logic
            if not hasattr(self, 'best_valid_loss'):  # Initialize on first epoch
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

            # Generation Evaluation
            if (epoch + 1) % 1 == 0:  # Every epoch
                self.evaluate_generation_quality(num_samples=2)

        end_train_time = time.time()
        print(f"--- Training Finished ({end_train_time - start_train_time:.2f}s) ---")

    def has_repetition(self, output_ids, n=3):
        """Check for n-gram repetition in output_ids."""
        output_ids = output_ids.tolist()  # Convert tensor to list for easier indexing
        for i in range(len(output_ids) - n):
            if all(output_ids[i+j] == output_ids[i+j+n] for j in range(n)):
                return True
        return False

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        print_memory_stats("Pre-generation")
        """Generates text with optional scaffold influence control"""
        if scaffold_weight is not None:  # Add this conditional
            self.set_scaffold_influence(scaffold_weight)

        start_time = time.time()
        scaffold_inputs = self.tokenizer(
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

        base_inputs = self.tokenizer(prompt, return_tensors='pt').to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):
             outputs = self.base_model.generate(
                 input_ids,
                 max_new_tokens=max_new_tokens,
                 pad_token_id=self.tokenizer.pad_token_id,
                 eos_token_id=self.tokenizer.eos_token_id,
                 **kwargs
             )

        self._temp_scaffold_context = None
        print_memory_stats("Post-generation")
        generated_ids = outputs[0][input_length:]
        if self.has_repetition(generated_ids, n=3):
            print("Warning: Repetition detected in output. Truncating at first repeat.")
            # Find the first repetition point and truncate
            for i in range(len(generated_ids) - 3):
                if all(generated_ids[i+j] == generated_ids[i+j+3] for j in range(3)):
                    generated_ids = generated_ids[:i+3]
                    break
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return response

    @torch.no_grad()
    def validate_epoch(self, valid_data):
        """Validation loss calculation"""
        self.scaffold_model.eval()
        total_loss, batches = 0, 0
        
        for i in range(0, len(valid_data), BATCH_SIZE):
            batch = valid_data[i:i+BATCH_SIZE]
            if not batch: continue
            
            # Reuse training forward logic
            prompts = [item['prompt'] for item in batch]
            completions = [item['completion'] for item in batch]
            full_texts = [p + c for p, c in zip(prompts, completions)]
            
            # Scaffold context
            scaffold_inputs = self.tokenizer(prompts, return_tensors='pt', 
                                           padding='max_length', truncation=True, 
                                           max_length=MAX_SEQ_LENGTH).to(DEVICE)
            scaffold_outputs = self.scaffold_model.base_model.transformer(**scaffold_inputs)
            self._temp_scaffold_context = scaffold_outputs.last_hidden_state
            
            # Base forward
            base_inputs = self.tokenizer(full_texts, return_tensors='pt',
                                       padding='max_length', truncation=True,
                                       max_length=MAX_SEQ_LENGTH).to(DEVICE)
            labels = base_inputs.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
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
    samples = random.sample(VALID_DATA, num_samples)
    print("\n=== Generation Evaluation ===")
    
    for example in samples:
        print(f"\nPrompt: {example['prompt']}")
        print(f"Expected: {example['completion']}")
        for weight in [0.0, 0.5, 1.0]:
            response = self.generate(example['prompt'], scaffold_weight=weight, 
                                   max_new_tokens=60, temperature=0.7)
            print(f"w={weight}: {response}")

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
                dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
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

# DOCUMENTATION
# After training:
# response = system.generate("How to make coffee?", scaffold_weight=0.7)

# To completely disable scaffold influence:
#response = system.generate("Explain quantum physics", scaffold_weight=0.0)
