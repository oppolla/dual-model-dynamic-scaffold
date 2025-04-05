import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import copy
import time

# --- Configuration (Bare Bones) ---
# Use small models suitable for consumer GPUs
BASE_MODEL_NAME = "gpt2"  # ~117M params
SCAFFOLD_MODEL_NAME = "gpt2" # ~117M params (Could be different, but keep small)
# Inject cross-attention into a couple of layers in the *base* model
# GPT-2 has 12 layers (indices 0-11). Let's pick middle and later ones.
CROSS_ATTN_LAYERS = [5, 10]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Simplified Cross-Attention Module ---

class SimpleCrossAttentionFuser(nn.Module):
    """
    Minimalist Fuser: Applies gated cross-attention.
    Assumes base_dim == scaffold_dim.
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        # Standard MultiheadAttention expects query, key, value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True # Expects (batch, seq, feature)
        )
        # Simple gate based on base model's hidden state
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim) # Add layernorm for stability

    def forward(self, base_hidden_state, scaffold_context):
        """
        Args:
            base_hidden_state (Tensor): Shape (batch, seq_len_base, hidden_dim)
            scaffold_context (Tensor): Shape (batch, seq_len_scaffold, hidden_dim)

        Returns:
            Tensor: Fused hidden state, shape (batch, seq_len_base, hidden_dim)
        """
        # Query: base_hidden_state
        # Key, Value: scaffold_context (use scaffold output as key/value)

        # Simple context pooling: Average scaffold states over sequence length
        # This provides a fixed-size representation of the scaffold output.
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        # Expand pooled context to match base sequence length for attention K/V
        # Shape becomes (batch, 1, hidden_dim) -> (batch, seq_len_base, hidden_dim)
        # Note: This is a simple approach; more sophisticated alignment could be used.
        # For attention, K/V don't strictly need same seq len as Q, but let's use
        # the *pooled* context as the K and V.
        # Shape becomes (batch, 1, hidden_dim)
        
        # Alternative: Use pooled context as Q to attend *to* the base_hidden_state
        # Q=pooled_scaffold_context, K=base_hidden_state, V=base_hidden_state
        # This might make more sense: what in the base state is relevant to the scaffold context?
        # Let's try this:
        # attn_output, _ = self.cross_attention(
        #     query=pooled_scaffold_context.repeat(1, base_hidden_state.size(1), 1), # Repeat pooled context for Q
        #     key=base_hidden_state,
        #     value=base_hidden_state
        # )
        
        # Original idea: Q=base, K=pooled_scaffold, V=pooled_scaffold
        attn_output, _ = self.cross_attention(
             query=base_hidden_state,
             key=pooled_scaffold_context, # Shape (batch, 1, hidden_dim)
             value=pooled_scaffold_context  # Shape (batch, 1, hidden_dim)
        )
        # attn_output shape: (batch, seq_len_base, hidden_dim)

        # Calculate gate based on base hidden state
        gate_values = self.gate(base_hidden_state) # Shape: (batch, seq_len_base, 1)

        # Combine: Add gated attention output to the original state
        # Apply layer normalization before residual connection
        fused_state = base_hidden_state + gate_values * attn_output
        fused_state = self.layer_norm(fused_state)

        return fused_state

# --- Bare Bones System ---

class BareBonesDMAO:
    def __init__(self):
        # --- Load Models (Frozen) ---
        print(f"Loading base model: {BASE_MODEL_NAME}")
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        # Load the full model for generation capabilities
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, config=self.base_config
        ).to(DEVICE)
        self.base_model.eval() # Set to evaluation mode
        for param in self.base_model.parameters(): # Freeze parameters
            param.requires_grad = False

        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        # Only need the core model part for hidden states, not the LM head
        self.scaffold_model = AutoModelForCausalLM.from_pretrained(
             SCAFFOLD_MODEL_NAME, config=self.scaffold_config
        ).to(DEVICE)
        self.scaffold_model.eval() # Set to evaluation mode
        for param in self.scaffold_model.parameters(): # Freeze parameters
            param.requires_grad = False

        # --- Load Tokenizers ---
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)

        # --- Handle Padding Tokens (Crucial for batching/generation) ---
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
            print(f"Set base pad token to EOS token: {self.base_tokenizer.eos_token}")
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
            # Scaffold model config doesn't strictly need pad_token_id if only used for hidden states
            print(f"Set scaffold pad token to EOS token: {self.scaffold_tokenizer.eos_token}")


        # --- Inject Cross-Attention ---
        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Injection complete.")

        # Temporary storage for scaffold context to bypass generate() limitations
        self._temp_scaffold_context = None

    def _get_model_layers(self, model):
        """Helper to get the main list of transformer layers"""
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h # GPT-2 structure
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers # Llama structure
        elif hasattr(model, 'layers'):
            return model.layers
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
             return model.decoder.layers # BART/T5 structure
        else:
            raise ValueError(f"Cannot determine layer structure for model: {model.__class__.__name__}")

    def _insert_cross_attention(self):
        """Injects the simplified cross-attention fuser into specified base model layers."""
        base_layers = self._get_model_layers(self.base_model)
        num_base_layers = len(base_layers)
        hidden_dim = self.base_config.hidden_size
        num_heads = self.base_config.num_attention_heads

        # Check if scaffold hidden dim matches base hidden dim
        if self.scaffold_config.hidden_size != hidden_dim:
            print(f"Warning: Scaffold hidden size ({self.scaffold_config.hidden_size}) "
                  f"differs from base hidden size ({hidden_dim}). "
                  f"Cross-attention may fail or require projection.")
            # Add a projection layer here if needed, or ensure models have same dim

        print(f"Base model hidden_dim: {hidden_dim}, num_heads: {num_heads}")
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

            # --- Modified Layer Wrapper ---
            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module, parent_system):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                    self._parent_system = parent_system # Reference to access temporary context

                def forward(self, hidden_states, **kwargs):
                    # Pass data through the original layer first
                    # Layer output might be a tuple (hidden_state, presents) for GPT2
                    outputs = self.orig_layer(hidden_states, **kwargs)

                    # Determine the main hidden state output
                    if isinstance(outputs, tuple):
                        base_hidden_state_output = outputs[0]
                    else:
                        base_hidden_state_output = outputs

                    # --- Context Passing Workaround ---
                    # Check if scaffold context is available from the parent system
                    scaffold_context = getattr(self._parent_system, '_temp_scaffold_context', None)

                    if scaffold_context is not None:
                        # Apply cross-attention fusion
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)

                        # Reconstruct the output tuple if necessary
                        if isinstance(outputs, tuple):
                            # Replace hidden state in tuple: (fused_state, other_outputs...)
                            final_outputs = (fused_hidden_state,) + outputs[1:]
                        else:
                            final_outputs = fused_hidden_state
                        return final_outputs
                    else:
                        # No context available, return original output
                        # print(f"Layer {layer_idx}: No scaffold context found.") # Debug
                        return outputs

            # Replace the layer in the base model
            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Successfully injected wrapper into layer {layer_idx}")


    @torch.no_grad() # Disable gradient calculations for inference
    def generate(self, prompt, max_new_tokens=50, **kwargs):
        """
        Generates text using the base model, influenced by the scaffold model's context.
        """
        start_time = time.time()

        # 1. Process input with Scaffold Model
        scaffold_inputs = self.scaffold_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.scaffold_config.n_positions # Use model's max length
        ).to(DEVICE)

        # Use torch.autocast for potential memory/speed benefits on Ampere+ cards
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):
            # Get scaffold hidden states (output_hidden_states=True needed)
            # We need the underlying transformer model if using AutoModelForCausalLM
            scaffold_core_model = self.scaffold_model.transformer \
                                  if hasattr(self.scaffold_model, 'transformer') \
                                  else self.scaffold_model.model # Adjust based on scaffold architecture
            
            scaffold_outputs = scaffold_core_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            # Use the last hidden state as the context
            scaffold_hidden_states = scaffold_outputs.hidden_states[-1] # Shape: (batch, seq_len, hidden_dim)

        # --- Context Passing Workaround ---
        # Store the context temporarily where the ModifiedLayer can access it
        self._temp_scaffold_context = scaffold_hidden_states
        # print(f"Stored scaffold context, shape: {self._temp_scaffold_context.shape}") # Debug

        # 2. Process input with Base Model for Generation
        base_inputs = self.base_tokenizer(
            prompt, return_tensors='pt'
        ).to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        # 3. Generate using the modified Base Model
        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type=='cuda' else torch.bfloat16):
            # The generate call will trigger the forward methods of the layers,
            # including our ModifiedLayer wrappers, which will access _temp_scaffold_context
             outputs = self.base_model.generate(
                 input_ids,
                 max_new_tokens=max_new_tokens,
                 pad_token_id=self.base_tokenizer.pad_token_id,
                 eos_token_id=self.base_tokenizer.eos_token_id,
                 **kwargs # Pass other generation params like temperature, top_k etc.
             )

        # --- Cleanup Temporary Context ---
        self._temp_scaffold_context = None

        # Decode the generated tokens (excluding the prompt)
        generated_ids = outputs[0][input_length:]
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return response

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nInitializing Bare Bones DMAO System...")
    try:
        dmao_system = BareBonesDMAO()
        print("\nSystem Ready. Enter prompts to generate responses.")
        print("Type 'quit' or 'exit' to stop.")

        while True:
            prompt = input("\nEnter Prompt: ")
            if prompt.lower() in ['quit', 'exit']:
                break
            if not prompt:
                continue

            # Example generation parameters (optional)
            gen_params = {
                 'temperature': 0.7,
                 'top_k': 50,
                 'do_sample': True # Sample for less deterministic output
            }

            print("\n--- Generating Response ---")
            response = dmao_system.generate(prompt, max_new_tokens=60, **gen_params)
            print("\nResponse:")
            print(response)
            print("-" * 20)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # Clean up GPU memory (optional, helps if running in a loop/notebook)
        del dmao_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
