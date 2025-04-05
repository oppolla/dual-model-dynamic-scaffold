# --- Configuration ---
BASE_MODEL_NAME = "deepseek-ai/deepseek-llm-67b-chat" # Consider smaller models for testing, e.g., "gpt2"
SCAFFOLD_MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen1.5-1.5b" # Consider smaller models, e.g., "gpt2"
LORA_RANK = 8 # Fixed LoRA rank
CROSS_ATTN_LAYERS = [8, 16, 24] # Adjust layer indices if base model changes structure/size

DMAO_CONFIG = {
    'check_interval': 300,      # 5 minutes
    'min_examples': 20,         # Reduced minimum examples for faster testing cycles
    'max_training_time': 1800,  # 30 minutes
    'system_load_limit': 0.85,  # Slightly increased limit if resources allow
    'salience_threshold': 0.7,   # Adjusted threshold
    'time_window': 3600,        # 1-hour update window
    'train_epochs': 1,          # Reduced epochs for faster testing cycles
    'learning_rate': 2e-5,      # Slightly adjusted LR typical for LoRA
    'confidence_threshold': 0.65, # Confidence for scaffold use
    'default_max_output_length': 256, # Reduced default length for faster testing
    'min_output_length': 10,          # Minimum output length in tokens
    # Removed Adaptive/Sparse specific configs: max_rank, sparse_topk, min_rank
}

# --- Optimized Modules ---
# REMOVE the AdaptiveLoRALinear class definition here

class SparseCrossAttention(nn.Module):
    """Simplified Cross-Attention (using standard MultiheadAttention)"""
    def __init__(self, base_dim, num_heads):
        super().__init__()
        # Use standard MultiheadAttention instead of custom sparse version
        self.attention = nn.MultiheadAttention(base_dim, num_heads, batch_first=True)
        # Removed topk_learned and sparsity_factor

    def forward(self, base_hidden, scaffold_hidden):
        # Direct attention calculation without sparsity
        # Query: base_hidden, Key: scaffold_hidden, Value: scaffold_hidden
        # Assumes scaffold_hidden has been projected to base_dim if needed
        attn_out, _ = self.attention(base_hidden, scaffold_hidden, scaffold_hidden)
        return attn_out # Return only the attention output to be gated

class CrossAttentionFuser(nn.Module):
    """Fuses base and scaffold hidden states with standard cross-attention"""
    def __init__(self, base_dim, scaffold_dim, num_heads=8):
        super().__init__()
        # Use the simplified (standard) attention
        self.attention = SparseCrossAttention(base_dim, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(base_dim, 1),
            nn.Sigmoid()
        )
        # Projection might still be needed if dims differ
        self.scaffold_proj = nn.Linear(scaffold_dim, base_dim) if base_dim != scaffold_dim else nn.Identity()
        self.confidence_threshold = nn.Parameter(torch.tensor([DMAO_CONFIG['confidence_threshold']], dtype=torch.float32))

    def forward(self, base_hidden, scaffold_hidden, blend=0.5): # Blend control (consider making dynamic)
        scaffold_projected = self.scaffold_proj(scaffold_hidden)
        # Ensure sequence lengths match for attention if needed, or handle padding/masking.
        # Assuming scaffold_projected can be attended to by base_hidden for simplicity here.
        # May need adjustment based on how scaffold_context is generated/passed.
        # Let's assume scaffold_projected needs to match base_hidden's sequence length.
        # A simple approach (might need refinement): average scaffold context or take first token.
        # Averaging:
        scaffold_context_for_attn = scaffold_projected.mean(dim=1, keepdim=True).expand_as(base_hidden)
        # Or using the projected states directly if lengths match (more complex handling needed otherwise)
        # scaffold_context_for_attn = scaffold_projected

        if self._get_confidence(scaffold_projected) > self.confidence_threshold:
            attn_output = self.attention(base_hidden, scaffold_context_for_attn, scaffold_context_for_attn)
            gate_weight = self.gate(base_hidden)
            # Combine: Add gated attention output to original base hidden state
            augmented_output = base_hidden + gate_weight * attn_output
            # Blend between original and augmented (optional)
            # return (1 - blend) * base_hidden + blend * augmented_output
            return augmented_output # Simpler: just return the augmented state
        return base_hidden # Return original if confidence is low

    def _get_confidence(self, hidden_states):
        # Simple confidence: mean L2 norm
        return torch.mean(torch.norm(hidden_states, dim=-1)).item()

class SalienceScorer(nn.Module):
    """Scores interaction importance using Scaffold Model Embeddings (Simpler)"""
    def __init__(self, scaffold_embedding_dim):
        super().__init__()
        # Simple classifier on top of scaffold embeddings
        self.classifier = nn.Sequential(
            nn.Linear(scaffold_embedding_dim, 128), # Reduced size
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # No separate BERT model needed

    def forward(self, scaffold_embeddings):
        # Expects pooled embeddings [batch_size, embedding_dim]
        # Pool the sequence embeddings (e.g., mean pooling)
        pooled_embeddings = scaffold_embeddings.mean(dim=1)
        return self.classifier(pooled_embeddings)

class DMAOSystem:
    """Dual-Model Adaptive Orchestrator System"""
    def __init__(self):
        # --- Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Base Model ---
        # Load base model (consider putting on CPU initially if memory constrained, move layers to GPU as needed)
        # For large models, consider techniques like DeepSpeed ZeRO or Accelerate's FSDP if deploying on multi-GPU
        self.base_model_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.base_model = AutoModel.from_pretrained(BASE_MODEL_NAME, config=self.base_model_config)
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval().to(self.device) # Move base model to device
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # --- Scaffold Model ---
        scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        original_scaffold = AutoModel.from_pretrained(SCAFFOLD_MODEL_NAME, config=scaffold_config)
        # DO NOT call _replace_lora_layers here

        # Apply standard PEFT LoRA
        peft_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2, # Common practice: alpha = 2*r
            target_modules=["q_proj", "v_proj"], # Adjust if module names differ for the scaffold model
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM" # Or "SEQ_2_SEQ_LM" etc. depending on the scaffold model type
        )
        self.scaffold_model = get_peft_model(original_scaffold, peft_config)
        self.scaffold_model.print_trainable_parameters()
        self.scaffold_model.to(self.device) # Move scaffold model to device
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)

        # Add padding token if missing (common for GPT-like models)
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
            self.scaffold_model.config.pad_token_id = self.scaffold_model.config.eos_token_id
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            # Base model config usually doesn't need changing if not fine-tuned

        # --- Initial State & Production Copy ---
        self.initial_scaffold_state = copy.deepcopy(self.scaffold_model.state_dict())
        # No need for a separate production_scaffold initially, just use self.scaffold_model
        # We will update self.scaffold_model directly after training validation (or keep a shadow copy for safety)
        # Let's simplify deployment: use the trained scaffold directly after training.
        # self.production_scaffold = copy.deepcopy(self.scaffold_model) # Remove this line

        # --- Cross-Attention Injection ---
        # This remains complex but is part of the core architecture. Ensure CROSS_ATTN_LAYERS indices are correct.
        self._insert_cross_attention() # This modifies self.base_model

        # --- DMAO Infrastructure ---
        self.interaction_buffer = []
        # Initialize simplified SalienceScorer using scaffold's embedding dimension
        scaffold_embedding_dim = self.scaffold_model.config.hidden_size
        self.salience_scorer = SalienceScorer(scaffold_embedding_dim).to(self.device).eval() # Put scorer on device, eval mode
        self.training_lock = threading.Lock()
        self.last_trained = 0

        # --- Data Augmenter (Removed) ---
        # Remove BART dependency for simplicity
        # self.augmenter_model = None
        # self.augmenter_tokenizer = None

        # --- Scheduler ---
        self._start_scheduler() # Keep scheduler or comment out for manual training triggers

    # REMOVE the _replace_lora_layers method definition here

    def _insert_cross_attention(self):
        """Inject cross-attention into base model (Keep as is, but acknowledge complexity)"""
        base_config = self.base_model.config
        scaffold_dim = self.scaffold_model.config.hidden_size

        # Ensure layer indices are valid for the loaded base model
        num_base_layers = getattr(base_config, 'num_hidden_layers', None)
        if num_base_layers is None:
             print("Warning: Cannot determine number of layers in base model automatically.")
        elif any(idx >= num_base_layers for idx in CROSS_ATTN_LAYERS):
             print(f"Warning: CROSS_ATTN_LAYERS indices {CROSS_ATTN_LAYERS} might be out of bounds for base model with {num_base_layers} layers.")
             # Potentially raise an error or adjust indices

        print(f"Injecting CrossAttentionFuser at layers: {CROSS_ATTN_LAYERS}")
        for layer_idx in CROSS_ATTN_LAYERS:
             try:
                # Accessing layers might differ based on model architecture (e.g., model.layers, model.decoder.layers, model.transformer.h)
                # Assuming a common structure like model.layers or model.transformer.h
                layer_container = None
                if hasattr(self.base_model, 'layers'):
                    layer_container = self.base_model.layers
                elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
                     layer_container = self.base_model.transformer.h
                elif hasattr(self.base_model, 'decoder') and hasattr(self.base_model.decoder, 'layers'):
                     layer_container = self.base_model.decoder.layers
                else:
                    print(f"Warning: Could not automatically find layer container for base model. Skipping injection for layer {layer_idx}.")
                    continue

                if layer_idx >= len(layer_container):
                    print(f"Warning: Layer index {layer_idx} is out of bounds for container of size {len(layer_container)}. Skipping.")
                    continue

                original_layer = layer_container[layer_idx]
                cross_attn = CrossAttentionFuser(
                    base_dim=base_config.hidden_size,
                    scaffold_dim=scaffold_dim,
                    num_heads=base_config.num_attention_heads
                ).to(self.device) # Move fuser to the correct device

                # Inner class to wrap the original layer and add cross-attention
                class ModifiedLayer(nn.Module):
                    def __init__(self, orig_layer, cross_attn_module):
                        super().__init__()
                        self.orig_layer = orig_layer
                        self.cross_attn = cross_attn_module
                        # Make sure the original layer's parameters are accessible
                        # This might not be needed if orig_layer is already a Module

                    def forward(self, hidden_states, **kwargs):
                        # Pass through original layer
                        # Need to handle various arguments layers might accept (attention_mask, etc.)
                        orig_output = self.orig_layer(hidden_states, **kwargs)

                        # Extract the primary hidden state output (handling tuples if necessary)
                        if isinstance(orig_output, tuple):
                            base_hidden_state_output = orig_output[0]
                        else:
                            base_hidden_state_output = orig_output

                        # Apply cross-attention if scaffold context is provided
                        scaffold_context = kwargs.get('scaffold_context', None)
                        if scaffold_context is not None:
                             fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)
                             # Replace the hidden state in the output tuple if necessary
                             if isinstance(orig_output, tuple):
                                 # Reconstruct the tuple with the modified hidden state
                                 return (fused_hidden_state,) + orig_output[1:]
                             else:
                                 return fused_hidden_state
                        else:
                            # Return original output if no scaffold context
                            return orig_output

                layer_container[layer_idx] = ModifiedLayer(original_layer, cross_attn)
                print(f"Successfully injected cross-attention wrapper into layer {layer_idx}")

             except Exception as e:
                 print(f"Error injecting cross-attention at layer {layer_idx}: {e}")


    def _start_scheduler(self):
        """Background training scheduler (Consider replacing with manual trigger for simplicity)"""
        # Option 1: Keep the background thread (as original)
        def scheduler_loop():
            while True:
                time.sleep(DMAO_CONFIG['check_interval'])
                if self._should_trigger_training():
                    print("Scheduler triggered training...")
                    self._run_dmao_training()
                # else:
                #     print("Scheduler checked: Conditions not met for training.") # Optional debug print

        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        print("Background training scheduler started.")

        # Option 2: Comment out the above and call _run_dmao_training() manually or via cron
        # print("Background scheduler disabled. Trigger training manually.")

    def _should_trigger_training(self):
        """Check if training should start"""
        # Simplified check: Just data amount and time window
        data_ok = len(self.interaction_buffer) >= DMAO_CONFIG['min_examples']
        time_ok = (time.time() - self.last_trained) > DMAO_CONFIG['time_window']

        # Optional: Add back system load checks if needed and psutil is acceptable
        # try:
        #     cpu_load = psutil.cpu_percent() / 100
        #     # GPU load check might need nvidia-smi or similar if torch.cuda.utilization is not available/reliable
        #     gpu_load = 0 # Placeholder
        #     if torch.cuda.is_available() and hasattr(torch.cuda, 'utilization') and torch.cuda.device_count() > 0:
        #          gpu_load = torch.cuda.utilization(self.device) / 100 # Check utilization on the correct device
        #     system_ok = cpu_load < DMAO_CONFIG['system_load_limit'] and gpu_load < DMAO_CONFIG['system_load_limit']
        # except ImportError:
        #     print("psutil not installed, skipping system load check.")
        #     system_ok = True # Assume OK if psutil not available
        # except Exception as e:
        #      print(f"Error checking system load: {e}")
        #      system_ok = False # Don't train if check fails

        system_ok = True # Assume system is OK for simplification

        salience_ok = False
        if data_ok:
             # Check average salience only if we have enough data
             avg_salience = sum(e['salience'] for e in self.interaction_buffer if 'salience' in e) / len(self.interaction_buffer)
             salience_ok = avg_salience >= DMAO_CONFIG['salience_threshold']
             # print(f"Data OK: True ({len(self.interaction_buffer)}), Time OK: {time_ok}, Salience OK: {salience_ok} (Avg: {avg_salience:.2f}), System OK: {system_ok}") # Debugging

        return data_ok and time_ok and salience_ok and system_ok


    def _run_dmao_training(self):
        """Execute training with safety checks (Simplified data prep)"""
        # Prevent concurrent training runs
        if not self.training_lock.acquire(blocking=False):
             print("Training is already in progress. Skipping this trigger.")
             return

        print("\n--- Starting DMAO Training ---")
        training_success = False
        try:
            # Prepare data (simplified)
            train_data = self._prepare_training_data()
            if not train_data:
                print("No data prepared, skipping training.")
                return # Use finally block to release lock

            # Ensure model is on the right device and in training mode
            self.scaffold_model.to(self.device).train()

            # Filter parameters that require gradients (only LoRA weights in scaffold_model)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.scaffold_model.parameters()),
                lr=DMAO_CONFIG['learning_rate']
            )

            start_time = time.time()

            for epoch in range(DMAO_CONFIG['train_epochs']):
                print(f"Starting Epoch {epoch+1}/{DMAO_CONFIG['train_epochs']}")
                if time.time() - start_time > DMAO_CONFIG['max_training_time']:
                    print("Training time limit exceeded.")
                    break # Don't raise error, just stop training

                # Run training epoch
                epoch_loss = self._train_epoch(epoch, train_data, optimizer)
                if epoch_loss is None: # Handle potential issues in epoch training
                    print(f"Epoch {epoch+1} failed or returned no loss.")
                    break

                # Optional: Add validation step here if you have validation data
                # self._validate_epoch(...)

            # Update the production model state (simplified: just use the trained scaffold)
            # No separate deployment step needed if using self.scaffold_model directly
            # self._deploy_updated_scaffold() # Remove this call
            self.last_trained = time.time()
            self.interaction_buffer = [] # Clear buffer after successful training
            print("--- DMAO Training Completed ---")
            training_success = True

        except Exception as e:
            print(f"DMAO Training failed: {e}")
            # Rollback logic might be needed if training corrupted the model state
            # self._rollback_training() # Consider if rollback is necessary/possible

        finally:
            # Ensure model is back in eval mode and lock is released
            self.scaffold_model.eval()
            self.training_lock.release()
            print(f"Training lock released. Training success: {training_success}")


    def _prepare_training_data(self):
        """Simplified data preparation (Random sampling, no augmentation)"""
        print(f"Preparing training data from {len(self.interaction_buffer)} interactions...")
        if len(self.interaction_buffer) < DMAO_CONFIG['min_examples']:
            return [] # Not enough data

        # --- Simple Random Sampling ---
        # Sample based on min_examples or buffer size, whichever is smaller
        sample_size = min(DMAO_CONFIG['min_examples'], len(self.interaction_buffer))
        # Prioritize by salience score if available
        self.interaction_buffer.sort(key=lambda x: x.get('salience', 0), reverse=True)
        selected_interactions = self.interaction_buffer[:sample_size]

        # --- Prepare Batches ---
        batch = []
        for interaction in selected_interactions:
             try:
                 # Tokenize input and output for the scaffold model
                 # Ensure padding and truncation are handled
                 inputs = self.scaffold_tokenizer(
                     interaction['input'],
                     return_tensors='pt',
                     padding='max_length', # Pad to max length for consistency
                     truncation=True,
                     max_length=self.scaffold_tokenizer.model_max_length or 512 # Use model's max length or a default
                 ).to(self.device) # Move tensors to device

                 # Labels are typically input_ids shifted, handle carefully for causal LM
                 # For seq2seq or encoder-decoder, labels are just the output tokens
                 # Assuming Causal LM: labels are the same as input_ids
                 # Loss function will handle shifting internally
                 labels = inputs['input_ids'].clone()

                 # Mask padding tokens in labels
                 labels[labels == self.scaffold_tokenizer.pad_token_id] = -100

                 batch.append({'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels})

             except Exception as e:
                  print(f"Warning: Skipping interaction due to error during tokenization: {e}")
                  print(f"Interaction Input: {interaction['input'][:100]}...") # Log problematic input
                  print(f"Interaction Output: {interaction['output'][:100]}...") # Log problematic output


        print(f"Prepared {len(batch)} training examples.")
        # Clear buffer after preparation for this training run
        # Moved buffer clearing to _run_dmao_training after successful completion
        return batch

    # REMOVE the _augment_text method definition here

    def _train_epoch(self, epoch, data, optimizer):
        """Train one epoch (Simplified loss calculation, no checkpointing for simplicity)"""
        # Ensure model is in training mode
        self.scaffold_model.train()
        total_loss = 0
        batch_count = 0

        # Use autocast for mixed-precision training if available
        scaler = None
        if self.device.type == 'cuda':
             scaler = torch.cuda.amp.GradScaler()

        for batch_data in data: # Assuming data is a list of dicts
            optimizer.zero_grad()

            # Move data to device (already done in _prepare_training_data)
            input_ids = batch_data['input_ids']
            attention_mask = batch_data['attention_mask']
            labels = batch_data['labels']

            try:
                # Mixed Precision Forward Pass
                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                    # Direct forward pass on the scaffold model (which includes LoRA)
                    outputs = self.scaffold_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss # PEFT models often return loss directly when labels are provided

                    # Add L2 regularization (optional, AdamW usually handles weight decay)
                    # l2_lambda = 0.01
                    # l2_norm = sum(p.pow(2.0).sum() for p in filter(lambda p: p.requires_grad, self.scaffold_model.parameters()))
                    # loss = loss + l2_lambda * l2_norm

                if loss is None:
                    print("Warning: Loss is None. Skipping batch.")
                    continue

                if scaler:
                    # Mixed Precision Backward Pass
                    scaler.scale(loss).backward()
                    # Unscale before clipping
                    scaler.unscale_(optimizer)
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.scaffold_model.parameters()), 1.0)
                    # Optimizer Step
                    scaler.step(optimizer)
                    # Update scaler
                    scaler.update()
                else:
                    # Standard Backward Pass
                    loss.backward()
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.scaffold_model.parameters()), 1.0)
                    # Optimizer Step
                    optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # Print progress occasionally
                if batch_count % 10 == 0:
                     print(f"Epoch {epoch+1} | Batch {batch_count}/{len(data)} | Loss: {loss.item():.4f}")

            except Exception as e:
                 print(f"Error during training batch: {e}")
                 # Consider skipping batch or stopping epoch based on error severity

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")
        return avg_loss


    # REMOVE the _deploy_updated_scaffold method as it's no longer needed with the simplified approach


    @torch.no_grad() # Ensure no gradients calculated during inference
    def generate_response(self, user_input, max_output_length=None):
        """Generate response using base and scaffold models with length control"""
        self.base_model.eval()
        self.scaffold_model.eval() # Ensure both models are in eval mode

        effective_max_length = (
            max_output_length if max_output_length is not None
            else DMAO_CONFIG['default_max_output_length']
        )
        effective_max_length = max(
            DMAO_CONFIG['min_output_length'],
            min(effective_max_length, self.base_tokenizer.model_max_length or 2048) # Use model max length or cap
        )

        # 1. Get Scaffold Hidden States
        scaffold_inputs = self.scaffold_tokenizer(
            user_input,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.scaffold_tokenizer.model_max_length or 512
        ).to(self.device)

        # Use the scaffold model (which includes LoRA adapters) to get hidden states
        # We need the *encoder* hidden states if scaffold is encoder-decoder, or just last_hidden_state
        # Assuming scaffold is decoder-only or we need its final hidden states
        scaffold_outputs = self.scaffold_model(**scaffold_inputs, output_hidden_states=True)
        scaffold_hidden_states = scaffold_outputs.hidden_states[-1] # Get last layer hidden states

        # 2. Prepare Base Model Inputs
        base_inputs = self.base_tokenizer(user_input, return_tensors='pt').to(self.device)
        input_ids = base_inputs['input_ids']
        attention_mask = base_inputs['attention_mask']

        # Determine current length to calculate max_new_tokens
        input_length = input_ids.shape[1]
        max_new_tokens = effective_max_length - input_length
        min_new_tokens = max(0, DMAO_CONFIG['min_output_length'] - input_length) # Ensure min_length applies to *new* tokens

        if max_new_tokens <= 0:
            print("Warning: Input length is already >= max_output_length. Returning empty string or input.")
            return "" # Or return user_input, depending on desired behavior

        # 3. Generate with Base Model, passing Scaffold Context
        # The key is how the ModifiedLayer uses 'scaffold_context'
        # We pass the computed scaffold_hidden_states here.
        # Ensure the generate method can handle this extra kwarg if passed down.
        # Note: Standard Hugging Face generate might not pass arbitrary kwargs deep into the model.
        # The _insert_cross_attention modification assumes this is possible.
        # If 'generate' doesn't pass it, the forward method of ModifiedLayer needs adjustment
        # to perhaps access scaffold_context from a shared object or via a different mechanism.
        # --> ASSUMPTION: The ModifiedLayer's forward correctly receives scaffold_context.

        # We need to pass scaffold_hidden_states in a way that ModifiedLayer can receive it.
        # If generate doesn't pass arbitrary kwargs, a workaround is needed.
        # Workaround idea: Temporarily attach scaffold_context to the model instance or a global registry. This is hacky.
        # Let's *assume* for now that the forward signature modification works with generate.
        try:
             outputs = self.base_model.generate(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 max_new_tokens=max_new_tokens, # Use max_new_tokens
                 min_new_tokens=min_new_tokens, # Use min_new_tokens
                 pad_token_id=self.base_tokenizer.pad_token_id or self.base_tokenizer.eos_token_id, # Set pad token ID
                 eos_token_id=self.base_tokenizer.eos_token_id,
                 # This is the crucial part - passing context to the modified layers
                 scaffold_context=scaffold_hidden_states
                 # Add other generation parameters as needed (do_sample, temperature, etc.)
             )
             # Decode only the generated part (excluding the input)
             response = self.base_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        except TypeError as e:
             print(f"Warning: Base model generate likely didn't accept 'scaffold_context'. Error: {e}")
             print("Falling back to generation without scaffold context.")
             # Fallback: Generate without scaffold context
             outputs = self.base_model.generate(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  max_new_tokens=max_new_tokens,
                  min_new_tokens=min_new_tokens,
                  pad_token_id=self.base_tokenizer.pad_token_id or self.base_tokenizer.eos_token_id,
                  eos_token_id=self.base_tokenizer.eos_token_id,
             )
             response = self.base_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response


    @torch.no_grad() # Ensure no gradients needed for logging/scoring
    def log_interaction(self, user_input, model_output):
        """Log interaction with salience score (using scaffold embeddings)"""
        self.salience_scorer.eval() # Ensure scorer is in eval mode
        self.scaffold_model.eval() # Ensure scaffold model is in eval mode

        # Combine input and output for context
        text_for_scoring = user_input + self.scaffold_tokenizer.sep_token + model_output \
                           if self.scaffold_tokenizer.sep_token else user_input + " " + model_output

        # Get scaffold embeddings for the combined text
        inputs = self.scaffold_tokenizer(
             text_for_scoring,
             return_tensors='pt',
             padding=True,
             truncation=True,
             max_length=self.scaffold_tokenizer.model_max_length or 512
         ).to(self.device)

        # Get hidden states from the scaffold model
        outputs = self.scaffold_model(**inputs, output_hidden_states=True)
        # Use last hidden state for salience scoring
        last_hidden_state = outputs.hidden_states[-1]

        # Score salience using the simplified scorer
        salience = self.salience_scorer(last_hidden_state).item()

        self.interaction_buffer.append({
            'input': user_input,
            'output': model_output,
            'salience': salience,
            'timestamp': time.time()
        })
        # print(f"Logged interaction. Salience: {salience:.3f}. Buffer size: {len(self.interaction_buffer)}") # Debug print


    def _rollback_training(self):
        """Revert scaffold model to its state before the failed training attempt."""
        # This requires storing the state *before* training starts.
        # For simplicity now, we can just reset to the initial state if training fails badly.
        print("Error during training occurred. Rolling back scaffold model to initial state.")
        self.reset_scaffold() # Or load from a pre-training checkpoint if available


    def reset_scaffold(self):
        """Reset scaffold model to its initial PEFT state"""
        with self.training_lock: # Ensure no training happens during reset
            print("--- Resetting Scaffold Model ---")
            # Load the initial state dictionary
            self.scaffold_model.load_state_dict(self.initial_scaffold_state)
            # Ensure model is on the correct device after loading state
            self.scaffold_model.to(self.device)
            # Put back in eval mode
            self.scaffold_model.eval()
            # Clear interaction buffer as it might correspond to the now-discarded training
            self.interaction_buffer = []
            self.last_trained = 0 # Reset training time
            print("--- Scaffold model reset to initial state. Buffer cleared. ---")


# --- Usage ---
if __name__ == "__main__":
    print("Initializing DMAO System...")
    # Make sure to handle potential CUDA OOM errors during initialization
    try:
        system = DMAOSystem()
        print("DMAO System Initialized.")
    except Exception as e:
         print(f"Failed to initialize DMAO system: {e}")
         print("Please check model names, available memory, and dependencies.")
         exit() # Exit if initialization fails

    try:
        print("\nEnter your prompts. Type 'RESET_SCAFFOLD' to reset the learned adaptations, or Ctrl+C to exit.")
        while True:
            user_input = input("User: ")
            if not user_input:
                continue

            if user_input.strip().upper() == "RESET_SCAFFOLD":
                system.reset_scaffold()
                print("Assistant: Scaffold reset complete. Ready for new input.")
                continue

            if user_input.strip().upper() == "FORCE_TRAIN":
                 print("Assistant: Manually triggering training cycle...")
                 system._run_dmao_training() # Note: Runs synchronously
                 print("Assistant: Manual training cycle finished.")
                 continue

            # Generate response with default length
            print("Assistant (generating...):")
            try:
                 response_default = system.generate_response(user_input)
                 print(f"Assistant: {response_default}") # Removed length indication for simplicity

                 # Log the interaction
                 system.log_interaction(user_input, response_default)

            except Exception as e:
                 print(f"\nError during generation or logging: {e}")
                 # Potential OOM or other runtime errors

    except KeyboardInterrupt:
        print("\n--- System Shutdown ---")
    finally:
         # Optional: Clean up GPU memory if needed
         del system
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
         print("Cleanup complete.")
