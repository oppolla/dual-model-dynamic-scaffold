# Console Commands
## `quit` or `exit`
- What it does: Stops the script and exits the program cleanly.
- How it works: Breaks the input loop, triggers cleanup (freeing GPU memory), and shuts down.
- Use case: When you’re done testing or training and want to close everything.
- Example: Type quit → Output: Exiting. (plus cleanup messages).
## `train`
- What it does: Runs the training cycle on TRAIN_DATA and VALID_DATA using settings from config.json (e.g., epochs, batch size, sigmoid parameters).
- How it works: Calls run_training_cycle, applies the tunable sigmoid life curve (via get_life_curve_weight), trains the scaffold model with LoRA, and logs progress. If in dry-run mode, it stops after one epoch.
- Use case: Train the model with your sample_log.jsonl data to adjust the scaffold’s influence.
- Example: Type train → Output:
```
Data exposure: 5 samples | Scaffold influence weight: 0.881
--- Starting Training (3 epochs) ---
Epoch 1/3
Step 1/3 | Loss: 2.3456
```
##  `int8`
- What it does: Sets the quantization mode to INT8 (8-bit integer) and reinitializes the system.
- How it works: Updates quantization_mode to "int8", then restarts the BareBonesDMAO_Learn instance to apply the change (reloading models in INT8).
- Use case: Reduce memory usage on your RTX 3070 if you’re hitting GPU memory limits (e.g., with larger batches or models).
- Example: Type int8 → Output:
```
Quantization mode set to 'int8'. Restart system to apply quantization.
Re-initializing system with INT8 quantization...
```
## `int4`
- What it does: Sets the quantization mode to INT4 (4-bit integer) and reinitializes the system.
- How it works: Same as int8, but uses "int4" for even lower memory usage (via bitsandbytes).
- Use case: Further optimize memory if INT8 isn’t enough or for experimenting with efficiency.
- Example: Type int4 → Output:
```
Quantization mode set to 'int4'. Restart system to apply quantization.
Re-initializing system with INT4 quantization...
```
## `fp16`
- What it does: Sets the quantization mode to FP16 (16-bit floating point) and reinitializes the system.
- How it works: Switches to "fp16" (the default), then reloads the system with half-precision floats.
- Use case: Restore default precision if you’ve switched to INT8/INT4, balancing speed and accuracy.
- Example: Type fp16 → Output:
```
Quantization mode set to 'fp16'. Restart system to apply quantization.
Re-initializing system with FP16 quantization...
```
## `dynamic`
- What it does: Enables dynamic layer selection for cross-attention and reinitializes the system.
- How it works: Sets USE_DYNAMIC_LAYERS to True, uses LAYER_SELECTION_MODE (e.g., "balanced") to pick layers dynamically, then restarts the system.
- Use case: Experiment with adaptive layer injection instead of fixed CROSS_ATTN_LAYERS.
- Example: Type dynamic → Output:
```
Dynamic layer selection enabled. Restart system to apply.
Re-initializing system with dynamic layers...
```
## `fixed`
- What it does: Disables dynamic layer selection (uses fixed layers from config.json) and reinitializes the system.
- How it works: Sets USE_DYNAMIC_LAYERS to False, sticks to CROSS_ATTN_LAYERS, then reloads the system.
- Use case: Revert to your predefined layers (e.g., [5, 7]) for consistency.
- Example: Type fixed → Output:
```
Dynamic layer selection disabled. Restart system to apply.
Re-initializing system with fixed layers...
```
## `new`
- What it does: Starts a new conversation, resetting the conversation history.
- How it works: Calls new_conversation, clears the ConversationHistory (last 10 messages), assigns a new conversation_id, and clears the scaffold cache.
- Use case: Begin a fresh interaction without past prompts influencing the logs.
- Example: Type new → Output:
```
New conversation started with ID: [new_uuid] (Previous ID: [old_uuid])
```
## Any Other Input (Prompt)
What it does: Treats the input as a prompt and generates a response using the current model state.
How it works: Calls generate with the input, applies the scaffold’s influence (tuned by the sigmoid), logs the response in log.jsonl, and prints it. Uses defaults like max_new_tokens=60, temperature=0.7.
Use case: Test the model’s output after training or tweak its behavior live.
Example: Type Hello! → Output:
```
--- Generating Response ---
Using quantization mode: fp16
Generation took 0.45 seconds.
Response:
Hi there! How can I assist you today?
```
## [Empty Input] (just pressing Enter)
What it does: Does nothing and waits for the next input.
How it works: The loop skips empty strings with if not user_cmd: continue.
Use case: Avoids errors if you accidentally hit Enter without typing.
```Example: Press Enter → Output: (just a new prompt line \nEnter command or prompt: )```
## Additional Notes
Startup Flag: If you run the script with --dry-run (e.g., python main.py --dry-run), it enables dry-run mode automatically. This limits training to one epoch, caps samples at 2, and validates the architecture without full training. You’ll see:
Dry run mode activated (max_samples=2, max_length=64)
Where Defined: These commands are in the if-elif-else block starting at line 1268 in the script:

python
if cmd in ['quit', 'exit']:
    break
elif cmd == 'train':
    dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
# ... and so on
How to Use Them
Start the Script:
cd C:\Users\YourName\AIProject
venv\Scripts\activate
python main.py

See the Prompt: After initialization, you’ll get:

System Ready.
Commands: 'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', 'new', or enter a prompt.
Enter command or prompt:

## Try Commands:
train to start training with your sample_log.jsonl.
int8 to switch to lower memory usage, then train again.
Hello! to generate a response.
quit to exit.

## Config Guide:

### core_config
- `"base_model_name": "gpt2",  // Base model for generation (string, e.g., "gpt2", "gpt2-medium"). Defines the core language model; larger models increase capability but demand more resources.`
- `"scaffold_model_name": "gpt2",  // Scaffold model for cross-attention and adaptation (string, e.g., "gpt2"). Guides the base model; matching base model simplifies token mapping.`
- `"cross_attn_layers": [5, 7],  // Layers in base model where cross-attention is injected (list of integers). Targets specific layers for scaffold influence; must be valid for base model’s architecture.`
- `"use_dynamic_layers": false,  // Enables dynamic layer selection based on mode (true/false). If true, overrides cross_attn_layers with dynamic rules; false uses fixed list.`
- `"layer_selection_mode": "balanced",  // Method for dynamic layer selection ("early", "late", "balanced", "custom"). Early: first third, late: last third, balanced: middle third, custom: uses custom_layers.`
- `"custom_layers": null,  // Custom layer indices for "custom" mode (list of integers or null). Overrides other modes if set; must match base model layer count.`
- `"valid_split_ratio": 0.2,  // Fraction of data for validation (0.0-1.0). Higher values reduce training data but improve validation; 0.2 balances the split.`
- `"random_seed": 42,  // Seed for random operations (integer). Ensures reproducibility; change for different shuffles or initializations.`
- `"quantization": "fp16"  // Precision for model loading ("fp16", "int8", "int4"). FP16 is fastest with full precision, INT8/INT4 reduce memory but may lose accuracy.`

## lora_config
- `"lora_rank": 8,  // Rank of LoRA adaptation matrices (integer, e.g., 4-16). Higher ranks increase adaptability but add parameters; 8 is a balanced default.`
- `"lora_alpha": 16,  // Scaling factor for LoRA updates (integer, e.g., 8-32). Amplifies LoRA effects; higher values enhance adaptation at risk of instability.`
- `"lora_dropout": 0.1,  // Dropout rate in LoRA layers (0.0-0.5). Prevents overfitting; higher values regularize more but may slow learning.`
- `"lora_target_modules": ["c_attn", "c_proj", "c_fc"]  // Transformer modules to apply LoRA (list of strings). Targets attention and feed-forward layers; adjust for model-specific names (e.g., "q_proj", "v_proj" for some models).`

## training_config
- `"learning_rate": 0.0003,  // Initial learning rate for training (float, e.g., 1e-5 to 5e-4). Higher values speed learning but risk overshooting; 0.0003 suits small models.`
- `"train_epochs": 3,  // Number of training epochs (integer, e.g., 1-10). More epochs refine the model but increase time; 3 balances learning and speed.`
- `"batch_size": 1,  // Number of samples per training batch (integer, e.g., 1-8). Larger batches improve stability but demand more memory; 1 fits small setups.`
- `"max_seq_length": 128,  // Maximum sequence length for inputs (integer, e.g., 64-1024). Longer sequences capture more context but strain memory; 128 is compact.`
- `"sigmoid_scale": 0.5,  // Steepness of lifecycle sigmoid curve (float, e.g., 0.1-1.0). Higher values sharpen growth transitions; 0.5 keeps it smooth.`
- `"sigmoid_shift": 5.0,  // Midpoint shift of lifecycle sigmoid (float, e.g., 1.0-10.0). Delays peak influence; 5.0 centers it moderately in the lifecycle.`
- `"lifecycle_capacity_factor": 0.01,  // Scales LoRA parameter capacity to define lifecycle length (0.001-0.1). Higher values shorten the lifecycle, accelerating transitions from young (curious) to old (melancholic).`
- `"lifecycle_curve": "sigmoid_linear"  // Defines lifecycle weight curve: "sigmoid_linear" (smooth growth, linear decay) or "exponential" (fast growth, sharp decay). Affects how scaffold influence evolves over data exposure.`

## controls_config
- `"sleep_conf_threshold": 0.7,  // Minimum avg confidence to trigger sleep training (0.5-0.9). Lowered by restlessness; higher values delay sleep, letting confidence build.`
- `"sleep_time_factor": 1.0,  // Scales sleep delay based on data exposure (0.5-5.0). Higher values increase time between sleep cycles, slowing adaptation.`
- `"sleep_log_min": 10,  // Minimum log entries required for sleep training (5-20). Higher values ensure more data before sleep, stabilizing training.`
- `"dream_swing_var": 0.1,  // Confidence variance threshold to trigger dreaming (0.05-0.2). Higher values need bigger confidence swings, making dreams rarer but more chaotic.`
- `"dream_lifecycle_delta": 0.1,  // Temperament score change threshold for dreaming (0.05-0.2). Higher values require larger mood shifts, tying dreams to lifecycle peaks.`
- `"dream_temperament_on": true,  // Enables dreaming based on temperament history trends (true/false). When true, significant mood trends trigger dreams, adding emotional depth.`
- `"dream_noise_scale": 0.05,  // Base noise scale for dream perturbations (0.01-0.1). Higher values increase chaos in dreams, subtly altering scaffold behavior.`
- `"temp_eager_threshold": 0.8,  // Confidence threshold for eager temperament (0.7-0.9). Obsolete but kept for compatibility; temperament_score > 0.5 now defines curious.`
- `"temp_sluggish_threshold": 0.6,  // Confidence threshold for sluggish temperament (0.4-0.6). Obsolete but kept; temperament_score <= -0.5 now defines melancholic.`
- `"temp_mood_influence": 0.0,  // Strength of temperament’s effect on generation temperature (0-1). Higher values let mood (via temperament_score) swing temperature ±0.3, adding emotional flavor.`
- `"scaffold_weight_cap": 1.0,  // Maximum scaffold influence weight (0.5-1.0). Limits how much scaffold (and dreams) affect base model outputs, capping chaos.`
- `"base_temperature": 0.7,  // Base generation temperature (0.5-1.5). Sets default randomness, modified by temp_mood_influence and temperament_score.`
- `"save_path_prefix": "state",  // File prefix for saving/loading state (string). Defines where model state, dream memory, and temperament are stored.`
- `"dream_memory_weight": 0.1,  // Influence of dream memory on generation (0-0.5). Higher values blend past dreams into outputs, creating a subconscious echo.`
- `"dream_memory_maxlen": 10,  // Maximum number of dream layers stored (5-20). Larger values retain more dream history, deepening the subconscious effect.`
- `"dream_prompt_weight": 0.5,  // Balance of prompt similarity vs. recency in dream selection (0-1). Higher values favor prompts like the last input, grounding dreams in user interaction.`
- `"dream_novelty_boost": 0.03,  // Extra noise for novel prompts in dreams (0-0.05). Increases chaos for unseen inputs, encouraging exploration.`
- `"temp_curiosity_boost": 0.5,  // Boost to temperament_score in young lifecycle (< 25%) (0-0.5). Higher values push early behavior toward curious (+1.0), speeding exploration.`
- `"temp_restless_drop": 0.1,  // Drop in sleep_conf_threshold during restlessness (-0.5 to 0.0) (0-0.5). Higher values make restless moods sleep more, stabilizing mood swings.`
- `"temp_melancholy_noise": 0.02,  // Extra dream noise in melancholic state (≤ -0.5) (0-0.05). Higher values add chaos to dreams during low moods, reflecting introspection.`
- `"conf_feedback_strength": 0.5  // Strength of confidence feedback on temperament_score (0-1). Higher values let confidence swing mood more, tying performance to emotion.`


