# SOVLSystem User Manual

This manual provides a comprehensive guide to using the `SOVLSystem`, a custom AI framework built with PyTorch and Transformers. It integrates a base model (e.g., GPT-2) with a scaffold model, enhanced by cross-attention, LoRA adapters, and advanced features like dreaming, gestation, and temperament tracking. The system is designed for training, generation, and experimentation with configurable controls.

**Current Date:** April 09, 2025 (as per system context)  
**Codebase Reference:** Provided Python script with continuous knowledge updates.

---

## Getting Started

### Prerequisites
- **Python**: 3.8+
- **Dependencies**: `torch`, `transformers`, `peft`, `bitsandbytes`, `json`, `sys`, `os`, `uuid`, `threading`
- **Hardware**: GPU recommended (e.g., RTX 3070); CPU fallback available
- **Files**:
  - `config.json`: Configuration file (see below)
  - `sample_log.jsonl`: Training data in JSONL format (prompt-response pairs)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd SOVLSystem

Create Virtual Environment:
bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Install Dependencies:
bash

pip install torch transformers peft bitsandbytes

Run the Script:
bash

python main.py

Add --dry-run for a lightweight test:
bash

python main.py --dry-run

Initial Output
Upon startup, the system initializes and displays:

Initializing SOVL System...
Using device: cuda (or cpu)
Loading base model: gpt2
Loading scaffold model: gpt2
System Ready.
Commands: 'quit', 'exit', 'train', 'int8', 'int4', 'fp16', 'dynamic', 'fixed', 'new', 'save', 'load', 'sleep', 'dream', 'temp', 'blend', 'lifecycle', 'cross', 'scaffold_mem', 'token_mem', 'both_mem', 'no_mem', or a prompt.
Enter command or prompt:

Console Commands
quit or exit
What it does: Stops the script and exits cleanly.

How it works: Breaks the input loop, calls cleanup() to save state and free GPU memory, then shuts down.

Use case: Exit after testing or training.

Example:

quit
Output: Cleanup completed.
        Exiting.

train
What it does: Runs a training cycle on TRAIN_DATA and VALID_DATA.

How it works: Executes run_training_cycle() with settings from config.json (e.g., train_epochs, batch_size). Trains the scaffold model with LoRA, applies lifecycle weighting, and logs progress. In dry-run mode, it limits to one epoch.

Use case: Train the model with sample_log.jsonl to refine scaffold influence.

Example:

train
Output: Data exposure: 10 | Scaffold influence: 0.632
        --- Training (3 epochs) ---
        Epoch 1/3
        Step 1/9 | Loss: 2.3456

int8
What it does: Sets quantization to INT8 (8-bit integer) and reinitializes the system.

How it works: Updates quantization_mode to "int8", reloads models using bitsandbytes for lower memory usage.

Use case: Reduce GPU memory footprint for larger batches or models.

Example:

int8
Output: Quantization mode set to 'int8'. Restart system to apply.
        Re-initializing with INT8 quantization...

int4
What it does: Sets quantization to INT4 (4-bit integer) and reinitializes the system.

How it works: Updates quantization_mode to "int4", reloads models with extreme memory optimization.

Use case: Maximize memory efficiency for resource-constrained environments.

Example:

int4
Output: Quantization mode set to 'int4'. Restart system to apply.
        Re-initializing with INT4 quantization...

fp16
What it does: Sets quantization to FP16 (16-bit floating point) and reinitializes the system.

How it works: Updates quantization_mode to "fp16" (default), reloads models with half-precision floats.

Use case: Balance speed and accuracy (default mode).

Example:

fp16
Output: Quantization mode set to 'fp16'. Restart system to apply.
        Re-initializing with FP16 quantization...

dynamic
What it does: Enables dynamic layer selection for cross-attention and reinitializes the system.

How it works: Sets USE_DYNAMIC_LAYERS to True, uses LAYER_SELECTION_MODE (e.g., "balanced") to pick layers dynamically, then restarts.

Use case: Adapt cross-attention layers based on model size or task.

Example:

dynamic
Output: Dynamic layers enabled. Restart to apply.
        Re-initializing with dynamic layers...

fixed
What it does: Disables dynamic layer selection and reinitializes the system.

How it works: Sets USE_DYNAMIC_LAYERS to False, uses fixed CROSS_ATTN_LAYERS from config.json, then restarts.

Use case: Use predefined layers (e.g., [5, 7]) for consistency.

Example:

fixed
Output: Dynamic layers disabled. Restart to apply.
        Re-initializing with fixed layers...

new
What it does: Starts a new conversation, resetting history.

How it works: Calls new_conversation(), clears ConversationHistory (max 10 messages), assigns a new conversation_id, and clears scaffold cache.

Use case: Reset context for a fresh interaction.

Example:

new
Output: New conversation: [new_uuid] (Previous: [old_uuid])

save
What it does: Saves the current system state to files.

How it works: Calls save_state(), saves scaffold weights, cross-attention weights, token map, and metadata to files with a prefix (default: "state" or custom via argument).

Use case: Preserve trained model state for later use.

Example:

save my_model
Output: State saved to my_model_*.pth/json

load
What it does: Loads a saved system state from files.

How it works: Calls load_state(), restores scaffold weights, cross-attention weights, token map, and metadata from files with a prefix (default: "state" or custom via argument).

Use case: Resume a previous session or experiment.

Example:

load my_model
Output: Scaffold state loaded.
        Cross-attention state loaded.
        Token map loaded.
        Metadata loaded.

sleep <conf> <time> <log>
What it does: Adjusts sleep/gestation parameters.

How it works: Updates sleep_conf_threshold (0.5–0.9), sleep_time_factor (0.5–5.0), and sleep_log_min (5–20) for triggering gestation.

Use case: Tune when the system enters gestation mode based on confidence and log size.

Example:

sleep 0.75 2.0 15
Output: Sleep params: conf=0.75, time_factor=2.0, log_min=15

dream <swing> <delta> <temp_on> <noise> <mem_weight> <mem_maxlen> <prompt_weight> <novelty_boost> <memory_decay> <prune_threshold>
What it does: Tunes dreaming behavior parameters.

How it works: Adjusts 10 parameters (see controls_config below for details) to control dream generation and memory.

Use case: Customize how the system "dreams" to adapt its scaffold model.

Example:

dream 0.15 0.12 true 0.07 0.2 12 0.6 0.04 0.9 0.15
Output: Dream params: swing_var=0.15, lifecycle_delta=0.12, temperament_on=True, noise_scale=0.07, memory_weight=0.2, memory_maxlen=12, prompt_weight=0.6, novelty_boost=0.04, memory_decay=0.9, prune_threshold=0.15

temp <eager> <sluggish> <influence> <curiosity> <restless> <melancholy> <conf_strength> <smoothing_factor>
What it does: Adjusts temperament parameters.

How it works: Updates 8 parameters (see controls_config below) to influence generation temperature and behavior.

Use case: Fine-tune the system’s "mood" and responsiveness.

Example:

temp 0.85 0.55 0.2 0.4 0.15 0.03 0.6 0.1
Output: Temperament params: eager=0.85, sluggish=0.55, mood_influence=0.2, curiosity_boost=0.4, restless_drop=0.15, melancholy_noise=0.03, conf_feedback_strength=0.6, smoothing_factor=0.1

blend <weight> <temp>
What it does: Sets global scaffold weight cap and base temperature.

How it works: Updates scaffold_weight_cap (0.5–1.0) and base_temperature (0.5–1.5).

Use case: Control overall scaffold influence and generation randomness.

Example:

blend 0.95 0.8
Output: Global blend: weight_cap=0.95, base_temp=0.8

lifecycle <capacity> <curve>
What it does: Tunes lifecycle parameters.

How it works: Updates lifecycle_capacity_factor (0.001–0.1) and lifecycle_curve ("sigmoid_linear" or "exponential").

Use case: Adjust how scaffold influence grows with data exposure.

Example:

lifecycle 0.02 exponential
Output: Lifecycle params: capacity_factor=0.02, curve=exponential

cross weight <float> | blend <float> | layers <float...> | confidence | temperament | off
What it does: Tunes cross-attention settings.

How it works:
weight <float>: Sets uniform influence weight (e.g., 0.8).

blend <float>: Sets blend strength (0.0–1.0).

layers <float...>: Sets per-layer weights (must match layer count).

confidence: Enables dynamic weighting based on confidence.

temperament: Enables dynamic weighting based on temperament.

off: Disables dynamic weighting.

Use case: Customize how scaffold affects base model.

Example:

cross weight 0.7
Output: Scaffold influence: weight=0.70, blend_strength=unchanged

cross layers 0.5 0.6 0.7
Output: Scaffold influence: weight=per-layer, blend_strength=unchanged

scaffold_mem, token_mem, both_mem, no_mem
What it does: Toggles memory usage modes.

How it works:
scaffold_mem: Enables scaffold memory only.

token_mem: Enables token map memory only.

both_mem: Enables both.

no_mem: Disables both.

Use case: Control memory-driven adaptation.

Example:

both_mem
Output: Memory toggled: Scaffold=True, Token Map=True

Any Other Input (Prompt)
What it does: Generates a response using the current model state.

How it works: Calls generate() with defaults (max_new_tokens=60, temperature=0.7, top_k=50), logs to log.jsonl, and applies scaffold influence, temperament, and dreaming if enabled.

Use case: Test model output interactively.

Example:

Hello!
Output: --- Generating Response ---
        Response: Hi there! How can I assist you today?
        --------------------

[Empty Input] (Pressing Enter)
What it does: Skips and waits for next input.

How it works: Loop ignores empty strings with if not user_cmd: continue.

Example:

[Enter]
Output: (new prompt line)

Configuration (config.json)
core_config
base_model_name: Base model (e.g., "gpt2"). Defines the primary language model.

scaffold_model_name: Scaffold model (e.g., "gpt2"). Guides the base model via cross-attention.

cross_attn_layers: Fixed layers for cross-attention (e.g., [5, 7]). Ignored if use_dynamic_layers is true.

use_dynamic_layers: If true, selects layers dynamically based on layer_selection_mode.

layer_selection_mode: Dynamic layer strategy ("early", "late", "balanced", "custom").

custom_layers: Custom layer list for "custom" mode (e.g., [0, 2, 4]).

valid_split_ratio: Validation data split (e.g., 0.2 = 20%).

random_seed: Seed for reproducibility (e.g., 42).

quantization: Precision mode ("fp16", "int8", "int4").

lora_config
lora_rank: LoRA rank (e.g., 8). Controls adaptation capacity.

lora_alpha: LoRA scaling factor (e.g., 16).

lora_dropout: Dropout rate for LoRA (e.g., 0.1).

lora_target_modules: Modules to adapt (e.g., ["q_proj", "v_proj"]).

training_config
learning_rate: Learning rate (e.g., 2e-5).

train_epochs: Number of epochs (e.g., 3).

batch_size: Batch size (e.g., 2).

max_seq_length: Max token length (e.g., 512).

sigmoid_scale: Sigmoid scaling for lifecycle (e.g., 0.5).

sigmoid_shift: Sigmoid shift for lifecycle (e.g., 5.0).

lifecycle_capacity_factor: Capacity factor for lifecycle weighting (e.g., 0.01).

lifecycle_curve: Lifecycle curve type ("sigmoid_linear", "exponential").

accumulation_steps: Gradient accumulation steps (e.g., 4).

exposure_gain_eager: Exposure gain when eager (e.g., 3).

exposure_gain_default: Default exposure gain (e.g., 2).

max_patience: Early stopping patience (e.g., 2).

dry_run: If true, enables dry-run mode.

dry_run_params:
max_samples: Max samples in dry run (e.g., 2).

max_length: Max length in dry run (e.g., 128).

validate_architecture: If true, validates architecture.

skip_training: If true, skips full training.

controls_config
sleep_conf_threshold: Confidence threshold for gestation (0.5–0.9, e.g., 0.7). Triggers gestation if average confidence exceeds this.

sleep_time_factor: Time factor for gestation (0.5–5.0, e.g., 1.0). Scales sleep duration.

sleep_log_min: Minimum log entries for gestation (5–20, e.g., 10).

dream_swing_var: Variance threshold for dreaming (0.05–0.2, e.g., 0.1). Triggers dreaming if confidence varies widely.

dream_lifecycle_delta: Lifecycle change for dreaming (0.05–0.2, e.g., 0.1). Triggers if temperament shifts significantly.

dream_temperament_on: If true, temperament affects dreaming (e.g., true).

dream_noise_scale: Noise scale for dreaming (0.01–0.1, e.g., 0.05). Adds randomness to dream states.

temp_eager_threshold: Eager temperament threshold (0.7–0.9, e.g., 0.8). Above this, system is "curious."

temp_sluggish_threshold: Sluggish threshold (0.4–0.6, e.g., 0.6). Below this, system is "restless."

temp_mood_influence: Mood impact on temperature (0–1, e.g., 0.0). Adjusts generation randomness.

scaffold_weight_cap: Max scaffold influence (0.5–1.0, e.g., 0.9).

base_temperature: Default generation temperature (0.5–1.5, e.g., 0.7).

save_path_prefix: File prefix for saving state (e.g., "state").

dream_memory_weight: Dream memory influence (0–0.5, e.g., 0.1). Blends past dreams into scaffold context.

dream_memory_maxlen: Max dream memory size (5–20, e.g., 10).

dream_prompt_weight: Prompt similarity weight in dreams (0–1, e.g., 0.5).

dream_novelty_boost: Novelty boost for new prompts (0–0.05, e.g., 0.03).

temp_curiosity_boost: Curiosity boost for temperament (0–0.5, e.g., 0.5).

temp_restless_drop: Restless drop for temperament (0–0.5, e.g., 0.1).

temp_melancholy_noise: Noise when melancholic (0–0.05, e.g., 0.02).

conf_feedback_strength: Confidence feedback strength (0–1, e.g., 0.5). Affects temperament updates.

temp_smoothing_factor: Temperament smoothing (0–1, e.g., 0.0). Reduces abrupt changes.

dream_memory_decay: Dream memory decay rate (0–1, e.g., 0.95). Reduces old dream weights.

dream_prune_threshold: Threshold to prune dreams (0–1, e.g., 0.1).

use_scaffold_memory: If true, uses scaffold memory for adaptation (e.g., true).

use_token_map_memory: If true, uses token map memory (e.g., true).

memory_decay_rate: Memory decay rate (0–1, e.g., 0.95).

dynamic_cross_attn_mode: Dynamic cross-attention mode (null, "confidence", "temperament").

has_woken: If true, system has woken up (e.g., false).

is_sleeping: If true, system is in gestation (e.g., false).

confidence_history_maxlen: Max confidence history size (e.g., 5).

temperament_history_maxlen: Max temperament history size (e.g., 5).

enable_dreaming: If true, enables dreaming (e.g., true).

enable_temperament: If true, enables temperament (e.g., true).

enable_confidence_tracking: If true, tracks confidence (e.g., true).

enable_gestation: If true, enables gestation (e.g., true).

enable_sleep_training: If true, enables sleep training (e.g., true).

enable_cross_attention: If true, enables cross-attention (e.g., true).

enable_dynamic_cross_attention: If true, enables dynamic cross-attention (e.g., true).

enable_lora_adapters: If true, uses LoRA (e.g., true).

enable_repetition_check: If true, checks for repetition (e.g., true).

enable_prompt_driven_dreams: If true, dreams are prompt-driven (e.g., true).

enable_lifecycle_weighting: If true, uses lifecycle weighting (e.g., true).

