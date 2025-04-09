```markdown
# SOVLSystem User Manual

## Console Commands

### `quit` or `exit`

**What it does:**  
Stops the script and exits cleanly.

**How it works:**  
Breaks the input loop, calls `cleanup()` to save state and free GPU memory, then shuts down.

**Use case:**  
Exit after testing or training.

**Example:**
```
quit
Output: Cleanup completed.
        Exiting.
```

---

### `train`

**What it does:**  
Runs a training cycle on `TRAIN_DATA` and `VALID_DATA`.

**How it works:**  
Executes `run_training_cycle()` with settings from `config.json` (e.g., `train_epochs`, `batch_size`). Trains the scaffold model with LoRA, applies lifecycle weighting, and logs progress. In dry-run mode, it limits to one epoch.

**Use case:**  
Train the model with `sample_log.jsonl` to refine scaffold influence.

**Example:**
```
train
Output: Data exposure: 10 | Scaffold influence: 0.632
        --- Training (3 epochs) ---
        Epoch 1/3
        Step 1/9 | Loss: 2.3456
```

---

### `int8`

**What it does:**  
Sets quantization to INT8 (8-bit integer) and reinitializes the system.

**How it works:**  
Updates `quantization_mode` to `"int8"`, reloads models using bitsandbytes for lower memory usage.

**Use case:**  
Reduce GPU memory footprint for larger batches or models.

**Example:**
```
int8
Output: Quantization mode set to 'int8'. Restart system to apply.
        Re-initializing with INT8 quantization...
```

---

### `int4`

**What it does:**  
Sets quantization to INT4 (4-bit integer) and reinitializes the system.

**How it works:**  
Updates `quantization_mode` to `"int4"`, reloads models with extreme memory optimization.

**Use case:**  
Maximize memory efficiency for resource-constrained environments.

**Example:**
```
int4
Output: Quantization mode set to 'int4'. Restart system to apply.
        Re-initializing with INT4 quantization...
```

---

### `fp16`

**What it does:**  
Sets quantization to FP16 (16-bit floating point) and reinitializes the system.

**How it works:**  
Updates `quantization_mode` to `"fp16"` (default), reloads models with half-precision floats.

**Use case:**  
Balance speed and accuracy (default mode).

**Example:**
```
fp16
Output: Quantization mode set to 'fp16'. Restart system to apply.
        Re-initializing with FP16 quantization...
```

---

### `dynamic`

**What it does:**  
Enables dynamic layer selection for cross-attention and reinitializes the system.

**How it works:**  
Sets `USE_DYNAMIC_LAYERS` to `True`, uses `LAYER_SELECTION_MODE` (e.g., `"balanced"`) to pick layers dynamically, then restarts.

**Use case:**  
Adapt cross-attention layers based on model size or task.

**Example:**
```
dynamic
Output: Dynamic layers enabled. Restart to apply.
        Re-initializing with dynamic layers...
```

---

### `fixed`

**What it does:**  
Disables dynamic layer selection and reinitializes the system.

**How it works:**  
Sets `USE_DYNAMIC_LAYERS` to `False`, uses fixed `CROSS_ATTN_LAYERS` from `config.json`, then restarts.

**Use case:**  
Use predefined layers (e.g., `[5, 7]`) for consistency.

**Example:**
```
fixed
Output: Dynamic layers disabled. Restart to apply.
        Re-initializing with fixed layers...
```

---

### `new`

**What it does:**  
Starts a new conversation, resetting history.

**How it works:**  
Calls `new_conversation()`, clears `ConversationHistory` (max 10 messages), assigns a new `conversation_id`, and clears scaffold cache.

**Use case:**  
Reset context for a fresh interaction.

**Example:**
```
new
Output: New conversation: [new_uuid] (Previous: [old_uuid])
```

---

### `save`

**What it does:**  
Saves the current system state to files.

**How it works:**  
Calls `save_state()`, saves scaffold weights, cross-attention weights, token map, and metadata to files with a prefix (default: `"state"` or custom via argument).

**Use case:**  
Preserve trained model state for later use.

**Example:**
```
save my_model
Output: State saved to my_model_*.pth/json
```

---

### `load`

**What it does:**  
Loads a saved system state from files.

**How it works:**  
Calls `load_state()`, restores scaffold weights, cross-attention weights, token map, and metadata from files with a prefix.

**Use case:**  
Resume a previous session or experiment.

**Example:**
```
load my_model
Output: Scaffold state loaded.
        Cross-attention state loaded.
        Token map loaded.
        Metadata loaded.
```

---

### `sleep <conf> <time> <log>`

**What it does:**  
Adjusts sleep/gestation parameters.

**How it works:**  
Updates `sleep_conf_threshold` (0.5–0.9), `sleep_time_factor` (0.5–5.0), and `sleep_log_min` (5–20) for triggering gestation.

**Use case:**  
Tune when the system enters gestation mode based on confidence and log size.

**Example:**
```
sleep 0.75 2.0 15
Output: Sleep params: conf=0.75, time_factor=2.0, log_min=15
```

---

### `dream <swing> <delta> <temp_on> <noise> <mem_weight> <mem_maxlen> <prompt_weight> <novelty_boost> <memory_decay> <prune_threshold>`

**What it does:**  
Tunes dreaming behavior parameters.

**How it works:**  
Adjusts 10 parameters to control dream generation and memory.

**Use case:**  
Customize how the system "dreams" to adapt its scaffold model.

**Example:**
```
dream 0.15 0.12 true 0.07 0.2 12 0.6 0.04 0.9 0.15
Output: Dream params: swing_var=0.15, lifecycle_delta=0.12, temperament_on=True, noise_scale=0.07, memory_weight=0.2, memory_maxlen=12, prompt_weight=0.6, novelty_boost=0.04, memory_decay=0.9, prune_threshold=0.15
```

---

### `temp <eager> <sluggish> <influence> <curiosity> <restless> <melancholy> <conf_strength> <smoothing_factor>`

**What it does:**  
Adjusts temperament parameters.

**How it works:**  
Updates 8 parameters to influence generation temperature and behavior.

**Use case:**  
Fine-tune the system’s "mood" and responsiveness.

**Example:**
```
temp 0.85 0.55 0.2 0.4 0.15 0.03 0.6 0.1
Output: Temperament params: eager=0.85, sluggish=0.55, mood_influence=0.2, curiosity_boost=0.4, restless_drop=0.15, melancholy_noise=0.03, conf_feedback_strength=0.6, smoothing_factor=0.1
```

---

### `blend <weight> <temp>`

**What it does:**  
Sets global scaffold weight cap and base temperature.

**How it works:**  
Updates `scaffold_weight_cap` (0.5–1.0) and `base_temperature` (0.5–1.5).

**Use case:**  
Control overall scaffold influence and generation randomness.

**Example:**
```
blend 0.95 0.8
Output: Global blend: weight_cap=0.95, base_temp=0.8
```

---

### `lifecycle <capacity> <curve>`

**What it does:**  
Tunes lifecycle parameters.

**How it works:**  
Updates `lifecycle_capacity_factor` (0.001–0.1) and `lifecycle_curve` ("sigmoid_linear" or "exponential").

**Use case:**  
Adjust how scaffold influence grows with data exposure.

**Example:**
```
lifecycle 0.02 exponential
Output: Lifecycle params: capacity_factor=0.02, curve=exponential
```

---

### `cross weight <float> | blend <float> | layers <float...> | confidence | temperament | off`

**What it does:**  
Tunes cross-attention settings.

**How it works:**

- `weight <float>`: Sets uniform influence weight (e.g., 0.8).  
- `blend <float>`: Sets blend strength (0.0–1.0).  
- `layers <float...>`: Sets per-layer weights.  
- `confidence`: Enables dynamic weighting based on confidence.  
- `temperament`: Enables dynamic weighting based on temperament.  
- `off`: Disables dynamic weighting.

**Use case:**  
Customize how scaffold affects base model.

**Examples:**
```
cross weight 0.7
Output: Scaffold influence: weight=0.70, blend_strength=unchanged

cross layers 0.5 0.6 0.7
Output: Scaffold influence: weight=per-layer, blend_strength=unchanged
```

---

### `scaffold_mem`, `token_mem`, `both_mem`, `no_mem`

**What it does:**  
Toggles memory usage modes.

**How it works:**

- `scaffold_mem`: Enables scaffold memory only.  
- `token_mem`: Enables token map memory only.  
- `both_mem`: Enables both.  
- `no_mem`: Disables both.

**Use case:**  
Control memory-driven adaptation.

**Example:**
```
both_mem
Output: Memory toggled: Scaffold=True, Token Map=True
```

---

### Any Other Input (Prompt)

**What it does:**  
Generates a response using the current model state.

**How it works:**  
Calls `generate()` with defaults, logs to `log.jsonl`, and applies scaffold influence, temperament, and dreaming if enabled.

**Use case:**  
Test model output interactively.

**Example:**
```
Hello!
Output: --- Generating Response ---
        Response: Hi there! How can I assist you today?
        --------------------
```

---

### [Empty Input] (Pressing Enter)

**What it does:**  
Skips and waits for next input.

**How it works:**  
Loop ignores empty strings.

**Example:**
```
[Enter]
Output: (new prompt line)
```
```

Let me know if you'd like the configuration section converted next!
```
