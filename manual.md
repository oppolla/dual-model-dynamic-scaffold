# Console Commands
## quit or exit
What it does: Stops the script and exits the program cleanly.

How it works: Breaks the input loop, triggers cleanup (freeing GPU memory), and shuts down.

Use case: When you’re done testing or training and want to close everything.

Example: Type quit → Output: Exiting. (plus cleanup messages).

## train
What it does: Runs the training cycle on TRAIN_DATA and VALID_DATA using settings from config.json (e.g., epochs, batch size, sigmoid parameters).

How it works: Calls run_training_cycle, applies the tunable sigmoid life curve (via get_life_curve_weight), trains the scaffold model with LoRA, and logs progress. If in dry-run mode, it stops after one epoch.

Use case: Train the model with your sample_log.jsonl data to adjust the scaffold’s influence.

Example: Type train → Output:

Data exposure: 5 samples | Scaffold influence weight: 0.881
--- Starting Training (3 epochs) ---
Epoch 1/3
Step 1/3 | Loss: 2.3456

##  int8
What it does: Sets the quantization mode to INT8 (8-bit integer) and reinitializes the system.

How it works: Updates quantization_mode to "int8", then restarts the BareBonesDMAO_Learn instance to apply the change (reloading models in INT8).

Use case: Reduce memory usage on your RTX 3070 if you’re hitting GPU memory limits (e.g., with larger batches or models).

Example: Type int8 → Output:

Quantization mode set to 'int8'. Restart system to apply quantization.
Re-initializing system with INT8 quantization...

## int4
What it does: Sets the quantization mode to INT4 (4-bit integer) and reinitializes the system.

How it works: Same as int8, but uses "int4" for even lower memory usage (via bitsandbytes).

Use case: Further optimize memory if INT8 isn’t enough or for experimenting with efficiency.

Example: Type int4 → Output:

Quantization mode set to 'int4'. Restart system to apply quantization.
Re-initializing system with INT4 quantization...

## fp16
What it does: Sets the quantization mode to FP16 (16-bit floating point) and reinitializes the system.

How it works: Switches to "fp16" (the default), then reloads the system with half-precision floats.

Use case: Restore default precision if you’ve switched to INT8/INT4, balancing speed and accuracy.

Example: Type fp16 → Output:

Quantization mode set to 'fp16'. Restart system to apply quantization.
Re-initializing system with FP16 quantization...

## dynamic
What it does: Enables dynamic layer selection for cross-attention and reinitializes the system.

How it works: Sets USE_DYNAMIC_LAYERS to True, uses LAYER_SELECTION_MODE (e.g., "balanced") to pick layers dynamically, then restarts the system.

Use case: Experiment with adaptive layer injection instead of fixed CROSS_ATTN_LAYERS.

Example: Type dynamic → Output:

Dynamic layer selection enabled. Restart system to apply.
Re-initializing system with dynamic layers...

## fixed
What it does: Disables dynamic layer selection (uses fixed layers from config.json) and reinitializes the system.

How it works: Sets USE_DYNAMIC_LAYERS to False, sticks to CROSS_ATTN_LAYERS, then reloads the system.

Use case: Revert to your predefined layers (e.g., [5, 7]) for consistency.

Example: Type fixed → Output:

Dynamic layer selection disabled. Restart system to apply.
Re-initializing system with fixed layers...

## new
What it does: Starts a new conversation, resetting the conversation history.

How it works: Calls new_conversation, clears the ConversationHistory (last 10 messages), assigns a new conversation_id, and clears the scaffold cache.

Use case: Begin a fresh interaction without past prompts influencing the logs.

Example: Type new → Output:

New conversation started with ID: [new_uuid] (Previous ID: [old_uuid])

## Any Other Input (Prompt)
What it does: Treats the input as a prompt and generates a response using the current model state.

How it works: Calls generate with the input, applies the scaffold’s influence (tuned by the sigmoid), logs the response in log.jsonl, and prints it. Uses defaults like max_new_tokens=60, temperature=0.7.

Use case: Test the model’s output after training or tweak its behavior live.
```
Example: Type Hello! → Output:

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

Example: Press Enter → Output: (just a new prompt line \nEnter command or prompt: )

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

