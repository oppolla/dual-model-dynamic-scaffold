# Documentation Guide for BareBonesDMAO_Learn Script

This script implements a machine learning system using **PyTorch** and the **Hugging Face Transformers library**. It combines a frozen base GPT-2 model with a scaffold GPT-2 model enhanced by **LoRA (Low-Rank Adaptation)** for efficient fine-tuning. The system injects **cross-attention mechanisms** into the base model to leverage information from the scaffold model, enabling a hybrid approach to text generation and training. The script supports training on a dataset and generating text based on prompts, with configurable scaffold influence.

---

## Overview

The `BareBonesDMAO_Learn` class is the core of the system, handling:

- **Base Model**: A frozen GPT-2 model that does not update during training.
- **Scaffold Model**: A GPT-2 model with LoRA adapters for efficient fine-tuning.
- **Cross-Attention**: Injected into specific layers of the base model to integrate scaffold model context.
- **Training**: A custom loop to fine-tune the scaffold model on a dataset.
- **Generation**: Text generation with adjustable scaffold influence.

The script is designed for tasks requiring fine-tuned language generation while leveraging a stable base model.

---

## Configuration

The script uses several parameters defined at the top to configure the models and training process:

### Model Configuration
- **`BASE_MODEL_NAME`**: Name of the base model (e.g., `"gpt2"`, ~117M parameters, frozen).
- **`SCAFFOLD_MODEL_NAME`**: Name of the scaffold model (e.g., `"gpt2"`, fine-tuned with LoRA).
- **`CROSS_ATTN_LAYERS`**: List of layer indices in the base model for cross-attention injection (e.g., `[5, 7]`; GPT-2 has layers 0-11).
- **`DEVICE`**: Automatically set to `"cuda"` if GPU is available, otherwise `"cpu"`.

### LoRA Configuration
- **`LORA_RANK`**: Rank of low-rank matrices (e.g., `8`).
- **`LORA_ALPHA`**: Scaling factor for LoRA (e.g., `16`, typically `2 * LORA_RANK`).
- **`LORA_DROPOUT`**: Dropout rate for LoRA adapters (e.g., `0.1`).
- **`LORA_TARGET_MODULES`**: Model modules to apply LoRA to (e.g., `["c_attn", "c_proj", "c_fc"]`).

### Training Configuration
- **`LEARNING_RATE`**: Optimizer learning rate (e.g., `3e-4`, higher for LoRA).
- **`TRAIN_EPOCHS`**: Number of training epochs (e.g., `3`).
- **`BATCH_SIZE`**: Batch size for training (e.g., `1`, small to manage memory).
- **`MAX_SEQ_LENGTH`**: Maximum sequence length for inputs (e.g., `128`).
- **`VALID_SPLIT_RATIO`**: Fraction of dataset for validation (e.g., `0.2`).
- **`RANDOM_SEED`**: Seed for reproducibility (e.g., `42`).

---

## Model Setup

### Base Model
- **Loading**: Loaded via `AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)`.
- **Configuration**: Uses `AutoConfig` to match the model specification.
- **State**: Moved to `DEVICE`, set to evaluation mode, and parameters are frozen (`requires_grad = False`).

### Scaffold Model
- **Loading**: Loaded similarly to the base model.
- **LoRA Application**: Wrapped with LoRA adapters using `get_peft_model` from the **PEFT library**, targeting specified modules.
- **State**: Moved to `DEVICE` and set up for training.

### Tokenizer
- **Loading**: A single tokenizer is loaded from `BASE_MODEL_NAME` using `AutoTokenizer`.
- **Padding**: If no pad token exists, it’s set to the EOS token, ensuring consistency across models.

---

## LoRA Application

**LoRA (Low-Rank Adaptation)** enables efficient fine-tuning of the scaffold model:
- **Parameters**: Configured with `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, and `LORA_TARGET_MODULES`.
- **Integration**: Applied via `get_peft_model`, allowing only the adapters to be trained, reducing memory and computation costs.
- **Task Type**: Set to `TaskType.CAUSAL_LM` for causal language modeling.

---

## Cross-Attention Injection

Cross-attention is injected into the base model to fuse scaffold model context:
- **Target Layers**: Specified by `CROSS_ATTN_LAYERS`.
- **Module**: Uses `SimpleCrossAttentionFuser`, a custom module with:
  - Multi-head attention (`nn.MultiheadAttention`) to process base and scaffold hidden states.
  - A gating mechanism (`nn.Sequential` with `nn.Linear` and `nn.Sigmoid`) to control fusion.
  - Layer normalization (`nn.LayerNorm`) for stability.
  - An `influence_weight` parameter (0-1) to adjust scaffold impact.
- **Wrapper**: Each target layer is wrapped in a `ModifiedLayer` class that applies the fuser when scaffold context is available.
- **State**: Fuser parameters are frozen to focus training on LoRA adapters.

---

## Training Process

The training process fine-tunes the scaffold model’s LoRA adapters:

### Data Preparation
- **Dataset**: Uses `TRAIN_DATA` (assumed to contain `prompt` and `completion` fields), split into training and validation sets based on `VALID_SPLIT_RATIO`.
- **Batching**: Data is shuffled and batched with `BATCH_SIZE`.

### Optimizer Setup
- **Optimizer**: `AdamW` optimizes only the scaffold model’s trainable parameters (LoRA adapters).
- **Scheduler**: A linear scheduler with no warmup is configured based on total training steps.

### Training Loop (`run_training_cycle`)
1. **Batch Processing**:
   - Scaffold model computes hidden states for prompts.
   - Base model processes full text (prompt + completion) with cross-attention.
   - Loss is calculated using cross-entropy on the completion part, masking prompts and padding.
2. **Gradient Accumulation**: Gradients are accumulated over 4 steps before optimization to manage memory.
3. **Updates**: The optimizer and scheduler step periodically, updating LoRA parameters.
4. **Validation**: After each epoch, validation loss is computed on `VALID_DATA`.
5. **Early Stopping**: Stops if validation loss doesn’t improve for 2 epochs (`max_patience`).
6. **Evaluation**: Generates sample responses every epoch to assess quality.

---

## Generation

The `generate` method produces text from a prompt:
- **Inputs**:
  - `prompt`: Text to generate from.
  - `max_new_tokens`: Maximum tokens to generate (e.g., `50`).
  - `scaffold_weight`: Optional float (0-1) to control scaffold influence (default: `1.0`).
  - Additional `kwargs` for `generate` (e.g., `temperature`, `top_k`).
- **Process**:
  1. Encodes the prompt with the tokenizer.
  2. Computes scaffold hidden states and stores them temporarily.
  3. Generates text using the base model’s `generate` method, leveraging cross-attention.
  4. Checks for repetition (3-gram) and truncates if detected.
  5. Decodes the output, skipping special tokens.
- **Output**: Returns the generated text as a string.

---

## Usage

### Initialization (Python)

dmao_system = BareBonesDMAO_Learn()

Training
python

dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)

Starts the training cycle on the provided dataset.
Generation
python

response = dmao_system.generate("How to make coffee?", max_new_tokens=60, scaffold_weight=0.7)
print(response)

Generates a response with specified scaffold influence.
Interactive Mode
Run the script directly:
bash

python script.py

Commands:
train: Start training.

<prompt>: Generate a response (e.g., "How to make coffee?").

quit or exit: Exit the script.

Additional Features
Memory Monitoring: print_memory_stats tracks GPU memory usage (allocated and reserved).

Repetition Check: Ensures generation quality by truncating repetitive outputs.

Mixed Precision: Uses torch.autocast for efficiency on GPU (float16) or CPU (bfloat16).
