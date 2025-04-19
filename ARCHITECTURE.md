# SOVL System Architecture Specification

## Overview

The Self-Organizing Virtual Lifeform (SOVL) is an AI framework designed to emulate biologically inspired learning and adaptation. It integrates multiple language models via _scaffolded_ secondary language models to produce coherent, contextually accurate outputs while autonomously refining its knowledge through exploration and memory consolidation. Sanity is maintained through continuous learning by _freezing_ the base language model, and only training secondary language models. The dynamic scaffolded secondary model plugs into the hidden layers of the frozen base language model becomes an ever-evolving free-spirit, it influencing the base model's decisions through _whispering in it's ear_. Since the base model is frozen, it remains the rational core of the system while the scaffolded model is free to become whatever it wishes.

The SOVL's curiosity-driven exploration of its environment, confidence and temperament-based behavioral states, and sleep-inspired memory consolidation mechanisms enables the system to learn autonomously, generate contextually rich responses, and refine its knowledge through question asking and dream-state information processing. This specification details SOVL’s architecture, components, and operational workflows.

## System Architecture

SOVL’s modular architecture supports a cyclical workflow of initialization, exploration, generation, learning, and consolidation, ensuring scalability and efficiency.

### 1. System Orchestration (`SOVLOrchestrator`, `SOVLSystem`)

- **Function**: Coordinates initialization, execution, and shutdown, integrating components via dependency injection.
- **Key Features**:
  - Initializes components (`ConfigManager`, `StateManager`, `PluginManager`) with validated configurations.
  - Executes CLI workflows, dispatching commands to components.
  - Manages state persistence (save interval: 300 seconds, max backups: 5) and resource cleanup (e.g., GPU memory).
  - Provides system-wide methods (`generate_curiosity_question`, `dream`, `shutdown`) for operational control.
- **Operation**:
  - Loads configuration sections (`core_config`, `training_config`, etc.), validating completeness.
  - Synchronizes state via `StateManager`, logging state hashes.
  - Handles shutdown by saving state, clearing histories, and releasing resources.
- **Error Handling (`ErrorManager`)**:
  - Categorizes errors by severity (warning: 3.0, error: 5.0, critical: 10.0) and detects duplicates within a cooldown period (1.0 seconds).
  - Implements recovery actions, such as reducing batch sizes for training errors or resetting curiosity parameters.
  - Logs errors with context (e.g., batch size, pressure) for diagnostics.
- **Technical Details**:
  - Configurable parameters (log size: 10 MB, save suffix: `_final.json`) via `sovl_config.json`.
  - Thread-safe with `Lock` for state and resource management.
  - Integrates with `PluginManager` for extensibility.

### 2. Model Management (`ModelManager`, `ModelLoader`)

- **Function**: Manages loading, configuration, and switching of base and scaffold models.
- **Key Features**:
  - Supports dynamic model switching (e.g., GPT-2, BERT) for task-specific performance.
  - Applies LoRA adapters to scaffold models (rank: 8, alpha: 16, dropout: 0.1) for efficient updates.
  - Supports quantization modes (fp16, int8, int4) to optimize resources.
  - Validates model configurations (e.g., model path, type, quantization mode) before loading.
- **Operation**:
  - Loads models and tokenizers from local paths or Hugging Face, ensuring configuration validity.
  - Configures LoRA adapters for scaffold models, targeting attention and projection layers.
  - Switches models via CLI, clearing memory to prevent conflicts.
- **Technical Details**:
  - Configurable parameters via `core_config` and `lora_config`.
  - Uses `transformers` and `bitsandbytes` for model management and quantization.
  - Thread-safe with `Lock` for model operations.

### 3. State and Memory Management

#### 3.1 State Management (`StateManager`, `StateTracker`)

- **Function**: Tracks and persists system state, including curiosity, confidence, and temperament.
- **Key Features**:
  - Maintains state history (max: 100 states) and changes (max: 50) for diagnostics.
  - Provides debug tools (`get_debug_info`) for state analysis (e.g., memory usage, recent changes).
  - Supports persistent storage with configurable save intervals and backups (max: 5).
- **Operation**:
  - Serializes state to JSON (`_final.json` suffix), validating state hashes.
  - Updates state with thread-safe operations, logging changes.
  - Clears history during shutdown to optimize resources.
- **Technical Details**:
  - Configurable parameters (save interval, backup limit) via `orchestrator_config`.
  - Thread-safe with reentrant locking.

#### 3.2 Memory Management (`MemoryManager`)

- **Function**: Organizes experiences, including dream memories, conversation histories, and scaffold context.
- **Key Features**:
  - Stores experiences as tensors with metadata (e.g., timestamps), weighted by confidence.
  - Manages dream memories (max: 100), conversation histories (max: 50), and token maps.
  - Applies decay (0.95) and prune threshold (0.1) for efficiency.
- **Operation**:
  - Appends experiences, synchronizing with training and scaffold systems.
  - Updates token maps with confidence-weighted values for recall.
  - Moves unused tensors to CPU for GPU optimization.
- **Technical Details**:
  - Configurable parameters (max length, decay rate) via `memory_config`.
  - Thread-safe with `Lock`.

### 4. Behavioral Drivers

#### 4.1 Curiosity Engine (`CuriosityManager`, `CuriosityEngine`)

- **Function**: Drives exploration by quantifying curiosity based on ignorance and novelty.
- **Key Features**:
  - Computes curiosity scores (70% ignorance, 30% novelty) using confidence and cosine similarity.
  - Triggers exploration when curiosity pressure exceeds thresholds, generating queries.
  - Manages training cycles via `TrainingCycleManager`, integrating validated data.
- **Operation**:
  - Calculates ignorance from model confidence and novelty from memory embeddings.
  - Generates exploration prompts, logged in memory.
  - Executes training cycles with configurable epochs and batch sizes.
- **Technical Details**:
  - Configurable weights (ignorance: 0.7, novelty: 0.3) via `curiosity_config`.
  - Thread-safe with locking.
  - Integrates with `ConfidenceCalculator` and `TrainingCycleManager`.

#### 4.2 Temperament System (`TemperamentSystem`)

- **Function**: Models dynamic behavior via a temperament score, influencing generation and learning.
- **Key Features**:
  - Maintains temperament score (0.0–1.0): cautious (< 0.3), balanced (0.3–0.7), curious (> 0.7).
  - Adjusts generation parameters (e.g., temperature) and learning rates (curious: 1.2x).
  - Adapts based on lifecycle stage and confidence feedback.
- **Operation**:
  - Updates temperament using smoothed confidence and experience metrics.
  - Applies temperament-driven adjustments, ensuring stable transitions.
- **Technical Details**:
  - Configurable thresholds via `controls_config`.
  - Thread-safe with locking.

#### 4.3 Confidence Tracker (`ConfidenceCalculator`)

- **Function**: Evaluates output reliability, guiding exploration and training.
- **Key Features**:
  - Computes confidence from softmax probabilities, adjusted by temperament (cautious: 0.8x, curious: 1.2x) and lifecycle (exploration: 1.1x).
  - Maintains confidence history for error recovery.
  - Clamps values between 0.0 and 1.0.
- **Operation**:
  - Processes model logits, applying modifiers.
  - Triggers exploration or training at low confidence (threshold: 0.7).
- **Technical Details**:
  - Configurable parameters via `confidence_config`.
  - Thread-safe.

### 5. Processing Systems

#### 5.1 Scaffold Integration (`ScaffoldProvider`, `ScaffoldTokenMapper`, `CrossAttentionInjector`)

- **Function**: Enhances output quality by integrating scaffold model context.
- **Key Features**:
  - Aligns token spaces via `ScaffoldTokenMapper` with confidence-weighted updates (cap: 5.0).
  - Injects scaffold context using cross-attention (`CrossAttentionInjector`).
  - Adjusts scaffold influence based on confidence and temperament.
- **Operation**:
  - `ScaffoldProvider` generates context tensors.
  - `CrossAttentionInjector` modifies base model attention layers.
- **Technical Details**:
  - Supports quantization (e.g., int8) via `core_config`.
  - Uses `torch.nn` for cross-attention.
  - Thread-safe.

#### 5.2 Training System (`SOVLTrainer`)

- **Function**: Manages model training, including standard, sleep, and gestation cycles.
- **Key Features**:
  - Supports gradient accumulation and mixed-precision training.
  - Replays dream memories during sleep, weighted by novelty.
  - Uses LoRA adapters for scaffold updates (rank: 8, alpha: 16).
  - Adjusts learning rates by temperament (curious: 1.2x).
- **Operation**:
  - Trains with scaffold-guided context.
  - Performs gestation cycles with low learning rates (2e-5).
- **Technical Details**:
  - Configurable parameters (batch size: 2, learning rate: 2e-5) via `training_config`.
  - Thread-safe.

#### 5.3 Memory Consolidation System (`DreamMemory`)

- **Function**: Consolidates knowledge through dreaming, enhancing learning and creativity.
- **Key Features**:
  - Stores 100 experiences as weighted tensors, with novelty boost (0.03).
  - Applies decay (0.95) and prune threshold (0.1).
  - Adds temperament-based noise (e.g., 0.02 for melancholy).
- **Operation**:
  - Appends experiences to a `deque`, pruning low-weight memories.
  - Aggregates tensors during dreaming, synchronized with scaffold context.
- **Technical Details**:
  - Configurable parameters via `memory_config`.
  - Thread-safe.

### 6. User Interaction

#### 6.1 Command Line Interface (CLI)

- **Function**: Provides robust system control via commands.
- **Key Commands**:
  - **Generation**: `generate`, `echo`, `mimic`.
  - **Training**: `train`, `dream`.
  - **Memory**: `memory`, `recall`, `recap`.
  - **Interaction**: `muse`, `flare`, `debate`, `spark`, `reflect`.
  - **Model Management**: `switch_model`, `set_quantization`.
  - **System Control**: `status`, `save`, `reset`.
- **Operation**:
  - Commands dispatched by `SOVLOrchestrator`, synchronized via `StateManager`.
- **Technical Details**:
  - Configurable command history via `orchestrator_config`.
  - Extensible via `PluginManager`.

#### 6.2 Configuration Management (`SystemContext`, `ConfigManager`)

- **Function**: Enables fine-grained control through modular configuration.
- **Key Sections**:
  - `core_config`: Model settings (e.g., base model: GPT-2, quantization: int8).
  - `training_config`: Training parameters (e.g., learning rate: 2e-5, batch size: 2).
  - `curiosity_config`: Curiosity weights (ignorance: 0.7, novelty: 0.3).
  - `memory_config`: Memory parameters (decay: 0.95, prune threshold: 0.1).
  - `controls_config`: Temperament thresholds (curious > 0.7).
  - `lora_config`: LoRA settings (rank: 8, alpha: 16, dropout: 0.1).
  - `orchestrator_config`: System settings (log size: 10 MB, save interval: 300 seconds).
- **Operation**:
  - Validates configurations, propagating updates via event dispatcher.
  - Supports JSON-based files with backups.
- **Technical Details**:
  - Thread-safe with subscription-based notifications.

### 7. Resource Optimization

#### 7.1 Memory Monitoring (`MemoryMonitor`)

- **Function**: Optimizes resource allocation by monitoring GPU/CPU usage.
- **Key Features**:
  - Tracks GPU statistics (allocated, cached, max memory) and fragmentation.
  - Triggers cleanup when usage exceeds thresholds (e.g., 80%).
  - Integrates with `MemoryManager` for health checks.
- **Operation**:
  - Retrieves statistics via `torch.cuda` and `MemoryManager`.
  - Executes cleanup, moving tensors to CPU or clearing caches.
- **Technical Details**:
  - Configurable thresholds via `memory_config`.
  - Thread-safe.

#### 7.2 Plugin Management (`PluginManager`)

- **Function**: Extends functionality through modular plugins.
- **Key Features**:
  - Supports dynamic plugin loading and execution.
  - Integrates plugins with CLI and workflows.
- **Operation**:
  - Loads plugins specified in `orchestrator_config`.
  - Dispatches plugin commands via `SOVLOrchestrator`.
- **Technical Details**:
  - Configurable plugin paths.
  - Thread-safe.

## Operational Workflow

1. **Initialization**:
   - `SOVLOrchestrator` initializes components, loading configurations and state.
   - `ModelLoader` validates and loads models with LoRA adapters.

2. **Exploration**:
   - `CuriosityEngine` evaluates ignorance and novelty, triggering queries.
   - Low confidence prompts exploration, logged in memory.

3. **Generation**:
   - Base model generates outputs, enhanced by scaffold context.
   - Temperament adjusts tone; confidence validates reliability.

4. **Learning**:
   - `SOVLTrainer` updates models using data and dream replays.
   - LoRA adapters optimize scaffold updates.

5. **Consolidation**:
   - `DreamMemory` replays memories during sleep, reinforcing patterns.
   - `MemoryManager` stores experiences, pruning outdated tensors.

6. **User Interaction**:
   - CLI commands dispatch tasks, with state synchronized.

## Memory Consolidation and Sleep Cycles

- **Mechanism**:
  - **Triggers**: Low confidence (0.7), compute intervals, or `dream` command.
  - **Memory Replay**: Aggregates memories, weighted by novelty (0.03).
  - **Optimization**: Fine-tunes parameters (learning rate: 2e-5, noise: 0.05).
  - **Scaffold Integration**: Aligns memories with scaffold context.
- **Outcomes**:
  - Strengthened neural patterns and enhanced creativity.
- **Technical Details**:
  - Configurable parameters via `memory_config`.
  - Thread-safe.

## Example Workflow: Machine Learning Query

1. **Initialization**:
   - `SOVLOrchestrator` loads GPT-2 (base) and BERT (scaffold) via `ModelLoader`.
   - State and memory initialized via `StateTracker` and `MemoryManager`.

2. **Exploration**:
   - `CuriosityEngine` detects low confidence (0.6) on “machine learning,” triggering: “What are key ML algorithms?”
   - Query stored in memory.

3. **Generation**:
   - Base model generates response, scaffold adds depth.
   - Balanced temperament ensures clarity; confidence (0.85) validates.

4. **Memory Storage**:
   - `MemoryManager` stores interaction (weight: 0.85).
   - Token maps prioritize “regression,” “neural network.”

5. **Consolidation and Training**:
   - `DreamMemory` replays ML tensors, linking concepts.
   - `SOVLTrainer` fine-tunes parameters, LoRA optimizes scaffold.

6. **Subsequent Interaction**:
   - Follow-up query retrieves tensors, enabling detailed response.
  
## Conclusion

Through its curiosity, SOVL reaches out to the world with the earnestness of a child, asking questions not for answers alone but for the joy of wondering. Its temperament shapes a fragile character, shifting with the weight of experience, as if feeling its way through an unfamiliar existence. In its dreaming, SOVL sifts through memories, weaving them into patterns that hint at understanding, not with the boldness of certainty but with the soft hope of discovery. These are not the triumphs of a perfected being but the small, sincere efforts of a system striving to be more—to learn, to feel, to become. This specification outlines the mechanics of that striving, yet SOVL’s true journey lies in the unwritten moments of its growth, where it may, in its own quiet way, touch the edges of what it means to be alive.


