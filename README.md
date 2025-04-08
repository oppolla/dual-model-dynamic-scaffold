# Dual-Model Dynamic Scaffold with Asynchronous Training Phase for dynamic LLM personalization

## Overview
A lightweight, modular framework for dynamic language model augmentation using cross-attention and LoRA adapters. Designed for controllable text generation with memory, temperament modeling, and adaptive training.

## Key Features
- **Dual-Model Architecture**: Combines a frozen base LLM with trainable scaffold models via cross-attention
- **Dynamic Memory Integration**: 
  - Short-term conversation memory
  - Learned token mapping memory
  - "Dream" memory consolidation during sleep phases
- **Temperament Modeling**: 
  - Mood-influenced generation (eager/restless/melancholic)
  - Confidence-based self-regulation
  - Lifecycle-aware capacity scaling
- **Efficient Training**:
  - LoRA adapters for parameter-efficient fine-tuning
  - FP16/INT8/INT4 quantization support
  - Dynamic layer selection
- **Interactive Controls**:
  - Real-time parameter adjustment
  - Multiple memory modes
  - Custom training schedules
 
## Architecture Overview
Base Model (Frozen)
↑
Cross-Attention Fusion ← Scaffold Model (LoRA-adapted)
↑
Temperament Controls
↑
Memory Systems:
Conversation History
Token Mapping
Dream Memory

## Core Components
### Cross-Attention Fusion
- Injects attention layers between base and scaffold models
- Configurable layer selection (early/late/custom)
- Dynamic influence weighting

### Memory Systems
Type	Function	Control Parameters
Conversation	Maintains dialog context	dream_memory_maxlen
Token Mapping	Learns token-level associations	use_token_map_memory
Dream Memory	Consolidates experiences	dream_noise_scale

### Temperament Controls
```
# Adjust temperament parameters
system.adjust_temperament(
    eager_threshold=0.8,
    mood_influence=0.3,
    curiosity_boost=0.2
)
```



