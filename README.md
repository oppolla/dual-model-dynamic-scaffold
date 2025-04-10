# SOVL System (Self-Organizing Virtual Lifeform)

## Overview
The SOVL System (Self-Organizing Virtual Lifeform) is an prototype AI architecture that combines a large base model with a smaller "scaffold" model for efficient, adaptive inference. The system dynamically coordinates between models using cross-attention and continuously improves through background training.

## Key Components of Prototype Implementation

### Core Models

- **Base Model:** Large frozen LLM (`deepseek-llm-67b-chat`)

- **Scaffold Model:** Smaller adaptable model (`deepseek-r1-distill-qwen1.5-1.5b`) with dynamic LoRA layers


## Innovative Modules

**AdaptiveLoRALinear:** Dynamically adjusts LoRA rank based on learned importance

**SparseCrossAttention:** Efficient top-k attention between models

**CrossAttentionFuser:** Intelligently combines base and scaffold outputs

**SalienceScorer:** Evaluates interaction importance using BERT

## Key Features

### Dynamic Adaptation:

- LoRA layers automatically adjust their rank

- Cross-attention gates scaffold contributions based on confidence

### Continuous Learning:

- Background training scheduler

- Cluster-based data sampling

- Automatic rollback on failure

### Resource Awareness:

- System load monitoring

- Training time limits

- Gradient checkpointing

## Configuration

The system has some configurable parameters including:

- Training intervals (default: 5 minutes)

- Minimum training examples (default: 50)

- System load limits (default: 70%)

- Salience thresholds (default: 0.75)

- Training epochs (default: 3)

- More planned (see TODO.md)

```
system = ASCSystem()
response = system.generate_response(user_input)
system.log_interaction(user_input, response)
```

## Requirements

- PyTorch

- Transformers

- PEFT (Parameter-Efficient Fine-Tuning)

- Scikit-learn (for clustering)

- PSutil (for system monitoring)
