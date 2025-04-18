# Step-by-Step Guide to Running SOVL System

## Prerequisites

1. **Python Environment**:
   - Python 3.8 or higher
   - Virtual environment recommended (but not required)

2. **Dependencies**:
   ```bash
   pip install torch transformers peft bitsandbytes
   ```

3. **Hardware Requirements**:
   - For GPU usage: NVIDIA GPU with CUDA support
   - Minimum 16GB RAM recommended
   - Sufficient disk space for model storage and checkpoints

## Step 1: Prepare the Configuration File

Create a `sovl_config.json` file with the following structure:

```json
{
  "core_config": {
    "base_model_name": "SmolLM2-360M",
    "base_model_path": null,
    "scaffold_model_name": "SmolLM2-135M",
    "scaffold_model_path": null,
    "cross_attn_layers": [],
    "use_dynamic_layers": false,
    "layer_selection_mode": "balanced",
    "custom_layers": null,
    "valid_split_ratio": 0.2,
    "random_seed": 42,
    "quantization": "fp16",
    "hidden_size": 768,
    "num_heads": 12,
    "gradient_checkpointing": true,
    "initializer_range": 0.02,
    "migration_mode": true
  },
}
```

Replace `/path/to/your/local/model` with the actual path to your model.

## Step 2: Running the System

The system can be run in several modes:

### Basic Run
```bash
python run_sovl.py --config sovl_config.json --device cuda
```

### Available Command Line Arguments:
- `--config`: Path to configuration file (required)
- `--device`: Device to use ("cuda" or "cpu")
- `--mode`: Operation mode ("train", "generate", or "dream")
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--train-data`: Path to training data file
- `--valid-data`: Path to validation data file
- `--test`: Run in test mode
- `--verbose`: Enable verbose logging
- `--monitor-interval`: Monitoring update interval in seconds
- `--checkpoint-interval`: Checkpoint interval in epochs
- `--resume-from-checkpoint`: Path to checkpoint file to resume from
- `--validate-every`: Run validation every N epochs
- `--max-patience`: Max epochs without validation improvement
- `--max-checkpoints`: Maximum number of checkpoints to keep

## Step 3: System Operation Modes

### Training Mode
```bash
python run_sovl.py --config sovl_config.json --mode train --epochs 10
```

### Generation Mode
```bash
python run_sovl.py --config sovl_config.json --mode generate
```

### Dream Mode
```bash
python run_sovl.py --config sovl_config.json --mode dream
```

## Step 4: Monitoring and Interaction

The system provides several ways to monitor and interact with the running instance:

### Log Files
- Logs are automatically created in the `output` directory
- Format: `sovl_run_YYYYMMDD_HHMMSS.log`
- Log rotation is enabled (max 5 files, 10MB each)

### CLI Commands
Once the system is running, you can use these commands:

#### System Control
- `monitor start/stop/status`: Control the monitoring system
- `status`: Check system health (memory, training, etc.)
- `config get/set`: View or modify configurations

#### Training
- `train [epochs]`: Start training (e.g., `train 10`)
- `dream`: Run a dream cycle to consolidate memories

#### Memory Management
- `memory`: View memory usage and statistics
- `recall`: Access stored memories
- `forget`: Clear specific memories
- `recap`: Get a summary of recent memories

#### Debugging
- `log view [n]`: Show recent logs
- `panic`: Force a system reset if errors occur
- `glitch`: Simulate error conditions for testing

## Step 5: Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**:
   - Verify the model path in `sovl_config.json`
   - Ensure the model is compatible with `transformers`
   - Check for sufficient disk space

2. **CUDA/GPU Issues**:
   - Verify CUDA installation: `nvidia-smi`
   - Check GPU memory requirements
   - Fall back to CPU if needed: `--device cpu`

3. **Memory Issues**:
   - Adjust `max_memory_mb` in config
   - Monitor memory usage with `memory` command
   - Consider reducing batch size

4. **Configuration Errors**:
   - Validate all required sections are present
   - Check value ranges for parameters
   - Ensure proper JSON formatting

### Error Logging
- Check the latest log file in the `output` directory
- Use `log view` command to see recent errors
- Monitor system status with `status` command

## Step 6: Best Practices for Testing

1. **Start Small**:
   - Begin with CPU mode to verify basic functionality
   - Use small batch sizes initially
   - Test with minimal epochs

2. **Monitor Resources**:
   - Watch memory usage during training
   - Check GPU utilization
   - Monitor disk space for checkpoints

3. **Regular Checkpoints**:
   - Set appropriate checkpoint intervals
   - Keep backup files for recovery
   - Test checkpoint loading functionality

4. **Error Handling**:
   - Test graceful shutdown
   - Verify error recovery mechanisms
   - Check log rotation and cleanup



