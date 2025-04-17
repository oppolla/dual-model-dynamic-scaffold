Step-by-Step Guide to Loading a Local Model and Running SOVL

Prerequisites

Python Environment: Ensure you have Python 3.8+ installed.

Dependencies: Install the required packages:

`pip install torch transformers peft bitsandbytes`

Local Model: Have a Hugging Face model (e.g., `Llama-2-7b`) downloaded locally.

Step 1: Prepare the Configuration File

Create a `sovl_config.json` file in your workspace with the following structure:

```
{
  "model": {
    "model_path": "/path/to/your/local/model",
    "model_type": "llama",
    "quantization_mode": "4bit"
  },
  "state": {
    "state_file": "sovl_state.json",
    "backup_interval": 300,
    "max_history": 100
  },
  "error_config": {
    "error_cooldown": 1.0,
    "warning_threshold": 3.0,
    "error_threshold": 5.0,
    "critical_threshold": 10.0
  }
}
```

Replace `/path/to/your/local/model` with the actual path to your model.

Step 2: Initialize the System

Create a Python script (e.g., `run_sovl.py`) with the following code:

```
from sovl_main import SystemContext, SOVLSystem, ModelLoader, StateTracker, ErrorManager, MemoryMonitor, CuriosityEngine

def main():
    # Step 1: Initialize System Context
    context = SystemContext(config_path="sovl_config.json", device="cuda")

    # Step 2: Load the Model
    model_loader = ModelLoader(context)
    model = model_loader.load_model()

    # Step 3: Initialize Core Components
    state_tracker = StateTracker(context)
    error_manager = ErrorManager(context, state_tracker)
    memory_monitor = MemoryMonitor(context)
    curiosity_engine = CuriosityEngine(
        config_handler=context.config_handler,
        model_loader=model_loader,
        state_tracker=state_tracker,
        error_manager=error_manager,
        logger=context.logger,
        device=context.device
    )

    # Step 4: Initialize the SOVL System
    sovl_system = SOVLSystem(
        context=context,
        config_handler=context.config_handler,
        model_loader=model_loader,
        curiosity_engine=curiosity_engine,
        memory_monitor=memory_monitor,
        state_tracker=state_tracker,
        error_manager=error_manager
    )

    # Step 5: Run the System
    print("SOVL System initialized successfully!")
    # Example: Run a training cycle
    sovl_system.curiosity_engine.run_training_cycle(
        train_data=None,  # Replace with your training data
        valid_data=None,  # Replace with validation data
        epochs=10,
        batch_size=32
    )

if __name__ == "__main__":
    main()
```

### Step 3: Run the Script

Execute the script:

`python run_sovl.py`

Step 4: Verify the System

Check Logs: The system will log events to the console and/or a log file (if configured).

Monitor Metrics: Use the `SystemMonitor` (from `sovl_monitor.py`) to track memory, training progress, and curiosity scores.

CLI Interaction: If you've integrated the CLI (`sovl_cli.py`), you can interact with the system using commands like:

`train:` Start training.

`monitor start`: Begin monitoring system metrics.
`status:` Check system status.

## Troubleshooting

Model Loading Errors:
Ensure the model path in `sovl_config.json` is correct.

Verify the model is compatible with `transformers` (e.g., it should have a config.json file).
CUDA Errors:
Ensure you have a compatible GPU and CUDA installed.
Fall back to CPU by setting `device="cpu"` in `SystemContext`.

Configuration Issues:
Validate `sovl_config.json` using the `ConfigHandler` in `sovl_main.py`.

## Step 5: Integrate the CLI for Interactive Control

Now that the SOVL system is running, you can interact with it using the Command Line Interface (CLI) from `sovl_cli.py`. Here's how to set it up:

1. Update the `run_sovl.py` Script
   
Modify your `run_sovl.py` to include the CLI initialization:

```
from sovl_cli import CommandHandler, run_cli  # Add these imports

def main():
    # ... (previous initialization code) ...

    # Step 5: Initialize the CLI
    cli_handler = CommandHandler(sovl_system)
    
    # Option 1: Run the CLI in interactive mode
    run_cli(cli_handler)  # Starts an interactive command loop

    # Option 2: Run specific commands programmatically
    # Example: Start monitoring
    cli_handler.execute("monitor", ["start"])

if __name__ == "__main__":
    main()
```
2. Key CLI Commands
3. 
Once the CLI is active, you can use these commands:

System Control:

`monitor start/stop/status:` Control the monitoring system.

`status:` Check system health (memory, training, etc.).

`config get/set:` View or modify configurations (e.g., config set monitor.update_interval 2.0).

Training:

`train [epochs]:` Start training (e.g., train 10).

`dream:` Run a dream cycle to consolidate memories.

Debugging:

`log view [n]:` Show recent logs (e.g., log view 5).

`panic:` Force a system reset if errors occur.

4. Example Workflow
5. 
Start the CLI:

`python run_sovl.py`

Begin Monitoring:

`Enter command: monitor start`

Check Status:

`Enter command: status`

Output:

```
   --- System Status ---
   Memory Usage: 4.2 GB / 16.0 GB
   Training Progress: 0%
   Monitoring: Active (update every 1.0s)
```

Run Training:

`Enter command: train 5`

## Common Issues

### Configuration Errors

Symptoms:
SystemInitializationError or ConfigValidationError on startup.
Missing metrics in monitoring or unexpected default values.
Likely Causes:
Invalid paths in sovl_config.json (e.g., model path typo).
Missing required fields (e.g., model_type, quantization_mode).
Invalid values (e.g., quantization_mode="5bit" instead of "4bit").
How to Fix:
Validate your sovl_config.json against the schema in sovl_config.py.
Use the ConfigHandler.validate() method to check for errors:
```
  from sovl_config import ConfigHandler
  handler = ConfigHandler("sovl_config.json")
  if not handler.validate():
      print("Configuration errors:", handler.get_validation_errors())
```

### Model Loading Failures

Symptoms:
ModelLoadingError or CUDA out-of-memory errors.
The system hangs during initialization.
Likely Causes:
Incorrect model path or incompatible model format.
Insufficient GPU memory for the specified quantization_mode.
Missing dependencies (e.g., bitsandbytes for 4-bit quantization).
How to Fix:
Verify the model directory contains:
config.json
pytorch_model.bin (or .safetensors)
Tokenizer files.
Reduce quantization (e.g., switch from 4bit to 8bit in config).
Test loading the model directly with transformers:
```
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("/your/model/path")
```



