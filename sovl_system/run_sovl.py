import argparse
import os
import sys
import torch
import traceback
import json
import signal
import atexit
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sovl_main import SystemContext, SOVLSystem, ModelLoader, StateTracker, ErrorManager, MemoryMonitor, CuriosityEngine
from sovl_io import load_training_data, InsufficientDataError
from sovl_monitor import SystemMonitor
from sovl_cli import CommandHandler, run_cli
from sovl_utils import safe_compare
from sovl_logger import Logger, LoggerConfig

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_DATA = None
VALID_DATA = None

COMMAND_CATEGORIES = {
    "System": ["quit", "exit", "save", "load", "reset", "status", "help", "monitor"],
    "Training": ["train", "dream"],
    "Generation": ["generate", "echo", "mimic"],
    "Memory": ["memory", "recall", "forget", "recap"],
    "Interaction": ["muse", "flare", "debate", "spark", "reflect"],
    "Debug": ["log", "config", "panic", "glitch"],
    "Advanced": ["tune", "rewind"],
    "History": ["history"]
}

# Configure logging
logger_config = LoggerConfig(
    log_file=f'sovl_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    max_size_mb=10,
    compress_old=True,
    max_in_memory_logs=1000,
    rotation_count=5
)
logger = Logger(logger_config)

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.log_event(
        event_type="signal_received",
        message=f"Received signal {signum}, initiating graceful shutdown...",
        level="info"
    )
    cleanup_resources()
    sys.exit(0)

def register_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def validate_config_file(config_path: str) -> bool:
    """Validate the configuration file format and required fields."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_sections = ['model', 'state', 'error_config', 'monitor']
        for section in required_sections:
            if section not in config:
                logger.log_error(
                    error_msg=f"Missing required configuration section: {section}",
                    error_type="config_validation_error"
                )
                return False
                
        # Validate model configuration
        if 'model_path' not in config['model']:
            logger.log_error(
                error_msg="Missing required model configuration: model_path",
                error_type="config_validation_error"
            )
            return False
            
        # Validate monitor configuration
        monitor_config = config.get('monitor', {})
        if not isinstance(monitor_config.get('update_interval', 1.0), (int, float)):
            logger.log_error(
                error_msg="Invalid monitor.update_interval in configuration",
                error_type="config_validation_error"
            )
            return False
            
        return True
    except json.JSONDecodeError:
        logger.log_error(
            error_msg=f"Invalid JSON format in configuration file: {config_path}",
            error_type="config_validation_error"
        )
        return False
    except Exception as e:
        logger.log_error(
            error_msg=f"Error validating configuration file: {str(e)}",
            error_type="config_validation_error"
        )
        return False

def initialize_context(args) -> SystemContext:
    """Initialize system context with enhanced validation and error handling."""
    try:
        # Validate configuration file
        if not os.path.exists(args.config):
            logger.log_error(
                error_msg=f"Configuration file not found: {args.config}",
                error_type="config_validation_error"
            )
            sys.exit(1)
            
        if not validate_config_file(args.config):
            logger.log_error(
                error_msg="Configuration validation failed",
                error_type="config_validation_error"
            )
            sys.exit(1)
            
        # Validate device
        if args.device == "cuda":
            if not torch.cuda.is_available():
                logger.log_error(
                    error_msg="CUDA is not available. Please use --device cpu",
                    error_type="device_validation_error"
                )
                sys.exit(1)
            logger.log_event(
                event_type="device_selected",
                message=f"Using CUDA device: {torch.cuda.get_device_name(0)}",
                level="info"
            )
        else:
            logger.log_event(
                event_type="device_selected",
                message="Using CPU device",
                level="info"
            )
            
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        return SystemContext(config_path=args.config, device=args.device)
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to initialize context: {str(e)}",
            error_type="context_initialization_error"
        )
        sys.exit(1)

def initialize_components(context) -> tuple:
    """Initialize core SOVL components with enhanced error handling and progress tracking."""
    try:
        # Initialize model loader
        logger.log_event(
            event_type="component_initialization",
            message="Initializing model loader...",
            level="info"
        )
        model_loader = ModelLoader(context)
        
        # Load model
        logger.log_event(
            event_type="component_initialization",
            message="Loading model...",
            level="info"
        )
        model = model_loader.load_model()
        
        # Initialize state tracker
        logger.log_event(
            event_type="component_initialization",
            message="Initializing state tracker...",
            level="info"
        )
        state_tracker = StateTracker(context)
        
        # Initialize error manager
        logger.log_event(
            event_type="component_initialization",
            message="Initializing error manager...",
            level="info"
        )
        error_manager = ErrorManager(context, state_tracker)
        
        # Initialize memory monitor
        logger.log_event(
            event_type="component_initialization",
            message="Initializing memory monitor...",
            level="info"
        )
        memory_monitor = MemoryMonitor(context)
        
        # Initialize curiosity engine
        logger.log_event(
            event_type="component_initialization",
            message="Initializing curiosity engine...",
            level="info"
        )
        curiosity_engine = CuriosityEngine(
            config_handler=context.config_handler,
            model_loader=model_loader,
            state_tracker=state_tracker,
            error_manager=error_manager,
            logger=context.logger,
            device=context.device
        )
        
        logger.log_event(
            event_type="component_initialization",
            message="All components initialized successfully",
            level="info"
        )
        return model_loader, model, state_tracker, error_manager, memory_monitor, curiosity_engine
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to initialize components: {str(e)}",
            error_type="component_initialization_error"
        )
        raise

def cleanup_resources(context=None, model=None):
    """Release system resources with enhanced logging and error handling."""
    try:
        logger.log_event(
            event_type="cleanup",
            message="Starting cleanup...",
            level="info"
        )
        
        if model is not None:
            logger.log_event(
                event_type="cleanup",
                message="Cleaning up model...",
                level="info"
            )
            del model
            
        if torch.cuda.is_available():
            logger.log_event(
                event_type="cleanup",
                message="Clearing CUDA cache...",
                level="info"
            )
            torch.cuda.empty_cache()
            
        if context is not None:
            logger.log_event(
                event_type="cleanup",
                message="Cleaning up context...",
                level="info"
            )
            context.cleanup()
            
        logger.log_event(
            event_type="cleanup",
            message="Cleanup completed successfully",
            level="info"
        )
    except Exception as e:
        logger.log_error(
            error_msg=f"Error during cleanup: {str(e)}",
            error_type="cleanup_error"
        )

def run_system(args, context, model, model_loader, state_tracker, error_manager, memory_monitor, curiosity_engine):
    """Run the SOVL system with enhanced monitoring and error handling."""
    try:
        logger.log_event(
            event_type="system_start",
            message="Initializing SOVL system...",
            level="info"
        )
        sovl_system = SOVLSystem(
            context=context,
            config_handler=context.config_handler,
            model_loader=model_loader,
            curiosity_engine=curiosity_engine,
            memory_monitor=memory_monitor,
            state_tracker=state_tracker,
            error_manager=error_manager
        )
        
        logger.log_event(
            event_type="system_start",
            message="Enabling memory management...",
            level="info"
        )
        sovl_system.toggle_memory(True)

        # Initialize system monitor
        logger.log_event(
            event_type="system_start",
            message="Initializing system monitor...",
            level="info"
        )
        system_monitor = SystemMonitor(
            memory_manager=memory_monitor,
            training_manager=curiosity_engine.training_manager,
            curiosity_manager=curiosity_engine.curiosity_manager,
            config_manager=context.config_handler,
            logger=context.logger
        )
        system_monitor.start_monitoring()

        # Start CLI in a separate thread
        logger.log_event(
            event_type="system_start",
            message="Starting interactive CLI...",
            level="info"
        )
        cli_thread = threading.Thread(
            target=run_cli,
            args=(sovl_system,),
            daemon=True  # Terminates when the main thread exits
        )
        cli_thread.start()

        # Pre-flight checks
        logger.log_event(
            event_type="system_start",
            message="Running pre-flight checks...",
            level="info"
        )
        model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**2
        logger.log_event(
            event_type="system_start",
            message=f"Model size: {model_size:.2f} MB",
            level="info"
        )
        
        if not memory_monitor.check_memory_health(model_size):
            logger.log_error(
                error_msg="Insufficient memory to run the system",
                error_type="memory_error"
            )
            sys.exit(1)

        if args.test:
            logger.log_event(
                event_type="system_start",
                message="Running in test mode...",
                level="info"
            )
            if not context.config_handler.validate():
                logger.log_error(
                    error_msg="Configuration validation failed",
                    error_type="config_validation_error"
                )
                sys.exit(1)
            logger.log_event(
                event_type="system_start",
                message="Test mode completed successfully",
                level="info"
            )
            return

        # Run the selected mode
        if args.mode == "train":
            logger.log_event(
                event_type="system_start",
                message="Starting training mode...",
                level="info"
            )
            train_data = load_training_data(args.train_data) if args.train_data else None
            valid_data = load_training_data(args.valid_data) if args.valid_data else None
            
            logger.log_event(
                event_type="system_start",
                message=f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}",
                level="info"
            )
            results = sovl_system.curiosity_engine.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            logger.log_event(
                event_type="system_start",
                message=f"Training completed. Results: {results}",
                level="info"
            )
            
        elif args.mode == "generate":
            logger.log_event(
                event_type="system_start",
                message="Starting generation mode...",
                level="info"
            )
            question = sovl_system.generate_curiosity_question()
            logger.log_event(
                event_type="system_start",
                message=f"Generated question: {question}",
                level="info"
            )
            
        elif args.mode == "dream":
            logger.log_event(
                event_type="system_start",
                message="Starting dream mode...",
                level="info"
            )
            success = sovl_system.dream()
            logger.log_event(
                event_type="system_start",
                message=f"Dream cycle {'succeeded' if success else 'failed'}",
                level="info"
            )
            
    except Exception as e:
        logger.log_error(
            error_msg=f"Error during system execution: {str(e)}",
            error_type="system_execution_error"
        )
        raise
    finally:
        # Stop monitoring when done
        if 'system_monitor' in locals():
            system_monitor.stop_monitoring()
        # Wait for CLI thread to finish
        if 'cli_thread' in locals() and cli_thread.is_alive():
            cli_thread.join(timeout=1.0)

def execute_command(sovl_system: SOVLSystem, command: str, args: List[str] = None) -> bool:
    """Execute a command with proper error handling and logging."""
    try:
        if not args:
            args = []
            
        if command == "quit" or command == "exit":
            print("Exiting...")
            return True
            
        elif command == "save":
            path = args[0] if args else None
            print(f"Saving state{' to ' + path if path else ' to default location'}...")
            if hasattr(sovl_system, 'save_state'):
                sovl_system.save_state(path)
                print("State saved.")
            else:
                print("Error: 'save_state' method not found.")
                
        elif command == "load":
            path = args[0] if args else None
            print(f"Loading state{' from ' + path if path else ' from default location'}...")
            if hasattr(sovl_system, 'load_state'):
                sovl_system.load_state(path)
                print("State loaded.")
            else:
                print("Error: 'load_state' method not found.")
                
        elif command == "reset":
            print("Resetting system state...")
            if hasattr(sovl_system, 'cleanup'):
                sovl_system.cleanup()
                print("Cleanup complete.")
            else:
                print("Warning: 'cleanup' method not found.")
                
        elif command == "status":
            print("\n--- System Status ---")
            if hasattr(sovl_system, 'state_tracker'):
                state = sovl_system.state_tracker.get_state()
                print(f"Conversation ID: {state.history.conversation_id}")
                print(f"Temperament: {state.temperament_score:.2f}")
                print(f"Last Confidence: {state.confidence_history[-1]:.2f if state.confidence_history else 'N/A'}")
                print(f"Data Exposure: {state.training_state.data_exposure}")
                print(f"Last Trained: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.training_state.last_trained)) if state.training_state.last_trained else 'Never'}")
                print(f"Gestating: {'Yes' if state.is_sleeping else 'No'}")
                
        elif command == "monitor":
            if not args:
                print("Monitoring commands: start, stop, status, history")
                return False
                
            action = args[0].lower()
            if not hasattr(sovl_system, 'monitor'):
                print("Error: System monitor not initialized")
                return False
                
            monitor = sovl_system.monitor
            
            if action == "start":
                monitor.start_monitoring()
                print("Monitoring started")
            elif action == "stop":
                monitor.stop_monitoring()
                print("Monitoring stopped")
            elif action == "status":
                print("\n--- Monitoring Status ---")
                print(f"Active: {'Yes' if monitor._monitor_thread and monitor._monitor_thread.is_alive() else 'No'}")
                print(f"Update Interval: {monitor.update_interval}s")
                enabled_metrics = monitor.config_manager.get("monitor.enabled_metrics", ["memory", "training", "curiosity"])
                print(f"Enabled Metrics: {', '.join(enabled_metrics)}")
                print("------------------------")
            elif action == "history":
                if not monitor._metrics_history:
                    print("No monitoring history available")
                    return False
                    
                print("\n--- Last 10 Metrics Snapshots ---")
                for i, metric in enumerate(list(monitor._metrics_history)[-10:]):
                    print(f"\nSnapshot {i+1}:")
                    if "memory" in metric:
                        print(f"  Memory: {metric['memory']['allocated_mb']:.1f}MB / {metric['memory']['total_memory_mb']:.1f}MB")
                    if "training" in metric:
                        print(f"  Training Progress: {metric['training']['progress']:.1f}%")
                    if "curiosity" in metric:
                        print(f"  Curiosity Score: {metric['curiosity']['score']:.1f}")
                print("\n------------------------")
            else:
                print(f"Invalid monitor command: {action}")
                print("Valid commands: start, stop, status, history")
                
        elif command == "train":
            epochs = TRAIN_EPOCHS
            dry_run = "--dry-run" in args
            non_flag_args = [arg for arg in args if arg != "--dry-run"]
            if non_flag_args:
                epochs = int(non_flag_args[0])
                
            print(f"Starting training for {epochs} epochs...")
            if dry_run and hasattr(sovl_system, 'enable_dry_run'):
                sovl_system.enable_dry_run()
                
            sovl_system.run_training_cycle(
                TRAIN_DATA,
                VALID_DATA,
                epochs=epochs,
                batch_size=BATCH_SIZE
            )
            
        elif command == "dream":
            print("Initiating dream cycle...")
            if hasattr(sovl_system, 'dream'):
                if sovl_system.dream():
                    print("Dream cycle completed successfully.")
                else:
                    print("Error: Dream cycle failed.")
            else:
                print("Error: Dream cycle not supported.")
                
        elif command == "generate":
            if not args:
                print("Error: 'generate' requires a prompt.")
                return False
            max_tokens = 60
            if args[-1].isdigit():
                max_tokens = int(args[-1])
                prompt = ' '.join(args[:-1])
            else:
                prompt = ' '.join(args)
                
            print(f"Generating response for: {prompt}...")
            response = sovl_system.generate(prompt, max_new_tokens=max_tokens)
            print(f"Response: {response}")
            
        elif command == "echo":
            if not args:
                print("Error: 'echo' requires text.")
                return False
            text = ' '.join(args)
            print(f"You said: '{text}'")
            
        elif command == "mimic":
            if not args:
                print("Error: Please provide text to mimic")
                return False
            text = ' '.join(args)
            print(f"Mimicking: '{text}'")
            response = sovl_system.mimic(text)
            print(f"Response: {response}")
            
        elif command == "memory":
            if not args:
                stats = sovl_system.get_memory_stats()
                print("\nMemory Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
            elif args[0].lower() in ["on", "off"]:
                enable = args[0].lower() == "on"
                if sovl_system.toggle_memory(enable):
                    print(f"Memory management {'enabled' if enable else 'disabled'}.")
                else:
                    print("Error: Failed to toggle memory management.")
            else:
                print("Error: Invalid memory command. Use 'memory' for stats or 'memory on/off' to toggle.")
                
        elif command == "recall":
            if not args:
                print("Error: 'recall' requires a query.")
                return False
            query = ' '.join(args)
            print(f"Recalling information related to: '{query}'...")
            response = sovl_system.recall(query)
            print(f"Response: {response}")
            
        elif command == "forget":
            if not args:
                print("Error: 'forget' requires a topic.")
                return False
            topic = ' '.join(args)
            print(f"Processing request to forget: '{topic}'...")
            response = sovl_system.forget(topic)
            print(f"Response: {response}")
            
        elif command == "recap":
            num_to_recap = int(args[0]) if args and args[0].isdigit() else 5
            print(f"Generating recap of last {num_to_recap} interactions...")
            response = sovl_system.recap(num_to_recap)
            print(f"Recap: {response}")
            
        elif command == "muse":
            print("Musing on a topic...")
            response = sovl_system.muse()
            print(f"Response: {response}")
            
        elif command == "flare":
            print("Triggering emotional flare...")
            response = sovl_system.flare()
            print(f"Response: {response}")
            
        elif command == "debate":
            if not args:
                print("Error: 'debate' requires a topic.")
                return False
            topic = ' '.join(args)
            print(f"Initiating debate on: '{topic}'")
            response = sovl_system.debate(topic)
            print(f"Response: {response}")
            
        elif command == "spark":
            print("Generating a curiosity-driven question...")
            response = sovl_system.spark()
            print(f"Response: {response}")
            
        elif command == "reflect":
            print("Reflecting on recent interactions...")
            response = sovl_system.reflect()
            print(f"Response: {response}")
            
        elif command == "log":
            num_entries = 5
            if args and args[0] == "view":
                num_entries = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
            print(f"\n--- Last {num_entries} Log Entries ---")
            logs = sovl_system.logger.read()[-num_entries:]
            for log in reversed(logs):
                print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log.get('timestamp', 0)))}")
                print(f"Event: {log.get('event', 'Interaction')}")
                print(f"Details: {log.get('details', 'N/A')}")
                print("-" * 20)
                
        elif command == "config":
            if not args:
                config = sovl_system.config_manager.get_state()
                print("\nCurrent Configuration:")
                for key, value in config.items():
                    print(f"{key}: {value}")
            elif len(args) == 1:
                key = args[0]
                value = sovl_system.config_manager.get(key)
                if value is not None:
                    print(f"{key}: {value}")
                else:
                    print(f"Configuration key '{key}' not found")
            elif len(args) == 2:
                key, new_value = args
                if sovl_system.config_manager.update(key, new_value):
                    print(f"Updated {key} to {new_value}")
                else:
                    print(f"Failed to update {key}")
            else:
                print("Invalid configuration command. Use 'config' to show all, 'config <key>' to get a value, or 'config <key> <value>' to set a value")
                
        elif command == "panic":
            print("Initiating emergency shutdown...")
            sovl_system.emergency_shutdown()
            return True
            
        elif command == "glitch":
            print("Simulating system glitch...")
            sovl_system.simulate_glitch()
            
        elif command == "tune":
            if not args:
                print("Error: Usage: tune <parameter> [value]")
                return False
            parameter, value_str = args[0].lower(), args[1] if len(args) > 1 else None
            if parameter == "cross_attention":
                weight = float(value_str) if value_str else None
                if weight is not None and not (0.0 <= weight <= 1.0):
                    print("Cross-attention weight must be between 0.0 and 1.0")
                    return False
                print(f"Setting cross-attention weight to {weight if weight else 'default'}...")
                if hasattr(sovl_system, 'tune_cross_attention'):
                    sovl_system.tune_cross_attention(weight=weight)
                    print("Cross-attention weight set.")
                else:
                    print("Error: 'tune_cross_attention' method not found.")
            else:
                print(f"Error: Unknown parameter '{parameter}'. Available: cross_attention")
                
        elif command == "rewind":
            print("Rewinding system state...")
            sovl_system.rewind()
            
        elif command == "history":
            num_entries = 10
            search_term = None
            if args:
                if args[0].isdigit():
                    num_entries = int(args[0])
                else:
                    search_term = ' '.join(args)
                    
            if hasattr(sovl_system, 'cmd_history'):
                entries = sovl_system.cmd_history.search(search_term) if search_term else sovl_system.cmd_history.get_last(num_entries)
                print(f"\n{'Commands matching' if search_term else 'Last'} {len(entries)} commands:")
                for entry in entries:
                    print(entry)
            else:
                print("Command history not available.")
                
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for available commands")
            
        return False
        
    except Exception as e:
        logger.log_error(
            error_msg=f"Error executing command {command}: {str(e)}",
            error_type="command_execution_error"
        )
        print(f"Error: {str(e)}")
        return False

def main():
    """Main entry point with enhanced argument parsing and error handling."""
    parser = argparse.ArgumentParser(
        description="Run the SOVL AI system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        default="sovl_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "generate", "dream"],
        help="Operation mode"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--train-data",
        help="Path to training data file"
    )
    parser.add_argument(
        "--valid-data",
        help="Path to validation data file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system in test mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="Monitoring update interval in seconds"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Register signal handlers
    register_signal_handlers()
    
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    logger.info("Starting SOVL system...")
    logger.info(f"Configuration: {args}")
    
    model = None
    context = None
    try:
        context = initialize_context(args)
        components = initialize_components(context)
        model = components[1]  # Extract model from components
        run_system(args, context, *components)
        logger.info("SOVL system completed successfully")
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to run SOVL system: {str(e)}",
            error_type="main_execution_error"
        )
        if hasattr(context, 'logger'):
            context.logger.log_error(
                error_msg=f"Failed to run SOVL system: {str(e)}",
                error_type="main_execution_error",
                stack_trace=traceback.format_exc(),
                additional_info={"config_path": args.config}
            )
        sys.exit(1)
    finally:
        cleanup_resources(context, model)

if __name__ == "__main__":
    main()
