import argparse
import os
import sys
import torch
import traceback
import logging
import json
import signal
import atexit
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sovl_main import SystemContext, SOVLSystem, ModelLoader, StateTracker, ErrorManager, MemoryMonitor, CuriosityEngine
from sovl_io import load_training_data, InsufficientDataError
from sovl_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'sovl_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
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
                logger.error(f"Missing required configuration section: {section}")
                return False
                
        # Validate model configuration
        if 'model_path' not in config['model']:
            logger.error("Missing required model configuration: model_path")
            return False
            
        # Validate monitor configuration
        monitor_config = config.get('monitor', {})
        if not isinstance(monitor_config.get('update_interval', 1.0), (int, float)):
            logger.error("Invalid monitor.update_interval in configuration")
            return False
            
        return True
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in configuration file: {config_path}")
        return False
    except Exception as e:
        logger.error(f"Error validating configuration file: {str(e)}")
        return False

def initialize_context(args) -> SystemContext:
    """Initialize system context with enhanced validation and error handling."""
    try:
        # Validate configuration file
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
            
        if not validate_config_file(args.config):
            logger.error("Configuration validation failed")
            sys.exit(1)
            
        # Validate device
        if args.device == "cuda":
            if not torch.cuda.is_available():
                logger.error("CUDA is not available. Please use --device cpu")
                sys.exit(1)
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU device")
            
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        return SystemContext(config_path=args.config, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize context: {str(e)}")
        sys.exit(1)

def initialize_components(context) -> tuple:
    """Initialize core SOVL components with enhanced error handling and progress tracking."""
    try:
        # Initialize model loader
        logger.info("Initializing model loader...")
        model_loader = ModelLoader(context)
        
        # Load model
        logger.info("Loading model...")
        model = model_loader.load_model()
        
        # Initialize state tracker
        logger.info("Initializing state tracker...")
        state_tracker = StateTracker(context)
        
        # Initialize error manager
        logger.info("Initializing error manager...")
        error_manager = ErrorManager(context, state_tracker)
        
        # Initialize memory monitor
        logger.info("Initializing memory monitor...")
        memory_monitor = MemoryMonitor(context)
        
        # Initialize curiosity engine
        logger.info("Initializing curiosity engine...")
        curiosity_engine = CuriosityEngine(
            config_handler=context.config_handler,
            model_loader=model_loader,
            state_tracker=state_tracker,
            error_manager=error_manager,
            logger=context.logger,
            device=context.device
        )
        
        logger.info("All components initialized successfully")
        return model_loader, model, state_tracker, error_manager, memory_monitor, curiosity_engine
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

def cleanup_resources(context=None, model=None):
    """Release system resources with enhanced logging and error handling."""
    try:
        logger.info("Starting cleanup...")
        
        if model is not None:
            logger.info("Cleaning up model...")
            del model
            
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
        if context is not None:
            logger.info("Cleaning up context...")
            context.cleanup()
            
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def run_system(args, context, model, model_loader, state_tracker, error_manager, memory_monitor, curiosity_engine):
    """Run the SOVL system with enhanced monitoring and error handling."""
    try:
        logger.info("Initializing SOVL system...")
        sovl_system = SOVLSystem(
            context=context,
            config_handler=context.config_handler,
            model_loader=model_loader,
            curiosity_engine=curiosity_engine,
            memory_monitor=memory_monitor,
            state_tracker=state_tracker,
            error_manager=error_manager
        )
        
        logger.info("Enabling memory management...")
        sovl_system.toggle_memory(True)

        # Initialize system monitor
        logger.info("Initializing system monitor...")
        system_monitor = SystemMonitor(
            memory_manager=memory_monitor,
            training_manager=curiosity_engine.training_manager,
            curiosity_manager=curiosity_engine.curiosity_manager,
            config_manager=context.config_handler,
            logger=context.logger
        )
        system_monitor.start_monitoring()

        # Pre-flight checks
        logger.info("Running pre-flight checks...")
        model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**2
        logger.info(f"Model size: {model_size:.2f} MB")
        
        if not memory_monitor.check_memory_health(model_size):
            logger.error("Insufficient memory to run the system")
            sys.exit(1)

        if args.test:
            logger.info("Running in test mode...")
            if not context.config_handler.validate():
                logger.error("Configuration validation failed")
                sys.exit(1)
            logger.info("Test mode completed successfully")
            return

        # Run the selected mode
        if args.mode == "train":
            logger.info("Starting training mode...")
            train_data = load_training_data(args.train_data) if args.train_data else None
            valid_data = load_training_data(args.valid_data) if args.valid_data else None
            
            logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
            results = sovl_system.curiosity_engine.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            logger.info(f"Training completed. Results: {results}")
            
        elif args.mode == "generate":
            logger.info("Starting generation mode...")
            question = sovl_system.generate_curiosity_question()
            logger.info(f"Generated question: {question}")
            
        elif args.mode == "dream":
            logger.info("Starting dream mode...")
            success = sovl_system.dream()
            logger.info(f"Dream cycle {'succeeded' if success else 'failed'}")
            
    except Exception as e:
        logger.error(f"Error during system execution: {str(e)}")
        raise
    finally:
        # Stop monitoring when done
        if 'system_monitor' in locals():
            system_monitor.stop_monitoring()

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
        logger.error(f"Failed to run SOVL system: {str(e)}")
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
