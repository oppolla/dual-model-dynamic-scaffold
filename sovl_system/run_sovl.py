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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sovl_main import SystemContext, SOVLSystem, ModelLoader, StateTracker, ErrorManager, MemoryMonitor, CuriosityEngine
from sovl_io import load_training_data, InsufficientDataError
from sovl_monitor import SystemMonitor
from sovl_cli import CommandHandler, run_cli
from sovl_utils import (
    safe_compare, memory_usage, log_memory_usage, dynamic_batch_size,
    detect_repetitions, adjust_temperature, synchronized,
    validate_components, sync_component_states, validate_component_states,
    initialize_component_state
)
from sovl_logger import Logger, LoggerConfig
from sovl_config import ConfigManager
from sovl_conductor import SOVLOrchestrator
from sovl_memory import MemoryManager

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

class SOVLRunner:
    """Main class to manage SOVL system execution with improved separation of concerns."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.context = None
        self.model = None
        self.components = None
        self.orchestrator = None
        
    @staticmethod
    def _setup_logger() -> Logger:
        """Configure and return logger instance."""
        logger_config = LoggerConfig(
            log_file=f'sovl_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            max_size_mb=10,
            compress_old=True,
            max_in_memory_logs=1000,
            rotation_count=5
        )
        return Logger(logger_config)
    
    @staticmethod
    def _handle_signal(signum: int, frame: Any, logger: Logger, cleanup_fn: callable):
        """Handle system signals for graceful shutdown."""
        logger.log_event(
            event_type="signal_received",
            message=f"Received signal {signum}, initiating graceful shutdown...",
            level="info"
        )
        cleanup_fn()
        sys.exit(0)
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal_handler = lambda signum, frame: self._handle_signal(
            signum, frame, self.logger, self.cleanup
        )
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @staticmethod
    def _validate_config_file(config_path: str, logger: Logger) -> bool:
        """Validate configuration file format and required fields."""
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
                    
            if 'model_path' not in config['model']:
                logger.log_error(
                    error_msg="Missing required model configuration: model_path",
                    error_type="config_validation_error"
                )
                return False
                
            monitor_config = config.get('monitor', {})
            if not isinstance(monitor_config.get('update_interval', 1.0), (int, float)):
                logger.log_error(
                    error_msg="Invalid monitor.update_interval in configuration",
                    error_type="config_validation_error"
                )
                return False
                
            return True
        except json.JSONDecodeError as e:
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
    
    def _initialize_context(self, args: argparse.Namespace) -> SystemContext:
        """Initialize system context with validation and error handling."""
        if not os.path.exists(args.config):
            self.logger.log_error(
                error_msg=f"Configuration file not found: {args.config}",
                error_type="config_validation_error"
            )
            sys.exit(1)
            
        if not self._validate_config_file(args.config, self.logger):
            self.logger.log_error(
                error_msg="Configuration validation failed",
                error_type="config_validation_error"
            )
            sys.exit(1)
            
        config_manager = ConfigManager(args.config, self.logger)
        
        if args.device == "cuda" and not torch.cuda.is_available():
            self.logger.log_error(
                error_msg="CUDA is not available. Please use --device cpu",
                error_type="device_validation_error"
            )
            sys.exit(1)
            
        self.logger.log_event(
            event_type="device_selected",
            message=f"Using {'CUDA device: ' + torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU device'}",
            level="info"
        )
        
        Path("output").mkdir(exist_ok=True)
        
        return SystemContext(
            config_path=args.config,
            device=args.device,
            config_manager=config_manager
        )
    
    def _initialize_components(self, context: SystemContext) -> Tuple:
        """Initialize core SOVL components with progress tracking."""
        components = []
        component_classes = [
            (ModelLoader, "model loader"),
            (None, "model"),  # Model is loaded separately
            (StateTracker, "state tracker"),
            (ErrorManager, "error manager"),
            (MemoryMonitor, "memory monitor"),
            (CuriosityEngine, "curiosity engine"),
            (MemoryManager, "memory manager")
        ]
        
        for component_class, name in component_classes:
            if component_class is None:  # Handle model loading
                self.logger.log_event(
                    event_type="component_initialization",
                    message="Loading model...",
                    level="info"
                )
                model = components[0].load_model()
                components.append(model)
            else:
                self.logger.log_event(
                    event_type="component_initialization",
                    message=f"Initializing {name}...",
                    level="info"
                )
                if name == "curiosity engine":
                    component = component_class(
                        config_handler=context.config_handler,
                        model_loader=components[0],
                        state_tracker=components[2],
                        error_manager=components[3],
                        logger=context.logger,
                        device=context.device
                    )
                elif name == "memory manager":
                    component = component_class(
                        context.config_manager,
                        context.device,
                        context.logger
                    )
                else:
                    component = component_class(context)
                components.append(component)
        
        validate_components(*components)
        initialize_component_state(components[2], components)  # StateTracker is components[2]
        
        self.logger.log_event(
            event_type="component_initialization",
            message="All components initialized successfully",
            level="info"
        )
        return tuple(components)
    
    def cleanup(self):
        """Release system resources with logging and error handling."""
        try:
            self.logger.log_event(
                event_type="cleanup",
                message="Starting cleanup...",
                level="info"
            )
            
            if self.model:
                self.logger.log_event(
                    event_type="cleanup",
                    message="Cleaning up model...",
                    level="info"
                )
                del self.model
                self.model = None
                
            if torch.cuda.is_available():
                self.logger.log_event(
                    event_type="cleanup",
                    message="Clearing CUDA cache...",
                    level="info"
                )
                torch.cuda.empty_cache()
                
            if self.context:
                self.logger.log_event(
                    event_type="cleanup",
                    message="Cleaning up context...",
                    level="info"
                )
                self.context.cleanup()
                self.context = None
                
            self.logger.log_event(
                event_type="cleanup",
                message="Cleanup completed successfully",
                level="info"
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during cleanup: {str(e)}",
                error_type="cleanup_error"
            )
    
    def _run_system(self, args: argparse.Namespace):
        """Run the SOVL system with monitoring and error handling."""
        try:
            self.logger.log_event(
                event_type="system_start",
                message="Initializing SOVL system...",
                level="info"
            )
            
            self.orchestrator = SOVLOrchestrator(
                config_path=args.config,
                log_file=self.logger.config.log_file
            )
            self.orchestrator.initialize_system()
            self.orchestrator.run()
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during system execution: {str(e)}",
                error_type="system_execution_error"
            )
            raise
    
    def execute_command(self, sovl_system: SOVLSystem, command: str, args: List[str] = None) -> bool:
        """Execute a command with proper error handling and logging."""
        try:
            args = args or []
            cmd_handler = CommandHandler(sovl_system, self.logger)
            return cmd_handler.handle_command(command, args)
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error executing command {command}: {str(e)}",
                error_type="command_execution_error"
            )
            print(f"Error: {str(e)}")
            return False
    
    def run(self):
        """Main execution method with enhanced argument parsing."""
        parser = argparse.ArgumentParser(
            description="Run the SOVL AI system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument("--config", default="sovl_config.json", help="Path to configuration file")
        parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
        parser.add_argument("--mode", default="train", choices=["train", "generate", "dream"], help="Operation mode")
        parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
        parser.add_argument("--train-data", help="Path to training data file")
        parser.add_argument("--valid-data", help="Path to validation data file")
        parser.add_argument("--test", action="store_true", help="Run system in test mode")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        parser.add_argument("--monitor-interval", type=float, default=1.0, help="Monitoring update interval in seconds")
        
        args = parser.parse_args()
        
        if args.verbose:
            self.logger.setLevel(logging.DEBUG)
            
        self._register_signal_handlers()
        atexit.register(self.cleanup)
        
        self.logger.info("Starting SOVL system...")
        self.logger.info(f"Configuration: {args}")
        
        try:
            self.context = self._initialize_context(args)
            self.components = self._initialize_components(self.context)
            self.model = self.components[1]
            self._run_system(args)
            self.logger.info("SOVL system completed successfully")
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to run SOVL system: {str(e)}",
                error_type="main_execution_error",
                stack_trace=traceback.format_exc(),
                additional_info={"config_path": args.config}
            )
            sys.exit(1)
        finally:
            self.cleanup()

def main():
    """Entry point for the SOVL system."""
    runner = SOVLRunner()
    runner.run()

if __name__ == "__main__":
    main()
