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
from sovl_trainer import SOVLTrainer, TrainingCycleManager

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
        self.last_checkpoint_time = None
        self.checkpoint_interval = 3600  # 1 hour default interval
        
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
    
    def save_checkpoint(self, force: bool = False) -> bool:
        """Save system state checkpoint.
        
        Args:
            force: If True, save checkpoint regardless of interval
            
        Returns:
            bool: True if checkpoint was saved, False otherwise
        """
        current_time = time.time()
        if not force and self.last_checkpoint_time is not None:
            if current_time - self.last_checkpoint_time < self.checkpoint_interval:
                return False
                
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message="Saving system checkpoint...",
                level="info"
            )
            
            # Save model state
            if self.model is not None:
                model_path = Path("checkpoints") / f"model_{int(current_time)}.pt"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                
            # Save system state
            state_path = Path("checkpoints") / f"state_{int(current_time)}.json"
            state_data = {
                "timestamp": current_time,
                "model_path": str(model_path) if self.model is not None else None,
                "components": {name: component.to_dict() for name, component in self.components.items()}
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
            self.last_checkpoint_time = current_time
            self.logger.log_event(
                event_type="checkpoint",
                message=f"Checkpoint saved successfully to {state_path}",
                level="info"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to save checkpoint: {str(e)}",
                error_type="checkpoint_error"
            )
            return False
            
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load system state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message=f"Loading checkpoint from {checkpoint_path}...",
                level="info"
            )
            
            with open(checkpoint_path, 'r') as f:
                state_data = json.load(f)
                
            # Load model state
            if state_data["model_path"] is not None:
                model_path = Path(state_data["model_path"])
                if model_path.exists():
                    self.model.load_state_dict(torch.load(model_path))
                    
            # Load component states
            for name, component_data in state_data["components"].items():
                if name in self.components:
                    self.components[name].from_dict(component_data)
                    
            self.last_checkpoint_time = state_data["timestamp"]
            self.logger.log_event(
                event_type="checkpoint",
                message="Checkpoint loaded successfully",
                level="info"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to load checkpoint: {str(e)}",
                error_type="checkpoint_error"
            )
            return False
    
    def run(self):
        """Main execution method with enhanced argument parsing."""
        try:
            # Parse command line arguments
            parser = argparse.ArgumentParser(description='Run SOVL system')
            parser.add_argument('--config', type=str, default='config.json',
                              help='Path to configuration file')
            parser.add_argument('--device', type=str, default='cuda',
                              help='Device to run on (cuda or cpu)')
            parser.add_argument('--mode', type=str, default='train',
                              help='Operation mode (train, generate, dream)')
            parser.add_argument('--epochs', type=int, default=10,
                              help='Number of training epochs')
            parser.add_argument('--batch_size', type=int, default=32,
                              help='Training batch size')
            parser.add_argument('--train_data', type=str,
                              help='Path to training data')
            parser.add_argument('--valid_data', type=str,
                              help='Path to validation data')
            parser.add_argument('--checkpoint', type=str,
                              help='Path to checkpoint file to load')
            parser.add_argument('--checkpoint_interval', type=int, default=3600,
                              help='Checkpoint interval in seconds')
            parser.add_argument('--validate_every', type=int, default=100,
                              help='Validate every N steps')
            args = parser.parse_args()
            
            # Set checkpoint interval
            self.checkpoint_interval = args.checkpoint_interval
            
            # Initialize system
            self.context = self._initialize_context(args)
            self.components = self._initialize_components(self.context)
            self.model = self.components[1]
            
            # Load checkpoint if specified
            if args.checkpoint:
                if not self.load_checkpoint(args.checkpoint):
                    self.logger.log_error(
                        error_msg="Failed to load checkpoint, starting fresh",
                        error_type="checkpoint_error"
                    )
            
            # Initialize orchestrator
            self.orchestrator = SOVLOrchestrator(
                model=self.model,
                components=self.components,
                context=self.context
            )
            
            # Run system
            if args.mode == 'train':
                self.logger.log_event(
                    event_type="training",
                    message="Starting training...",
                    level="info"
                )
                
                # Load training and validation data
                train_data = load_training_data(args.train_data) if args.train_data else []
                valid_data = load_training_data(args.valid_data) if args.valid_data else []
                
                if not train_data:
                    raise ValueError("No training data provided")
                
                # Training loop with validation
                for epoch in range(args.epochs):
                    self.logger.log_event(
                        event_type="epoch_start",
                        message=f"Starting epoch {epoch + 1}/{args.epochs}",
                        level="info"
                    )
                    
                    # Training phase
                    train_loss = self.orchestrator.train(
                        epochs=1,  # Train one epoch at a time
                        batch_size=args.batch_size,
                        train_data=train_data,
                        valid_data=valid_data,
                        checkpoint_callback=self.save_checkpoint,
                        validate_every=args.validate_every
                    )
                    
                    # Validation phase
                    if valid_data:
                        valid_loss, metrics = self.orchestrator.validate(valid_data)
                        self.logger.log_event(
                            event_type="validation",
                            message=f"Epoch {epoch + 1} validation results",
                            level="info",
                            additional_info={
                                "train_loss": train_loss,
                                "valid_loss": valid_loss,
                                "metrics": metrics
                            }
                        )
                    
                    # Save checkpoint if needed
                    self.save_checkpoint()
                    
            elif args.mode == 'generate':
                self.logger.log_event(
                    event_type="generation",
                    message="Starting generation...",
                    level="info"
                )
                self.orchestrator.generate()
            elif args.mode == 'dream':
                self.logger.log_event(
                    event_type="dreaming",
                    message="Starting dreaming...",
                    level="info"
                )
                self.orchestrator.dream()
            else:
                raise ValueError(f"Invalid mode: {args.mode}")
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"System error: {str(e)}",
                error_type="system_error"
            )
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint(force=True)
            self.cleanup()

def main():
    """Entry point for the SOVL system."""
    runner = SOVLRunner()
    runner.run()

if __name__ == "__main__":
    main()
