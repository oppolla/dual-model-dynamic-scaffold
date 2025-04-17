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
from sovl_state import StateManager, SOVLState
from sovl_error import ErrorManager, ErrorContext

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_DATA = None
VALID_DATA = None
CHECKPOINT_INTERVAL = 1  # Save checkpoint every epoch by default
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
        self.state_manager = None
        self.error_manager = None
        self.last_checkpoint_time = None
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        self.metrics_history = []
        self.best_validation_loss = float('inf')
        self.patience = 0
        self.max_patience = 3
        
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
                if name == "error manager":
                    component = ErrorManager(
                        context=context,
                        state_tracker=components[2],
                        config_manager=context.config_manager,
                        error_cooldown=1.0,
                        max_recent_errors=100
                    )
                    self.error_manager = component
                elif name == "curiosity engine":
                    component = component_class(
                        config_handler=context.config_handler,
                        model_loader=components[0],
                        state_tracker=components[2],
                        error_manager=self.error_manager,
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
        initialize_component_state(components[2], components)
        
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
    
    def save_checkpoint(self, force: bool = False, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Save system state checkpoint.
        
        Args:
            force: If True, save checkpoint regardless of interval
            optimizer: Optional optimizer to save its state
            
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
            model_path = None
            if self.model is not None:
                model_path = Path("checkpoints") / f"model_{int(current_time)}.pt"
                model_path.parent.mkdir(exist_ok=True)
                torch.save(self.model.state_dict(), model_path)
                
            # Save system state
            state_path = Path("checkpoints") / f"state_{int(current_time)}.json"
            state_data = {
                "timestamp": current_time,
                "model_path": str(model_path) if model_path else None,
                "components": {i: component.to_dict() for i, component in enumerate(self.components)},
                "optimizer_state": optimizer.state_dict() if optimizer else None,
                "best_validation_loss": self.best_validation_loss,
                "patience": self.patience,
                "metrics_history": self.metrics_history
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
            
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Load system state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to load its state
            
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
            for idx, component_data in state_data["components"].items():
                idx = int(idx)  # JSON keys are strings
                if idx < len(self.components):
                    self.components[idx].from_dict(component_data)
                    
            # Load optimizer state
            if optimizer and state_data.get("optimizer_state"):
                optimizer.load_state_dict(state_data["optimizer_state"])
                
            # Load training metadata
            self.best_validation_loss = state_data.get("best_validation_loss", float('inf'))
            self.patience = state_data.get("patience", 0)
            self.metrics_history = state_data.get("metrics_history", [])
                    
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
    
    def cleanup_old_checkpoints(self, max_checkpoints: int = 5):
        """Remove old checkpoints to manage disk space."""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoints = sorted(checkpoint_dir.glob("state_*.json"), key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()
                model_path = checkpoint_dir / f"model_{old_checkpoint.stem.split('_')[1]}.pt"
                if model_path.exists():
                    model_path.unlink()
                self.logger.log_event(
                    event_type="checkpoint_cleanup",
                    message=f"Removed old checkpoint: {old_checkpoint}",
                    level="info"
                )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to clean up old checkpoints: {str(e)}",
                error_type="checkpoint_cleanup_error"
            )

    def _run_validation(self, valid_data: List[Dict[str, Any]], batch_size: int) -> Dict[str, float]:
        """Run validation loop and compute metrics."""
        try:
            self.logger.log_event(
                event_type="validation_start",
                message="Starting validation...",
                level="info"
            )
            
            self.model.eval()
            total_loss = 0.0
            total_batches = 0
            metrics = {metric: 0.0 for metric in self.context.config_manager.get("training.metrics_to_track", ["loss", "accuracy", "confidence"])}
            
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    try:
                        batch = valid_data[i:i + batch_size]
                        if not batch:
                            continue
                            
                        # Prepare batch
                        inputs = self._prepare_batch(batch)
                        
                        # Forward pass
                        outputs = self.model(**inputs)
                        loss = self._calculate_loss(outputs, inputs)
                        
                        # Update metrics
                        total_loss += loss.item()
                        total_batches += 1
                        
                        # Calculate additional metrics
                        if "accuracy" in metrics:
                            preds = outputs.logits.argmax(dim=-1)
                            mask = inputs["labels"] != -100
                            correct = (preds[mask] == inputs["labels"][mask]).sum().item()
                            total = mask.sum().item()
                            metrics["accuracy"] += correct / total if total > 0 else 0.0
                            
                        if "perplexity" in metrics:
                            metrics["perplexity"] += torch.exp(loss).item()
                            
                        if "confidence" in metrics:
                            probs = torch.softmax(outputs.logits, dim=-1)
                            max_probs = probs.max(dim=-1)[0]
                            metrics["confidence"] += max_probs[mask].mean().item()
                            
                    except Exception as e:
                        error_context = ErrorContext(
                            error_type="validation_error",
                            error_message=str(e),
                            stack_trace=traceback.format_exc(),
                            timestamp=datetime.now(),
                            additional_info={
                                "batch_index": i,
                                "batch_size": batch_size,
                                "total_batches": total_batches
                            }
                        )
                        self.error_manager.handle_error(
                            error=e,
                            error_type="validation",
                            context=error_context
                        )
                        continue
            
            # Average metrics
            if total_batches > 0:
                metrics["loss"] = total_loss / total_batches
                for metric in metrics:
                    if metric != "loss":
                        metrics[metric] /= total_batches
            
            # Log validation results
            self.logger.log_event(
                event_type="validation_complete",
                message="Validation completed",
                level="info",
                additional_info={
                    "metrics": metrics,
                    "total_batches": total_batches
                }
            )
            
            return metrics
            
        except Exception as e:
            error_context = ErrorContext(
                error_type="critical_validation_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                timestamp=datetime.now(),
                additional_info={
                    "total_batches": total_batches,
                    "metrics": metrics
                }
            )
            self.error_manager.handle_error(
                error=e,
                error_type="validation",
                context=error_context
            )
            raise

    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for model input."""
        try:
            # Tokenize and prepare inputs
            texts = [item["text"] for item in batch]
            inputs = self.context.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.context.config_manager.get("training.max_seq_length", 512),
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.context.device) for k, v in inputs.items()}
            
            # Prepare labels
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][inputs["labels"] == self.context.tokenizer.pad_token_id] = -100
            
            return inputs
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Batch preparation error: {str(e)}",
                error_type="batch_preparation_error"
            )
            raise

    def _calculate_loss(self, outputs: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for the model outputs."""
        try:
            return torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                inputs["labels"].view(-1),
                ignore_index=-100
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Loss calculation error: {str(e)}",
                error_type="loss_calculation_error"
            )
            raise

    def _update_metrics_history(self, metrics: Dict[str, float], epoch: int) -> None:
        """Update metrics history and handle early stopping."""
        try:
            # Add metrics to history
            self.metrics_history.append({
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            # Check for early stopping
            if metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = metrics["loss"]
                self.patience = 0
            else:
                self.patience += 1
                
            # Log metrics update
            self.logger.log_event(
                event_type="metrics_update",
                message=f"Metrics updated for epoch {epoch}",
                level="info",
                additional_info={
                    "metrics": metrics,
                    "best_validation_loss": self.best_validation_loss,
                    "patience": self.patience
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Metrics update error: {str(e)}",
                error_type="metrics_update_error"
            )
            raise

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
        parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL, 
                          help="Checkpoint interval in epochs")
        parser.add_argument("--resume-from-checkpoint", help="Path to checkpoint file to resume from")
        parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N epochs")
        parser.add_argument("--max-patience", type=int, default=3, help="Max epochs without validation improvement")
        parser.add_argument("--max-checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
        
        args = parser.parse_args()
        
        if args.verbose:
            self.logger.setLevel(logging.DEBUG)
            
        self.checkpoint_interval = args.checkpoint_interval
        self.max_patience = args.max_patience
            
        self._register_signal_handlers()
        atexit.register(self.cleanup)
        
        self.logger.info("Starting SOVL system...")
        self.logger.info(f"Configuration: {args}")
        
        try:
            self.context = self._initialize_context(args)
            self.components = self._initialize_components(self.context)
            self.model = self.components[1]
            
            # Initialize state manager
            self.state_manager = StateManager(
                self.context.config_manager,
                self.logger,
                self.context.device
            )
            self.state_manager.initialize_state()
            
            # Load checkpoint if specified
            optimizer = None  # Note: Optimizer should be initialized in SOVLOrchestrator or passed
            if args.resume_from_checkpoint:
                if not self.load_checkpoint(args.resume_from_checkpoint, optimizer):
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
                        epochs=1,
                        batch_size=args.batch_size,
                        train_data=train_data,
                        valid_data=valid_data,
                        checkpoint_callback=lambda: self.save_checkpoint(optimizer=optimizer),
                        validate_every=args.validate_every
                    )
                    
                    # Validation phase
                    if valid_data and (epoch + 1) % args.validate_every == 0:
                        valid_loss, metrics = self.orchestrator.validate(valid_data)
                        self._update_metrics_history(metrics, epoch + 1)
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
                        if self.patience >= self.max_patience:
                            self.logger.log_event(
                                event_type="early_stopping",
                                message=f"Early stopping triggered after {self.patience} epochs without improvement",
                                level="info"
                            )
                            break
                    
                    # Save checkpoint and clean up old ones
                    self.save_checkpoint(optimizer=optimizer)
                    self.cleanup_old_checkpoints(args.max_checkpoints)
                    
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
            self.save_checkpoint(force=True, optimizer=optimizer)
            self.cleanup_old_checkpoints(args.max_checkpoints)
            self.cleanup()

def main():
    """Entry point for the SOVL system."""
    runner = SOVLRunner()
    runner.run()

if __name__ == "__main__":
    main()
