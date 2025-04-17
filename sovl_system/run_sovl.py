import argparse
import os
import sys
import torch
import traceback
from sovl_main import SystemContext, SOVLSystem, ModelLoader, StateTracker, ErrorManager, MemoryMonitor, CuriosityEngine
from sovl_io import load_training_data, InsufficientDataError

def initialize_context(args):
    """Initialize system context with configuration and device validation."""
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found.")
        sys.exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA is not available. Please use --device cpu.")
        sys.exit(1)
    return SystemContext(config_path=args.config, device=args.device)

def initialize_components(context):
    """Initialize core SOVL components."""
    model_loader = ModelLoader(context)
    model = model_loader.load_model()
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
    return model_loader, model, state_tracker, error_manager, memory_monitor, curiosity_engine

def cleanup(context, model):
    """Release system resources."""
    context.logger.record_event(
        event_type="cleanup",
        message="Releasing system resources",
        level="info"
    )
    if model is not None:
        del model
    torch.cuda.empty_cache()

def run_system(args, context, model, model_loader, state_tracker, error_manager, memory_monitor, curiosity_engine):
    """Run the SOVL system in the specified mode."""
    sovl_system = SOVLSystem(
        context=context,
        config_handler=context.config_handler,
        model_loader=model_loader,
        curiosity_engine=curiosity_engine,
        memory_monitor=memory_monitor,
        state_tracker=state_tracker,
        error_manager=error_manager
    )
    sovl_system.toggle_memory(True)  # Enable memory management

    # Pre-flight check
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024**2
    if not memory_monitor.check_memory_health(model_size):
        print("Error: Insufficient memory to run the system.")
        sys.exit(1)

    if args.test:
        print("Running in test mode...")
        if not context.config_handler.validate():
            print("Error: Configuration validation failed.")
            sys.exit(1)
        print("Test mode completed successfully.")
        return

    if args.mode == "train":
        train_data = load_training_data(args.train_data) if args.train_data else None
        valid_data = load_training_data(args.valid_data) if args.valid_data else None
        results = sovl_system.curiosity_engine.run_training_cycle(
            train_data=train_data,
            valid_data=valid_data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        print(f"Training completed. Results: {results}")
        context.logger.record_event(
            event_type="training_completed",
            message="Training cycle completed",
            level="info",
            additional_info={"results": results}
        )
    elif args.mode == "generate":
        question = sovl_system.generate_curiosity_question()
        print(f"Generated question: {question}")
    elif args.mode == "dream":
        success = sovl_system.dream()
        print(f"Dream cycle {'succeeded' if success else 'failed'}")

def main():
    """Initialize and run the SOVL AI system with specified configuration and mode."""
    parser = argparse.ArgumentParser(description="Run the SOVL AI system")
    parser.add_argument("--config", default="sovl_config.json", help="Path to configuration file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--mode", default="train", choices=["train", "generate", "dream"], help="Operation mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--train-data", help="Path to training data file")
    parser.add_argument("--valid-data", help="Path to validation data file")
    parser.add_argument("--test", action="store_true", help="Run system in test mode")
    args = parser.parse_args()

    model = None
    try:
        context = initialize_context(args)
        components = initialize_components(context)
        model = components[1]  # Extract model from components
        run_system(args, context, *components)
    except Exception as e:
        context.logger.log_error(
            error_msg=f"Failed to run SOVL system: {str(e)}",
            error_type="main_execution_error",
            stack_trace=traceback.format_exc(),
            additional_info={"config_path": args.config}
        )
        print(f"Error: Failed to run SOVL system. Check logs for details.")
        sys.exit(1)
    finally:
        cleanup(context, model)

if __name__ == "__main__":
    main()
