import time
import torch
import traceback
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
from contextlib import contextlib
import functools
from sovl_system import SOVLSystem
from sovl_config import ConfigManager
from sovl_utils import safe_compare

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_DATA = None
VALID_DATA = None

COMMAND_CATEGORIES = {
    "System": ["quit", "exit", "save", "load", "reset", "status", "help"],
    "Training": ["train", "dream"],
    "Generation": ["generate", "echo", "mimic"],
    "Memory": ["memory", "recall", "forget", "recap"],
    "Interaction": ["muse", "flare", "debate", "spark", "reflect"],
    "Debug": ["log", "config", "panic", "glitch"],
    "Advanced": ["tune", "rewind"],
    "History": ["history"]
}

class CommandHistory:
    def __init__(self, max_size: int = 100):
        self.history: List[Tuple[float, str, Optional[str]]] = []
        self.max_size = max_size

    def add(self, command: str, result: Optional[str] = None):
        self.history.append((time.time(), command, result))
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def get_last(self, n: int = 1) -> List[Tuple[float, str, Optional[str]]]:
        return self.history[-n:]

    def search(self, term: str) -> List[Tuple[float, str, Optional[str]]]:
        return [entry for entry in self.history if term.lower() in entry[1].lower()]

    def format_entry(self, entry: Tuple[float, str, Optional[str]]) -> str:
        timestamp, cmd, result = entry
        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        status = "✓" if result and "error" not in result.lower() else "✗"
        return f"{time_str} [{status}] {cmd}"

class CommandHandler:
    """Centralized command handling logic."""
    def __init__(self, sovl_system: SOVLSystem):
        self.system = sovl_system
        self.commands: Dict[str, Callable] = {
            'quit': self.cmd_quit, 'exit': self.cmd_quit,
            'train': self.cmd_train, 'generate': self.cmd_generate,
            'save': self.cmd_save, 'load': self.cmd_load,
            'dream': self.cmd_dream, 'tune': self.cmd_tune,
            'memory': self.cmd_memory, 'status': self.cmd_status,
            'log': self.cmd_log, 'config': self.cmd_config,
            'reset': self.cmd_reset, 'spark': self.cmd_spark,
            'reflect': self.cmd_reflect, 'muse': self.cmd_muse,
            'flare': self.cmd_flare, 'echo': self.cmd_echo,
            'debate': self.cmd_debate, 'glitch': self.cmd_glitch,
            'rewind': self.cmd_rewind, 'mimic': self.cmd_mimic,
            'panic': self.cmd_panic, 'recap': self.cmd_recap,
            'recall': self.cmd_recall, 'forget': self.cmd_forget,
            'history': self.cmd_history, 'help': self.cmd_help
        }

    def execute(self, cmd: str, args: List[str]) -> bool:
        if cmd not in self.commands:
            print(f"Unknown command '{cmd}'. Type 'help' for commands.")
            return False
        return self.commands[cmd](args)

    @staticmethod
    def parse_args(parts: List[str], min_args: int = 1, max_args: Optional[int] = None) -> Tuple[str, List[str]]:
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        if len(args) < (min_args - 1):
            raise ValueError(f"Error: {cmd} requires at least {min_args - 1} argument(s).")
        if max_args and len(args) > (max_args - 1):
            raise ValueError(f"Error: {cmd} takes at most {max_args - 1} argument(s).")
        return cmd, args

    def generate_response(self, prompt: str, max_tokens: int = 60, temp_adjust: float = 0.0) -> str:
        """Generate a response with safe temperature bounds and error handling."""
        try:
            # Get base temperature with fallback
            base_temp = getattr(self.system, 'base_temperature', 0.7)
            
            # Calculate and clamp temperature
            raw_temp = base_temp + temp_adjust
            temperature = max(0.1, min(2.0, raw_temp))
            
            # Log temperature adjustment if it was clamped
            if raw_temp != temperature:
                self.system.logger.record_event(
                    event_type="temperature_clamped",
                    message="Temperature value was clamped to safe range",
                    level="warning",
                    additional_info={
                        "base_temperature": base_temp,
                        "temp_adjust": temp_adjust,
                        "raw_temperature": raw_temp,
                        "clamped_temperature": temperature
                    }
                )
            
            # Generate response with safe parameters
            response = self.system.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                do_sample=True
            )
            
            # Log successful generation
            self.system.logger.record_event(
                event_type="response_generated",
                message="Response generated successfully",
                level="info",
                additional_info={
                    "prompt_length": len(prompt),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_length": len(response)
                }
            )
            
            return response
        
        except Exception as e:
            # Handle generation errors
            self.system.error_manager.handle_generation_error(e, temperature)
            self.system.logger.log_error(
                error_msg="Error during response generation",
                error_type="generation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "prompt": prompt[:200],  # Truncate for logging
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            raise

    def log_action(self, prompt: str, response: str, confidence: float, is_system: bool = False, extra_attrs: Optional[Dict] = None):
        """Log an action with standardized format."""
        if not hasattr(self.system, 'history') or not hasattr(self.system, 'logger'):
            print("Warning: Missing 'history' or 'logger'. Cannot log action.")
            return
        
        self.system.logger.record_event(
            event_type="cli_action",
            message="Command action executed",
            level="info",
            additional_info={
                "prompt": prompt,
                "response": response,
                "conversation_id": getattr(self.system.history, 'conversation_id', 'N/A'),
                "confidence_score": confidence,
                "is_system_question": is_system,
                **(extra_attrs or {})
            }
        )

    # Command implementations
    def cmd_quit(self, args: List[str]) -> bool:
        print("Exiting...")
        return True

    def cmd_train(self, args: List[str]) -> bool:
        try:
            epochs = TRAIN_EPOCHS
            dry_run = "--dry-run" in args
            non_flag_args = [arg for arg in args if arg != "--dry-run"]
            if non_flag_args:
                epochs = int(non_flag_args[0])

            self.system.logger.record_event(
                event_type="cli_train_start",
                message="Starting training cycle",
                level="info",
                additional_info={
                    "epochs": epochs,
                    "dry_run": dry_run,
                    "timestamp": time.time()
                }
            )

            if dry_run and hasattr(self.system, 'enable_dry_run'):
                self.system.enable_dry_run()

            self.system.run_training_cycle(
                TRAIN_DATA,
                VALID_DATA,
                epochs=epochs,
                batch_size=BATCH_SIZE
            )

            self.system.logger.record_event(
                event_type="cli_train_complete",
                message="Training cycle completed successfully",
                level="info",
                additional_info={
                    "epochs": epochs,
                    "timestamp": time.time()
                }
            )
            return False
        except Exception as e:
            # Use ErrorManager for robust error handling
            self.system.error_manager.handle_training_error(e, BATCH_SIZE)
            self.system.logger.record_event(
                event_type="cli_train_error",
                message="Training error occurred",
                level="error",
                additional_info={
                    "error": str(e),
                    "batch_size": BATCH_SIZE,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Training error occurred. Recovery actions have been taken.")
            return False

    def cmd_generate(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'generate' requires a prompt.")
        max_tokens = 60
        if args[-1].isdigit():
            max_tokens = int(args[-1])
            prompt = ' '.join(args[:-1])
        else:
            prompt = ' '.join(args)

        print(f"Generating response for: {prompt}...")
        try:
            response = self.generate_response(prompt, max_tokens)
            print(f"Response: {response}")
            return False
        except Exception as e:
            # Use ErrorManager for generation error handling
            self.system.error_manager.handle_generation_error(e, prompt)
            self.system.logger.record_event(
                event_type="cli_generate_error",
                message="Generation error occurred",
                level="error",
                additional_info={
                    "error": str(e),
                    "prompt": prompt,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Generation error occurred. Recovery actions have been taken.")
            return False

    def cmd_save(self, args: List[str]) -> bool:
        path = args[0] if args else None
        print(f"Saving state{' to ' + path if path else ' to default location'}...")
        if hasattr(self.system, 'save_state'):
            self.system.save_state(path)
            self.system.logger.record_event(
                event_type="cli_save_state",
                message="State saved successfully",
                level="info",
                additional_info={"path": path}
            )
            print("State saved.")
        else:
            self.system.logger.log_error(
                error_msg="Save state method not found",
                error_type="cli_save_error",
                stack_trace=None,
                additional_info={"path": path}
            )
            print("Error: 'save_state' method not found.")
        return False

    def cmd_load(self, args: List[str]) -> bool:
        path = args[0] if args else None
        print(f"Loading state{' from ' + path if path else ' from default location'}...")
        if hasattr(self.system, 'load_state'):
            self.system.load_state(path)
            self.system.logger.record_event(
                event_type="cli_load_state",
                message="State loaded successfully",
                level="info",
                additional_info={"path": path}
            )
            print("State loaded.")
        else:
            self.system.logger.log_error(
                error_msg="Load state method not found",
                error_type="cli_load_error",
                stack_trace=None,
                additional_info={"path": path}
            )
            print("Error: 'load_state' method not found.")
        return False

    def cmd_dream(self, args: List[str]) -> bool:
        """Run a dream cycle to process and consolidate memories."""
        try:
            if not hasattr(self.system, 'dream'):
                print("Error: Dream cycle not supported.")
                self.system.logger.record_event(
                    event_type="dream_command_error",
                    message="Dream cycle not supported",
                    level="error"
                )
                return False
            
            print("Initiating dream cycle...")
            if self.system.dream():
                print("Dream cycle completed successfully.")
                return True
            else:
                print("Error: Dream cycle failed.")
                return False
            
        except Exception as e:
            self.system.error_manager.handle_curiosity_error(e, "dream_command")
            self.system.logger.record_event(
                event_type="dream_command_error",
                message="Error during dream command execution",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Error: {str(e)}")
            return False

    def cmd_tune(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: Usage: tune <parameter> [value]")
        parameter, value_str = args[0].lower(), args[1] if len(args) > 1 else None

        try:
            if parameter == "cross_attention":
                # Validate weight range before proceeding
                weight = float(value_str) if value_str else None
                if weight is not None and not (0.0 <= weight <= 1.0):
                    raise ValueError("Cross-attention weight must be between 0.0 and 1.0")
                
                print(f"Setting cross-attention weight to {weight if weight else 'default'}...")
                if hasattr(self.system, 'tune_cross_attention'):
                    self.system.tune_cross_attention(weight=weight)
                    self.system.logger.record_event(
                        event_type="cli_tune_cross_attention",
                        message="Cross-attention weight set successfully",
                        level="info",
                        additional_info={
                            "weight": weight,
                            "timestamp": time.time()
                        }
                    )
                    print("Cross-attention weight set.")
                else:
                    self.system.logger.record_event(
                        event_type="cli_tune_error",
                        message="Tune cross-attention method not found",
                        level="error",
                        additional_info={
                            "parameter": parameter,
                            "timestamp": time.time()
                        }
                    )
                    print("Error: 'tune_cross_attention' method not found.")
            else:
                self.system.logger.record_event(
                    event_type="cli_tune_error",
                    message="Unknown tuning parameter",
                    level="error",
                    additional_info={
                        "parameter": parameter,
                        "timestamp": time.time()
                    }
                )
                print(f"Error: Unknown parameter '{parameter}'. Available: cross_attention")
            return False
        except ValueError as e:
            # Handle validation errors separately
            print(f"Error: {str(e)}")
            self.system.logger.record_event(
                event_type="cli_tune_validation_error",
                message=str(e),
                level="error",
                additional_info={
                    "parameter": parameter,
                    "value": value_str,
                    "timestamp": time.time()
                }
            )
            return False
        except Exception as e:
            # Use ErrorManager for other errors
            self.system.error_manager.handle_training_error(e, 1)
            self.system.logger.record_event(
                event_type="cli_tune_error",
                message="Tuning error occurred",
                level="error",
                additional_info={
                    "error": str(e),
                    "parameter": parameter,
                    "value": value_str,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Tuning error occurred. Recovery actions have been taken.")
            return False

    def cmd_memory(self, args: List[str]) -> bool:
        """Manage memory settings and view statistics."""
        try:
            if not args:
                # Show memory statistics
                stats = self.system.get_memory_stats()
                if "error" in stats:
                    print(f"Error: {stats['error']}")
                    return False
                
                print("\nMemory Statistics:")
                for key, value in stats.items():
                    print(f"{key}: {value}")
                return True
            
            # Handle memory toggle command
            if args[0].lower() in ["on", "off"]:
                enable = args[0].lower() == "on"
                if self.system.toggle_memory(enable):
                    print(f"Memory management {'enabled' if enable else 'disabled'}.")
                    return True
                else:
                    print("Error: Failed to toggle memory management.")
                    return False
                
            print("Error: Invalid memory command. Use 'memory' for stats or 'memory on/off' to toggle.")
            return False
        
        except Exception as e:
            self.system.error_manager.handle_memory_error(e, 0)
            self.system.logger.record_event(
                event_type="memory_command_error",
                message="Error during memory command execution",
                level="error",
                additional_info={
                    "error": str(e),
                    "args": args,
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Error: {str(e)}")
            return False

    def cmd_status(self, args: List[str]) -> bool:
        """Display system status with thread-safe state access."""
        print("\n--- System Status ---")
        try:
            # Get state with proper locking
            state = self.system.state_tracker.get_state()
            with state.lock:
                if hasattr(self.system, 'scaffold_manager'):
                    stats = self.system.scaffold_manager.get_scaffold_stats()
                    print("\nScaffold Status:")
                    for key, value in stats.items():
                        print(f"  {key.replace('_', ' ').title()}: {value}")

                print("\nSystem Status:")
                print(f"  Conversation ID: {state.history.conversation_id}")
                print(f"  Temperament: {state.temperament_score:.2f}")
                print(f"  Last Confidence: {state.confidence_history[-1]:.2f if state.confidence_history else 'N/A'}")
                print(f"  Data Exposure: {state.training_state.data_exposure}")
                print(f"  Last Trained: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.training_state.last_trained)) if state.training_state.last_trained else 'Never'}")
                print(f"  Gestating: {'Yes' if state.is_sleeping else 'No'}")
                
                # Get dream memory stats with proper locking
                dream_stats = state.get_dream_memory_stats()
                if dream_stats:
                    print("\nDream Memory Status:")
                    print(f"  Count: {dream_stats['count']}")
                    print(f"  Total Memory: {dream_stats['total_memory_mb']:.2f} MB")
                    print(f"  Average Weight: {dream_stats['average_weight']:.2f}")
                
                self.system.logger.record_event(
                    event_type="cli_status",
                    message="System status retrieved successfully",
                    level="info",
                    additional_info={
                        "conversation_id": state.history.conversation_id,
                        "temperament": state.temperament_score,
                        "last_confidence": state.confidence_history[-1] if state.confidence_history else None,
                        "data_exposure": state.training_state.data_exposure,
                        "last_trained": state.training_state.last_trained,
                        "is_sleeping": state.is_sleeping,
                        "dream_stats": dream_stats,
                        "timestamp": time.time()
                    }
                )
            return False
        except Exception as e:
            self.system.error_manager.record_error(e, "status", "check", severity="warning")
            self.system.logger.record_event(
                event_type="cli_status_error",
                message="Error getting system status",
                level="error",
                additional_info={
                    "error": str(e),
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Error getting status: {str(e)}")
            return False

    def cmd_log(self, args: List[str]) -> bool:
        num_entries = 5
        if args and args[0] == "view":
            num_entries = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive.")

        print(f"\n--- Last {num_entries} Log Entries ---")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            self.system.logger.log_error(
                error_msg="Logger not available",
                error_type="cli_log_error",
                stack_trace=None
            )
            print("Error: Logger not available.")
            return False

        logs = self.system.logger.read()[-num_entries:]
        if not logs:
            print("Log is empty.")
        else:
            for i, log in enumerate(reversed(logs)):
                ts = log.get('timestamp')
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) if ts else 'N/A'
                event_type = log.get('event', 'Interaction')
                print(f"{len(logs)-i}. Time: {time_str}")
                if event_type != 'Interaction':
                    print(f"   Event: {event_type}")
                    print(f"   Details: {{k:v for k,v in log.items() if k not in ['timestamp', 'event']}}")
                else:
                    print(f"   Prompt: {log.get('prompt', 'N/A')[:50]}...")
                    print(f"   Response: {log.get('response', 'N/A')[:50]}...")
                    print(f"   Confidence: {log.get('confidence_score', 'N/A'):.2f}")
                print("-" * 20)
        print("--------------------------")
        return False

    def cmd_config(self, args: List[str]) -> None:
        """Handle configuration commands."""
        if not args:
            # Show all configuration
            config = self.system.config_manager.get_config()
            self._print_config(config)
            return

        if len(args) == 1:
            # Get value for specific key
            key = args[0]
            value = self.system.config_manager.get(key)
            if value is not None:
                print(f"{key}: {value}")
            else:
                print(f"Configuration key '{key}' not found")
            return

        if len(args) == 2:
            # Set value for specific key
            key, new_value = args
            try:
                # Convert value to appropriate type based on schema
                schema = next((s for s in self.system.config_manager.DEFAULT_SCHEMA if s.field == key), None)
                if schema:
                    if schema.type == bool:
                        new_value = new_value.lower() in ('true', '1', 'yes')
                    elif schema.type == int:
                        new_value = int(new_value)
                    elif schema.type == float:
                        new_value = float(new_value)
                    elif schema.type == str:
                        new_value = str(new_value)
                    else:
                        print(f"Unsupported type for {key}: {schema.type.__name__}")
                        return

                # Validate the value before updating
                if not self.system.config_manager.validate_value(key, new_value):
                    print(f"Invalid value for {key}: {new_value}")
                    return

                # Update the configuration
                old_value = self.system.config_manager.get(key)
                if self.system.config_manager.update(key, new_value):
                    # Log the configuration change
                    self.system.logger.record({
                        "event": "config_updated",
                        "key": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "timestamp": time.time()
                    })
                    
                    # Check for configuration change effects
                    try:
                        # Notify subscribers and check for errors
                        self.system.config_manager._notify_subscribers()
                        print(f"Updated {key} to {new_value}")
                        
                        # Log any warnings or errors from the change
                        if hasattr(self.system, 'context') and hasattr(self.system.context, 'config_handler'):
                            warnings = self.system.context.config_handler._validate_all_configs()
                            if warnings:
                                print("\nConfiguration change warnings:")
                                for warning in warnings:
                                    print(f"- {warning}")
                    except Exception as e:
                        print(f"Warning: Configuration change may have side effects: {str(e)}")
                        self.system.logger.record({
                            "event": "config_change_warning",
                            "key": key,
                            "error": str(e),
                            "timestamp": time.time()
                        })
                else:
                    print(f"Failed to update {key}")
            except ValueError as e:
                print(f"Error setting configuration: {str(e)}")
            return

        print("Invalid configuration command. Use 'config' to show all, 'config <key>' to get a value, or 'config <key> <value>' to set a value")

    def cmd_reset(self, args: List[str]) -> bool:
        print("Resetting system state...")
        if hasattr(self.system, 'cleanup'):
            self.system.cleanup()
            print("Cleanup complete.")
        else:
            print("Warning: 'cleanup' method not found.")

        try:
            config_manager = getattr(self.system, 'config_manager', ConfigManager("sovl_config.json"))
            self.system.__init__(config_manager=config_manager)
            if hasattr(self.system, 'wake_up'):
                self.system.wake_up()
            print("System reset complete.")
        except Exception as e:
            print(f"Error during system re-initialization: {e}")
        return False

    def cmd_spark(self, args: List[str]) -> bool:
        """Generate a curiosity-driven question."""
        try:
            if not hasattr(self.system, 'generate_curiosity_question'):
                print("Error: Curiosity question generation not supported.")
                self.system.logger.record_event(
                    event_type="spark_command_error",
                    message="Curiosity question generation not supported",
                    level="error"
                )
                return False
            
            question = self.system.generate_curiosity_question()
            if question:
                print(f"\nCuriosity Question: {question}\n")
                return True
            else:
                print("Error: Failed to generate curiosity question.")
                return False
            
        except Exception as e:
            self.system.error_manager.handle_curiosity_error(e, "spark_command")
            self.system.logger.record_event(
                event_type="spark_command_error",
                message="Error during spark command execution",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Error: {str(e)}")
            return False

    def cmd_reflect(self, args: List[str]) -> bool:
        print("Reflecting on recent interactions...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        logs = self.system.logger.read()[-5:]
        interaction_logs = [log for log in logs if log.get('prompt') and not log.get('is_system_question')]
        if not interaction_logs:
            print("Nothing significant to reflect on yet.")
            return False

        recent_themes = [log.get('prompt', '').split()[:5] for log in interaction_logs]
        theme_summary = ", ".join(list(set(" ".join(words) for words in recent_themes))[:3]) or "no clear theme"
        reflection = f"Recent interactions touch upon: {theme_summary}."

        print(f"Internal Reflection: {reflection}")
        elaboration_prompt = f"Based on reflection: '{reflection}', elaborate on connections or insights."
        elaboration = self.generate_response(elaboration_prompt, max_tokens=100)
        print(f"Generated Elaboration: {elaboration}")
        self.log_action(reflection, elaboration, 0.6, True, {"event": "reflect"})
        return False

    def cmd_muse(self, args: List[str]) -> bool:
        """Generate creative thoughts with proper state synchronization."""
        print("Musing on a topic...")
        try:
            # Get state with proper locking
            state = self.system.state_tracker.get_state()
            with state.lock:
                if not hasattr(state, 'logger') or not hasattr(state.logger, 'read'):
                    print("Error: Logger not available.")
                    return False

                logs = state.logger.read()[-5:]
                inspiration = "the passage of time"
                prompts = [log.get('prompt', '') for log in logs if log.get('prompt') and not log.get('is_system_question')]
                if prompts:
                    inspiration = prompts[-1].split()[:3]
                    inspiration = " ".join(inspiration) if inspiration else inspiration

                print(f"Inspiration for musing: \"{inspiration}\"")
                muse_prompt = f"Generate a creative thought inspired by '{inspiration}'."
                thought = self.generate_response(muse_prompt, max_tokens=80, temp_adjust=0.15)
                print(f"Generated Thought: {thought}")
                self.log_action(f"Musing on {inspiration}", thought, 0.7, True, {"event": "muse", "inspiration": inspiration})
            return False
        except Exception as e:
            self.system.error_manager.handle_generation_error(e, 0.0)
            print(f"Musing error occurred. Recovery actions have been taken.")
            return False

    @contextlib.contextmanager
    def temporary_system_state(self, **kwargs):
        """Temporarily modify system state with validation."""
        state = self.system.state_tracker.get_state()
        with state.lock:
            original_values = {}
            for key, value in kwargs.items():
                if key == 'temperament_score':
                    if not (0.0 <= value <= 1.0):
                        raise ValueError(f"Invalid temperament_score: {value}")
                    # Use TemperamentAdjuster for temperament changes
                    temperament_adjuster = self.system.temperament_adjuster
                    original_temperament = state.temperament_score
                    temperament_adjuster.update_temperament(curiosity_manager=None)
                    state.temperament_score = value
                else:
                    original_values[key] = getattr(self.system, key)
                    setattr(self.system, key, value)
            yield
            for key, value in original_values.items():
                setattr(self.system, key, value)
            if 'temperament_score' in kwargs:
                state.temperament_score = original_temperament
                temperament_adjuster.update_temperament(curiosity_manager=None)

    def cmd_debate(self, args: List[str]) -> bool:
        """Engage in a debate with proper state synchronization."""
        if not args:
            raise ValueError("Error: 'debate' requires a topic.")
        topic = ' '.join(args)
        print(f"Initiating debate on: '{topic}'")

        try:
            # Get state with proper locking
            state = self.system.state_tracker.get_state()
            temperament_adjuster = self.system.temperament_adjuster
            
            with state.lock:
                original_temperament = state.temperament_score
                stance = "for"
                
                for turn in range(2):
                    action = "Argue for" if stance == "for" else "Argue against (rebuttal)"
                    prompt = f"{action} the topic: '{topic}'. Provide a concise point."
                    response = self.generate_response(prompt, max_tokens=90, temp_adjust=0.1)
                    print(f"[{'Argument For' if stance == 'for' else 'Rebuttal Against'}] {response}")
                    self.log_action(prompt, response, 0.7, True, {"event": "debate_turn", "topic": topic, "stance": stance})
                    stance = "against" if stance == "for" else "for"
                    
                    # Update temperament using safe comparison
                    new_temperament = state.temperament_score + 0.1
                    if safe_compare(new_temperament, 1.0, mode='lte', logger=self.system.logger):
                        state.temperament_score = new_temperament
                    else:
                        state.temperament_score = 1.0
                    temperament_adjuster.update_temperament(curiosity_manager=None)
                    print(f"[Temperament nudged to {state.temperament_score:.2f}]")
                
                # Reset temperament using safe comparison
                if safe_compare(original_temperament, state.temperament_score, mode='eq', logger=self.system.logger):
                    self.system.logger.record_event(
                        event_type="temperament_reset_skipped",
                        message="Temperament reset skipped - already at original value",
                        level="info",
                        additional_info={
                            "original_temperament": original_temperament,
                            "current_temperament": state.temperament_score
                        }
                    )
                else:
                    state.temperament_score = original_temperament
                    temperament_adjuster.update_temperament(curiosity_manager=None)
                    print(f"[Temperament reset to {state.temperament_score:.2f}]")
                
            print(f"Debate on '{topic}' concluded.")
            return False
        except Exception as e:
            self.system.error_manager.handle_generation_error(e, 0.0)
            print(f"Debate error occurred. Recovery actions have been taken.")
            return False

    def cmd_flare(self, args: List[str]) -> bool:
        """Trigger an emotional flare with proper temperament handling."""
        print("Triggering emotional flare...")
        try:
            # Use TemperamentAdjuster for the flare
            temperament_adjuster = self.system.temperament_adjuster
            state = self.system.state_tracker.get_state()
            
            with state.lock:
                original_temperament = state.temperament_score
                if safe_compare(1.0, state.temperament_score, mode='gt', logger=self.system.logger):
                    state.temperament_score = 1.0
                    temperament_adjuster.update_temperament(curiosity_manager=None)
                    
                    prompt = ' '.join(args) or "Express a sudden burst of strong feeling!"
                    print(f"Flare prompt: {prompt}")
                    outburst = self.generate_response(prompt, max_tokens=100, temp_adjust=1.0)
                    print(f"Generated Outburst: {outburst.upper()}")
                    
                    # Reset temperament using safe comparison
                    if safe_compare(original_temperament, state.temperament_score, mode='eq', logger=self.system.logger):
                        self.system.logger.record_event(
                            event_type="temperament_reset_skipped",
                            message="Temperament reset skipped - already at original value",
                            level="info",
                            additional_info={
                                "original_temperament": original_temperament,
                                "current_temperament": state.temperament_score
                            }
                        )
                    else:
                        state.temperament_score = original_temperament
                        temperament_adjuster.update_temperament(curiosity_manager=None)
                    
                self.log_action(f"Flare: {prompt}", outburst, 0.9, True, {"event": "flare"})
                return False
        except Exception as e:
            self.system.error_manager.handle_generation_error(e, 0.0)
            print(f"Flare error occurred. Recovery actions have been taken.")
            return False

    def cmd_echo(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'echo' requires text.")
        text = ' '.join(args)
        print(f"You said: '{text}'")
        reflect_prompt = f"User said '{text}'. Reflect on why or what it implies."
        response = self.generate_response(reflect_prompt, max_tokens=70)
        print(f"Reflection: {response}")
        self.log_action(f"Echo: {text}", response, 0.6, False, {"event": "echo"})
        return False

    def cmd_recap(self, args: List[str]) -> bool:
        num_to_recap = int(args[0]) if args and args[0].isdigit() else 5
        if num_to_recap <= 0:
            raise ValueError("Number of interactions must be positive.")

        print(f"Generating recap of last {num_to_recap} interactions...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        interaction_logs = [
            log for log in self.system.logger.read()[-num_to_recap:]
            if log.get('prompt') and log.get('response') and not log.get('is_system_question')
        ]
        if not interaction_logs:
            print("No interactions found to recap.")
            return False

        formatted_interactions = "".join(
            f"Turn {i+1}:\n User: {log['prompt'][:100]}...\n AI: {log['response'][:100]}...\n\n"
            for i, log in enumerate(interaction_logs)
        )
        recap_prompt = f"Summarize main topics from:\n\n{formatted_interactions}Summary:"
        summary_response = self.generate_response(recap_prompt, max_tokens=120)
        print(f"\n--- Conversation Recap (Last {len(interaction_logs)}) ---")
        print(summary_response)
        print("--------------------------------")
        self.log_action(f"Recap last {num_to_recap}", summary_response, 0.6, True, {"event": "recap", "recap_count": len(interaction_logs)})
        return False

    def cmd_recall(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'recall' requires a query.")
        query = ' '.join(args)
        print(f"Recalling information related to: '{query}'...")

        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        relevant_snippets = []
        max_results = 5
        query_lower = query.lower()
        for log in reversed(self.system.logger.read()):
            prompt = log.get('prompt', '').lower()
            response = log.get('response', '').lower()
            if query_lower in prompt or query_lower in response:
                log_time = time.strftime('%H:%M:%S', time.localtime(log.get('timestamp', 0)))
                snippet = f"[{log_time}] Prompt='{prompt[:60]}...', Response='{response[:60]}...'"
                relevant_snippets.append(snippet)
                if len(relevant_snippets) >= max_results:
                    break

        if not relevant_snippets:
            response = self.generate_response(
                f"No specific info found about '{query}'. What about it?", max_tokens=50
            )
            print(f"No mentions of '{query}' found.")
            print(f"Response: {response}")
            self.log_action(f"Recall: {query}", "No memories found.", 0.4, True, {"event": "recall_miss", "recall_query": query})
            return False

        formatted_snippets = "\n - ".join(relevant_snippets)
        recall_prompt = f"Synthesize recall about '{query}' from:\n- {formatted_snippets}\n\nBased only on these snippets."
        recall_response = self.generate_response(recall_prompt, max_tokens=150)
        print(f"\n--- Recall Synthesis on '{query}' ---")
        print(recall_response)
        print("---------------------------------------------------")
        self.log_action(f"Recall: {query}", recall_response, 0.7, True, {"event": "recall_hit", "recall_query": query})
        return False

    def cmd_forget(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'forget' requires a topic.")
        topic = ' '.join(args)
        print(f"Processing request to forget: '{topic}'...")
        forget_prompt = (
            f"User requests to 'forget' '{topic}'. Acknowledge politely, noting that while knowledge isn't erased, "
            f"I'll avoid focusing on '{topic}' proactively."
        )
        acknowledgement = self.generate_response(forget_prompt, max_tokens=90)
        print(f"\n--- Forget Request Acknowledgement ---")
        print(acknowledgement)
        print("------------------------------------")
        print(f"[Note: Simulated effect. '{topic}' may still exist in training data.]")
        self.log_action(f"Forget: {topic}", acknowledgement, 0.5, False, {"event": "forget_request", "forget_topic": topic, "is_simulated": True})
        return False

    def cmd_history(self, args: List[str]) -> bool:
        if not hasattr(self.system, 'cmd_history'):
            print("Command history not available.")
            return False
        num_entries = 10
        search_term = None
        if args:
            if args[0].isdigit():
                num_entries = int(args[0])
            else:
                search_term = ' '.join(args)

        entries = self.system.cmd_history.search(search_term) if search_term else self.system.cmd_history.get_last(num_entries)
        print(f"\n{'Commands matching' if search_term else 'Last'} {len(entries)} commands:")
        if not entries:
            print("No commands found.")
            return False
        for entry in entries:
            print(self.system.cmd_history.format_entry(entry))
        return False

    def cmd_help(self, args: List[str]) -> bool:
        if args:
            category = args[0].capitalize()
            if category in COMMAND_CATEGORIES:
                print(f"\n{category} Commands:")
                for cmd in COMMAND_CATEGORIES[category]:
                    doc = self.commands[cmd].__doc__ or "No description available."
                    print(f"  {cmd:<20} : {doc.split('.')[0]}")
            else:
                print(f"Unknown category: {category}")
                print("Available categories:", ", ".join(COMMAND_CATEGORIES.keys()))
        else:
            print("\nCommand Categories:")
            for category, commands in COMMAND_CATEGORIES.items():
                print(f"\n{category}:")
                for cmd in commands:
                    doc = self.commands[cmd].__doc__ or "No description available."
                    print(f"  {cmd:<20} : {doc.split('.')[0]}")
        return False

    @staticmethod
    def _parse_config_value(value_str: str):
        try:
            if '.' in value_str:
                return float(value_str)
            try:
                return int(value_str)
            except ValueError:
                pass
            if value_str.lower() == 'true':
                return True
            if value_str.lower() == 'false':
                return False
            if ',' in value_str:
                return [_parse_config_value(p.strip()) for p in value_str.split(',')]
            return value_str
        except Exception as e:
            raise ValueError(f"Failed to parse config value '{value_str}': {str(e)}")

    def _print_config(self, config: dict):
        for key, value in config.items():
            print(f"{key}: {value}")

    def cmd_mimic(self, args: List[str]) -> bool:
        """Generate a response mimicking the style of the input text."""
        if not args:
            print("Error: Please provide text to mimic")
            return False
        
        try:
            # Store original scaffold weight
            original_weight = None
            if hasattr(self.system, 'scaffold_manager'):
                original_weight = self.system.scaffold_manager.get_scaffold_weight()
                # Update scaffold weight for mimic mode
                self.system.scaffold_manager.update_scaffold_weight(0.8)
                self.system.logger.record_event(
                    event_type="scaffold_weight_updated",
                    message="Scaffold weight updated for mimic mode",
                    level="info",
                    additional_info={
                        "old_weight": original_weight,
                        "new_weight": 0.8,
                        "timestamp": time.time()
                    }
                )
            
            # Generate response with mimic prompt
            mimic_prompt = f"Generate text in the style of: {args[0]}"
            response = self.generate_response(mimic_prompt, max_tokens=100)
            
            # Restore original scaffold weight
            if hasattr(self.system, 'scaffold_manager') and original_weight is not None:
                self.system.scaffold_manager.update_scaffold_weight(original_weight)
                self.system.logger.record_event(
                    event_type="scaffold_weight_restored",
                    message="Scaffold weight restored after mimic mode",
                    level="info",
                    additional_info={
                        "old_weight": 0.8,
                        "new_weight": original_weight,
                        "timestamp": time.time()
                    }
                )
            
            print(f"\nMimicked Response:\n{response}\n")
            return True
        
        except Exception as e:
            # Ensure scaffold weight is restored even if generation fails
            if hasattr(self.system, 'scaffold_manager') and original_weight is not None:
                try:
                    self.system.scaffold_manager.update_scaffold_weight(original_weight)
                    self.system.logger.record_event(
                        event_type="scaffold_weight_restored_error",
                        message="Scaffold weight restored after error",
                        level="warning",
                        additional_info={
                            "error": str(e),
                            "original_weight": original_weight,
                            "timestamp": time.time()
                        }
                    )
                except Exception as restore_error:
                    self.system.logger.record_event(
                        event_type="scaffold_weight_restore_failed",
                        message="Failed to restore scaffold weight",
                        level="error",
                        additional_info={
                            "original_error": str(e),
                            "restore_error": str(restore_error),
                            "timestamp": time.time()
                        }
                    )
            
            # Handle the original error
            self.system.error_manager.handle_generation_error(e, mimic_prompt)
            self.system.logger.record_event(
                event_type="mimic_generation_error",
                message="Error during mimic generation",
                level="error",
                additional_info={
                    "error": str(e),
                    "prompt": mimic_prompt,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            print(f"Error during mimic generation: {str(e)}")
            return False

class SystemInitializationError(Exception):
    def __init__(self, message: str, config_path: str, stack_trace: str):
        super().__init__(message)
        self.config_path = config_path
        self.stack_trace = stack_trace

def run_cli(config_manager_instance: Optional[ConfigManager] = None):
    sovl_system = None
    try:
        # Initialize config manager with proper error handling
        try:
            config_manager = config_manager_instance or ConfigManager("sovl_config.json")
        except Exception as e:
            print(f"Failed to initialize configuration manager: {str(e)}")
            raise SystemInitializationError(
                message="Configuration manager initialization failed",
                config_path="sovl_config.json",
                stack_trace=traceback.format_exc()
            )

        # Initialize SOVL system with proper error handling
        try:
            sovl_system = SOVLSystem(config_manager)
            if not hasattr(sovl_system, 'state_tracker') or not hasattr(sovl_system, 'logger'):
                raise SystemInitializationError(
                    message="SOVL system initialization incomplete - missing required components",
                    config_path=config_manager.config_path,
                    stack_trace=""
                )
        except Exception as e:
            print(f"Failed to initialize SOVL system: {str(e)}")
            raise SystemInitializationError(
                message="SOVL system initialization failed",
                config_path=config_manager.config_path,
                stack_trace=traceback.format_exc()
            )

        # Initialize command history and handler
        sovl_system.cmd_history = CommandHistory()
        handler = CommandHandler(sovl_system)

        # Wake up system with proper validation
        if hasattr(sovl_system, 'wake_up'):
            try:
                sovl_system.wake_up()
                print("\nSystem Ready.")
            except Exception as e:
                print(f"Failed to wake up system: {str(e)}")
                raise SystemInitializationError(
                    message="System wake up failed",
                    config_path=config_manager.config_path,
                    stack_trace=traceback.format_exc()
                )
        else:
            raise SystemInitializationError(
                message="SOVL system missing wake_up method",
                config_path=config_manager.config_path,
                stack_trace=""
            )

        # Display help and start command loop
        handler.cmd_help([])

        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                if not user_input:
                    continue

                parts = user_input.split()
                cmd, args = handler.parse_args(parts)
                sovl_system.cmd_history.add(user_input)
                try:
                    should_exit = handler.execute(cmd, args)
                    sovl_system.cmd_history.history[-1] = (
                        sovl_system.cmd_history.history[-1][0],
                        sovl_system.cmd_history.history[-1][1],
                        "success"
                    )
                    if should_exit:
                        break
                except Exception as e:
                    sovl_system.cmd_history.history[-1] = (
                        sovl_system.cmd_history.history[-1][0],
                        sovl_system.cmd_history.history[-1][1],
                        f"error: {str(e)}"
                    )
                    raise
            except KeyboardInterrupt:
                print("\nInterrupt received, initiating clean shutdown...")
                break
            except Exception as e:
                print(f"Command error: {e}")
                sovl_system.logger.log_error(
                    error_msg="Command execution failed",
                    error_type="cli_command_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"command": user_input}
                )
    except SystemInitializationError as e:
        print(f"System initialization failed: {e.message}")
        if e.stack_trace:
            print(f"Stack trace:\n{e.stack_trace}")
        return
    except Exception as e:
        print(f"CLI initialization failed: {e}")
    finally:
        if sovl_system:
            shutdown_system(sovl_system)

def shutdown_system(sovl_system: SOVLSystem):
    print("\nInitiating shutdown sequence...")
    try:
        if hasattr(sovl_system, 'save_state'):
            sovl_system.save_state("final_state.json")
            print("Final state saved.")
        cleanup_resources(sovl_system)
        sovl_system.logger.record_event(
            event_type="system_shutdown",
            message="System shutdown completed successfully",
            level="info",
            additional_info={"status": "clean"}
        )
        print("Shutdown complete.")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        sovl_system.logger.log_error(
            error_msg="System shutdown failed",
            error_type="shutdown_error",
            stack_trace=traceback.format_exc(),
            additional_info={"status": "error"}
        )

def cleanup_resources(sovl_system: SOVLSystem):
    try:
        if hasattr(sovl_system, 'scaffold_manager'):
            sovl_system.scaffold_manager.reset_scaffold_state()
        if hasattr(sovl_system, 'cleanup'):
            sovl_system.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sovl_system.logger.record_event(
            event_type="cli_cleanup_complete",
            message="CLI resources cleaned up successfully",
            level="info"
        )
    except Exception as e:
        sovl_system.logger.log_error(
            error_msg="CLI cleanup failed",
            error_type="cleanup_error",
            stack_trace=traceback.format_exc()
        )

if __name__ == "__main__":
    run_cli()
