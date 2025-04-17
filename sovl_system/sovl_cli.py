import time
import torch
import traceback
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
from contextlib import contextlib
import functools
from sovl_main import SOVLSystem
from sovl_config import ConfigManager
from sovl_utils import safe_compare
import readline
import rlcompleter
from collections import deque
import cmd
import sys
import logging

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

class CommandHistory:
    """Manages command history with search functionality."""
    def __init__(self, max_size: int = 100):
        self.history = deque(maxlen=max_size)
        self.current_index = -1

    def add(self, command: str):
        """Add a command to history."""
        self.history.append(command)
        self.current_index = -1

    def get_previous(self) -> Optional[str]:
        """Get previous command in history."""
        if not self.history:
            return None
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
        return self.history[-(self.current_index + 1)]

    def get_next(self) -> Optional[str]:
        """Get next command in history."""
        if not self.history or self.current_index < 0:
            return None
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[-(self.current_index + 1)]
        self.current_index = -1
        return ""

    def search(self, query: str) -> List[str]:
        """Search through command history."""
        return [cmd for cmd in self.history if query.lower() in cmd.lower()]

    def clear(self):
        """Clear command history."""
        self.history.clear()
        self.current_index = -1

class CommandHandler(cmd.Cmd):
    """Enhanced command handler with history and search capabilities."""
    prompt = 'sovl> '
    
    def __init__(self, sovl_system: SOVLSystem):
        super().__init__()
        self.sovl_system = sovl_system
        self.history = deque(maxlen=100)
        self.current_index = -1
        
    def preloop(self):
        """Initialize the command handler."""
        print("SOVL Interactive CLI")
        print("Type 'help' for available commands")
        
    def postcmd(self, stop, line):
        """Called after each command execution."""
        self.current_index = -1
        return stop
        
    def do_help(self, arg):
        """Show help for commands."""
        if arg:
            # Show help for specific command
            try:
                func = getattr(self, 'help_' + arg)
            except AttributeError:
                try:
                    doc = getattr(self, 'do_' + arg).__doc__
                    if doc:
                        print(doc)
                        return
                except AttributeError:
                    pass
                print(f"No help available for {arg}")
            else:
                func()
        else:
            # Show all commands
            print("\nAvailable commands:")
            print("------------------")
            print("help [command]    - Show help for commands")
            print("status           - Show system status")
            print("pause            - Pause the current operation")
            print("resume           - Resume the current operation")
            print("metrics          - Show current metrics")
            print("config           - Show current configuration")
            print("exit             - Exit the CLI")
            print("history          - Show command history")
            print("search <query>   - Search command history")
            
    def do_status(self, arg):
        """Show system status."""
        status = self.sovl_system.get_status()
        print("\nSystem Status:")
        print("-------------")
        for key, value in status.items():
            print(f"{key}: {value}")
            
    def do_pause(self, arg):
        """Pause the current operation."""
        if self.sovl_system.pause():
            print("Operation paused")
        else:
            print("No operation to pause")
            
    def do_resume(self, arg):
        """Resume the current operation."""
        if self.sovl_system.resume():
            print("Operation resumed")
        else:
            print("No operation to resume")
            
    def do_metrics(self, arg):
        """Show current metrics."""
        metrics = self.sovl_system.get_metrics()
        print("\nCurrent Metrics:")
        print("---------------")
        for key, value in metrics.items():
            print(f"{key}: {value}")
            
    def do_config(self, arg):
        """Show current configuration."""
        config = self.sovl_system.get_config()
        print("\nCurrent Configuration:")
        print("---------------------")
        for key, value in config.items():
            print(f"{key}: {value}")
            
    def do_exit(self, arg):
        """Exit the CLI."""
        print("Exiting CLI...")
        return True
        
    def do_history(self, arg):
        """Show command history."""
        print("\nCommand History:")
        print("---------------")
        for i, cmd in enumerate(self.history, 1):
            print(f"{i}: {cmd}")
            
    def do_search(self, arg):
        """Search command history."""
        if not arg:
            print("Please provide a search query")
            return
        matches = [cmd for cmd in self.history if arg.lower() in cmd.lower()]
        if matches:
            print("\nMatching Commands:")
            print("-----------------")
            for cmd in matches:
                print(cmd)
        else:
            print("No matching commands found")
            
    def default(self, line):
        """Handle unknown commands."""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")
        
    def emptyline(self):
        """Do nothing on empty line."""
        pass
        
    def precmd(self, line):
        """Called before command execution."""
        if line:
            self.history.append(line)
        return line

def run_cli(sovl_system: SOVLSystem):
    """Run the CLI interface."""
    try:
        handler = CommandHandler(sovl_system)
        handler.cmdloop()
    except KeyboardInterrupt:
        print("\nCLI terminated by user")
    except Exception as e:
        logging.error(f"CLI error: {str(e)}")
        print(f"Error: {str(e)}")

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
        handler.do_help([])

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
