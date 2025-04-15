import time
import traceback
from typing import Optional, Dict, Any, TYPE_CHECKING
from sovl_config import ConfigManager
from sovl_cli import run_cli
from sovl_logger import Logger

if TYPE_CHECKING:
    from sovl_main import SOVLSystem

class SOVLOrchestrator:
    """
    Orchestrates the initialization, execution, and shutdown of the SOVL system.

    Responsible for setting up core components (ConfigManager, SOVLSystem),
    selecting execution modes (e.g., CLI), and ensuring clean shutdown with
    state saving and resource cleanup.
    """
    # Constants for configuration
    DEFAULT_CONFIG_PATH: str = "sovl_config.json"
    DEFAULT_LOG_FILE: str = "sovl_orchestrator_logs.jsonl"
    LOG_MAX_SIZE_MB: int = 10
    SAVE_PATH_SUFFIX: str = "_final.json"

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, log_file: str = DEFAULT_LOG_FILE) -> None:
        """
        Initialize the orchestrator with configuration and logging.

        Args:
            config_path: Path to the configuration file.
            log_file: Path to the orchestrator's log file.

        Raises:
            RuntimeError: If initialization of ConfigManager or SOVLSystem fails.
        """
        self._initialize_logger(log_file)
        self._log_event("orchestrator_init_start", {"config_path": config_path})

        try:
            self.config_manager = ConfigManager(config_path, self.logger)
            self.system: Optional['SOVLSystem'] = None
            self._log_event("orchestrator_init_success")
        except Exception as e:
            self._log_error("Orchestrator initialization failed", e)
            raise RuntimeError(f"Failed to initialize orchestrator: {str(e)}") from e

    def _create_config_manager(self, config_path: str) -> ConfigManager:
        """Create and initialize the configuration manager."""
        try:
            return ConfigManager(config_path, self.logger)
        except Exception as e:
            self._log_error("Config manager creation failed", e)
            raise

    def _initialize_logger(self, log_file: str) -> None:
        """Initialize the logger with specified log file."""
        try:
            self.logger = Logger(log_file, max_size_mb=self.LOG_MAX_SIZE_MB)
        except Exception as e:
            print(f"Failed to initialize logger: {str(e)}")
            raise

    def _log_event(self, event_type: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Log an event with optional additional information."""
        try:
            self.logger.record_event(
                event_type=event_type,
                message=f"Orchestrator event: {event_type}",
                level="info",
                additional_info=additional_info
            )
        except Exception as e:
            print(f"Failed to log event: {str(e)}")

    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with stack trace."""
        try:
            self.logger.log_error(
                error_msg=message,
                error_type="orchestrator_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error": str(error)}
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")

    def initialize_system(self) -> None:
        """Initialize the SOVL system with the current configuration."""
        try:
            if self.system is None:
                from sovl_main import SOVLSystem  # Import here to break circular dependency
                self.system = SOVLSystem(self.config_manager.config_path)
                self._log_event("system_initialized")
        except Exception as e:
            self._log_error("System initialization failed", e)
            raise

    def run(self) -> None:
        """Run the SOVL system in the appropriate mode."""
        try:
            self.initialize_system()
            if self.system is None:
                raise RuntimeError("System not initialized")
            
            # Run in CLI mode by default
            run_cli(self.config_manager)
            
        except Exception as e:
            self._log_error("System execution failed", e)
            raise

    def shutdown(self) -> None:
        """Shutdown the system and save state."""
        try:
            if self.system is not None:
                # Save final state
                self.system.state_tracker.state.save_state()
                self._log_event("system_shutdown")
        except Exception as e:
            self._log_error("System shutdown failed", e)
            raise

# Main block
if __name__ == "__main__":
    orchestrator = SOVLOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        print(f"Error running SOVL system: {str(e)}")
        raise
    finally:
        orchestrator.shutdown()
