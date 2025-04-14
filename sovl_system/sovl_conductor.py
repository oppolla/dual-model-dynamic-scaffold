import time
import traceback
from typing import Optional, Dict, Any
from sovl_config import ConfigManager
from sovl_main import SOVLSystem
from sovl_cli import run_cli
from sovl_logger import Logger
import torch

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
            self.config_manager: ConfigManager = self._create_config_manager(config_path)
            self.system: Optional[SOVLSystem] = self._create_system()
            self._log_event("orchestrator_init_success")
        except Exception as e:
            self._log_error("Orchestrator initialization failed", e)
            raise RuntimeError(f"Failed to initialize orchestrator: {str(e)}") from e

    def run(self, mode: str = "cli", **kwargs: Any) -> None:
        """
        Execute the system in the specified mode.

        Args:
            mode: Execution mode (currently supports 'cli').
            **kwargs: Additional arguments for the execution mode.

        Raises:
            ValueError: If an unsupported mode is provided.
            RuntimeError: If execution fails.
        """
        self._log_event("orchestrator_run_start", {"mode": mode})

        try:
            mode_handlers: Dict[str, callable] = {
                "cli": self._run_cli_mode
            }
            handler = mode_handlers.get(mode)
            if handler is None:
                raise ValueError(f"Unsupported execution mode: {mode}")
            handler(**kwargs)
            self._log_event("orchestrator_run_complete", {"mode": mode})
        except Exception as e:
            self._log_error(f"Run failed in mode {mode}", e)
            raise RuntimeError(f"Execution failed in mode {mode}: {str(e)}") from e

    def shutdown(self) -> None:
        """
        Perform clean shutdown, saving state and releasing resources.

        Raises:
            RuntimeError: If shutdown fails.
        """
        self._log_event("orchestrator_shutdown_start")

        try:
            self._save_system_state()
            self._cleanup_resources()
            self._log_event("orchestrator_shutdown_complete", {"status": "clean"})
        except Exception as e:
            self._log_error("Shutdown failed", e)
            raise RuntimeError(f"Shutdown failed: {str(e)}") from e

    def _initialize_logger(self, log_file: str) -> None:
        """Set up the logger for the orchestrator."""
        self.logger: Logger = Logger(
            log_file=log_file,
            max_size_mb=self.LOG_MAX_SIZE_MB,
            compress_old=True
        )

    def _create_config_manager(self, config_path: str) -> ConfigManager:
        """Create and return a ConfigManager instance."""
        return ConfigManager(config_path)

    def _create_system(self) -> SOVLSystem:
        """Create and return a SOVLSystem instance."""
        return SOVLSystem(self.config_manager)

    def _run_cli_mode(self, **kwargs: Any) -> None:
        """Execute the CLI mode by calling run_cli."""
        run_cli(config_manager=self.config_manager, system=self.system, **kwargs)

    def _save_system_state(self) -> None:
        """Save the system state if the system supports it."""
        if hasattr(self.system, 'save_state'):
            save_path = self.config_manager.get("controls_config.save_path_prefix", "state") + self.SAVE_PATH_SUFFIX
            self.system.save_state(save_path)
            self._log_event("state_saved", {"save_path": save_path})

    def _cleanup_resources(self) -> None:
        """Clean up system resources, including scaffold state and GPU memory."""
        try:
            if hasattr(self.system, 'scaffold_manager'):
                self.system.scaffold_manager.reset_scaffold_state()
                self._log_event("scaffold_state_reset")

            if hasattr(self.system, 'cleanup'):
                self.system.cleanup()
                self._log_event("system_cleanup")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self._log_event("cuda_cache_cleared")
        except Exception as e:
            self._log_error("Resource cleanup failed", e)
            raise

    def _log_event(self, event: str, extra_attrs: Optional[Dict[str, Any]] = None) -> None:
        """Log an event with optional additional attributes."""
        log_entry = {
            "event": event,
            "timestamp": time.time()
        }
        if extra_attrs:
            log_entry.update(extra_attrs)
        self.logger.record(log_entry)

    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with stack trace."""
        self.logger.record({
            "error": f"{message}: {str(error)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })

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
