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

    This class is responsible for setting up the ConfigManager, SOVLSystem, and other
    core components, selecting the execution mode (e.g., CLI), and ensuring clean
    shutdown with state saving and resource cleanup.
    """
    def __init__(self, config_path: str = "sovl_config.json", log_file: str = "sovl_orchestrator_logs.jsonl"):
        """
        Initialize the orchestrator with configuration and logging.

        Args:
            config_path (str): Path to the configuration file.
            log_file (str): Path to the orchestrator's log file.
        """
        # Initialize logger
        self.logger = Logger(
            log_file=log_file,
            max_size_mb=10,
            compress_old=True
        )
        self.logger.record({
            "event": "orchestrator_init_start",
            "config_path": config_path,
            "timestamp": time.time()
        })

        try:
            # Initialize ConfigManager
            self.config_manager = ConfigManager(config_path)

            # Initialize SOVLSystem
            self.system = SOVLSystem(self.config_manager)

            self.logger.record({
                "event": "orchestrator_init_success",
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Orchestrator initialization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def run(self, mode: str = "cli", **kwargs) -> None:
        """
        Run the system in the specified mode.

        Args:
            mode (str): Execution mode ('cli' supported currently).
            **kwargs: Additional arguments for the execution mode.

        Raises:
            ValueError: If an unsupported mode is provided.
            Exception: For other execution errors, logged appropriately.
        """
        self.logger.record({
            "event": "orchestrator_run_start",
            "mode": mode,
            "timestamp": time.time()
        })

        try:
            if mode == "cli":
                # Pass config_manager and system to run_cli
                run_cli(config_manager=self.config_manager, system=self.system, **kwargs)
            else:
                raise ValueError(f"Unsupported execution mode: {mode}")

            self.logger.record({
                "event": "orchestrator_run_complete",
                "mode": mode,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Run failed in mode {mode}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def shutdown(self) -> None:
        """
        Perform clean shutdown, saving state and releasing resources.
        """
        self.logger.record({
            "event": "orchestrator_shutdown_start",
            "timestamp": time.time()
        })

        try:
            if hasattr(self.system, 'save_state'):
                save_path = self.config_manager.get("controls_config.save_path_prefix", "state") + "_final.json"
                self.system.save_state(save_path)
                self.logger.record({
                    "event": "state_saved",
                    "save_path": save_path,
                    "timestamp": time.time()
                })

            # Clean up system resources
            self._cleanup_resources()

            self.logger.record({
                "event": "orchestrator_shutdown_complete",
                "status": "clean",
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Shutdown failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _cleanup_resources(self) -> None:
        """
        Clean up system resources, including scaffold state and GPU memory.
        """
        try:
            if hasattr(self.system, 'scaffold_manager'):
                self.system.scaffold_manager.reset_scaffold_state()
                self.logger.record({
                    "event": "scaffold_state_reset",
                    "timestamp": time.time()
                })

            if hasattr(self.system, 'cleanup'):
                self.system.cleanup()
                self.logger.record({
                    "event": "system_cleanup",
                    "timestamp": time.time()
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.record({
                    "event": "cuda_cache_cleared",
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Resource cleanup failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise
