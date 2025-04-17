from typing import Dict, Any, List
from threading import Thread, Event, Lock
from collections import deque
from sovl_memory import MemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager
from sovl_config import ConfigManager
from sovl_logger import Logger
import time
import traceback

class SystemMonitor:
    """Real-time monitoring of system metrics."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        training_manager: TrainingCycleManager,
        curiosity_manager: CuriosityManager,
        config_manager: ConfigManager,
        logger: Logger,
        update_interval: float = None
    ):
        """
        Initialize the system monitor with required components.
        
        Args:
            memory_manager: Memory manager instance for memory metrics.
            training_manager: Training manager instance for training metrics.
            curiosity_manager: Curiosity manager instance for curiosity metrics.
            config_manager: Config manager for fetching configuration values.
            logger: Logger instance for logging events.
            update_interval: Time interval (in seconds) between metric updates.
        """
        self.memory_manager = memory_manager
        self.training_manager = training_manager
        self.curiosity_manager = curiosity_manager
        self.config_manager = config_manager
        self.logger = logger
        self.update_interval = update_interval or self.config_manager.get("monitor.update_interval", 1.0)
        self._stop_event = Event()
        self._monitor_thread = None
        self._lock = Lock()
        self._metrics_history = deque(maxlen=100)

    def start_monitoring(self) -> None:
        """Start the real-time monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.record_event(
                event_type="monitoring_already_running",
                message="Monitoring is already running.",
                level="warning"
            )
            return
        
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.record_event(
            event_type="monitoring_started",
            message="Monitoring started.",
            level="info"
        )

    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join()
            self.logger.record_event(
                event_type="monitoring_stopped",
                message="Monitoring stopped.",
                level="info"
            )
        else:
            self.logger.record_event(
                event_type="monitoring_not_running",
                message="Monitoring is not running.",
                level="warning"
            )

    def _monitor_loop(self) -> None:
        """Main loop for monitoring system metrics."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._display_metrics(metrics)
                with self._lock:
                    self._metrics_history.append(metrics)
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Error during monitoring: {str(e)}",
                    error_type="monitoring_error",
                    stack_trace=traceback.format_exc()
                )
            finally:
                time.sleep(self.update_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics from all components."""
        # Get enabled metrics first
        enabled_metrics = self.config_manager.get("monitor.enabled_metrics", ["memory", "training", "curiosity"])
        
        # Collect metrics without holding the lock
        metrics = {}
        if "memory" in enabled_metrics:
            metrics["memory"] = self.memory_manager.get_memory_stats()
        if "training" in enabled_metrics:
            metrics["training"] = self.training_manager.get_training_progress()
        if "curiosity" in enabled_metrics:
            metrics["curiosity"] = self.curiosity_manager.get_curiosity_scores()
        
        return metrics

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the metrics history in a thread-safe way."""
        with self._lock:
            return list(self._metrics_history)  # Return a copy to avoid race conditions

    def _display_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display collected metrics in a user-friendly format."""
        print("\n--- System Metrics ---")
        if "memory" in metrics:
            print(f"Memory Usage: {metrics['memory']['allocated_mb']:.2f} MB / {metrics['memory']['total_memory_mb']:.2f} MB")
        if "training" in metrics:
            print(f"Training Progress: {metrics['training']['progress']:.2f}%")
        if "curiosity" in metrics:
            print(f"Curiosity Score: {metrics['curiosity']['score']:.2f}")
        print("----------------------")
