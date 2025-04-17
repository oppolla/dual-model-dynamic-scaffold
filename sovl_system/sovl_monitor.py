from typing import Dict, Any
from threading import Thread, Event
from sovl_memory import MemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager

class SystemMonitor:
    """Real-time monitoring of system metrics."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        training_manager: TrainingCycleManager,
        curiosity_manager: CuriosityManager,
        update_interval: float = 1.0
    ):
        """
        Initialize the system monitor with required components.
        
        Args:
            memory_manager: Memory manager instance for memory metrics.
            training_manager: Training manager instance for training metrics.
            curiosity_manager: Curiosity manager instance for curiosity metrics.
            update_interval: Time interval (in seconds) between metric updates.
        """
        self.memory_manager = memory_manager
        self.training_manager = training_manager
        self.curiosity_manager = curiosity_manager
        self.update_interval = update_interval
        self._stop_event = Event()
        self._monitor_thread = None

    def start_monitoring(self) -> None:
        """Start the real-time monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            print("Monitoring is already running.")
            return
        
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("Monitoring started.")

    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join()
            print("Monitoring stopped.")
        else:
            print("Monitoring is not running.")

    def _monitor_loop(self) -> None:
        """Main loop for monitoring system metrics."""
        while not self._stop_event.is_set():
            metrics = self._collect_metrics()
            self._display_metrics(metrics)
            time.sleep(self.update_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics from all components."""
        return {
            "memory": self.memory_manager.get_memory_stats(),
            "training": self.training_manager.get_training_progress(),
            "curiosity": self.curiosity_manager.get_curiosity_scores()
        }

    def _display_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display collected metrics in a user-friendly format."""
        print("\n--- System Metrics ---")
        print(f"Memory Usage: {metrics['memory']['allocated_mb']:.2f} MB / {metrics['memory']['total_memory_mb']:.2f} MB")
        print(f"Training Progress: {metrics['training']['progress']:.2f}%")
        print(f"Curiosity Score: {metrics['curiosity']['score']:.2f}")
        print("----------------------")
