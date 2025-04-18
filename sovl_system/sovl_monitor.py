from typing import Dict, Any, List
from threading import Thread, Event, Lock
from collections import deque
from sovl_memory import MemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_events import MemoryEventDispatcher, MemoryEventTypes
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
        memory_dispatcher: MemoryEventDispatcher,
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
            memory_dispatcher: MemoryEventDispatcher instance for memory event handling.
            update_interval: Time interval (in seconds) between metric updates.
        """
        self.memory_manager = memory_manager
        self.training_manager = training_manager
        self.curiosity_manager = curiosity_manager
        self.config_manager = config_manager
        self.logger = logger
        self.memory_dispatcher = memory_dispatcher
        self.update_interval = update_interval or self.config_manager.get("monitor.update_interval", 1.0)
        self._stop_event = Event()
        self._monitor_thread = None
        self._lock = Lock()
        self._metrics_history = deque(maxlen=100)
        self._memory_events_history = deque(maxlen=100)
        
        # Register memory event handlers
        self._register_memory_event_handlers()

    def _register_memory_event_handlers(self) -> None:
        """Register handlers for memory-related events."""
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_INITIALIZED, self._handle_memory_initialized)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_CONFIG_UPDATED, self._handle_memory_config_update)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_THRESHOLD_REACHED, self._handle_memory_threshold)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_CLEANUP_STARTED, self._handle_memory_cleanup)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_STATS_UPDATED, self._handle_memory_stats_update)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_ERROR, self._handle_memory_error)

    async def _handle_memory_initialized(self, event_data: Dict[str, Any]) -> None:
        """Handle memory initialization events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_INITIALIZED,
                'config': event_data.get('config', {})
            })
        self.logger.record_event(
            event_type="memory_initialized",
            message="Memory system initialized",
            level="info",
            config=event_data.get('config', {})
        )

    async def _handle_memory_config_update(self, event_data: Dict[str, Any]) -> None:
        """Handle memory configuration update events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_CONFIG_UPDATED,
                'changes': event_data.get('changes', {})
            })
        self.logger.record_event(
            event_type="memory_config_updated",
            message="Memory configuration updated",
            level="info",
            changes=event_data.get('changes', {})
        )

    async def _handle_memory_threshold(self, event_data: Dict[str, Any]) -> None:
        """Handle memory threshold reached events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_THRESHOLD_REACHED,
                'threshold': event_data.get('threshold'),
                'current_usage': event_data.get('current_usage')
            })
        self.logger.record_event(
            event_type="memory_threshold_reached",
            message="Memory threshold reached",
            level="warning",
            threshold=event_data.get('threshold'),
            current_usage=event_data.get('current_usage')
        )

    async def _handle_memory_cleanup(self, event_data: Dict[str, Any]) -> None:
        """Handle memory cleanup events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_CLEANUP_STARTED,
                'threshold': event_data.get('threshold'),
                'current_usage': event_data.get('current_usage')
            })
        self.logger.record_event(
            event_type="memory_cleanup_started",
            message="Memory cleanup initiated",
            level="info",
            threshold=event_data.get('threshold'),
            current_usage=event_data.get('current_usage')
        )

    async def _handle_memory_stats_update(self, event_data: Dict[str, Any]) -> None:
        """Handle memory statistics update events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_STATS_UPDATED,
                'stats': event_data.get('stats', {})
            })
        # Update metrics history with new memory stats
        self._update_metrics_with_memory_stats(event_data.get('stats', {}))

    async def _handle_memory_error(self, event_data: Dict[str, Any]) -> None:
        """Handle memory error events."""
        with self._lock:
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_ERROR,
                'error_msg': event_data.get('error_msg'),
                'error_type': event_data.get('error_type')
            })
        self.logger.log_error(
            error_msg=event_data.get('error_msg', 'Unknown memory error'),
            error_type=event_data.get('error_type', 'memory_error'),
            stack_trace=event_data.get('stack_trace')
        )

    def _update_metrics_with_memory_stats(self, memory_stats: Dict[str, Any]) -> None:
        """Update metrics history with new memory statistics."""
        with self._lock:
            if self._metrics_history:
                last_metrics = self._metrics_history[-1]
                last_metrics['memory'] = memory_stats
                self._metrics_history[-1] = last_metrics

    def get_memory_events_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the memory events history in a thread-safe way."""
        with self._lock:
            return list(self._memory_events_history)

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
            # Display recent memory events
            recent_events = self.get_memory_events_history()[-5:]  # Show last 5 events
            if recent_events:
                print("\nRecent Memory Events:")
                for event in recent_events:
                    print(f"- {event['event_type']} at {time.strftime('%H:%M:%S', time.localtime(event['timestamp']))}")
        if "training" in metrics:
            print(f"Training Progress: {metrics['training']['progress']:.2f}%")
        if "curiosity" in metrics:
            print(f"Curiosity Score: {metrics['curiosity']['score']:.2f}")
        print("----------------------")
