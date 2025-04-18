from typing import Dict, Any, List
from threading import Thread, Event, Lock
from collections import deque
from sovl_memory import MemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_events import MemoryEventDispatcher, MemoryEventTypes
from sovl_state import SOVLState, StateManager
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
        state_manager: StateManager,
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
            state_manager: StateManager instance for state-related metrics.
            update_interval: Time interval (in seconds) between metric updates.
        """
        self.memory_manager = memory_manager
        self.training_manager = training_manager
        self.curiosity_manager = curiosity_manager
        self.config_manager = config_manager
        self.logger = logger
        self.memory_dispatcher = memory_dispatcher
        self.state_manager = state_manager
        self.update_interval = update_interval or self.config_manager.get("monitor.update_interval", 1.0)
        self._stop_event = Event()
        self._monitor_thread = None
        self._lock = Lock()
        self._metrics_history = deque(maxlen=100)
        self._memory_events_history = deque(maxlen=100)
        self._state_events_history = deque(maxlen=100)
        
        # Register memory event handlers
        self._register_memory_event_handlers()
        # Register state event handlers
        self._register_state_event_handlers()

    def _register_memory_event_handlers(self) -> None:
        """Register handlers for memory-related events."""
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_INITIALIZED, self._handle_memory_initialized)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_CONFIG_UPDATED, self._handle_memory_config_update)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_THRESHOLD_REACHED, self._handle_memory_threshold)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_CLEANUP_STARTED, self._handle_memory_cleanup)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_STATS_UPDATED, self._handle_memory_stats_update)
        self.memory_dispatcher.subscribe(MemoryEventTypes.MEMORY_ERROR, self._handle_memory_error)

    def _register_state_event_handlers(self) -> None:
        """Register handlers for state-related events."""
        self.state_manager.get_state().add_state_change_callback(self._handle_state_change)
        self.state_manager.get_state().add_confidence_callback(self._handle_confidence_change)
        self.state_manager.get_state().add_memory_usage_callback(self._handle_memory_usage_change)

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

    async def _handle_state_change(self, event_data: Dict[str, Any]) -> None:
        """Handle state change events."""
        with self._lock:
            self._state_events_history.append({
                'timestamp': time.time(),
                'event_type': 'state_change',
                'changes': event_data.get('changes', {})
            })
        self.logger.record_event(
            event_type="state_change",
            message="System state changed",
            level="info",
            changes=event_data.get('changes', {})
        )

    async def _handle_confidence_change(self, event_data: Dict[str, Any]) -> None:
        """Handle confidence change events."""
        with self._lock:
            self._state_events_history.append({
                'timestamp': time.time(),
                'event_type': 'confidence_change',
                'confidence': event_data.get('confidence', 0.0)
            })
        self.logger.record_event(
            event_type="confidence_change",
            message="Confidence level changed",
            level="info",
            confidence=event_data.get('confidence', 0.0)
        )

    async def _handle_memory_usage_change(self, event_data: Dict[str, Any]) -> None:
        """Handle memory usage change events."""
        with self._lock:
            self._state_events_history.append({
                'timestamp': time.time(),
                'event_type': 'memory_usage_change',
                'usage': event_data.get('usage', 0.0)
            })
        self.logger.record_event(
            event_type="memory_usage_change",
            message="Memory usage changed",
            level="info",
            usage=event_data.get('usage', 0.0)
        )

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
        metrics = {
            'memory': self.memory_manager.get_memory_stats(),
            'training': self.training_manager.get_training_progress(),
            'curiosity': self.curiosity_manager.get_curiosity_scores(),
            'state': self._collect_state_metrics()
        }
        return metrics

    def _collect_state_metrics(self) -> Dict[str, Any]:
        """Collect state-related metrics."""
        state = self.state_manager.get_state()
        return {
            'confidence_history': list(state.get_confidence_history()),
            'memory_usage': state._calculate_memory_usage(),
            'state_hash': state.state_hash(),
            'cache_size': len(state._cache),
            'dream_memory_size': len(state._dream_memory)
        }

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the metrics history in a thread-safe way."""
        with self._lock:
            return list(self._metrics_history)  # Return a copy to avoid race conditions

    def _display_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display collected metrics in a user-friendly format."""
        print("\n=== System Metrics ===")
        
        # Display memory metrics
        print("\nMemory Metrics:")
        memory_stats = metrics['memory']
        print(f"  Allocated: {memory_stats['allocated_mb']:.2f} MB")
        print(f"  Reserved: {memory_stats['reserved_mb']:.2f} MB")
        print(f"  Cache: {memory_stats['cache_mb']:.2f} MB")
        
        # Display training metrics
        print("\nTraining Metrics:")
        training_progress = metrics['training']
        print(f"  Progress: {training_progress['progress']:.2f}%")
        print(f"  Current Loss: {training_progress['current_loss']:.4f}")
        
        # Display curiosity metrics
        print("\nCuriosity Metrics:")
        curiosity_scores = metrics['curiosity']
        print(f"  Score: {curiosity_scores['score']:.2f}")
        print(f"  Pressure: {curiosity_scores['pressure']:.2f}")
        
        # Display state metrics
        print("\nState Metrics:")
        state_metrics = metrics['state']
        print(f"  Memory Usage: {state_metrics['memory_usage']:.2f}%")
        print(f"  Cache Size: {state_metrics['cache_size']}")
        print(f"  Dream Memory Size: {state_metrics['dream_memory_size']}")
        print(f"  State Hash: {state_metrics['state_hash'][:8]}...")
        
        # Display recent state events
        print("\nRecent State Events:")
        with self._lock:
            recent_events = list(self._state_events_history)[-5:]  # Show last 5 events
        for event in recent_events:
            print(f"  [{time.strftime('%H:%M:%S', time.localtime(event['timestamp']))}] {event['event_type']}")
        
        print("\n=====================")

    def get_state_events_history(self) -> List[Dict[str, Any]]:
        """Get a copy of the state events history in a thread-safe way."""
        with self._lock:
            return list(self._state_events_history)
