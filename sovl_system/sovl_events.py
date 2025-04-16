from collections import defaultdict
import logging
import re
import time
from typing import Callable, Any, Optional, Set, Dict
from threading import Lock
from contextlib import contextmanager
import asyncio

# Type alias for callbacks
EventCallback = Callable[..., None]

# Event type validation regex (alphanumeric, underscores, hyphens, dots)
EVENT_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

class EventDispatcher:
    """Manages event subscriptions and notifications in a thread-safe manner.

    This class provides a generic event handling system that allows components to
    subscribe to specific event types and receive notifications when those events occur.
    It supports prioritized callbacks, async notifications, event metadata, and duplicate
    subscription detection.

    Attributes:
        _subscribers: Dict mapping event types to lists of (priority, callback) tuples.
        _lock: Thread lock for thread-safe operations.
        _logger: Logger instance for error and debug logging.
        _notification_depth: Tracks active notifications to defer changes.
        _deferred_changes: Stores deferred unsubscriptions during notifications.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the event dispatcher.

        Args:
            logger: Optional logger instance for error logging. If None, uses the default logging module.
        """
        self._subscribers = defaultdict(list)  # List of (priority, callback) tuples
        self._lock = Lock()
        self._logger = logger or logging.getLogger(__name__)
        self._notification_depth = 0  # Tracks active notifications
        self._deferred_changes: Dict[str, Set[EventCallback]] = defaultdict(set)  # Deferred unsubscriptions

    @contextmanager
    def _locked(self):
        """Context manager for thread-safe operations."""
        with self._lock:
            yield

    def _validate_event_type(self, event_type: str) -> None:
        """Validate the event type format.

        Args:
            event_type: The event type to validate.

        Raises:
            ValueError: If the event type is invalid.
        """
        if not event_type or not isinstance(event_type, str):
            self._logger.error("Event type must be a non-empty string")
            raise ValueError("Event type must be a non-empty string")
        if not EVENT_TYPE_PATTERN.match(event_type):
            self._logger.error(f"Invalid event type format: {event_type}")
            raise ValueError("Event type must match pattern [a-zA-Z0-9_-.]+")

    def subscribe(self, event_type: str, callback: EventCallback, priority: int = 0) -> None:
        """Subscribe a callback to an event type with an optional priority.

        Args:
            event_type: The type of event to subscribe to (e.g., 'config_change').
            callback: The function to call when the event occurs. Must be callable.
            priority: Higher priority callbacks are executed first (default: 0).

        Raises:
            TypeError: If the callback is not callable.
            ValueError: If the event_type is invalid.
        """
        if not callable(callback):
            self._logger.error(f"Invalid callback for event {event_type}: must be callable")
            raise TypeError("Callback must be callable")
        self._validate_event_type(event_type)

        with self._locked():
            # Check for duplicate subscription
            for prio, cb in self._subscribers[event_type]:
                if cb == callback:
                    self._logger.warning(
                        f"Duplicate subscription detected for event {event_type}: {callback.__qualname__}"
                    )
                    return
            self._subscribers[event_type].append((priority, callback))
            # Sort by priority (descending) for consistent execution order
            self._subscribers[event_type].sort(key=lambda x: x[0], reverse=True)
            self._logger.debug(f"Subscribed callback {callback.__qualname__} to event {event_type} with priority {priority}")

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Unsubscribe a callback from an event type.

        Args:
            event_type: The type of event to unsubscribe from.
            callback: The function to remove from the subscribers list.

        Raises:
            ValueError: If the event_type is invalid.
        """
        self._validate_event_type(event_type)

        with self._locked():
            if self._notification_depth > 0:
                # Defer unsubscription during notification
                self._deferred_changes[event_type].add(callback)
                self._logger.debug(f"Deferred unsubscription for {event_type}: {callback.__qualname__}")
                return
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    (prio, cb) for prio, cb in self._subscribers[event_type] if cb != callback
                ]
                self._logger.debug(f"Unsubscribed callback {callback.__qualname__} from event {event_type}")
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]  # Clean up empty lists

    def notify(self, event_type: str, *args, include_metadata: bool = False, **kwargs) -> None:
        """Notify all subscribers of an event synchronously.

        Args:
            event_type: The type of event to notify.
            include_metadata: If True, includes event metadata (e.g., timestamp, event_id).
            *args: Positional arguments to pass to the callback functions.
            **kwargs: Keyword arguments to pass to the callback functions.

        Raises:
            ValueError: If the event_type is invalid.
        """
        self._validate_event_type(event_type)

        with self._locked():
            self._notification_depth += 1
            subscribers = [(prio, cb) for prio, cb in self._subscribers.get(event_type, [])]

        try:
            metadata = {
                "event_id": f"{event_type}_{time.time()}",
                "timestamp": time.time(),
            } if include_metadata else {}

            for _, callback in subscribers:
                try:
                    if include_metadata:
                        callback(*args, metadata=metadata, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    self._logger.error(
                        f"Error in event handler for {event_type} in {callback.__qualname__}: {str(e)}",
                        exc_info=True
                    )
        finally:
            with self._locked():
                self._notification_depth -= 1
                if self._notification_depth == 0:
                    # Process deferred changes
                    for evt, callbacks in self._deferred_changes.items():
                        for cb in callbacks:
                            self._subscribers[evt] = [
                                (prio, c) for prio, c in self._subscribers[evt] if c != cb
                            ]
                            if not self._subscribers[evt]:
                                del self._subscribers[evt]
                    self._deferred_changes.clear()

    async def async_notify(self, event_type: str, *args, include_metadata: bool = False, **kwargs) -> None:
        """Notify all subscribers of an event asynchronously.

        Args:
            event_type: The type of event to notify.
            include_metadata: If True, includes event metadata (e.g., timestamp, event_id).
            *args: Positional arguments to pass to the callback functions.
            **kwargs: Keyword arguments to pass to the callback functions.

        Raises:
            ValueError: If the event_type is invalid.
        """
        self._validate_event_type(event_type)

        with self._locked():
            self._notification_depth += 1
            subscribers = [(prio, cb) for prio, cb in self._subscribers.get(event_type, [])]

        try:
            metadata = {
                "event_id": f"{event_type}_{time.time()}",
                "timestamp": time.time(),
            } if include_metadata else {}

            for _, callback in subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        if include_metadata:
                            await callback(*args, metadata=metadata, **kwargs)
                        else:
                            await callback(*args, **kwargs)
                    else:
                        if include_metadata:
                            callback(*args, metadata=metadata, **kwargs)
                        else:
                            callback(*args, **kwargs)
                except Exception as e:
                    self._logger.error(
                        f"Error in async event handler for {event_type} in {callback.__qualname__}: {str(e)}",
                        exc_info=True
                    )
        finally:
            with self._locked():
                self._notification_depth -= 1
                if self._notification_depth == 0:
                    # Process deferred changes
                    for evt, callbacks in self._deferred_changes.items():
                        for cb in callbacks:
                            self._subscribers[evt] = [
                                (prio, c) for prio, c in self._subscribers[evt] if c != cb
                            ]
                            if not self._subscribers[evt]:
                                del self._subscribers[evt]
                    self._deferred_changes.clear()

    def get_subscribers(self, event_type: str) -> Set[EventCallback]:
        """Get the set of callbacks subscribed to a specific event type.

        Args:
            event_type: The type of event to query.

        Returns:
            A set of callback functions subscribed to the event type.

        Raises:
            ValueError: If the event_type is invalid.
        """
        self._validate_event_type(event_type)

        with self._locked():
            return {cb for _, cb in self._subscribers.get(event_type, [])}

    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """Clear subscribers for a specific event type or all events.

        Args:
            event_type: The type of event to clear subscribers for. If None, clears all subscribers.

        Raises:
            ValueError: If the event_type is invalid.
        """
        with self._locked():
            if event_type is None:
                self._subscribers.clear()
                self._deferred_changes.clear()
                self._logger.debug("Cleared all event subscribers")
            else:
                self._validate_event_type(event_type)
                if event_type in self._subscribers:
                    del self._subscribers[event_type]
                    self._deferred_changes.pop(event_type, None)
                    self._logger.debug(f"Cleared subscribers for event {event_type}")

    def has_subscribers(self, event_type: str) -> bool:
        """Check if there are any subscribers for a specific event type.

        Args:
            event_type: The type of event to check.

        Returns:
            True if there are subscribers for the event type, False otherwise.

        Raises:
            ValueError: If the event_type is invalid.
        """
        self._validate_event_type(event_type)

        with self._locked():
            return event_type in self._subscribers and len(self._subscribers[event_type]) > 0

    def cleanup_stale_events(self) -> None:
        """Remove event types with no subscribers to free memory."""
        with self._locked():
            stale_events = [evt for evt, subs in self._subscribers.items() if not subs]
            for evt in stale_events:
                del self._subscribers[evt]
                self._deferred_changes.pop(evt, None)
                self._logger.debug(f"Cleaned up stale event type: {evt}")
