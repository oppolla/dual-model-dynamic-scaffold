import asyncio
import logging
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast # Added Tuple, cast

# Type alias for callbacks - clearer name
EventHandler = Callable[..., Any] # Changed name, allow return value typing

# Event type validation regex (alphanumeric, underscores, hyphens, dots)
# Defined as a module-level constant for clarity
EVENT_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

# Logger setup
_logger = logging.getLogger(__name__) # Module-level logger setup

class EventDispatcher:
    """
    Manages event subscriptions and notifications in a thread-safe manner.

    This class provides a robust event handling system allowing components
    to subscribe to specific event types and receive notifications when those
    events occur.

    Features:
        - Thread-safe operations using locks.
        - Prioritized event handlers (higher priority executed first).
        - Synchronous (`notify`) and asynchronous (`async_notify`) notification.
        - Optional event metadata (timestamp, event_id).
        - Validation of event types and handlers.
        - Duplicate subscription detection and warning.
        - Deferred unsubscription: Prevents errors if handlers unsubscribe
          during a notification cycle.
        - Cleanup methods for stale events or all subscribers.
    """

    # Using __slots__ can slightly reduce memory footprint for instances,
    # especially if many dispatchers are created. It also prevents accidental
    # creation of new instance attributes outside __init__.
    __slots__ = (
        '_subscribers',
        '_lock',
        '_logger',
        '_notification_depth',
        '_deferred_unsubscriptions',
    )

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the EventDispatcher.

        Args:
            logger: Optional logger instance. If None, uses a logger named
                    based on this module (__name__).
        """
        # Stores subscribers as: { event_type: List[Tuple[priority, EventHandler]] }
        self._subscribers: Dict[str, List[Tuple[int, EventHandler]]] = defaultdict(list)
        self._lock = Lock()
        self._logger = logger or _logger # Use provided or module logger
        self._notification_depth: int = 0  # Tracks nesting of notify/async_notify calls
        # Stores deferred unsubscriptions: { event_type: Set[EventHandler] }
        self._deferred_unsubscriptions: Dict[str, Set[EventHandler]] = defaultdict(set)

    @contextmanager
    def _locked(self):
        """Context manager for acquiring and releasing the internal lock."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def _validate_event_type(self, event_type: Any) -> str:
        """
        Validates the event type format and returns it if valid.

        Args:
            event_type: The event type to validate.

        Returns:
            The validated event type string.

        Raises:
            ValueError: If the event type is not a non-empty string or
                        does not match the required pattern.
        """
        if not isinstance(event_type, str) or not event_type:
            msg = "Event type must be a non-empty string"
            self._logger.error(msg)
            raise ValueError(msg)
        if not EVENT_TYPE_PATTERN.match(event_type):
            msg = f"Invalid event type format: '{event_type}'. Must match pattern [a-zA-Z0-9_.-]+"
            self._logger.error(msg)
            raise ValueError(msg)
        return event_type # Return validated string

    def _validate_handler(self, handler: Any) -> EventHandler:
        """
        Validates that the handler is callable.

        Args:
            handler: The event handler to validate.

        Returns:
            The validated event handler.

        Raises:
            TypeError: If the handler is not callable.
        """
        if not callable(handler):
            msg = f"Invalid event handler: {type(handler).__name__} is not callable."
            self._logger.error(msg)
            raise TypeError(msg)
        # Use cast to inform type checkers that it's now confirmed callable
        return cast(EventHandler, handler)

    def subscribe(self, event_type: str, handler: EventHandler, priority: int = 0) -> None:
        """
        Subscribes an event handler to an event type with optional priority.

        Handlers with higher priority values are executed first. If multiple
        handlers have the same priority, their execution order is not guaranteed
        relative to each other, but respects the overall priority sequence.

        Args:
            event_type: The type of event to subscribe to (e.g., 'user.created').
                        Validated against `EVENT_TYPE_PATTERN`.
            handler: The function or method to call when the event occurs.
                     Must be callable.
            priority: An integer representing the handler's priority. Higher
                      numbers execute earlier (default: 0).

        Raises:
            ValueError: If the event_type is invalid.
            TypeError: If the handler is not callable.
        """
        valid_event_type = self._validate_event_type(event_type)
        valid_handler = self._validate_handler(handler)
        handler_name = getattr(valid_handler, '__qualname__', repr(valid_handler))

        with self._locked():
            sub_list = self._subscribers[valid_event_type]

            # Check for duplicate subscription (same handler for the same event)
            if any(h == valid_handler for _, h in sub_list):
                self._logger.warning(
                    f"Handler '{handler_name}' is already subscribed to event '{valid_event_type}'. Ignoring duplicate."
                )
                return

            # Add and maintain sorted order by priority (descending)
            # Using bisect_left/insort could be slightly more performant on average
            # for large numbers of subscribers per event, but simple append + sort
            # is often clear enough and efficient for typical use cases.
            sub_list.append((priority, valid_handler))
            sub_list.sort(key=lambda item: item[0], reverse=True)
            self._logger.debug(
                f"Subscribed handler '{handler_name}' to event '{valid_event_type}' with priority {priority}"
            )

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribes an event handler from an event type.

        If called during a notification cycle for the *same* event type, the
        unsubscription is deferred until the cycle completes to avoid modifying
        the list of handlers currently being iterated over.

        Args:
            event_type: The type of event to unsubscribe from.
            handler: The specific handler function/method to remove.

        Raises:
            ValueError: If the event_type is invalid.
            TypeError: If the handler is not callable (though less likely to be caught
                       here if it wasn't subscribed correctly).
        """
        valid_event_type = self._validate_event_type(event_type)
        valid_handler = self._validate_handler(handler) # Basic check
        handler_name = getattr(valid_handler, '__qualname__', repr(valid_handler))

        with self._locked():
            if self._notification_depth > 0 and valid_event_type in self._subscribers:
                # Check if this event type is *potentially* being notified.
                # It's safer to defer if *any* notification is in progress,
                # although strictly necessary only if the current notification
                # is for valid_event_type. This simpler check avoids complexity.
                self._deferred_unsubscriptions[valid_event_type].add(valid_handler)
                self._logger.debug(
                    f"Deferred unsubscription for handler '{handler_name}' from event '{valid_event_type}'"
                )
                return

            # Perform immediate unsubscription if no notification is active
            # or if the event type wasn't found anyway.
            if valid_event_type in self._subscribers:
                original_count = len(self._subscribers[valid_event_type])
                # Create a new list excluding the target handler
                self._subscribers[valid_event_type] = [
                    (prio, h) for prio, h in self._subscribers[valid_event_type] if h != valid_handler
                ]
                new_count = len(self._subscribers[valid_event_type])

                if new_count < original_count:
                     self._logger.debug(
                        f"Unsubscribed handler '{handler_name}' from event '{valid_event_type}'"
                    )
                else:
                    self._logger.warning(
                        f"Attempted to unsubscribe handler '{handler_name}' from event '{valid_event_type}', but it was not found."
                    )

                # Clean up the event type entry if no subscribers remain
                if not self._subscribers[valid_event_type]:
                    del self._subscribers[valid_event_type]
                    # Also remove any potentially related deferred unsubscriptions
                    self._deferred_unsubscriptions.pop(valid_event_type, None)
            else:
                 self._logger.warning(
                     f"Attempted to unsubscribe from non-existent or empty event type '{valid_event_type}'."
                 )


    def _prepare_notification(self, event_type: str) -> List[Tuple[int, EventHandler]]:
        """Internal helper to prepare for notification."""
        valid_event_type = self._validate_event_type(event_type)
        with self._locked():
            self._notification_depth += 1
            # IMPORTANT: Create a *copy* of the subscriber list.
            # This allows releasing the lock while iterating and calling handlers,
            # preventing deadlocks if a handler tries to subscribe/unsubscribe.
            subscribers_copy = list(self._subscribers.get(valid_event_type, []))
        return subscribers_copy

    def _finalize_notification(self) -> None:
        """Internal helper to finalize notification and process deferred actions."""
        with self._locked():
            self._notification_depth -= 1
            if self._notification_depth == 0:
                # Process deferred unsubscriptions only when the outermost notification cycle ends
                if self._deferred_unsubscriptions:
                    self._process_deferred_unsubscriptions()

    def _process_deferred_unsubscriptions(self) -> None:
        """
        Processes handlers marked for deferred unsubscription.
        Must be called while holding the lock and when notification_depth is 0.
        """
        if not self._deferred_unsubscriptions:
            return

        self._logger.debug("Processing deferred unsubscriptions...")
        for event_type, handlers_to_remove in self._deferred_unsubscriptions.items():
            if event_type in self._subscribers:
                initial_len = len(self._subscribers[event_type])
                self._subscribers[event_type] = [
                    (prio, h) for prio, h in self._subscribers[event_type]
                    if h not in handlers_to_remove
                ]
                removed_count = initial_len - len(self._subscribers[event_type])
                if removed_count > 0:
                    self._logger.debug(
                        f"Processed {removed_count} deferred unsubscription(s) for event '{event_type}'."
                    )

                # Clean up if event type becomes empty
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    self._logger.debug(f"Cleaned up event type '{event_type}' after deferred unsubscriptions.")

        self._deferred_unsubscriptions.clear()
        self._logger.debug("Finished processing deferred unsubscriptions.")


    def notify(self, event_type: str, *args: Any, include_metadata: bool = False, **kwargs: Any) -> None:
        """
        Notifies all subscribed handlers of an event synchronously.

        Handlers are called sequentially in order of priority (descending).
        If a handler raises an exception, it is logged, and notification
        continues to the next handler.

        Args:
            event_type: The type of event being triggered.
            *args: Positional arguments to pass to each handler.
            include_metadata: If True, an additional `metadata` keyword argument
                              (containing timestamp, event_id) is passed to handlers.
            **kwargs: Keyword arguments to pass to each handler.

        Raises:
            ValueError: If the event_type is invalid.
        """
        subscribers_copy = self._prepare_notification(event_type)
        if not subscribers_copy:
            self._finalize_notification() # Decrement depth even if no subscribers
            return

        metadata = {}
        if include_metadata:
            now = time.time()
            metadata = {
                "event_id": f"{event_type}-{now:.6f}", # Use hyphen, add precision
                "timestamp": now,
            }

        # Call handlers *outside* the lock
        for _, handler in subscribers_copy:
            handler_name = getattr(handler, '__qualname__', repr(handler))
            try:
                 # Ensure async handlers aren't accidentally called synchronously without await
                if asyncio.iscoroutinefunction(handler):
                    self._logger.warning(
                        f"Attempted to call async handler '{handler_name}' for event '{event_type}' "
                        f"using synchronous notify(). Skipping handler. Use async_notify() instead."
                    )
                    continue # Skip this async handler

                call_kwargs = kwargs.copy()
                if include_metadata:
                    call_kwargs['metadata'] = metadata

                handler(*args, **call_kwargs)

            except Exception as e:
                self._logger.error(
                    f"Error executing synchronous handler '{handler_name}' for event '{event_type}': {e}",
                    exc_info=True # Include stack trace in log
                )
            # Continue to the next handler regardless of errors

        # Finalize (decrement depth, process deferred actions if depth becomes 0)
        self._finalize_notification()


    async def async_notify(self, event_type: str, *args: Any, include_metadata: bool = False, **kwargs: Any) -> None:
        """
        Notifies all subscribed handlers of an event asynchronously.

        Handlers are called sequentially in order of priority (descending).
        Coroutine handlers are awaited. Synchronous handlers are called directly.
        If a handler raises an exception, it is logged, and notification
        continues to the next handler.

        Args:
            event_type: The type of event being triggered.
            *args: Positional arguments to pass to each handler.
            include_metadata: If True, an additional `metadata` keyword argument
                              (containing timestamp, event_id) is passed to handlers.
            **kwargs: Keyword arguments to pass to each handler.

        Raises:
            ValueError: If the event_type is invalid.
        """
        subscribers_copy = self._prepare_notification(event_type)
        if not subscribers_copy:
            self._finalize_notification() # Decrement depth even if no subscribers
            return

        metadata = {}
        if include_metadata:
            now = time.time()
            metadata = {
                "event_id": f"{event_type}-{now:.6f}",
                "timestamp": now,
            }

        # Call/await handlers *outside* the lock
        for _, handler in subscribers_copy:
            handler_name = getattr(handler, '__qualname__', repr(handler))
            try:
                call_kwargs = kwargs.copy()
                if include_metadata:
                    call_kwargs['metadata'] = metadata

                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **call_kwargs)
                else:
                    handler(*args, **call_kwargs) # Call synchronous handler directly

            except Exception as e:
                self._logger.error(
                    f"Error executing handler '{handler_name}' during async notification for event '{event_type}': {e}",
                    exc_info=True
                )
            # Continue to the next handler regardless of errors

        # Finalize (decrement depth, process deferred actions if depth becomes 0)
        self._finalize_notification()


    def get_subscribers(self, event_type: str) -> Set[EventHandler]:
        """
        Gets the set of unique handlers subscribed to a specific event type.

        Args:
            event_type: The event type to query.

        Returns:
            A set containing the handler functions/methods subscribed.

        Raises:
            ValueError: If the event_type is invalid.
        """
        valid_event_type = self._validate_event_type(event_type)
        with self._locked():
            # Return a copy as a set
            return {h for _, h in self._subscribers.get(valid_event_type, [])}

    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """
        Clears subscribers for a specific event type or all event types.

        Warning: This immediately removes subscribers. If called during a
        notification cycle, handlers removed by this method might not finish
        executing if they haven't been called yet in that cycle. Deferred
        unsubscriptions related to the cleared events are also removed.

        Args:
            event_type: The event type to clear. If None, clears subscribers
                        for *all* event types.

        Raises:
            ValueError: If a specific event_type is provided and is invalid.
        """
        with self._locked():
            if event_type is None:
                # Clear all subscribers
                self._subscribers.clear()
                self._deferred_unsubscriptions.clear()
                self._logger.info("Cleared all event subscribers and deferred unsubscriptions.")
            else:
                # Clear subscribers for a specific event type
                valid_event_type = self._validate_event_type(event_type)
                if valid_event_type in self._subscribers:
                    del self._subscribers[valid_event_type]
                    # Also remove any pending deferred unsubscriptions for this event
                    self._deferred_unsubscriptions.pop(valid_event_type, None)
                    self._logger.info(f"Cleared subscribers for event '{valid_event_type}'.")
                else:
                    self._logger.info(f"No subscribers to clear for event '{valid_event_type}'.")


    def has_subscribers(self, event_type: str) -> bool:
        """
        Checks if an event type has any active subscribers.

        Args:
            event_type: The event type to check.

        Returns:
            True if at least one handler is subscribed, False otherwise.

        Raises:
            ValueError: If the event_type is invalid.
        """
        valid_event_type = self._validate_event_type(event_type)
        with self._locked():
            # Check key exists and list is not empty
            return valid_event_type in self._subscribers # defaultdict ensures list exists, but it might be empty

    def cleanup_stale_events(self) -> None:
        """
        Removes event type entries that have no subscribers.

        This can help free up minor amounts of memory if many event types
        were subscribed to but later became empty via `unsubscribe`.
        Also cleans up related empty entries in deferred unsubscriptions.
        """
        with self._locked():
            # Find keys in _subscribers pointing to empty lists
            # Need to create a list of keys to remove to avoid changing dict size during iteration
            stale_subscriber_keys = [
                evt for evt, handlers in self._subscribers.items() if not handlers
            ]
            if stale_subscriber_keys:
                for evt in stale_subscriber_keys:
                    del self._subscribers[evt]
                    # Also remove from deferred, though they should ideally be removed
                    # when the subscriber list becomes empty in unsubscribe/clear
                    self._deferred_unsubscriptions.pop(evt, None)
                    self._logger.debug(f"Cleaned up stale event type entry: '{evt}'")

            # Less common: Check for deferred unsubscription keys that no longer exist in subscribers
            stale_deferred_keys = [
                evt for evt in self._deferred_unsubscriptions if evt not in self._subscribers
            ]
            if stale_deferred_keys:
                 for evt in stale_deferred_keys:
                     del self._deferred_unsubscriptions[evt]
                     self._logger.debug(f"Cleaned up stale deferred unsubscription entry: '{evt}'")
