import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    additional_info: Dict[str, Any]

class ErrorManager:
    """Centralized error management system for SOVL."""
    
    def __init__(self, config_manager: Any, logger: Any):
        self.config_manager = config_manager
        self.logger = logger
        self.error_history = []
        self.max_error_history = config_manager.get("max_error_history", 1000)
        
    def _create_error_context(
        self,
        error: Exception,
        error_type: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Create standardized error context."""
        return ErrorContext(
            error_type=error_type,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            additional_info=additional_info or {}
        )
        
    def _log_error(self, context: ErrorContext) -> None:
        """Log error with standardized format."""
        self.logger.record_event(
            event_type=f"error_{context.error_type}",
            message=context.error_message,
            level="error",
            additional_info={
                "stack_trace": context.stack_trace,
                "timestamp": context.timestamp.isoformat(),
                **context.additional_info
            }
        )
        
        # Maintain error history
        self.error_history.append(context)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
            
    def handle_training_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle training-related errors."""
        error_context = self._create_error_context(
            error=error,
            error_type="training",
            additional_info=context
        )
        self._log_error(error_context)
        
        # Implement recovery logic
        try:
            # Add recovery steps here
            return False  # Indicate failure
        except Exception as recovery_error:
            self._log_error(self._create_error_context(
                error=recovery_error,
                error_type="recovery",
                additional_info={"original_error": error_context.error_message}
            ))
            return False
            
    def handle_scaffold_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle scaffold-related errors."""
        error_context = self._create_error_context(
            error=error,
            error_type="scaffold",
            additional_info=context
        )
        self._log_error(error_context)
        return False
        
    def handle_curiosity_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle curiosity-related errors."""
        error_context = self._create_error_context(
            error=error,
            error_type="curiosity",
            additional_info=context
        )
        self._log_error(error_context)
        return False
        
    def handle_memory_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle memory-related errors."""
        error_context = self._create_error_context(
            error=error,
            error_type="memory",
            additional_info=context
        )
        self._log_error(error_context)
        return False
        
    def get_error_history(self, limit: Optional[int] = None) -> List[ErrorContext]:
        """Get error history with optional limit."""
        if limit is None:
            return self.error_history
        return self.error_history[-limit:]
        
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear() 