from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import time
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorHandler
from sovl_io import load_training_data, InsufficientDataError

class DataProvider(ABC):
    """Abstract interface for data providers."""
    
    @abstractmethod
    def load_data(self, source: str, min_entries: int = 0) -> List[Dict[str, Any]]:
        """
        Load data from a specified source.
        
        Args:
            source: Identifier for the data source (e.g., file path, database URI).
            min_entries: Minimum number of entries required.
        
        Returns:
            List of data entries as dictionaries.
        
        Raises:
            InsufficientDataError: If not enough data is loaded.
            ValueError: If the source is invalid.
        """
        pass

    @abstractmethod
    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate the integrity of loaded data.
        
        Args:
            data: List of data entries to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        pass

class FileDataProvider(DataProvider):
    """Data provider for loading data from JSONL files."""
    
    def __init__(self, logger: Logger):
        self.logger = logger

    def load_data(self, source: str, min_entries: int = 0) -> List[Dict[str, Any]]:
        """Load data from a JSONL file."""
        try:
            data = load_training_data(source, min_entries=min_entries)
            self._log_load_success(source, len(data))
            return data
        except InsufficientDataError:
            raise
        except Exception as e:
            self._log_load_error(source, e)
            raise ValueError(f"Invalid data source {source}: {str(e)}")

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """Validate JSONL data entries."""
        if not data:
            self._log_validation_warning("Empty data provided for validation")
            return False
        
        for entry in data:
            if not self._is_valid_entry(entry):
                return False
        return True

    def _is_valid_entry(self, entry: Any) -> bool:
        """Check if a single data entry is valid."""
        if not isinstance(entry, dict):
            self._log_validation_warning(f"Invalid data entry: not a dictionary, got {str(entry)[:100]}")
            return False
        if "prompt" not in entry or not isinstance(entry["prompt"], str):
            self._log_validation_warning(f"Invalid data entry: missing or invalid prompt, got {str(entry)[:100]}")
            return False
        return True

    def _log_load_success(self, source: str, entry_count: int) -> None:
        """Log successful data load event."""
        self.logger.record({
            "event": "data_loaded",
            "source": source,
            "entry_count": entry_count,
            "timestamp": time.time(),
            "conversation_id": "data_load"
        })

    def _log_load_error(self, source: str, error: Exception) -> None:
        """Log data load error."""
        self.logger.record({
            "error": f"Failed to load data from {source}: {str(error)}",
            "timestamp": time.time(),
            "conversation_id": "data_load"
        })

    def _log_validation_warning(self, message: str) -> None:
        """Log validation warning."""
        self.logger.record({
            "warning": message,
            "timestamp": time.time(),
            "conversation_id": "data_validate"
        })

class DataManager:
    """Manages loading, validation, and splitting of training data."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, error_handler: ErrorHandler):
        """
        Initialize the DataManager.
        
        Args:
            config_manager: Configuration manager for settings.
            logger: Logger for recording events and errors.
            error_handler: Handler for managing errors.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.error_handler = error_handler
        self._initialize_config()
        self.provider: DataProvider = FileDataProvider(logger)

    def _initialize_config(self) -> None:
        """Initialize configuration settings."""
        core_config = self.config_manager.get_section("core_config")
        self.random_seed: int = core_config.get("random_seed", 42)
        self.default_split_ratio: float = core_config.get("valid_split_ratio", 0.2)
        self.default_source: str = core_config.get("data_source", "sovl_seed.jsonl")

    def set_provider(self, provider: DataProvider) -> None:
        """
        Set a custom data provider.
        
        Args:
            provider: Instance of a DataProvider implementation.
        """
        self.provider = provider
        self.logger.record({
            "event": "data_provider_set",
            "provider_type": provider.__class__.__name__,
            "timestamp": time.time(),
            "conversation_id": "data_config"
        })

    def load_and_split(
        self,
        source: Optional[str] = None,
        min_entries: int = 0,
        valid_split_ratio: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load data from a source and split into training and validation sets.
        
        Args:
            source: Data source (e.g., file path). Uses config default if None.
            min_entries: Minimum number of entries required.
            valid_split_ratio: Ratio for validation split. Uses config default if None.
        
        Returns:
            Tuple of (train_data, valid_data).
        """
        source = source or self.default_source
        valid_split_ratio = valid_split_ratio or self.default_split_ratio

        try:
            data = self._load_data(source, min_entries)
            if not data:
                return [], []

            if not self.provider.validate_data(data):
                self._handle_validation_error(source, len(data))
                return [], []

            train_data, valid_data = self._split_data(data, valid_split_ratio)
            self._log_split_success(source, len(data), len(train_data), len(valid_data), valid_split_ratio)
            return train_data, valid_data

        except InsufficientDataError as e:
            self._handle_data_error(e, source, min_entries)
            return [], []
        except Exception as e:
            self._handle_data_error(e, source, min_entries)
            return [], []

    def _load_data(self, source: str, min_entries: int) -> List[Dict[str, Any]]:
        """Load data with logging."""
        data = self.provider.load_data(source, min_entries)
        if not data:
            self.logger.record({
                "warning": f"No data loaded from {source}",
                "timestamp": time.time(),
                "conversation_id": "data_load"
            })
        return data

    def _split_data(
        self, data: List[Dict[str, Any]], split_ratio: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split data into training and validation sets.
        
        Args:
            data: List of data entries.
            split_ratio: Fraction of data to use for validation.
        
        Returns:
            Tuple of (train_data, valid_data).
        """
        if not data:
            return [], []
        
        random.seed(self.random_seed)
        shuffled_data = data.copy()  # Avoid modifying original data
        random.shuffle(shuffled_data)
        split_idx = int(len(shuffled_data) * (1 - split_ratio))
        return shuffled_data[:split_idx], shuffled_data[split_idx:]

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate data using the current provider.
        
        Args:
            data: List of data entries to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        try:
            return self.provider.validate_data(data)
        except Exception as e:
            self.error_handler.handle_data_error(
                error=e,
                context={"data_sample": str(data[:2])[:100]},
                conversation_id="data_validate"
            )
            return False

    def get_data_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about the data.
        
        Args:
            data: List of data entries.
        
        Returns:
            Dictionary with statistics (e.g., entry count, prompt lengths).
        """
        if not data:
            return {"entry_count": 0, "avg_prompt_length": 0.0, "has_response": False}
        
        stats = {
            "entry_count": len(data),
            "avg_prompt_length": 0.0,
            "has_response": False
        }
        
        prompt_lengths = [len(entry.get("prompt", "")) for entry in data]
        stats["has_response"] = any("response" in entry for entry in data)
        stats["avg_prompt_length"] = (
            sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
        )
        
        return stats

    def _handle_validation_error(self, source: str, entry_count: int) -> None:
        """Handle data validation errors."""
        self.error_handler.handle_data_error(
            error=ValueError("Data validation failed"),
            context={"source": source, "entry_count": entry_count},
            conversation_id="data_load"
        )

    def _handle_data_error(self, error: Exception, source: str, min_entries: int) -> None:
        """Handle general data errors."""
        self.error_handler.handle_data_error(
            error=error,
            context={"source": source, "min_entries": min_entries},
            conversation_id="data_load"
        )

    def _log_split_success(
        self, source: str, total: int, train: int, valid: int, split_ratio: float
    ) -> None:
        """Log successful data split event."""
        self.logger.record({
            "event": "data_loaded_and_split",
            "source": source,
            "total_entries": total,
            "train_entries": train,
            "valid_entries": valid,
            "split_ratio": split_ratio,
            "timestamp": time.time(),
            "conversation_id": "data_load"
        })
