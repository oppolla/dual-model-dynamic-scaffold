from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import json
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorHandler
from sovl_io import load_training_data, InsufficientDataError

class DataProvider(ABC):
    """Abstract interface for data providers."""
    @abstractmethod
    def load_data(self, source: str, min_entries: int = 0) -> List[Dict]:
        """
        Load data from a specified source.
        
        Args:
            source: Identifier for the data source (e.g., file path, database URI).
            min_entries: Minimum number of entries required.
        
        Returns:
            List of data entries (dictionaries).
        
        Raises:
            InsufficientDataError: If not enough data is loaded.
            ValueError: If the source is invalid.
        """
        pass

    @abstractmethod
    def validate_data(self, data: List[Dict]) -> bool:
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

    def load_data(self, source: str, min_entries: int = 0) -> List[Dict]:
        """Load data from a JSONL file."""
        try:
            data = load_training_data(source, min_entries=min_entries)
            self.logger.record({
                "event": "data_loaded",
                "source": source,
                "entry_count": len(data),
                "timestamp": time.time(),
                "conversation_id": "data_load"
            })
            return data
        except InsufficientDataError as e:
            raise
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load data from {source}: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": "data_load"
            })
            raise ValueError(f"Invalid data source {source}: {str(e)}")

    def validate_data(self, data: List[Dict]) -> bool:
        """Validate JSONL data entries."""
        if not data:
            self.logger.record({
                "warning": "Empty data provided for validation",
                "timestamp": time.time(),
                "conversation_id": "data_validate"
            })
            return False
        for entry in data:
            if not isinstance(entry, dict):
                self.logger.record({
                    "warning": "Invalid data entry: not a dictionary",
                    "entry": str(entry)[:100],
                    "timestamp": time.time(),
                    "conversation_id": "data_validate"
                })
                return False
            if "prompt" not in entry or not isinstance(entry["prompt"], str):
                self.logger.record({
                    "warning": "Invalid data entry: missing or invalid prompt",
                    "entry": str(entry)[:100],
                    "timestamp": time.time(),
                    "conversation_id": "data_validate"
                })
                return False
        return True

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
        self.core_config = config_manager.get_section("core_config")
        self.logger = logger
        self.error_handler = error_handler
        self.random_seed = self.core_config.get("random_seed", 42)
        self.default_split_ratio = self.core_config.get("valid_split_ratio", 0.2)
        self.default_source = self.core_config.get("data_source", "sovl_seed.jsonl")
        
        # Initialize default provider
        self.provider = FileDataProvider(logger)
        
    def set_provider(self, provider: DataProvider):
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

    def load_and_split(self, source: Optional[str] = None, min_entries: int = 0, 
                      valid_split_ratio: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
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
            # Load data
            data = self.provider.load_data(source, min_entries)
            
            if not data:
                self.logger.record({
                    "warning": f"No data loaded from {source}",
                    "timestamp": time.time(),
                    "conversation_id": "data_load"
                })
                return [], []
            
            # Validate data
            if not self.provider.validate_data(data):
                self.error_handler.handle_data_error(
                    error=ValueError("Data validation failed"),
                    context={"source": source, "entry_count": len(data)},
                    conversation_id="data_load"
                )
                return [], []
            
            # Split data
            train_data, valid_data = self._split_data(data, valid_split_ratio)
            
            self.logger.record({
                "event": "data_loaded_and_split",
                "source": source,
                "total_entries": len(data),
                "train_entries": len(train_data),
                "valid_entries": len(valid_data),
                "split_ratio": valid_split_ratio,
                "timestamp": time.time(),
                "conversation_id": "data_load"
            })
            return train_data, valid_data
            
        except InsufficientDataError as e:
            self.error_handler.handle_data_error(
                error=e,
                context={"source": source, "min_entries": min_entries},
                conversation_id="data_load"
            )
            return [], []
        except Exception as e:
            self.error_handler.handle_data_error(
                error=e,
                context={"source": source, "min_entries": min_entries},
                conversation_id="data_load"
            )
            return [], []

    def _split_data(self, data: List[Dict], split_ratio: float) -> Tuple[List[Dict], List[Dict]]:
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
        random.shuffle(data)
        split_idx = int(len(data) * (1 - split_ratio))
        return data[:split_idx], data[split_idx:]

    def validate_data(self, data: List[Dict]) -> bool:
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

    def get_data_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Compute statistics about the data.
        
        Args:
            data: List of data entries.
        
        Returns:
            Dictionary with statistics (e.g., entry count, prompt lengths).
        """
        if not data:
            return {"entry_count": 0, "avg_prompt_length": 0, "has_response": False}
        
        prompt_lengths = [len(entry.get("prompt", "")) for entry in data]
        has_response = any("response" in entry for entry in data)
        avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
        
        return {
            "entry_count": len(data),
            "avg_prompt_length": avg_prompt_length,
            "has_response": has_response
        }
