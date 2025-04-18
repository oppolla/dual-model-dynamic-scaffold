import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock
from sovl_config import ConfigManager, ConfigSchema, ConfigKey, ConfigKeys
from sovl_logger import Logger

@pytest.fixture
def temp_config_file():
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump({
            "core_config": {
                "base_model_name": "test_model",
                "scaffold_model_name": "test_scaffold"
            }
        }, f)
        f.flush()
        yield f.name
    # Cleanup
    os.unlink(f.name)

@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=Logger)
    logger.record = MagicMock()
    logger.record_event = MagicMock()
    return logger

@pytest.fixture
def config_manager(temp_config_file, mock_logger):
    return ConfigManager(temp_config_file, mock_logger)
