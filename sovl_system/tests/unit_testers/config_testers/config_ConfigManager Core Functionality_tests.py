def test_config_manager_initialization(temp_config_file, mock_logger):
    """Test that config manager loads config file on init"""
    manager = ConfigManager(temp_config_file, mock_logger)
    assert manager.get("core_config.base_model_name") == "test_model"
    assert manager.get("core_config.random_seed", 42) == 42  # Default value

def test_config_manager_get_methods(config_manager):
    """Test various get method scenarios"""
    # Existing value from file
    assert config_manager.get("core_config.base_model_name") == "test_model"
    
    # Default value
    assert config_manager.get("nonexistent.key", "default") == "default"
    
    # Type-safe key
    key = ConfigKey("core_config", "base_model_name")
    assert config_manager.get(key) == "test_model"

def test_config_manager_update_success(config_manager):
    """Test successful config updates"""
    assert config_manager.update("core_config.base_model_name", "new_model")
    assert config_manager.get("core_config.base_model_name") == "new_model"
    
    # Verify structured config was updated
    assert config_manager.store.structured_config["core_config"]["base_model_name"] == "new_model"

def test_config_manager_update_failure(config_manager):
    """Test failed config updates"""
    # Invalid type
    assert not config_manager.update("core_config.random_seed", "not_an_int")
    assert config_manager.get("core_config.random_seed") == 42  # Default
    
    # Unknown key
    assert not config_manager.update("nonexistent.key", "value")
    mock_logger.record.assert_called()

def test_config_manager_freeze_unfreeze(config_manager):
    """Test freeze/unfreeze functionality"""
    config_manager.freeze()
    assert not config_manager.update("core_config.base_model_name", "new_model")
    
    config_manager.unfreeze()
    assert config_manager.update("core_config.base_model_name", "new_model")

def test_config_manager_update_batch(config_manager):
    """Test batch updates"""
    updates = {
        "core_config.base_model_name": "batch_model",
        "training_config.learning_rate": 0.001
    }
    assert config_manager.update_batch(updates)
    
    assert config_manager.get("core_config.base_model_name") == "batch_model"
    assert config_manager.get("training_config.learning_rate") == 0.001

def test_config_manager_subscribers(config_manager):
    """Test subscriber notification system"""
    callback = MagicMock()
    config_manager.subscribe(callback)
    
    config_manager.update("core_config.base_model_name", "new_model")
    callback.assert_called_once()
    
    config_manager.unsubscribe(callback)
    config_manager.update("core_config.base_model_name", "another_model")
    callback.assert_called_once()  # No additional calls

def test_config_manager_diff(config_manager):
    """Test config diff functionality"""
    old_config = config_manager.store.flat_config.copy()
    config_manager.update("core_config.base_model_name", "diff_model")
    
    diff = config_manager.diff_config(old_config)
    assert "core_config" in diff
    assert diff["core_config"]["base_model_name"]["old"] == "test_model"
    assert diff["core_config"]["base_model_name"]["new"] == "diff_model"
