def test_config_manager_validation(config_manager):
    """Test configuration validation"""
    # Valid case
    assert config_manager.validate_value("core_config.base_model_name", "valid_model")
    
    # Invalid case
    assert not config_manager.validate_value("core_config.random_seed", "not_an_int")
    
    # Unknown key
    assert not config_manager.validate_value("nonexistent.key", "value")

def test_config_manager_register_schema(config_manager):
    """Test dynamic schema registration"""
    new_schema = ConfigSchema("new_section.field", int, default=100, range=(1, 200))
    config_manager.register_schema([new_schema])
    
    # Should now accept valid value
    assert config_manager.update("new_section.field", 150)
    assert config_manager.get("new_section.field") == 150
    
    # Should reject invalid value
    assert not config_manager.update("new_section.field", 250)

def test_config_manager_load_profile(config_manager, temp_config_file):
    """Test profile loading functionality"""
    profile_name = "test_profile"
    profile_file = f"{os.path.splitext(temp_config_file)[0]}_{profile_name}.json"
    
    # Create a profile file
    with open(profile_file, 'w') as f:
        json.dump({"core_config": {"base_model_name": "profile_model"}}, f)
    
    assert config_manager.load_profile(profile_name)
    assert config_manager.get("core_config.base_model_name") == "profile_model"
    
    # Cleanup
    os.unlink(profile_file)

def test_config_manager_tune_parameter(config_manager):
    """Test parameter tuning with min/max bounds"""
    assert config_manager.tune_parameter("core_config", "random_seed", 123, min_value=0, max_value=1000)
    assert config_manager.get("core_config.random_seed") == 123
    
    # Should fail when out of bounds
    assert not config_manager.tune_parameter("core_config", "random_seed", -1, min_value=0)
    assert config_manager.get("core_config.random_seed") == 123  # Unchanged

def test_config_manager_validate_with_model(config_manager):
    """Test model-based validation"""
    mock_model_config = MagicMock()
    mock_model_config.num_hidden_layers = 12
    
    # Valid case
    config_manager.update("core_config.cross_attn_layers", [5, 7])
    assert config_manager.validate_with_model(mock_model_config)
    
    # Invalid case
    config_manager.update("core_config.cross_attn_layers", [15])
    assert not config_manager.validate_with_model(mock_model_config)
