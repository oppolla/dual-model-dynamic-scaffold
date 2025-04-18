def test_file_handler_load_success(temp_config_file, mock_logger):
    """Test successful config file loading"""
    handler = FileHandler(temp_config_file, mock_logger)
    config = handler.load()
    assert "core_config" in config
    assert config["core_config"]["base_model_name"] == "test_model"

def test_file_handler_load_failure(mock_logger):
    """Test handling of missing config file"""
    handler = FileHandler("nonexistent.json", mock_logger)
    config = handler.load()
    assert config == {}
    mock_logger.record.assert_called()

def test_file_handler_save_success(temp_config_file, mock_logger):
    """Test successful config file saving"""
    handler = FileHandler(temp_config_file, mock_logger)
    test_config = {"test": {"value": 42}}
    assert handler.save(test_config)
    
    # Verify the file was written correctly
    with open(temp_config_file, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == test_config

def test_file_handler_save_failure(mock_logger):
    """Test handling of save failures"""
    # Create a read-only directory
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chmod(temp_dir, 0o555)  # Read-only
        read_only_file = os.path.join(temp_dir, "config.json")
        
        handler = FileHandler(read_only_file, mock_logger)
        assert not handler.save({"test": 1})
        mock_logger.record.assert_called()
