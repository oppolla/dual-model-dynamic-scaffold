def test_empty_config_file(mock_logger):
    """Test handling of empty config file"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as f:
        f.write("{}")
        f.flush()
        
        manager = ConfigManager(f.name, mock_logger)
        # Should use defaults
        assert manager.get("core_config.base_model_name") == "gpt2"

def test_corrupted_config_file(mock_logger):
    """Test handling of corrupted config file"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as f:
        f.write("{invalid json}")
        f.flush()
        
        manager = ConfigManager(f.name, mock_logger)
        # Should use defaults
        assert manager.get("core_config.base_model_name") == "gpt2"
        mock_logger.record.assert_called()

def test_concurrent_access(config_manager):
    """Test thread safety with concurrent access"""
    from threading import Thread
    
    results = []
    
    def worker():
        for i in range(100):
            config_manager.update("core_config.random_seed", i)
            results.append(config_manager.get("core_config.random_seed"))
    
    threads = [Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Final value should be consistent
    final_value = config_manager.get("core_config.random_seed")
    assert all(r == final_value for r in results[-5:])

def test_type_safe_keys(config_manager):
    """Test usage of type-safe ConfigKeys"""
    # Using the type-safe key
    assert config_manager.get(ConfigKeys.CONTROLS_MEMORY_THRESHOLD, 0.85) == 0.85
    
    # Update using type-safe key
    config_manager.update(str(ConfigKeys.CONTROLS_MEMORY_THRESHOLD), 0.9)
    assert config_manager.get(ConfigKeys.CONTROLS_MEMORY_THRESHOLD) == 0.9
