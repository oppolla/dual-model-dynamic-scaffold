def test_config_store_set_get_value():
    """Test basic value setting and getting"""
    store = ConfigStore()
    
    # Simple value
    store.set_value("section.field", 42)
    assert store.get_value("section.field", None) == 42
    assert store.structured_config["section"]["field"] == 42
    
    # Nested value
    store.set_value("training_config.dry_run_params.max_samples", 10)
    assert store.get_value("training_config.dry_run_params.max_samples", None) == 10
    assert store.structured_config["training_config"]["dry_run_params"]["max_samples"] == 10

def test_config_store_default_values():
    """Test default value fallback"""
    store = ConfigStore()
    assert store.get_value("nonexistent.section", "default") == "default"
    assert store.get_value("section.nonexistent", 100) == 100

def test_config_store_rebuild_structured():
    """Test rebuilding structured config from flat config"""
    store = ConfigStore()
    schemas = [
        ConfigSchema("section1.field1", int, default=1),
        ConfigSchema("section1.field2", str, default="a"),
        ConfigSchema("section2.nested.field", float, default=3.14)
    ]
    
    # Set some values
    store.set_value("section1.field1", 10)
    store.set_value("section1.field2", "b")
    store.set_value("section2.nested.field", 6.28)
    
    # Rebuild and verify
    store.rebuild_structured(schemas)
    assert store.structured_config["section1"]["field1"] == 10
    assert store.structured_config["section1"]["field2"] == "b"
    assert store.structured_config["section2"]["nested"]["field"] == 6.28
