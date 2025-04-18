def test_schema_validator_register(mock_logger):
    """Test that schemas are properly registered and stored"""
    validator = SchemaValidator(mock_logger)
    schemas = [
        ConfigSchema("test.field1", int, default=10),
        ConfigSchema("test.field2", str, default="default")
    ]
    validator.register(schemas)
    assert len(validator.schemas) == 2
    assert "test.field1" in validator.schemas
    assert validator.schemas["test.field1"].type == int

def test_schema_validator_validation_success(mock_logger):
    """Test successful validation cases"""
    validator = SchemaValidator(mock_logger)
    validator.register([
        ConfigSchema("test.int_field", int, default=5, range=(1, 10)),
        ConfigSchema("test.str_field", str, default="default", validator=lambda x: x in ["valid", "default"]),
        ConfigSchema("test.nullable_field", str, nullable=True)
    ])
    
    # Valid int in range
    valid, value = validator.validate("test.int_field", 5)
    assert valid and value == 5
    
    # Valid string from allowed values
    valid, value = validator.validate("test.str_field", "valid")
    assert valid and value == "valid"
    
    # Nullable field with None
    valid, value = validator.validate("test.nullable_field", None)
    assert valid and value is None

def test_schema_validator_validation_failures(mock_logger):
    """Test validation failure cases"""
    validator = SchemaValidator(mock_logger)
    validator.register([
        ConfigSchema("test.int_field", int, default=5, range=(1, 10), required=True),
        ConfigSchema("test.str_field", str, default="default", validator=lambda x: x in ["valid", "default"]),
        ConfigSchema("test.nullable_field", str, nullable=False)
    ])
    
    # Required field missing
    valid, value = validator.validate("test.int_field", None)
    assert not valid and value == 5
    mock_logger.record.assert_called()
    
    # Int out of range
    valid, value = validator.validate("test.int_field", 15)
    assert not valid and value == 5
    
    # Invalid string value
    valid, value = validator.validate("test.str_field", "invalid")
    assert not valid and value == "default"
    
    # Non-nullable field with None
    valid, value = validator.validate("test.nullable_field", None)
    assert not valid
