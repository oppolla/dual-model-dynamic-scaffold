import unittest
import tempfile
import os
import json
import torch
from unittest.mock import patch, MagicMock
from sovl_system import SOVLSystem
from sovl_config import ConfigManager, ConfigSchema
from sovl_logger import Logger

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_config.write(b'{}')
        self.temp_config.close()
        self.logger = MagicMock(spec=Logger)
        self.config_manager = ConfigManager(self.temp_config.name, self.logger)

    def tearDown(self):
        os.unlink(self.temp_config.name)

    def test_initialization(self):
        """Test config manager initializes with empty config"""
        self.assertIsInstance(self.config_manager, ConfigManager)
        self.assertEqual(self.config_manager.config, {})

    def test_schema_validation(self):
        """Test schema validation applies defaults"""
        # Test required field
        self.assertEqual(
            self.config_manager.get("core_config.base_model_name"),
            "gpt2"  # From schema default
        )
        
        # Test type validation
        self.config_manager.update("training_config.batch_size", "not_an_int")
        self.assertEqual(
            self.config_manager.get("training_config.batch_size"),
            1  # Reverts to default
        )
        
        # Test range validation
        self.config_manager.update("training_config.learning_rate", 100.0)
        self.assertEqual(
            self.config_manager.get("training_config.learning_rate"),
            0.0003  # Reverts to default
        )

    def test_config_persistence(self):
        """Test config saving and loading"""
        test_value = "test_model"
        self.config_manager.update("core_config.base_model_name", test_value)
        self.config_manager.save_config()
        
        # Create new instance to test loading
        new_manager = ConfigManager(self.temp_config.name, self.logger)
        self.assertEqual(
            new_manager.get("core_config.base_model_name"),
            test_value
        )

    def test_batch_updates(self):
        """Test atomic batch updates"""
        updates = {
            "core_config.base_model_name": "test_model",
            "training_config.batch_size": 2
        }
        success = self.config_manager.update_batch(updates)
        self.assertTrue(success)
        self.assertEqual(
            self.config_manager.get("core_config.base_model_name"),
            "test_model"
        )
        self.assertEqual(
            self.config_manager.get("training_config.batch_size"),
            2
        )

    def test_structured_access(self):
        """Test section-based config access"""
        section = self.config_manager.get_section("core_config")
        self.assertIsInstance(section, dict)
        self.assertEqual(section.get("base_model_name"), "gpt2")

class TestSOVLSystem(unittest.TestCase):
    def setUp(self):
        # Create temp config file
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        base_config = {
            "core_config": {
                "base_model_name": "gpt2",
                "scaffold_model_name": "gpt2",
                "quantization": "fp16"
            },
            "training_config": {
                "batch_size": 1,
                "dry_run": True  # Enable dry run for faster tests
            }
        }
        self.temp_config.write(json.dumps(base_config).encode())
        self.temp_config.close()
        
        # Mock logger
        self.logger = MagicMock(spec=Logger)
        
        # Patch model loading to use smaller test models
        self.patcher = patch('transformers.AutoModelForCausalLM.from_pretrained')
        self.mock_model = self.patcher.start()
        self.mock_model.return_value = MagicMock()  # Return mock model

    def tearDown(self):
        os.unlink(self.temp_config.name)
        self.patcher.stop()

    def test_system_initialization(self):
        """Test system initializes with config"""
        system = SOVLSystem(ConfigManager(self.temp_config.name, self.logger))
        self.assertIsInstance(system, SOVLSystem)
        
        # Verify config was applied
        self.assertEqual(system.base_model_name, "gpt2")
        self.assertTrue(system.dry_run)

    def test_generation_pipeline(self):
        """Test text generation workflow"""
        system = SOVLSystem(ConfigManager(self.temp_config.name, self.logger))
        
        # Mock tokenizer and model outputs
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        system.base_tokenizer = mock_tokenizer
        system.scaffold_tokenizer = mock_tokenizer
        
        # Mock model generate
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_output.scores = [torch.tensor([[0.1, 0.9]])]
        system.base_model.generate.return_value = mock_output
        
        # Test generation
        response = system.generate("Test prompt")
        self.assertIsInstance(response, str)
        
        # Verify logging occurred
        self.assertTrue(self.logger.record.called)

    def test_training_cycle(self):
        """Test training workflow"""
        system = SOVLSystem(ConfigManager(self.temp_config.name, self.logger))
        
        # Mock training data
        train_data = [{"prompt": "test", "completion": "response"}]
        valid_data = [{"prompt": "test", "completion": "response"}]
        
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = None
        system.trainer = mock_trainer
        
        # Test training
        system.run_training_cycle(train_data, valid_data)
        
        # Verify trainer was called
        mock_trainer.train.assert_called_once()

    def test_state_management(self):
        """Test state saving/loading"""
        system = SOVLSystem(ConfigManager(self.temp_config.name, self.logger))
        
        # Create temp state file
        with tempfile.NamedTemporaryFile() as temp_state:
            # Test saving
            system.save_state(temp_state.name)
            self.assertTrue(os.path.exists(temp_state.name + "_state.json"))
            
            # Test loading
            system.load_state(temp_state.name)
            
            # Verify state was loaded
            self.assertIsNotNone(system.state)

    def test_memory_management(self):
        """Test memory health checks"""
        system = SOVLSystem(ConfigManager(self.temp_config.name, self.logger))
        
        # Mock memory stats
        with patch('torch.cuda.memory_allocated') as mock_alloc:
            mock_alloc.return_value = 10 * 1024**3  # 10GB
            
            # Test memory check
            system.check_memory_health()
            
            # Verify adjustments were made
            self.assertLess(system.config_manager.get("training_config.batch_size"), 1)

class IntegrationTests(unittest.TestCase):
    def test_config_to_system_integration(self):
        """Test config changes propagate to system"""
        # Create temp config
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_config:
            base_config = {
                "core_config": {
                    "base_model_name": "gpt2",
                    "scaffold_model_name": "gpt2"
                },
                "training_config": {
                    "batch_size": 1,
                    "dry_run": True
                }
            }
            temp_config.write(json.dumps(base_config).encode())
            temp_config.close()
            
            # Initialize system
            logger = MagicMock(spec=Logger)
            config_manager = ConfigManager(temp_config.name, logger)
            system = SOVLSystem(config_manager)
            
            # Change config
            new_batch_size = 2
            config_manager.update("training_config.batch_size", new_batch_size)
            
            # Verify system updated
            self.assertEqual(system.training_config["batch_size"], new_batch_size)
            
            os.unlink(temp_config.name)

if __name__ == "__main__":
    unittest.main()
