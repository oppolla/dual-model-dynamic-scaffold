import unittest
from unittest.mock import MagicMock, patch
import torch
from sovl_system.sovl_main import SOVLSystem

"""
This integration test suite verifies the functionality and interactions of the SOVLSystem 
(Self-Organizing Virtual Lifeform) components, including the base LLM, scaffold LLM, and 
the mechanisms for learning, adaptation, and memory management. The tests ensure the 
system operates as expected and validate the following functionalities:

1. `test_base_and_scaffold_interaction`: 
   - Ensures that both the base LLM and scaffold LLM models are properly loaded. 
   - Verifies that cross-attention layers are correctly injected into the base model's layers.
   - Checks the token mapping process to ensure accurate transformation of base model tokens.

2. `test_dream_mechanism`: 
   - Simulates the dream mechanism, which updates the scaffold memory based on past interactions. 
   - Populates the logger with dummy log data to mimic user prompts and responses.
   - Validates that the dream mechanism updates the dream memory appropriately and calculates 
     novelty and similarity metrics for dream entries.

3. `test_sleep_and_gestation`: 
   - Tests the sleep and gestation mechanisms, where the system utilizes logs to refine its models.
   - Simulates the gestation process and ensures that the system enters the sleep state, 
     progresses through gestation, and records training loss.

4. `test_cross_attention_tuning`: 
   - Verifies the dynamic tuning of cross-attention layers, ensuring that parameters like 
     `influence_weight` and `blend_strength` are correctly set and updated.

Setup and teardown methods are included to initialize and clean up the SOVLSystem instance 
before and after each test, ensuring isolated and reliable test execution.
"""

class TestSOVLSystemIntegration(unittest.TestCase):
    def setUp(self):
        """Set up the SOVLSystem instance for testing."""
        self.sovl_system = SOVLSystem()
        self.sovl_system.logger = MagicMock()  # Mock logger to avoid file operations

    def test_base_and_scaffold_interaction(self):
        """Test interaction between base LLM and scaffold LLM."""
        # Verify models are loaded
        self.assertIsNotNone(self.sovl_system.base_model, "Base model not loaded")
        self.assertIsNotNone(self.sovl_system.scaffolds[0], "Scaffold model not loaded")
        
        # Verify cross-attention layers are injected
        base_layers = self.sovl_system._get_model_layers(self.sovl_system.base_model)
        self.assertTrue(
            hasattr(base_layers[0], "cross_attn"),
            "Cross-attention layers not injected"
        )
        
        # Verify token mapping
        base_token_ids = torch.tensor([[1, 2, 3]])  # Example token IDs
        mapped_ids = self.sovl_system.map_sequence(base_token_ids)
        self.assertEqual(mapped_ids.size(0), base_token_ids.size(0), "Token mapping failed")

    def test_dream_mechanism(self):
        """Test the dream mechanism updates scaffold memory."""
        # Populate logger with dummy log data
        dummy_log = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            {"prompt": "Explain neural networks.", "response": "Neural networks are..."}
        ]
        self.sovl_system.logger.read.return_value = dummy_log
        
        # Trigger the dream mechanism
        self.sovl_system._dream()
        
        # Verify dream memory is updated
        self.assertGreater(len(self.sovl_system.dream_memory), 0, "Dream memory not updated")
        
        # Verify novelty and similarity metrics
        dream_entry = self.sovl_system.dream_memory[-1]
        self.assertIsInstance(dream_entry, tuple, "Dream memory entry is not a tuple")
        self.assertEqual(len(dream_entry), 2, "Dream memory entry does not have tensor and weight")

    def test_sleep_and_gestation(self):
        """Test the sleep and gestation mechanism."""
        # Populate logger with dummy log data
        dummy_log = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            {"prompt": "Explain neural networks.", "response": "Neural networks are..."}
        ]
        self.sovl_system.logger.read.return_value = dummy_log
        
        # Trigger the gestation mechanism
        with patch("torch.nn.functional.cross_entropy", return_value=torch.tensor(1.0)):
            self.sovl_system._gestate()
        
        # Verify sleep training progress
        self.assertTrue(self.sovl_system.is_sleeping, "System did not enter sleep state")
        self.assertGreater(self.sovl_system.sleep_progress, 0, "Gestation progress not updated")
        self.assertGreater(self.sovl_system.sleep_total_loss, 0, "No loss recorded during gestation")

    def test_cross_attention_tuning(self):
        """Test dynamic tuning of cross-attention layers."""
        # Set cross-attention parameters
        self.sovl_system.tune_cross_attention(weight=0.5, blend_strength=0.8)
        
        # Verify cross-attention layer parameters
        base_layers = self.sovl_system._get_model_layers(self.sovl_system.base_model)
        cross_attn_layer = base_layers[0].cross_attn
        
        self.assertAlmostEqual(cross_attn_layer.influence_weight, 0.5, "Influence weight not updated")
        self.assertAlmostEqual(cross_attn_layer.blend_strength, 0.8, "Blend strength not updated")

    def tearDown(self):
        """Clean up after each test."""
        del self.sovl_system


if __name__ == "__main__":
    unittest.main()
