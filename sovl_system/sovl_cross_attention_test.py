import unittest
from transformers import AutoConfig
from sovl_system.sovl_main import SOVLSystem, get_cross_attention_layers

"""
This Python test file contains a suite of unit tests for the SOVL (Self-Organizing Virtual 
Lifeform) System, specifically focusing on the integration and behavior of cross-attention 
layers within the model. The SOVL System is an AI framework that employs a base language 
model (LLM) paired with a scaffolded secondary LLM to facilitate autonomous learning, 
leveraging "sleep" and "dream" mechanisms for continuous improvement.

The tests in this file are structured as follows:

1. TestSOVLSystem Class:
   This class inherits from `unittest.TestCase` and organizes the test cases for the SOVL System.

2. setUpClass Method:
   - Initializes a shared instance of the SOVL System for all tests.
   - This method is run once before all tests in the class.

3. Test Cases:
   - `test_configuration`:
     - Validates the configuration of cross-attention layers (`CROSS_ATTN_LAYERS`) and custom 
       layers (`CUSTOM_LAYERS`) in the model.
     - Ensures that the specified layer indices are valid and within bounds of the base model's 
       hidden layers.
   
   - `test_cross_attention_integration`:
     - Tests whether cross-attention layers are correctly integrated into the base model.
     - Ensures that each specified layer contains the expected `cross_attn` attribute.
     - Uses helper methods to inject and retrieve cross-attention layers.
   
   - `test_dynamic_mode`:
     - Evaluates the dynamic behavior of the cross-attention mechanism when operating under 
       different modes (e.g., "confidence", "temperament").
     - Checks that dynamic modes are correctly activated and deactivated.
     - Validates the presence of cross-attention attributes in the relevant layers for each 
       dynamic mode.

4. Dynamic Modes:
   - The SOVL System supports dynamic cross-attention modes that adapt based on external 
     parameters such as confidence or temperament.
   - This file includes tests to verify the correct behavior of these modes and their 
     interactions with the cross-attention layers.

5. Execution:
   - The `unittest.main()` function at the end of the file allows the tests to be run 
     independently as a script.

Overall, this test suite ensures the reliability and correctness of the SOVL System's 
cross-attention mechanisms, enabling seamless integration and dynamic adaptability within 
the AI framework.
"""

class TestSOVLSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the SOVL system for testing."""
        cls.sovl_system = SOVLSystem()

    def test_configuration(self):
        """Validate that all specified layers exist within the base model."""
        base_config = AutoConfig.from_pretrained(self.sovl_system.BASE_MODEL_NAME)
        num_layers = base_config.num_hidden_layers

        # Validate CROSS_ATTN_LAYERS
        for layer in self.sovl_system.CROSS_ATTN_LAYERS:
            self.assertTrue(0 <= layer < num_layers, f"Invalid CROSS_ATTN_LAYERS index: {layer}")

        # Validate CUSTOM_LAYERS if custom mode is enabled
        if self.sovl_system.LAYER_SELECTION_MODE == "custom":
            for layer in self.sovl_system.CUSTOM_LAYERS:
                self.assertTrue(0 <= layer < num_layers, f"Invalid CUSTOM_LAYERS index: {layer}")

    def test_cross_attention_integration(self):
        """Test cross-attention interaction between base and scaffold models."""
        # Inject cross-attention layers
        self.sovl_system._insert_cross_attention()

        # Validate that cross-attention layers exist in the base model
        base_layers = self.sovl_system._get_model_layers(self.sovl_system.base_model)
        cross_attention_layers = get_cross_attention_layers(self.sovl_system.base_model)

        for layer_idx in cross_attention_layers:
            if layer_idx >= len(base_layers):
                continue  # Skip invalid layers; they are already handled in the configuration
            layer = base_layers[layer_idx]
            self.assertTrue(
                hasattr(layer, "cross_attn"),
                f"Cross-attention not found in layer {layer_idx}"
            )

    def test_dynamic_mode(self):
        """Evaluate the behavior of dynamic modes (e.g., confidence, temperament)."""
        base_layers = self.sovl_system._get_model_layers(self.sovl_system.base_model)
        dynamic_modes = ["confidence", "temperament"]

        for mode in dynamic_modes:
            with self.subTest(mode=mode):
                # Enable dynamic mode
                self.sovl_system.tune_cross_attention(dynamic_mode=mode)
                self.assertEqual(self.sovl_system.dynamic_cross_attn_mode, mode)

                # Validate behavior for cross-attention layers
                for layer_idx in get_cross_attention_layers(self.sovl_system.base_model):
                    if layer_idx >= len(base_layers):
                        continue  # Skip invalid layers
                    layer = base_layers[layer_idx]
                    self.assertTrue(
                        hasattr(layer, "cross_attn"),
                        f"Cross-attention not found in layer {layer_idx} for mode {mode}"
                    )

        # Disable dynamic mode
        self.sovl_system.tune_cross_attention(dynamic_mode="off")
        self.assertIsNone(self.sovl_system.dynamic_cross_attn_mode, "Dynamic mode was not disabled correctly.")


if __name__ == "__main__":
    unittest.main()
