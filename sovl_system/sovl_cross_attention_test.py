import unittest
from transformers import AutoConfig
from sovl_system.sovl_main import SOVLSystem, get_cross_attention_layers


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
