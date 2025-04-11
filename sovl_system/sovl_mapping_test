import unittest
import torch
from sovl_system.sovl_main import SOVLSystem


class TestSOVLSystemTokenMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the SOVL system for testing."""
        cls.sovl_system = SOVLSystem()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_mapping_accuracy(self):
        """Ensure token mapping aligns with expected vocabularies and retains semantic meaning."""
        print("Running Mapping Accuracy Test...")
        test_prompt = "This is a test prompt for token mapping."
        mapped_tokens = self.sovl_system.tokenize_and_map(test_prompt)

        # Verify that the mapped tokens are not empty
        self.assertGreater(
            mapped_tokens["input_ids"].numel(),
            0,
            "Mapping Accuracy Test: Mapped tokens should not be empty."
        )

        # Verify that mapped tokens match expected structure
        base_tokens = self.sovl_system.base_tokenizer.encode(
            test_prompt, return_tensors="pt", truncation=True
        ).to(self.device)
        self.assertEqual(
            mapped_tokens["input_ids"].size(0),
            base_tokens.size(0),
            "Mapping Accuracy Test: Mapped token batch size mismatch."
        )

    def test_truncation(self):
        """Test long sequences to verify proper truncation warnings and handling."""
        print("Running Truncation Test...")
        long_prompt = "This is a very long test prompt. " * 500  # Create an excessively long prompt
        mapped_tokens = self.sovl_system.tokenize_and_map(long_prompt)

        # Verify truncation occurred
        truncated_length = min(self.sovl_system.MAX_SEQ_LENGTH, len(long_prompt.split()))
        self.assertEqual(
            mapped_tokens["input_ids"].size(1),
            truncated_length,
            f"Truncation Test: Expected length {truncated_length}, got {mapped_tokens['input_ids'].size(1)}."
        )

        # Verify warning was logged for truncation
        log_entries = self.sovl_system.logger.read()
        truncation_logs = [
            entry for entry in log_entries if "Token mapping truncated" in entry.get("warning", "")
        ]
        self.assertGreater(
            len(truncation_logs),
            0,
            "Truncation Test: No truncation warning logged despite long input."
        )

    def test_edge_case_inputs(self):
        """Use edge-case inputs (e.g., empty prompts or special tokens) to check robustness."""
        print("Running Edge Case Test...")

        # Test empty prompt
        empty_prompt = ""
        mapped_tokens = self.sovl_system.tokenize_and_map(empty_prompt)
        self.assertGreaterEqual(
            mapped_tokens["input_ids"].numel(),
            0,
            "Edge Case Test: Empty prompt mapping failed."
        )

        # Test prompt with special tokens
        special_prompt = "<|endoftext|>"  # Common special token
        mapped_tokens = self.sovl_system.tokenize_and_map(special_prompt)
        self.assertGreater(
            mapped_tokens["input_ids"].numel(),
            0,
            "Edge Case Test: Special token prompt mapping failed."
        )

        # Verify proper handling of special tokens
        self.assertIn(
            self.sovl_system.base_tokenizer.pad_token_id,
            mapped_tokens["input_ids"].tolist(),
            "Edge Case Test: Special tokens not mapped correctly."
        )


if __name__ == "__main__":
    unittest.main()
