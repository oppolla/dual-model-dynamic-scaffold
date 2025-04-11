import unittest
import torch
from sovl_system.sovl_main import SOVLSystem


class TestSOVLSystemMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the SOVL system for testing."""
        cls.sovl_system = SOVLSystem()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_stress_memory(self):
        """Simulate high memory usage with large datasets or long prompts to check for memory threshold breaches."""
        if self.device.type != "cuda":
            self.skipTest("CUDA is not available for stress testing.")

        # Generate a long prompt to simulate high memory usage
        long_prompt = "This is a test prompt. " * 1000  # Very long prompt
        print("Running Stress Test with a long prompt...")

        try:
            response = self.sovl_system.generate(long_prompt, max_new_tokens=500)
            self.assertIsNotNone(response, "Stress Test: Response should not be None.")
        except torch.cuda.OutOfMemoryError:
            self.fail("Stress Test: CUDA ran out of memory unexpectedly.")

    def test_memory_leak(self):
        """Monitor GPU memory before and after repeated operations to ensure proper cleanup."""
        if self.device.type != "cuda":
            self.skipTest("CUDA is not available for leak testing.")

        print("Running Memory Leak Test...")
        initial_memory = torch.cuda.memory_allocated(self.device)

        # Perform repeated operations
        for _ in range(10):  # Repeat the operation 10 times
            prompt = "Memory leak test prompt."
            response = self.sovl_system.generate(prompt, max_new_tokens=50)
            self.assertIsNotNone(response, "Memory Leak Test: Response should not be None.")

        final_memory = torch.cuda.memory_allocated(self.device)
        print(f"Initial Memory: {initial_memory / 1024**2:.2f} MB")
        print(f"Final Memory: {final_memory / 1024**2:.2f} MB")
        self.assertAlmostEqual(
            initial_memory, final_memory, delta=10 * 1024**2,
            msg="Memory Leak Test: Significant memory increase detected."
        )

    def test_edge_case_memory(self):
        """Test with minimal and maximal memory thresholds to ensure appropriate adjustments."""
        if self.device.type != "cuda":
            self.skipTest("CUDA is not available for edge case testing.")

        print("Running Edge Case Test...")

        # Minimal memory threshold test
        self.sovl_system.MEMORY_THRESHOLD = 0.1  # Set very low memory threshold
        try:
            self.sovl_system.check_memory_health()
        except Exception as e:
            self.fail(f"Edge Case Test (Minimal Threshold): Failed with exception: {e}")

        # Maximal memory threshold test
        self.sovl_system.MEMORY_THRESHOLD = 0.95  # Set very high memory threshold
        try:
            self.sovl_system.check_memory_health()
        except Exception as e:
            self.fail(f"Edge Case Test (Maximal Threshold): Failed with exception: {e}")

        print("Edge Case Test passed for both minimal and maximal thresholds.")


if __name__ == "__main__":
    unittest.main()
