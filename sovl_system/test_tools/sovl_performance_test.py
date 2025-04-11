import time
import torch

"""
This module contains performance tests for the SOVL system, focusing on its sleep/dream functionality 
and its ability to handle large input datasets effectively. The tests are designed to measure performance 
metrics such as execution time and GPU memory usage under stress conditions.

Functions:
1. performance_test_sleep_dream:
   - Tests the sleep/dream performance of the SOVL system.
   - Measures GPU memory usage and time taken during the sleep/dream process.
   - Loads a large dataset into the system's logger for stress testing.

2. stress_test_large_inputs:
   - Generates a large number of inputs with maximum length to stress-test the system.
   - Calls the performance_test_sleep_dream function to evaluate the system's performance under load.

Usage:
- Run the script to initialize the SOVL system and execute the stress test with large input datasets.
- Performance metrics will be displayed, including execution time and memory usage.
"""

def performance_test_sleep_dream(sovl_system, large_dataset):
    
    print("\n--- Starting Sleep/Dream Performance Test ---")

    # Load large dataset to logger for sleep training
    for data in large_dataset:
        sovl_system.logger.write({
            "prompt": data["prompt"],
            "response": data["response"],
            "timestamp": time.time(),
            "conversation_id": sovl_system.history.conversation_id
        })

    # Measure memory usage before starting
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    start_time = time.time()

    # Trigger the sleep process
    sovl_system._sleep_train()

    # Measure memory usage after completion
    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    end_time = time.time()

    # Output performance metrics
    print(f"Sleep/Dream Performance Metrics:")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print(f"GPU Memory Usage: {memory_after - memory_before:.2f} bytes")

def stress_test_large_inputs(sovl_system, num_inputs=1000, max_length=512):
    """
    Stress test the system with a large number of inputs and measure performance.

    Args:
        sovl_system: An instance of the SOVLSystem class.
        num_inputs: The number of inputs to generate.
        max_length: The maximum length of each input.
    """
    print("\n--- Starting Large Input Stress Test ---")

    # Generate large random inputs
    large_inputs = [
        {"prompt": "a" * max_length, "response": "b" * max_length} for _ in range(num_inputs)
    ]

    # Measure performance
    performance_test_sleep_dream(sovl_system, large_inputs)


if __name__ == "__main__":
    # Initialize the SOVL system
    print("\nInitializing SOVL System...")
    sovl_system = SOVLSystem()

    # Stress test with large inputs
    stress_test_large_inputs(sovl_system)
