from sovl_system.sovl_main import SOVLSystem

def stress_test():
    # Initialize the SOVL system
    sovl = SOVLSystem()
    
    print("-- Cold Start Test --")
    try:
        # Test generation immediately after initialization
        result = sovl.generate("Hello world", max_new_tokens=20)
        print("Generated:", result)
    except Exception as e:
        print("Error during Cold Start Test:", e)
    
    print("\n-- Training Stress Test --")
    try:
        # Run a short training cycle
        sovl.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=1)
        print("Training completed successfully.")
    except Exception as e:
        print("Error during Training Stress Test:", e)
    
    print("\n-- Memory Test --")
    try:
        # Generate repeatedly with a large input to stress memory
        for i in range(5):
            result = sovl.generate("Test" * 50, max_new_tokens=100)
            print(f"Iteration {i + 1} completed.")
    except Exception as e:
        print("Error during Memory Test:", e)
    
    print("\n-- Error Recovery Test --")
    try:
        # Force a potential error with extreme input
        result = sovl.generate("", max_new_tokens=500)
        print("Generated:", result)
    except Exception as e:
        print("Error during Error Recovery Test:", e)

if __name__ == "__main__":
    stress_test()
