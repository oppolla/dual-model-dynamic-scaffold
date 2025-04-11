import time
import torch
import traceback
from sovl_system import SOVLSystem  # Adjust if main file name differs

# Centralized constants
VALID_COMMANDS = [
    'quit', 'exit', 'train', 'generate', 'save', 'load', 'dream', 'tune',
    'memory', 'status', 'log', 'config', 'reset', 'spark', 'reflect', 'muse', 'flare'
]

# Utility functions
def validate_args(parts, expected_length, error_message):
    if len(parts) < expected_length:
        print(error_message)
        return False
    return True

# Command Handlers
def handle_train(sovl_system, parts):
    epochs = 10  # Default value; replace with TRAIN_EPOCHS if it's part of the system
    dry_run = False
    if len(parts) > 1:
        try:
            epochs = int(parts[1])
        except ValueError:
            print("Error: Epochs must be a number.")
            return
    if '--dry-run' in parts:
        dry_run = True
        sovl_system.enable_dry_run()
    print(f"Starting training for {epochs} epochs{' (dry run)' if dry_run else ''}...")
    sovl_system.run_training_cycle("TRAIN_DATA", "VALID_DATA", epochs=epochs, batch_size=32)  # Replace placeholders

def handle_generate(sovl_system, parts):
    if not validate_args(parts, 2, "Error: Please provide a prompt."):
        return
    prompt = ' '.join(parts[1:] if len(parts) == 2 else parts[1:-1])
    max_tokens = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 60
    print(f"Generating response for: {prompt}...")
    response = sovl_system.generate(prompt, max_new_tokens=max_tokens, temperature=sovl_system.base_temperature, top_k=50, do_sample=True)
    print(f"Response: {response}")

def handle_save(sovl_system, parts):
    path = parts[1] if len(parts) > 1 else None
    print(f"Saving state{' to ' + path if path else ''}...")
    sovl_system.save_state(path)

def handle_load(sovl_system, parts):
    path = parts[1] if len(parts) > 1 else None
    print(f"Loading state{' from ' + path if path else ''}...")
    sovl_system.load_state(path)

def handle_dream(sovl_system, parts):
    print("Triggering dream cycle...")
    sovl_system._dream()

def handle_memory(sovl_system, parts):
    if not validate_args(parts, 2, "Error: Use 'memory on' or 'memory off'."):
        return
    mode = 'both_mem' if parts[1] == 'on' else 'no_mem'
    print(f"Setting memory to {parts[1]}...")
    sovl_system.toggle_memory(mode)

def handle_status(sovl_system, parts):
    print("\n--- System Status ---")
    print(f"Conversation ID: {sovl_system.history.conversation_id}")
    print(f"Temperament: {sovl_system.temperament_score:.2f}")
    print(f"Confidence: {sovl_system.confidence_history[-1] if sovl_system.confidence_history else 'N/A'}")
    print(f"Memory: {'On' if sovl_system.use_scaffold_memory or sovl_system.use_token_map_memory else 'Off'}")
    print(f"Data Exposure: {sovl_system.data_exposure}")
    print(f"Last Trained: {sovl_system.last_trained}")
    print(f"Gestating: {'Yes' if sovl_system.is_sleeping else 'No'}")

# Map commands to handler functions
COMMAND_HANDLERS = {
    "train": handle_train,
    "generate": handle_generate,
    "save": handle_save,
    "load": handle_load,
    "dream": handle_dream,
    "memory": handle_memory,
    "status": handle_status,
    # Add other command handlers here
}

class SOVLCLI:
    def __init__(self):
        self.sovl_system = None

    def run(self):
        print("\nInitializing SOVL System...")
        try:
            self.sovl_system = SOVLSystem()
            print("\nSystem Ready.")
            print("Commands: " + ", ".join(VALID_COMMANDS))

            while True:
                user_input = input("\nEnter command: ").strip()
                parts = user_input.split()
                cmd = parts[0].lower() if parts else ""

                if cmd in ['quit', 'exit']:
                    print("Exiting...")
                    break

                handler = COMMAND_HANDLERS.get(cmd)
                if handler:
                    handler(self.sovl_system, parts)
                else:
                    print(f"Error: Unknown command. Valid commands: {', '.join(VALID_COMMANDS)}")

        except FileNotFoundError as e:
            print(f"\nFile error: {e}. Check 'sovl_config.json' and 'sovl_seed.jsonl'.")
        except torch.cuda.OutOfMemoryError:
            print("\nOut of GPU memory! Try smaller BATCH_SIZE, MAX_SEQ_LENGTH, or INT8/INT4.")
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
        finally:
            if self.sovl_system is not None:
                self.sovl_system.cleanup()
                del self.sovl_system
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nExiting.")

if __name__ == "__main__":
    cli = SOVLCLI()
    cli.run()
