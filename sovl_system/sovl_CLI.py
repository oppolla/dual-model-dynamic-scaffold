import time
import torch
import traceback
from sovl_system import SOVLSystem  # Adjust import based on your main file name

def run_cli():
    print("\nInitializing SOVL System...")
    sovl_system = None
    try:
        sovl_system = SOVLSystem()
        print("\nSystem Ready.")
        valid_commands = [
            'quit', 'exit', 'train', 'generate', 'save', 'load', 'dream',
            'tune', 'memory', 'status', 'log', 'config', 'reset', 'spark', 'reflect', 'muse', 'flare'
        ]
        print("Commands: quit, exit, train [epochs] [--dry-run], generate <prompt> [max_tokens], "
              "save [path], load [path], dream, tune cross [weight], memory <on|off>, "
              "status, log view, config <key> [value], reset, spark, reflect, muse, flare")

        while True:
            user_input = input("\nEnter command: ").strip()
            parts = user_input.split()
            cmd = parts[0].lower() if parts else ""

            if cmd in ['quit', 'exit']:
                print("Exiting...")
                break

            elif cmd == 'train':
                epochs = TRAIN_EPOCHS
                dry_run = False
                if len(parts) > 1:
                    try:
                        epochs = int(parts[1])
                    except ValueError:
                        print("Error: Epochs must be a number.")
                        continue
                if '--dry-run' in parts:
                    dry_run = True
                    sovl_system.enable_dry_run()
                print(f"Starting training for {epochs} epochs{' (dry run)' if dry_run else ''}...")
                sovl_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=epochs, batch_size=BATCH_SIZE)
                if dry_run:
                    print("Dry run complete.")
                    break

            elif cmd == 'generate':
                if len(parts) < 2:
                    print("Error: Please provide a prompt.")
                    continue
                prompt = ' '.join(parts[1:] if len(parts) == 2 else parts[1:-1])
                max_tokens = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 60
                print(f"Generating response for: {prompt}...")
                response = sovl_system.generate(prompt, max_new_tokens=max_tokens, temperature=sovl_system.base_temperature, top_k=50, do_sample=True)
                print(f"Response: {response}")

            elif cmd == 'save':
                path = parts[1] if len(parts) > 1 else None
                print(f"Saving state{' to ' + path if path else ''}...")
                sovl_system.save_state(path)

            elif cmd == 'load':
                path = parts[1] if len(parts) > 1 else None
                print(f"Loading state{' from ' + path if path else ''}...")
                sovl_system.load_state(path)

            elif cmd == 'dream':
                print("Triggering dream cycle...")
                sovl_system._dream()

            elif cmd == 'tune' and len(parts) >= 2 and parts[1] == 'cross':
                try:
                    weight = float(parts[2]) if len(parts) > 2 else None
                    print(f"Setting cross-attention weight to {weight if weight is not None else 'default'}...")
                    sovl_system.tune_cross_attention(weight=weight)
                except ValueError:
                    print("Error: Weight must be a number.")
                    continue

            elif cmd == 'memory':
                if len(parts) != 2 or parts[1] not in ['on', 'off']:
                    print("Error: Use 'memory on' or 'memory off'.")
                    continue
                mode = 'both_mem' if parts[1] == 'on' else 'no_mem'
                print(f"Setting memory to {parts[1]}...")
                sovl_system.toggle_memory(mode)

            elif cmd == 'status':
                print("\n--- System Status ---")
                print(f"Conversation ID: {sovl_system.history.conversation_id}")
                print(f"Temperament: {sovl_system.temperament_score:.2f}")
                print(f"Confidence: {sovl_system.confidence_history[-1] if sovl_system.confidence_history else 'N/A'}")
                print(f"Memory: {'On' if sovl_system.use_scaffold_memory or sovl_system.use_token_map_memory else 'Off'}")
                print(f"Data Exposure: {sovl_system.data_exposure}")
                print(f"Last Trained: {sovl_system.last_trained}")
                print(f"Gestating: {'Yes' if sovl_system.is_sleeping else 'No'}")

            elif cmd == 'log' and len(parts) == 2 and parts[1] == 'view':
                print("\n--- Last 5 Log Entries ---")
                logs = sovl_system.logger.read()[-5:]
                for log in logs:
                    print(f"Time: {log.get('timestamp', 'N/A')}, "
                          f"Prompt: {log.get('prompt', 'N/A')[:30]}..., "
                          f"Response: {log.get('response', 'N/A')[:30]}...")

            elif cmd == 'config':
                if len(parts) < 2:
                    print("Error: Please specify a config key.")
                    continue
                key = parts[1]
                if len(parts) == 2:
                    value = get_config_value(config, key, "Not found")
                    print(f"Config {key}: {value}")
                else:
                    try:
                        value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                        print(f"Setting {key} to {value} (Note: Changes apply on restart)")
                    except ValueError:
                        print("Error: Value must be a number.")
                        continue

            elif cmd == 'reset':
                print("Resetting system state...")
                sovl_system.cleanup()
                sovl_system = SOVLSystem()
                sovl_system.wake_up()

            elif cmd == 'spark':
                print("Sparking curiosity...")
                question = sovl_system.generate_curiosity_question()
                if not question:
                    print("No curious question generated. Try again later.")
                    continue
                print(f"Curiosity: {question}")
                response = sovl_system.generate(question, max_new_tokens=60, temperature=sovl_system.base_temperature, top_k=50, do_sample=True)
                print(f"Response: {response}")
                sovl_system.logger.write({
                    "prompt": question,
                    "response": response,
                    "timestamp": time.time(),
                    "conversation_id": sovl_system.history.conversation_id,
                    "confidence_score": 0.5,  # Default for spark
                    "is_system_question": True
                })

            elif cmd == 'reflect':
                print("Reflecting on recent interactions...")
                logs = sovl_system.logger.read()[-3:]  # Look at last 3 interactions
                if not logs:
                    print("Nothing to reflect on yet. Try generating some responses first.")
                    continue
                recent_prompts = [log.get('prompt', '') for log in logs if 'prompt' in log]
                if not recent_prompts:
                    reflection = "I haven't been prompted much lately."
                else:
                    reflection = f"I've noticed a lot of talk about {recent_prompts[-1].split()[0] if recent_prompts[-1] else 'things'} lately."
                print(f"Reflection: {reflection}")
                elaboration = sovl_system.generate(
                    f"Based on recent thoughts: {reflection}", 
                    max_new_tokens=60, 
                    temperature=sovl_system.base_temperature, 
                    top_k=50, 
                    do_sample=True
                )
                print(f"Elaboration: {elaboration}")
                sovl_system.logger.write({
                    "prompt": reflection,
                    "response": elaboration,
                    "timestamp": time.time(),
                    "conversation_id": sovl_system.history.conversation_id,
                    "confidence_score": 0.6,  # Slightly higher for reflection
                    "is_system_question": True
                })

            elif cmd == 'muse':
                print("Musing...")
                logs = sovl_system.logger.read()[-3:]  # Look at last 3 interactions
                if not logs:
                    inspiration = "silence"
                    print("Inspiration: silence (nothing recent to draw from)")
                else:
                    recent_prompts = [log.get('prompt', '') for log in logs if 'prompt' in log]
                    inspiration = recent_prompts[-1].split()[0] if recent_prompts and recent_prompts[-1] else "mystery"
                    print(f"Inspiration: \"{inspiration}\" (from recent interactions)")
                thought = sovl_system.generate(
                    f"A whimsical thought about {inspiration}:",
                    max_new_tokens=60,
                    temperature=sovl_system.base_temperature + 0.1,  # Slightly higher for creativity
                    top_k=50,
                    do_sample=True
                )
                print(f"Thought: {thought}")
                sovl_system.logger.write({
                    "prompt": f"Musing on {inspiration}",
                    "response": thought,
                    "timestamp": time.time(),
                    "conversation_id": sovl_system.history.conversation_id,
                    "confidence_score": 0.7,  # Higher for creative flair
                    "is_system_question": True
                })

            elif cmd == 'flare':
                print("Flaring up...")
                original_temperament = sovl_system.temperament_score  # Save current state
                sovl_system.temperament_score = 1.0  # Crank to max
                print("Temperament Cranked: 1.0 (MAX)")
                prompt = ' '.join(parts[1:]) if len(parts) > 1 else "THIS QUIET IS TOO MUCH!"
                outburst = sovl_system.generate(
                    prompt,
                    max_new_tokens=80,
                    temperature=1.5,  # High heat for chaos
                    top_k=50,
                    do_sample=True
                ).upper()  # Shout it out
                print(f"Outburst: {outburst}")
                sovl_system.temperament_score = original_temperament  # Reset to normal
                print("[Temperament resets to normal]")
                sovl_system.logger.write({
                    "prompt": f"Flare: {prompt}",
                    "response": outburst,
                    "timestamp": time.time(),
                    "conversation_id": sovl_system.history.conversation_id,
                    "confidence_score": 0.9,  # High for intensity
                    "is_system_question": True
                })

            else:
                print(f"Error: Unknown command. Valid commands: {', '.join(valid_commands)}")

    except FileNotFoundError as e:
        print(f"\nFile error: {e}. Check 'sovl_config.json' and 'sovl_seed.jsonl'.")
    except torch.cuda.OutOfMemoryError:
        print("\nOut of GPU memory! Try smaller BATCH_SIZE, MAX_SEQ_LENGTH, or INT8/INT4.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        if sovl_system is not None:
            sovl_system.cleanup()
            del sovl_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")
