import time
import torch
import traceback
from sovl_system import SOVLSystem  # Adjust if main file name differs

# Default constants (override if passed from main file)
TRAIN_EPOCHS = 10  # Example default, adjust as needed
BATCH_SIZE = 32    # Example default, adjust as needed
TRAIN_DATA = None  # Placeholder, assumes sovl_system provides it
VALID_DATA = None  # Placeholder, assumes sovl_system provides it

def parse_args(parts, min_args=1, max_args=None):
    """Helper to parse command arguments safely."""
    cmd = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []
    if len(args) < min_args - 1:
        raise ValueError(f"Error: {cmd} requires at least {min_args - 1} argument(s).")
    if max_args and len(args) > max_args - 1:
        raise ValueError(f"Error: {cmd} takes at most {max_args - 1} argument(s).")
    return cmd, args

def generate_response(sovl_system, prompt, max_tokens=60, temp_adjust=0.0):
    """Helper for consistent generation and logging."""
    response = sovl_system.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=sovl_system.base_temperature + temp_adjust,
        top_k=50,
        do_sample=True
    )
    return response

def log_action(sovl_system, prompt, response, confidence, is_system=False, extra_attrs=None):
    """Helper for consistent logging with optional extra attributes."""
    log_entry = {
        "prompt": prompt,
        "response": response,
        "timestamp": time.time(),
        "conversation_id": sovl_system.history.conversation_id,
        "confidence_score": confidence,
        "is_system_question": is_system
    }
    if extra_attrs:
        log_entry.update(extra_attrs)
    sovl_system.logger.write(log_entry)

# Command handlers
def cmd_quit(sovl_system, args):
    print("Exiting...")
    return True  # Signal to break the loop

def cmd_train(sovl_system, args):
    epochs = TRAIN_EPOCHS
    dry_run = "--dry-run" in args
    if args and args[0] != "--dry-run":
        epochs = int(args[0])
    if dry_run:
        sovl_system.enable_dry_run()
    print(f"Starting training for {epochs} epochs{' (dry run)' if dry_run else ''}...")
    sovl_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=epochs, batch_size=BATCH_SIZE)
    if dry_run:
        print("Dry run complete.")
        return True  # Break after dry run
    return False

def cmd_generate(sovl_system, args):
    prompt = ' '.join(args[:-1]) if args[-1].isdigit() else ' '.join(args)
    max_tokens = int(args[-1]) if args[-1].isdigit() else 60
    print(f"Generating response for: {prompt}...")
    response = generate_response(sovl_system, prompt, max_tokens)
    print(f"Response: {response}")

def cmd_save(sovl_system, args):
    path = args[0] if args else None
    print(f"Saving state{' to ' + path if path else ''}...")
    sovl_system.save_state(path)

def cmd_load(sovl_system, args):
    path = args[0] if args else None
    print(f"Loading state{' from ' + path if path else ''}...")
    sovl_system.load_state(path)

def cmd_dream(sovl_system, args):
    print("Triggering dream cycle...")
    sovl_system._dream()

def cmd_tune(sovl_system, args):
    if not args or args[0] != "cross":
        raise ValueError("Error: Use 'tune cross [weight]'.")
    weight = float(args[1]) if len(args) > 1 else None
    print(f"Setting cross-attention weight to {weight if weight is not None else 'default'}...")
    sovl_system.tune_cross_attention(weight=weight)

def cmd_memory(sovl_system, args):
    mode = args[0]
    if mode not in ['on', 'off']:
        raise ValueError("Error: Use 'memory on' or 'memory off'.")
    mode_val = 'both_mem' if mode == 'on' else 'no_mem'
    print(f"Setting memory to {mode}...")
    sovl_system.toggle_memory(mode_val)

def cmd_status(sovl_system, args):
    print("\n--- System Status ---")
    print(f"Conversation ID: {sovl_system.history.conversation_id}")
    print(f"Temperament: {sovl_system.temperament_score:.2f}")
    print(f"Confidence: {sovl_system.confidence_history[-1] if sovl_system.confidence_history else 'N/A'}")
    print(f"Memory: {'On' if sovl_system.use_scaffold_memory or sovl_system.use_token_map_memory else 'Off'}")
    print(f"Data Exposure: {sovl_system.data_exposure}")
    print(f"Last Trained: {sovl_system.last_trained}")
    print(f"Gestating: {'Yes' if sovl_system.is_sleeping else 'No'}")

def cmd_log(sovl_system, args):
    if args[0] != "view":
        raise ValueError("Error: Use 'log view'.")
    print("\n--- Last 5 Log Entries ---")
    logs = sovl_system.logger.read()[-5:]
    for log in logs:
        print(f"Time: {log.get('timestamp', 'N/A')}, "
              f"Prompt: {log.get('prompt', 'N/A')[:30]}..., "
              f"Response: {log.get('response', 'N/A')[:30]}...")

def cmd_config(sovl_system, args):
    key = args[0]
    if len(args) == 1:
        value = getattr(sovl_system, 'config', {}).get(key, "Not found")
        print(f"Config {key}: {value}")
    else:
        value = float(args[1]) if '.' in args[1] else int(args[1])
        print(f"Setting {key} to {value} (Note: Changes apply on restart)")

def cmd_reset(sovl_system, args):
    print("Resetting system state...")
    sovl_system.cleanup()
    sovl_system.__init__()  # Reinitialize (assumes no args needed)
    sovl_system.wake_up()

def cmd_spark(sovl_system, args):
    print("Sparking curiosity...")
    question = sovl_system.generate_curiosity_question()
    if not question:
        print("No curious question generated. Try again later.")
        return False
    print(f"Curiosity: {question}")
    response = generate_response(sovl_system, question)
    print(f"Response: {response}")
    log_action(sovl_system, question, response, 0.5, True)
    return False

def cmd_reflect(sovl_system, args):
    print("Reflecting on recent interactions...")
    logs = sovl_system.logger.read()[-3:]
    if not logs:
        print("Nothing to reflect on yet. Try generating some responses first.")
        return False
    recent_prompts = [log.get('prompt', '') for log in logs if 'prompt' in log]
    reflection = ("I haven't been prompted much lately." if not recent_prompts else
                  f"I've noticed a lot of talk about {recent_prompts[-1].split()[0] if recent_prompts[-1] else 'things'} lately.")
    print(f"Reflection: {reflection}")
    elaboration = generate_response(sovl_system, f"Based on recent thoughts: {reflection}")
    print(f"Elaboration: {elaboration}")
    log_action(sovl_system, reflection, elaboration, 0.6, True)
    return False

def cmd_muse(sovl_system, args):
    print("Musing...")
    logs = sovl_system.logger.read()[-3:]
    if not logs:
        inspiration = "silence"
        print("Inspiration: silence (nothing recent to draw from)")
    else:
        recent_prompts = [log.get('prompt', '') for log in logs if 'prompt' in log]
        inspiration = recent_prompts[-1].split()[0] if recent_prompts and recent_prompts[-1] else "mystery"
        print(f"Inspiration: \"{inspiration}\" (from recent interactions)")
    thought = generate_response(sovl_system, f"A whimsical thought about {inspiration}:", temp_adjust=0.1)
    print(f"Thought: {thought}")
    log_action(sovl_system, f"Musing on {inspiration}", thought, 0.7, True)
    return False

def cmd_flare(sovl_system, args):
    print("Flaring up...")
    original_temperament = sovl_system.temperament_score
    sovl_system.temperament_score = 1.0
    print("Temperament Cranked: 1.0 (MAX)")
    prompt = ' '.join(args) if args else "THIS QUIET IS TOO MUCH!"
    outburst = generate_response(sovl_system, prompt, max_tokens=80, temp_adjust=1.5 - sovl_system.base_temperature).upper()
    print(f"Outburst: {outburst}")
    sovl_system.temperament_score = original_temperament
    print("[Temperament resets to normal]")
    log_action(sovl_system, f"Flare: {prompt}", outburst, 0.9, True)
    return False

# New commands
def cmd_echo(sovl_system, args):
    text = ' '.join(args)
    print(f"You said '{text}'")
    response = generate_response(sovl_system, f"Reflecting on '{text}': Why say this now?")
    print(f"Response: {response}")
    log_action(sovl_system, f"Echo: {text}", response, 0.6, False, {"is_echo": True})
    return False

def cmd_debate(sovl_system, args):
    topic = ' '.join(args)
    print(f"Debating: {topic}")
    stance = "for"
    for turn in range(2):  # 2 turns: argument + rebuttal
        prompt = f"{'Argue for' if stance == 'for' else 'Argue against'} '{topic}':"
        response = generate_response(sovl_system, prompt, max_tokens=80)
        print(f"[{'Argument' if turn == 0 else 'Rebuttal'}] {response}")
        log_action(sovl_system, prompt, response, 0.7, True)
        stance = "against" if stance == "for" else "for"
        sovl_system.temperament_score = min(1.0, sovl_system.temperament_score + 0.2)  # Swing temperament
    return False

def cmd_glitch(sovl_system, args):
    prompt = ' '.join(args)
    print(f"Processing glitch: {prompt}")
    # Simulate noise (assumes enable_error_listening exists)
    sovl_system.enable_error_listening()  # Placeholder if not implemented
    response = generate_response(sovl_system, f"Fix this mess: {prompt}", temp_adjust=0.2)
    print(f"Response: I sense chaos. Did you mean something like '{response}'?")
    log_action(sovl_system, f"Glitch: {prompt}", response, 0.5, False)
    return False

def cmd_rewind(sovl_system, args):
    steps = int(args[0]) if args else 1
    print(f"Rewinding {steps} step(s)...")
    logs = sovl_system.logger.read()[-steps-1:-1]  # Get past interactions
    if not logs:
        print("No history to rewind.")
        return False
    past_prompt = logs[-1].get('prompt', 'unknown')
    past_response = logs[-1].get('response', 'lost')
    print(f"{steps} command(s) ago, you said: '{past_prompt}'. I replied: '{past_response}'")
    new_response = generate_response(sovl_system, f"Reinterpreting '{past_prompt}' now:")
    print(f"Now I think: {new_response}")
    log_action(sovl_system, f"Rewind {steps}: {past_prompt}", new_response, 0.6, True)
    return False

def cmd_mimic(sovl_system, args):
    style = args[0]
    prompt = ' '.join(args[1:])
    print(f"Mimicking {style} for: {prompt}")
    sovl_system.scaffold_weight = 0.8  # Boost style bias (assumes this attribute exists)
    response = generate_response(sovl_system, f"In the style of {style}: {prompt}")
    print(f"Response: {response}")
    sovl_system.scaffold_weight = 0.5  # Reset (assumed default)
    log_action(sovl_system, f"Mimic {style}: {prompt}", response, 0.7, False)
    return False

def cmd_panic(sovl_system, args):
    print("ERROR STATE: Rebooting synapses...")
    sovl_system.save_state("panic_save.json")  # Auto-save logs
    sovl_system.cleanup()
    sovl_system._reset_sleep_state()  # Assumes this exists
    sovl_system.__init__()  # Reinitialize
    print("[System reloaded]")
    return False

# Command dispatch table
COMMANDS = {
    'quit': cmd_quit, 'exit': cmd_quit, 'train': cmd_train, 'generate': cmd_generate,
    'save': cmd_save, 'load': cmd_load, 'dream': cmd_dream, 'tune': cmd_tune,
    'memory': cmd_memory, 'status': cmd_status, 'log': cmd_log, 'config': cmd_config,
    'reset': cmd_reset, 'spark': cmd_spark, 'reflect': cmd_reflect, 'muse': cmd_muse,
    'flare': cmd_flare, 'echo': cmd_echo, 'debate': cmd_debate, 'glitch': cmd_glitch,
    'rewind': cmd_rewind, 'mimic': cmd_mimic, 'panic': cmd_panic
}

def run_cli(config_manager=None):
    from sovl_config import ConfigManager  # Import here to avoid circular imports
    print("\nInitializing SOVL System...")
    sovl_system = None
    try:
        # Use provided config_manager or create a new one
        if config_manager is None:
            config_manager = ConfigManager("sovl_config.json")
        sovl_system = SOVLSystem(config_manager)
        print("\nSystem Ready.")
        valid_commands = list(COMMANDS.keys())
        print("Commands: quit, exit, train [epochs] [--dry-run], generate <prompt> [max_tokens], "
              "save [path], load [path], dream, tune cross [weight], memory <on|off], "
              "status, log view, config <key> [value], reset, spark, reflect, muse, flare, "
              "echo [text], debate [topic], glitch [prompt], rewind [steps], mimic [style] [prompt], panic")

        while True:
            user_input = input("\nEnter command: ").strip()
            try:
                cmd, args = parse_args(user_input.split())
                if cmd in COMMANDS:
                    should_exit = COMMANDS[cmd](sovl_system, args)
                    if should_exit:
                        break
                else:
                    print(f"Error: Unknown command. Valid commands: {', '.join(valid_commands)}")
            except ValueError as e:
                print(str(e))
            except Exception as e:
                print(f"Command error: {e}")

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
