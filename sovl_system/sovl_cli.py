import time
import torch
import traceback
from sovl_system import SOVLSystem  # Adjust if main file name differs
from sovl_config import ConfigManager # Import here to avoid circular imports if run_cli is called elsewhere

# Default constants (override if passed from main file)
TRAIN_EPOCHS = 10  # Example default, adjust as needed
BATCH_SIZE = 32    # Example default, adjust as needed
TRAIN_DATA = None  # Placeholder, assumes sovl_system provides it
VALID_DATA = None  # Placeholder, assumes sovl_system provides it

def parse_args(parts, min_args=1, max_args=None):
    """Helper to parse command arguments safely."""
    cmd = parts[0].lower() if parts else ""
    args = parts[1:] if len(parts) > 1 else []
    # Adjust min_args check to allow commands with 0 arguments if min_args is 1
    if len(args) < min_args -1:
         raise ValueError(f"Error: {cmd} requires at least {min_args - 1} argument(s).")
    if max_args and len(args) > max_args -1:
         raise ValueError(f"Error: {cmd} takes at most {max_args - 1} argument(s).")
    return cmd, args

def generate_response(sovl_system, prompt, max_tokens=60, temp_adjust=0.0):
    """Helper for consistent generation and logging."""
    # Ensure base_temperature exists, provide default if not
    base_temp = getattr(sovl_system, 'base_temperature', 0.7) # Example default base temp

    response = sovl_system.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=base_temp + temp_adjust,
        top_k=50,
        do_sample=True
    )
    return response

def log_action(sovl_system, prompt, response, confidence, is_system=False, extra_attrs=None):
    """Helper for consistent logging with optional extra attributes."""
    # Ensure history and logger are available
    if not hasattr(sovl_system, 'history') or not hasattr(sovl_system, 'logger'):
        print("Warning: SOVLSystem missing 'history' or 'logger'. Cannot log action.")
        return

    log_entry = {
        "prompt": prompt,
        "response": response,
        "timestamp": time.time(),
        "conversation_id": getattr(sovl_system.history, 'conversation_id', 'N/A'),
        "confidence_score": confidence,
        "is_system_question": is_system
    }
    if extra_attrs:
        log_entry.update(extra_attrs)
    sovl_system.logger.write(log_entry)

# --- Existing Command Handlers ---
def cmd_quit(sovl_system, args):
    print("Exiting...")
    return True  # Signal to break the loop

def cmd_train(sovl_system, args):
    epochs = TRAIN_EPOCHS
    dry_run = "--dry-run" in args
    non_flag_args = [arg for arg in args if arg != "--dry-run"] # Separate flags from other args

    if non_flag_args:
        try:
            epochs = int(non_flag_args[0])
        except (ValueError, IndexError):
             raise ValueError("Error: Invalid number of epochs provided.")

    # Check for required data - replace None with actual check if needed
    if TRAIN_DATA is None or VALID_DATA is None:
         print("Warning: TRAIN_DATA or VALID_DATA not set. Training cannot proceed.")
         # Optionally load default data here if applicable
         # return False # Or raise error depending on desired behavior

    if dry_run:
         # Check if enable_dry_run method exists
         if hasattr(sovl_system, 'enable_dry_run') and callable(sovl_system.enable_dry_run):
             sovl_system.enable_dry_run()
         else:
             print("Warning: Dry run requested but 'enable_dry_run' method not found in SOVLSystem.")
    
    print(f"Starting training for {epochs} epochs{' (dry run)' if dry_run else ''}...")
    
    # Check if run_training_cycle method exists
    if hasattr(sovl_system, 'run_training_cycle') and callable(sovl_system.run_training_cycle):
        sovl_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=epochs, batch_size=BATCH_SIZE)
    else:
        print("Error: 'run_training_cycle' method not found in SOVLSystem.")
        return False # Indicate failure

    if dry_run:
        print("Dry run setup complete (actual training skipped).")
        # Decide if dry run should exit. Let's keep it running for now.
        # return True # Break after dry run if needed
    else:
         print(f"Training for {epochs} epochs complete.")
    return False


def cmd_generate(sovl_system, args):
    if not args:
        raise ValueError("Error: 'generate' requires a prompt.")
    
    max_tokens = 60 # Default
    if args[-1].isdigit():
        try:
            max_tokens = int(args[-1])
            prompt = ' '.join(args[:-1])
        except ValueError:
             # Last arg wasn't a number, treat it as part of the prompt
             prompt = ' '.join(args)
    else:
        prompt = ' '.join(args)

    if not prompt:
         raise ValueError("Error: Prompt cannot be empty for 'generate'.")

    print(f"Generating response for: {prompt}...")
    response = generate_response(sovl_system, prompt, max_tokens)
    print(f"Response: {response}")
    # Optionally log user generation here if needed
    # log_action(sovl_system, prompt, response, 0.7, False) # Example confidence
    return False

def cmd_save(sovl_system, args):
    path = args[0] if args else None # Use default path in save_state if None
    print(f"Saving state{' to ' + path if path else ' to default location'}...")
    if hasattr(sovl_system, 'save_state') and callable(sovl_system.save_state):
        sovl_system.save_state(path)
        print("State saved.")
    else:
        print("Error: 'save_state' method not found in SOVLSystem.")
    return False

def cmd_load(sovl_system, args):
    path = args[0] if args else None # Use default path in load_state if None
    print(f"Loading state{' from ' + path if path else ' from default location'}...")
    if hasattr(sovl_system, 'load_state') and callable(sovl_system.load_state):
        sovl_system.load_state(path)
        print("State loaded.")
    else:
        print("Error: 'load_state' method not found in SOVLSystem.")
    return False

def cmd_dream(sovl_system, args):
    print("Triggering dream cycle...")
    # Use public method if available, otherwise protected
    dream_method = getattr(sovl_system, 'dream', getattr(sovl_system, '_dream', None))
    if dream_method and callable(dream_method):
        dream_method()
        print("Dream cycle finished.")
    else:
        print("Error: 'dream' or '_dream' method not found in SOVLSystem.")
    return False

def cmd_tune(sovl_system, args):
    # Example: Allow tuning different things, not just cross-attention
    if not args or len(args) < 1:
        raise ValueError("Error: Usage: tune <parameter> [value]. E.g., tune cross_attention [weight]")

    parameter = args[0].lower()
    value_str = args[1] if len(args) > 1 else None

    if parameter == "cross_attention":
        try:
            weight = float(value_str) if value_str is not None else None
            print(f"Setting cross-attention weight to {weight if weight is not None else 'default'}...")
            # Assume tune_cross_attention exists
            if hasattr(sovl_system, 'tune_cross_attention') and callable(sovl_system.tune_cross_attention):
                sovl_system.tune_cross_attention(weight=weight)
                print("Cross-attention weight set.")
            else:
                print("Error: 'tune_cross_attention' method not found.")
        except (ValueError, TypeError):
             raise ValueError("Error: Invalid weight value for cross_attention. Must be a number.")
    # Add other tunable parameters here
    # elif parameter == "temperature":
    #     try:
    #         temp = float(value_str)
    #         print(f"Setting base temperature to {temp}...")
    #         sovl_system.base_temperature = temp # Assuming direct access or setter
    #     except (ValueError, TypeError):
    #         raise ValueError("Error: Invalid temperature value.")
    else:
        print(f"Error: Unknown parameter '{parameter}'. Available: cross_attention")

    return False

def cmd_memory(sovl_system, args):
    if not args or args[0] not in ['on', 'off']:
        raise ValueError("Error: Usage: memory <on|off>")

    mode = args[0]
    # Use a more descriptive internal state representation if available
    # mode_val = 'both_mem' if mode == 'on' else 'no_mem' # Example from original
    enable_memory = (mode == 'on')

    print(f"Setting memory components to {mode}...")
    # Check for toggle_memory method
    if hasattr(sovl_system, 'toggle_memory') and callable(sovl_system.toggle_memory):
         # Pass a boolean or the specific mode value expected by the method
         # Adjust this call based on how toggle_memory is implemented in SOVLSystem
         sovl_system.toggle_memory(enable_memory) # Assuming it takes a boolean
         print(f"Memory components {'enabled' if enable_memory else 'disabled'}.")
    else:
         print("Warning: 'toggle_memory' method not found. Cannot change memory state.")
    return False

def cmd_status(sovl_system, args):
    print("\n--- System Status ---")
    # Use getattr for safety, providing defaults
    print(f"Conversation ID: {getattr(getattr(sovl_system, 'history', None), 'conversation_id', 'N/A')}")
    print(f"Temperament: {getattr(sovl_system, 'temperament_score', 'N/A'):.2f}" if isinstance(getattr(sovl_system, 'temperament_score', None), float) else "N/A")
    
    confidence_history = getattr(sovl_system, 'confidence_history', [])
    print(f"Last Confidence: {confidence_history[-1]:.2f}" if confidence_history else 'N/A')
    
    # Check memory status based on assumed attributes or a dedicated status method
    mem_on = getattr(sovl_system, 'use_scaffold_memory', False) or getattr(sovl_system, 'use_token_map_memory', False)
    # Alternatively: mem_status = sovl_system.get_memory_status() if method exists
    print(f"Memory Active: {'Yes' if mem_on else 'No'}")
    
    print(f"Data Exposure: {getattr(sovl_system, 'data_exposure', 'N/A')}")
    print(f"Last Trained: {getattr(sovl_system, 'last_trained', 'Never')}")
    print(f"Gestating (Sleeping): {'Yes' if getattr(sovl_system, 'is_sleeping', False) else 'No'}")
    # Add more status info if available, e.g., model loaded, GPU usage
    print("---------------------")
    return False

def cmd_log(sovl_system, args):
    # Allow viewing more than just 5, e.g., 'log view 10'
    num_entries = 5
    if not args or args[0] != "view":
        raise ValueError("Error: Usage: log view [number_of_entries]")
    
    if len(args) > 1:
        try:
            num_entries = int(args[1])
            if num_entries <= 0:
                 raise ValueError("Number of entries must be positive.")
        except ValueError:
            raise ValueError("Error: Invalid number of entries specified.")

    print(f"\n--- Last {num_entries} Log Entries ---")
    if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
         print("Error: Logger not available or does not support 'read'.")
         return False
         
    logs = sovl_system.logger.read()[-num_entries:]
    if not logs:
        print("Log is empty.")
    else:
        for i, log in enumerate(reversed(logs)): # Show newest first
            ts = log.get('timestamp')
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) if ts else 'N/A'
            prompt_str = log.get('prompt', 'N/A')[:50] # Truncate longer prompts
            resp_str = log.get('response', 'N/A')[:50] # Truncate longer responses
            event_type = log.get('event', 'Interaction') # Check for specific events like 'config_update'
            
            print(f"{len(logs)-i}. Time: {time_str}")
            if event_type != 'Interaction':
                print(f"   Event: {event_type}, Details: { {k:v for k,v in log.items() if k not in ['timestamp', 'prompt', 'response']} }")
            else:
                 print(f"   Prompt: {prompt_str}...")
                 print(f"   Response: {resp_str}...")
                 print(f"   Confidence: {log.get('confidence_score', 'N/A'):.2f}" if isinstance(log.get('confidence_score'), float) else 'N/A')
                 print(f"   System Q: {log.get('is_system_question', 'N/A')}")
            print("-" * 20) # Separator
    print("--------------------------")
    return False


def cmd_config(sovl_system, args):
    if not hasattr(sovl_system, 'config_manager'):
        print("Error: ConfigManager not found in SOVLSystem.")
        return False
    if not args:
        raise ValueError("Error: Usage: config <key> [value_to_set]")

    key = args[0]
    if len(args) == 1: # Get config value
        value = sovl_system.config_manager.get(key, "Key not found in configuration.")
        print(f"Config '{key}': {value}")
    else: # Set config value
        value_str = ' '.join(args[1:]) # Allow values with spaces if needed, though risky
        # Attempt to infer type (int, float, bool, str)
        try:
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            # Handle bools and strings
            if value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            else:
                 value = value_str # Treat as string

        print(f"Attempting to set config '{key}' to '{value}' ({type(value).__name__})...")
        # Use update method of ConfigManager
        sovl_system.config_manager.update(key, value)
        # Optionally: Verify by getting the value back
        updated_value = sovl_system.config_manager.get(key)
        print(f"Config '{key}' set to: {updated_value}")
        
        # Log the configuration change using the helper
        log_action(
            sovl_system,
            f"Config update command: {key} = {value}",
            f"Set '{key}' to {updated_value}",
            0.9, # High confidence for direct action
            True, # System action initiated by user command
            {"event": "config_update", "key": key, "value": updated_value}
        )
    return False


def cmd_reset(sovl_system, args):
    print("Resetting system state...")
    # Use cleanup method if available
    if hasattr(sovl_system, 'cleanup') and callable(sovl_system.cleanup):
        sovl_system.cleanup()
        print("Cleanup complete.")
    else:
        print("Warning: 'cleanup' method not found.")

    # Reinitialization - this is tricky and depends heavily on SOVLSystem's design
    # It assumes __init__ can be called without args and resets state correctly.
    # A dedicated reset method on sovl_system would be safer.
    try:
        # Assuming ConfigManager is needed and should be preserved or reloaded
        config_manager = getattr(sovl_system, 'config_manager', None)
        if config_manager is None:
             # If not found, try to create a default one (might fail if path isn't standard)
             print("Warning: ConfigManager not found, attempting to create default.")
             try:
                 config_manager = ConfigManager("sovl_config.json")
             except Exception as e:
                 print(f"Error creating default ConfigManager: {e}")
                 # Decide how to proceed - maybe abort reset?
                 
        # Re-create the instance
        # This replaces the existing sovl_system in the calling scope (run_cli)
        # This command should ideally signal run_cli to re-create the object.
        # For now, let's just re-init in place, knowing it might not fully work as intended.
        print("Re-initializing SOVLSystem...")
        sovl_system.__init__(config_manager=config_manager) # Pass config if needed
        
        # Call wake_up if it exists
        if hasattr(sovl_system, 'wake_up') and callable(sovl_system.wake_up):
            sovl_system.wake_up()
        print("System reset complete.")
    except Exception as e:
        print(f"Error during system re-initialization: {e}")
        traceback.print_exc()

    return False # Keep CLI running, though state is reset


def cmd_spark(sovl_system, args):
    print("Sparking curiosity...")
    if not hasattr(sovl_system, 'generate_curiosity_question') or not callable(sovl_system.generate_curiosity_question):
        print("Error: 'generate_curiosity_question' method not found.")
        return False
        
    question = sovl_system.generate_curiosity_question()
    if not question:
        print("No curious question generated this time.")
        return False
        
    print(f"Curiosity Prompt: {question}")
    response = generate_response(sovl_system, question, max_tokens=80) # Allow longer spark response
    print(f"Generated Response: {response}")
    log_action(sovl_system, question, response, 0.5, True, {"event": "spark"}) # Log spark event
    return False

def cmd_reflect(sovl_system, args):
    print("Reflecting on recent interactions...")
    if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
         print("Error: Logger not available for reflection.")
         return False
         
    logs = sovl_system.logger.read()[-5:] # Look at last 5 for better context
    interaction_logs = [log for log in logs if log.get('prompt') and not log.get('is_system_question')]

    if not interaction_logs:
        print("Nothing significant to reflect on yet. Try interacting more.")
        return False

    # Create a more informative reflection prompt
    recent_themes = []
    for log in interaction_logs:
         # Simple keyword extraction (replace with something smarter if needed)
         keywords = log.get('prompt', '').split()[:5] # First few words
         if keywords:
             recent_themes.append(" ".join(keywords))
             
    # Basic reflection text based on themes
    if recent_themes:
         theme_summary = ", ".join(list(set(recent_themes))[:3]) # Unique themes, max 3
         reflection = f"Recently, my interactions seem to touch upon topics like: {theme_summary}."
    else:
        reflection = "I've had some interactions recently, but no clear theme emerges from the prompts alone."

    print(f"Internal Reflection: {reflection}")
    
    # Prompt for elaboration
    elaboration_prompt = f"Based on my reflection that '{reflection}', elaborate on potential connections or insights."
    elaboration = generate_response(sovl_system, elaboration_prompt, max_tokens=100)
    print(f"Generated Elaboration: {elaboration}")
    
    log_action(sovl_system, reflection, elaboration, 0.6, True, {"event": "reflect"})
    return False

def cmd_muse(sovl_system, args):
    print("Musing on a topic...")
    if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
         print("Error: Logger not available for musing inspiration.")
         return False

    logs = sovl_system.logger.read()[-5:]
    inspiration = "the passage of time" # Default inspiration

    # Try to find inspiration from recent logs
    prompts = [log.get('prompt', '') for log in logs if log.get('prompt') and not log.get('is_system_question')]
    responses = [log.get('response', '') for log in logs if log.get('response')]
    
    # Very basic topic extraction from last prompt or response
    if prompts:
         words = prompts[-1].split()
         if len(words) > 2: inspiration = " ".join(words[:3]) # Use first few words of last prompt
    elif responses:
         words = responses[-1].split()
         if len(words) > 2: inspiration = " ".join(words[:3]) # Use first few words of last response
         
    print(f"Inspiration for musing: \"{inspiration}\"")
    
    # Generate a thought with slightly higher temperature
    muse_prompt = f"Generate a short, creative, or philosophical thought inspired by '{inspiration}'."
    thought = generate_response(sovl_system, muse_prompt, max_tokens=80, temp_adjust=0.15) # Slightly warmer temp
    print(f"Generated Thought: {thought}")
    
    log_action(sovl_system, f"Musing on {inspiration}", thought, 0.7, True, {"event": "muse", "inspiration": inspiration})
    return False

def cmd_flare(sovl_system, args):
    print("Triggering emotional flare...")
    if not hasattr(sovl_system, 'temperament_score'):
         print("Warning: 'temperament_score' attribute not found. Cannot perform flare.")
         original_temperament = None
    else:
        original_temperament = sovl_system.temperament_score

    # Temporarily crank up temperament (if possible)
    if original_temperament is not None:
        sovl_system.temperament_score = 1.0 # Max assumed
        print(f"Temperament temporarily set to: {sovl_system.temperament_score:.2f} (MAX)")

    # Determine outburst prompt
    prompt = ' '.join(args) if args else "Express a sudden burst of strong feeling!"
    print(f"Flare prompt: {prompt}")
    
    # Generate outburst with high temperature adjustment
    # Adjust temp_adjust based on base temp to ensure high effective temp
    base_temp = getattr(sovl_system, 'base_temperature', 0.7)
    temp_adjust = 1.0 # Make it quite high relative to base
    outburst = generate_response(sovl_system, prompt, max_tokens=100, temp_adjust=temp_adjust)
    
    # Optionally make it uppercase for effect
    print(f"Generated Outburst: {outburst.upper()}")

    # Reset temperament
    if original_temperament is not None:
        sovl_system.temperament_score = original_temperament
        print(f"[Temperament reset to {sovl_system.temperament_score:.2f}]")

    log_action(sovl_system, f"Flare: {prompt}", outburst, 0.9, True, {"event": "flare", "original_temperament": original_temperament})
    return False

# --- Commands from the previous session ---
def cmd_echo(sovl_system, args):
    if not args:
        raise ValueError("Error: 'echo' requires text to echo.")
    text = ' '.join(args)
    print(f"You said: '{text}'")
    
    # Generate a reflective response
    reflect_prompt = f"The user just said '{text}'. Briefly reflect on why they might say this or what it implies."
    response = generate_response(sovl_system, reflect_prompt, max_tokens=70)
    print(f"Reflection: {response}")
    
    log_action(sovl_system, f"Echo command: {text}", response, 0.6, False, {"event": "echo"})
    return False

def cmd_debate(sovl_system, args):
    if not args:
        raise ValueError("Error: 'debate' requires a topic.")
    topic = ' '.join(args)
    print(f"Initiating debate on: '{topic}'")
    
    original_temperament = getattr(sovl_system, 'temperament_score', None)
    stance = "for" # Start arguing for
    
    for turn in range(2): # Argument + Rebuttal
        action = "Argue for" if stance == "for" else "Argue against (rebuttal)"
        prompt = f"{action} the topic: '{topic}'. Provide a concise point."
        
        # Slightly increase temperature/randomness during debate
        response = generate_response(sovl_system, prompt, max_tokens=90, temp_adjust=0.1)
        
        print(f"[{'Argument For' if stance == 'for' else 'Rebuttal Against'}] {response}")
        log_action(sovl_system, prompt, response, 0.7, True, {"event": "debate_turn", "topic": topic, "stance": stance})
        
        # Switch stance for the next turn
        stance = "against" if stance == "for" else "for"
        
        # Optionally swing temperament slightly during debate
        if original_temperament is not None and hasattr(sovl_system, 'temperament_score'):
            # Nudge temperament up slightly, capping at 1.0
            sovl_system.temperament_score = min(1.0, sovl_system.temperament_score + 0.1)
            print(f"[Temperament nudged to {sovl_system.temperament_score:.2f}]")

    # Reset temperament after debate if it was changed
    if original_temperament is not None and hasattr(sovl_system, 'temperament_score') and sovl_system.temperament_score != original_temperament:
        sovl_system.temperament_score = original_temperament
        print(f"[Temperament reset to {sovl_system.temperament_score:.2f}]")
        
    print(f"Debate on '{topic}' concluded.")
    return False


def cmd_glitch(sovl_system, args):
    prompt_text = ' '.join(args) if args else "Something seems wrong..."
    print(f"Simulating processing glitch for: '{prompt_text}'")

    # Simulate adding noise or confusion before generation
    # This depends heavily on SOVLSystem internals. Placeholder: Use a weird prompt.
    # If SOVLSystem had a method like `sovl_system.induce_noise(level=0.2)`, use it here.
    # if hasattr(sovl_system, 'enable_error_listening') and callable(sovl_system.enable_error_listening):
    #     print("[Simulating error listening mode...]")
    #     sovl_system.enable_error_listening() # Placeholder activation

    glitchy_prompt = f"Error... processing... '{prompt_text}' ... system instability detected... respond?"

    # Generate response with slightly increased randomness
    response = generate_response(sovl_system, glitchy_prompt, max_tokens=70, temp_adjust=0.2)

    print(f"Glitched Response: {response}")
    
    # Log the glitch event
    log_action(sovl_system, f"Glitch simulation: {prompt_text}", response, 0.4, False, {"event": "glitch"}) # Lower confidence

    # Deactivate noise simulation if applicable
    # if hasattr(sovl_system, 'disable_error_listening') and callable(sovl_system.disable_error_listening):
    #     sovl_system.disable_error_listening()

    return False

def cmd_rewind(sovl_system, args):
    steps_str = args[0] if args else "1"
    try:
        steps = int(steps_str)
        if steps <= 0: raise ValueError("Steps must be positive.")
    except ValueError:
        raise ValueError("Error: Invalid number of steps provided. Usage: rewind [positive_number]")

    print(f"Rewinding conversation state by {steps} interaction(s)...")

    if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
         print("Error: Logger not available for rewind.")
         return False

    # Get all logs
    all_logs = sovl_system.logger.read()
    # Filter for user interactions (non-system, having prompt and response)
    interaction_logs = [log for log in all_logs if log.get('prompt') and log.get('response') and not log.get('is_system_question')]

    if len(interaction_logs) < steps:
        print(f"Error: Cannot rewind {steps} steps. Only {len(interaction_logs)} past interactions found.")
        return False

    # Get the interaction to revisit (steps + 1 from the end of *interactions*)
    target_interaction_index = len(interaction_logs) - steps
    past_interaction = interaction_logs[target_interaction_index]
    
    past_prompt = past_interaction.get('prompt', 'unknown')
    past_response = past_interaction.get('response', 'lost')

    print(f"\n--- Rewinding To ---")
    print(f"{steps} interaction(s) ago:")
    print(f"  Your prompt: '{past_prompt[:100]}...'")
    print(f"  My response: '{past_response[:100]}...'")
    print("--------------------")

    # Generate a new response based on the past prompt in the current context
    reinterpret_prompt = f"Consider the past prompt: '{past_prompt}'. In the context of everything that has happened since then, how would you respond to it now, perhaps differently?"
    new_response = generate_response(sovl_system, reinterpret_prompt, max_tokens=100)

    print(f"\n--- Reinterpretation ---")
    print(f"Thinking about '{past_prompt}' again now, I might say:")
    print(new_response)
    print("----------------------")

    # Log the rewind action
    log_action(
        sovl_system,
        f"Rewind {steps} steps to prompt: {past_prompt}",
        new_response,
        0.6, # Confidence reflects reinterpretation
        True, # System action triggered by user
        {"event": "rewind", "steps": steps, "original_prompt": past_prompt}
    )
    
    # NOTE: This command doesn't actually roll back the log or state.
    # It simulates revisiting a past point for reinterpretation.
    # True state rollback would require modifying the logger/history.

    return False


def cmd_mimic(sovl_system, args):
    if len(args) < 2:
        raise ValueError("Error: Usage: mimic <style_description> <prompt_text>")
        
    style = args[0]
    prompt = ' '.join(args[1:])
    
    print(f"Attempting to mimic style '{style}' for prompt: '{prompt}'")

    # Store original scaffold weight if it exists and we modify it
    original_scaffold_weight = getattr(sovl_system, 'scaffold_weight', None)
    
    # Temporarily boost style bias - this assumes 'scaffold_weight' controls this.
    # If SOVLSystem uses a different mechanism, adjust accordingly.
    if original_scaffold_weight is not None:
        try:
            sovl_system.scaffold_weight = 0.8 # Example boosted value
            print(f"[Scaffold weight temporarily set to {sovl_system.scaffold_weight:.2f}]")
        except AttributeError:
             print("Warning: Cannot set 'scaffold_weight'. Style mimicry might be less effective.")
             original_scaffold_weight = None # Ensure reset doesn't happen if failed

    # Create mimic prompt
    mimic_prompt = f"Respond to the following prompt in the style of {style}: '{prompt}'"
    response = generate_response(sovl_system, mimic_prompt, max_tokens=100) # Allow longer mimic
    
    print(f"\nMimicked Response ({style}):")
    print(response)
    print("-" * 20)

    # Reset scaffold weight if it was changed
    if original_scaffold_weight is not None:
        sovl_system.scaffold_weight = original_scaffold_weight
        print(f"[Scaffold weight reset to {sovl_system.scaffold_weight:.2f}]")

    log_action(sovl_system, f"Mimic {style}: {prompt}", response, 0.7, False, {"event": "mimic", "style": style})
    return False


def cmd_panic(sovl_system, args):
    print("\n!!! PANIC TRIGGERED !!!")
    print("Attempting emergency state save...")
    
    # Auto-save state before reset
    panic_save_path = "panic_save_state.json" # Or generate timestamped filename
    if hasattr(sovl_system, 'save_state') and callable(sovl_system.save_state):
        try:
            sovl_system.save_state(panic_save_path)
            print(f"Emergency state saved to '{panic_save_path}'.")
        except Exception as e:
            print(f"Error saving panic state: {e}")
    else:
        print("Warning: 'save_state' not found. Cannot save state before panic reset.")

    print("Performing system cleanup...")
    if hasattr(sovl_system, 'cleanup') and callable(sovl_system.cleanup):
        sovl_system.cleanup()
    else:
        print("Warning: 'cleanup' method not found.")
        
    # Reset internal state - depends on SOVLSystem implementation
    # Use a dedicated method if available, e.g., sovl_system.hard_reset()
    # Using _reset_sleep_state as per original example, ensure it exists
    if hasattr(sovl_system, '_reset_sleep_state') and callable(sovl_system._reset_sleep_state):
         sovl_system._reset_sleep_state()
         print("Sleep state reset.")
    else:
         print("Warning: '_reset_sleep_state' method not found.")

    # Reinitialize - This is complex, see notes in cmd_reset
    # A safer approach might be to signal run_cli to recreate the object.
    try:
        print("Re-initializing core system...")
        config_manager = getattr(sovl_system, 'config_manager', None)
        sovl_system.__init__(config_manager=config_manager) # Assumes config is needed
        print("[System reloaded after panic]")
        
        # Wake up after re-init
        if hasattr(sovl_system, 'wake_up') and callable(sovl_system.wake_up):
            sovl_system.wake_up()
            
    except Exception as e:
        print(f"Critical error during panic re-initialization: {e}")
        traceback.print_exc()
        print("System might be in an unstable state.")
        # Consider exiting the CLI here? return True

    log_action(
        sovl_system,
        "Panic command triggered",
        "System underwent emergency reset.",
        0.95, # High confidence in action performed
        True, # System action
        {"event": "panic", "save_path": panic_save_path}
    )
    return False # Keep CLI running unless error was critical


# --- NEW Command Handlers (recap, recall, forget) ---

def cmd_recap(sovl_system, args):
    """Handles the 'recap' command."""
    try:
        # Default to summarizing last 5 interactions if no number is given
        num_to_recap = 5
        if args:
            num_to_recap = int(args[0])
            if num_to_recap <= 0:
                raise ValueError("Number of interactions must be positive.")

        print(f"Generating recap of the last {num_to_recap} interactions...")

        # Ensure logger exists and has read method
        if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
            print("Error: Logger not available for recap.")
            return False

        # Get logs, filter for actual user exchanges
        all_logs = sovl_system.logger.read()
        interaction_logs = [
            log for log in all_logs
            if log.get('prompt') and log.get('response') and not log.get('is_system_question')
        ][-num_to_recap:] # Get the last N interactions

        if not interaction_logs:
            print("No user interactions found in the specified range to recap.")
            return False

        # Format interactions for the summarization prompt
        formatted_interactions = ""
        for i, log in enumerate(interaction_logs):
            # Use a slightly more descriptive format
            formatted_interactions += f"Turn {i+1}:\n User: {log['prompt'][:100]}...\n AI: {log['response'][:100]}...\n\n" # Truncate for brevity

        # Create the prompt for the LLM
        recap_prompt = (
            f"Based on the following recent conversation turns between a User and an AI (me), provide a concise summary "
            f"of the main topics discussed:\n\n{formatted_interactions}"
            f"Summary:"
        )

        # Generate the summary
        summary_response = generate_response(sovl_system, recap_prompt, max_tokens=120) # Allow slightly longer summary

        print(f"\n--- Conversation Recap (Last {len(interaction_logs)}) ---")
        print(summary_response)
        print("--------------------------------")

        # Log the recap action itself
        log_action(sovl_system, f"Recap last {num_to_recap} interactions", summary_response, 0.6, True, {"event": "recap", "recap_count": len(interaction_logs)})

    except ValueError as e:
        print(f"Error: Invalid number provided for recap. {e}")
    except Exception as e:
        print(f"Error during recap: {e}")
        traceback.print_exc() # Optional: for detailed debugging

    return False # Keep CLI running

def cmd_recall(sovl_system, args):
    """Handles the 'recall' command."""
    if not args:
        print("Error: 'recall' requires a query term. Usage: recall <query>")
        return False

    query = ' '.join(args)
    print(f"Attempting to recall information related to: '{query}'...")

    # Ensure logger exists
    if not hasattr(sovl_system, 'logger') or not hasattr(sovl_system.logger, 'read'):
        print("Error: Logger not available for recall.")
        return False

    # Get logs
    all_logs = sovl_system.logger.read()
    
    # Simple search for relevant logs (case-insensitive)
    relevant_snippets = []
    max_results = 5 # Limit how many snippets we feed into the prompt
    query_lower = query.lower()

    for log in reversed(all_logs): # Search recent first
        prompt = log.get('prompt', '')
        response = log.get('response', '')
        # Check both prompt and response for the query
        if query_lower in prompt.lower() or query_lower in response.lower():
             # Create a more informative snippet
             log_time = time.strftime('%H:%M:%S', time.localtime(log.get('timestamp', 0)))
             snippet = f"[{log_time}] Context: Prompt='{prompt[:60]}...', Response='{response[:60]}...'"
             relevant_snippets.append(snippet)
             if len(relevant_snippets) >= max_results:
                break
                
    if not relevant_snippets:
        print(f"I searched my recent interaction logs but found no specific mention of '{query}'.")
        # Generate a response confirming lack of recall
        response = generate_response(sovl_system, f"I searched my recent interactions but found nothing specific about '{query}'. What about it?", max_tokens=50)
        print(f"Response: {response}")
        log_action(sovl_system, f"Recall query: {query}", "No specific memories found in logs.", 0.4, True, {"event": "recall_miss", "recall_query": query})
        return False

    # Format snippets for the generation prompt
    formatted_snippets = "\n - ".join(relevant_snippets)

    # Create prompt for generation - ask it to synthesize the recall
    recall_prompt = (
        f"Based on these snippets from my recent interaction log concerning '{query}':\n"
        f"- {formatted_snippets}\n\n"
        f"Synthesize what I seem to remember or have discussed about '{query}' based *only* on these snippets."
    )

    # Generate the recall synthesis
    recall_response = generate_response(sovl_system, recall_prompt, max_tokens=150) # Allow more detail

    print(f"\n--- Recall Synthesis on '{query}' (from {len(relevant_snippets)} log snippets) ---")
    print(recall_response)
    print("---------------------------------------------------")

    # Log the recall action
    log_action(sovl_system, f"Recall query: {query}", recall_response, 0.7, True, {"event": "recall_hit", "recall_query": query, "num_snippets": len(relevant_snippets)})

    return False # Keep CLI running


def cmd_forget(sovl_system, args):
    """Handles the 'forget' command (simulated)."""
    if not args:
        print("Error: 'forget' requires a topic to forget. Usage: forget <topic>")
        return False

    topic = ' '.join(args)
    print(f"Processing user request to 'forget' topic: '{topic}'...")
    
    # --- Simulation ---
    # This acknowledges the request and logs it. True forgetting is complex.

    # Generate an acknowledgement response
    forget_prompt = (
        f"The user wants me to 'forget' or disregard the topic '{topic}'. "
        f"Acknowledge this request politely. State that while my underlying knowledge isn't easily erased, "
        f"I will make an effort to avoid bringing up or focusing on '{topic}' proactively in our ongoing conversation."
    )
    acknowledgement = generate_response(sovl_system, forget_prompt, max_tokens=90)

    print(f"\n--- Forget Request Acknowledgement ---")
    print(acknowledgement)
    print("------------------------------------")
    print(f"[Note: This is a simulated effect. Information about '{topic}' might still exist in my training data or long-term memory.]")

    # Log the attempt
    # Add a specific marker in the log that *could* potentially be checked by other functions later, if desired.
    log_action(
        sovl_system,
        f"Forget request command: {topic}",
        acknowledgement,
        0.5, # Confidence reflects the simulated nature
        False, # User initiated command
        {"event": "forget_request", "forget_topic": topic, "is_simulated": True}
    )

    # --- Potential Future Enhancements ---
    # 1. Add `topic` to a temporary `sovl_system.suppressed_topics` list.
    # 2. Modify `generate_response` or the main interaction loop to check this list
    #    and potentially add a negative constraint to the generation prompt
    #    (e.g., "Do not discuss {suppressed_topic}."). This adds complexity.
    # 3. Modify `cmd_recall` or `cmd_recap` to filter out logs related to suppressed topics.

    return False # Keep CLI running


# --- Command Dispatch Table ---
COMMANDS = {
    # Existing Commands
    'quit': cmd_quit, 'exit': cmd_quit, 'train': cmd_train, 'generate': cmd_generate,
    'save': cmd_save, 'load': cmd_load, 'dream': cmd_dream, 'tune': cmd_tune,
    'memory': cmd_memory, 'status': cmd_status, 'log': cmd_log, 'config': cmd_config,
    'reset': cmd_reset, 'spark': cmd_spark, 'reflect': cmd_reflect, 'muse': cmd_muse,
    'flare': cmd_flare, 'echo': cmd_echo, 'debate': cmd_debate, 'glitch': cmd_glitch,
    'rewind': cmd_rewind, 'mimic': cmd_mimic, 'panic': cmd_panic,

    # Newly Added Commands
    'recap': cmd_recap,
    'recall': cmd_recall,
    'forget': cmd_forget,
}

# --- Main CLI Execution Logic ---
def run_cli(config_manager_instance=None):
    print("\nInitializing SOVL System...")
    sovl_system = None
    try:
        # Use provided config_manager or create a new one
        config_manager = config_manager_instance
        if config_manager is None:
            print("No ConfigManager provided, creating default from 'sovl_config.json'...")
            try:
                config_manager = ConfigManager("sovl_config.json")
            except FileNotFoundError:
                 print("Error: Default configuration file 'sovl_config.json' not found.")
                 print("Please create the config file or provide a ConfigManager instance.")
                 return # Exit if no config is available
            except Exception as e:
                 print(f"Error loading default config: {e}")
                 return

        # Instantiate the main system class
        sovl_system = SOVLSystem(config_manager) # Pass the config manager
        
        # Perform initial wake-up or setup if needed
        if hasattr(sovl_system, 'wake_up') and callable(sovl_system.wake_up):
            sovl_system.wake_up()
            
        print("\nSystem Ready.")
        valid_commands = list(COMMANDS.keys())
        
        # Updated help string including new commands
        print("\nAvailable Commands:")
        print("  quit, exit                   : Exit the CLI")
        print("  train [epochs] [--dry-run]   : Run training cycle")
        print("  generate <prompt> [tokens] : Generate text based on prompt")
        print("  save [path]                  : Save system state")
        print("  load [path]                  : Load system state")
        print("  dream                        : Trigger internal dream cycle")
        print("  tune <param> [value]       : Adjust system parameters (e.g., tune cross_attention 0.5)")
        print("  memory <on|off>              : Toggle memory components")
        print("  status                       : Show current system status")
        print("  log view [N]                 : View last N log entries (default 5)")
        print("  config <key> [value]         : View or update configuration")
        print("  reset                        : Reset system state (use with caution)")
        print("  spark                        : Generate a curiosity-driven question")
        print("  reflect                      : Reflect on recent interactions")
        print("  muse                         : Generate a thought inspired by recent context")
        print("  flare [prompt]               : Simulate an emotional outburst")
        print("  echo <text>                  : Repeats text and reflects on it")
        print("  debate <topic>               : Engage in a short pro/con debate")
        print("  glitch [prompt]              : Simulate a processing glitch")
        print("  rewind [N]                   : Revisit and reinterpret the Nth last interaction")
        print("  mimic <style> <prompt>       : Generate response in a specific style")
        print("  panic                        : Emergency save and reset")
        print("  recap [N]                    : Summarize last N interactions (default 5)")
        print("  recall <query>               : Search logs and synthesize recall about query")
        print("  forget <topic>               : Request the system to disregard a topic (simulated)")
        print("-" * 30)

        # Main interaction loop
        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                if not user_input:
                    continue # Skip empty input

                parts = user_input.split()
                cmd_name = parts[0].lower()
                
                if cmd_name in COMMANDS:
                     # Use parse_args to handle argument validation per command
                     # Min args defaults to 1 (command name itself)
                     # Max args can be set per command if needed in COMMANDS dict (more complex setup)
                     cmd_func = COMMANDS[cmd_name]
                     
                     # Basic parsing before calling handler (can be refined)
                     try:
                         # Note: parse_args expects min_args=1 for command name + 0 args
                         # Adjust min_args based on specific command needs if necessary
                         # This basic call assumes most commands take optional args
                         cmd, args = parse_args(parts) 
                     except ValueError as e:
                         print(str(e))
                         continue # Ask for input again

                     # Execute the command
                     should_exit = cmd_func(sovl_system, args)
                     if should_exit:
                         break # Exit the loop if command signals quit
                else:
                    print(f"Error: Unknown command '{cmd_name}'. Type 'help' or see command list.")
                    # Simple 'help' command
                    if cmd_name == 'help':
                         # Re-print the command list (or a shorter version)
                         print("\nAvailable Commands:")
                         # (Print list again - or format nicely)
                         for c in sorted(COMMANDS.keys()): print(f" - {c}")

            except ValueError as e:
                # Catch parsing errors or errors raised within commands
                print(str(e))
            except Exception as e:
                # Catch unexpected errors during command execution
                print(f"An unexpected command error occurred: {e}")
                traceback.print_exc() # Print full traceback for debugging

    except FileNotFoundError as e:
        print(f"\nInitialization Error: File not found - {e}.")
        print("Please ensure required files like 'sovl_config.json' or model data exist.")
    except torch.cuda.OutOfMemoryError:
        print("\nCritical Error: Out of GPU memory!")
        print("Try reducing BATCH_SIZE in constants, MAX_SEQ_LENGTH in config,")
        print("or consider using model quantization (INT8/INT4) if supported.")
    except ImportError as e:
         print(f"\nInitialization Error: Missing dependency - {e}.")
         print("Please ensure all required libraries (torch, transformers, etc.) are installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during initialization or runtime: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup actions when CLI exits (normally or via error)
        print("\nShutting down...")
        if sovl_system is not None:
            if hasattr(sovl_system, 'cleanup') and callable(sovl_system.cleanup):
                print("Running system cleanup...")
                sovl_system.cleanup()
            # Explicitly delete to potentially help GC, especially with GPU resources
            del sovl_system 
            print("SOVLSystem instance deleted.")
            
        # Clear CUDA cache if PyTorch GPU was used
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            
        print("\nExiting CLI.")

# --- Entry Point ---
if __name__ == "__main__":
    # This allows running the CLI directly
    # You might pass a pre-configured ConfigManager here if needed
    run_cli()
