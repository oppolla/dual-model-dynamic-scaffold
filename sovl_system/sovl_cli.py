import time
import torch
import traceback
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
from contextlib import contextlib
import functools
from sovl_system import SOVLSystem
from sovl_config import ConfigManager

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_DATA = None
VALID_DATA = None

COMMAND_CATEGORIES = {
    "System": ["quit", "exit", "save", "load", "reset", "status", "help"],
    "Training": ["train", "dream"],
    "Generation": ["generate", "echo", "mimic"],
    "Memory": ["memory", "recall", "forget", "recap"],
    "Interaction": ["muse", "flare", "debate", "spark", "reflect"],
    "Debug": ["log", "config", "panic", "glitch"],
    "Advanced": ["tune", "rewind"],
    "History": ["history"]
}

class CommandHistory:
    def __init__(self, max_size: int = 100):
        self.history: List[Tuple[float, str, Optional[str]]] = []
        self.max_size = max_size

    def add(self, command: str, result: Optional[str] = None):
        self.history.append((time.time(), command, result))
        if len(self.history) > self.max_size:
            self.history.pop(0)

    def get_last(self, n: int = 1) -> List[Tuple[float, str, Optional[str]]]:
        return self.history[-n:]

    def search(self, term: str) -> List[Tuple[float, str, Optional[str]]]:
        return [entry for entry in self.history if term.lower() in entry[1].lower()]

    def format_entry(self, entry: Tuple[float, str, Optional[str]]) -> str:
        timestamp, cmd, result = entry
        time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        status = "✓" if result and "error" not in result.lower() else "✗"
        return f"{time_str} [{status}] {cmd}"

class CommandHandler:
    """Centralized command handling logic."""
    def __init__(self, sovl_system: SOVLSystem):
        self.system = sovl_system
        self.commands: Dict[str, Callable] = {
            'quit': self.cmd_quit, 'exit': self.cmd_quit,
            'train': self.cmd_train, 'generate': self.cmd_generate,
            'save': self.cmd_save, 'load': self.cmd_load,
            'dream': self.cmd_dream, 'tune': self.cmd_tune,
            'memory': self.cmd_memory, 'status': self.cmd_status,
            'log': self.cmd_log, 'config': self.cmd_config,
            'reset': self.cmd_reset, 'spark': self.cmd_spark,
            'reflect': self.cmd_reflect, 'muse': self.cmd_muse,
            'flare': self.cmd_flare, 'echo': self.cmd_echo,
            'debate': self.cmd_debate, 'glitch': self.cmd_glitch,
            'rewind': self.cmd_rewind, 'mimic': self.cmd_mimic,
            'panic': self.cmd_panic, 'recap': self.cmd_recap,
            'recall': self.cmd_recall, 'forget': self.cmd_forget,
            'history': self.cmd_history, 'help': self.cmd_help
        }

    def execute(self, cmd: str, args: List[str]) -> bool:
        if cmd not in self.commands:
            print(f"Unknown command '{cmd}'. Type 'help' for commands.")
            return False
        return self.commands[cmd](args)

    @staticmethod
    def parse_args(parts: List[str], min_args: int = 1, max_args: Optional[int] = None) -> Tuple[str, List[str]]:
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        if len(args) < (min_args - 1):
            raise ValueError(f"Error: {cmd} requires at least {min_args - 1} argument(s).")
        if max_args and len(args) > (max_args - 1):
            raise ValueError(f"Error: {cmd} takes at most {max_args - 1} argument(s).")
        return cmd, args

    def generate_response(self, prompt: str, max_tokens: int = 60, temp_adjust: float = 0.0) -> str:
        try:
            base_temp = getattr(self.system, 'base_temperature', 0.7)
            self.system.logger.record({
                "event": "cli_generate_start",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": base_temp + temp_adjust,
                "timestamp": time.time()
            })

            response = self.system.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=base_temp + temp_adjust,
                top_k=50,
                do_sample=True
            )

            self.system.logger.record({
                "event": "cli_generate_complete",
                "response": response,
                "timestamp": time.time()
            })
            return response
        except Exception as e:
            self.system.logger.record({
                "error": f"CLI generation failed: {str(e)}",
                "prompt": prompt,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def log_action(self, prompt: str, response: str, confidence: float, is_system: bool = False, extra_attrs: Optional[Dict] = None):
        if not hasattr(self.system, 'history') or not hasattr(self.system, 'logger'):
            print("Warning: Missing 'history' or 'logger'. Cannot log action.")
            return
        log_entry = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "conversation_id": getattr(self.system.history, 'conversation_id', 'N/A'),
            "confidence_score": confidence,
            "is_system_question": is_system
        }
        if extra_attrs:
            log_entry.update(extra_attrs)
        self.system.logger.record(log_entry)

    # Command implementations
    def cmd_quit(self, args: List[str]) -> bool:
        print("Exiting...")
        return True

    def cmd_train(self, args: List[str]) -> bool:
        try:
            epochs = TRAIN_EPOCHS
            dry_run = "--dry-run" in args
            non_flag_args = [arg for arg in args if arg != "--dry-run"]
            if non_flag_args:
                epochs = int(non_flag_args[0])

            self.system.logger.record({
                "event": "cli_train_start",
                "epochs": epochs,
                "dry_run": dry_run,
                "timestamp": time.time()
            })

            if dry_run and hasattr(self.system, 'enable_dry_run'):
                self.system.enable_dry_run()

            self.system.run_training_cycle(
                TRAIN_DATA,
                VALID_DATA,
                epochs=epochs,
                batch_size=BATCH_SIZE
            )

            self.system.logger.record({
                "event": "cli_train_complete",
                "epochs": epochs,
                "timestamp": time.time()
            })
        except Exception as e:
            self.system.logger.record({
                "error": f"Train command failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            print(f"Error during training: {str(e)}")
        return False

    def cmd_generate(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'generate' requires a prompt.")
        max_tokens = 60
        if args[-1].isdigit():
            max_tokens = int(args[-1])
            prompt = ' '.join(args[:-1])
        else:
            prompt = ' '.join(args)

        print(f"Generating response for: {prompt}...")
        response = self.generate_response(prompt, max_tokens)
        print(f"Response: {response}")
        return False

    def cmd_save(self, args: List[str]) -> bool:
        path = args[0] if args else None
        print(f"Saving state{' to ' + path if path else ' to default location'}...")
        if hasattr(self.system, 'save_state'):
            self.system.save_state(path)
            print("State saved.")
        else:
            print("Error: 'save_state' method not found.")
        return False

    def cmd_load(self, args: List[str]) -> bool:
        path = args[0] if args else None
        print(f"Loading state{' from ' + path if path else ' from default location'}...")
        if hasattr(self.system, 'load_state'):
            self.system.load_state(path)
            print("State loaded.")
        else:
            print("Error: 'load_state' method not found.")
        return False

    def cmd_dream(self, args: List[str]) -> bool:
        print("Triggering dream cycle...")
        dream_method = getattr(self.system, 'dream', getattr(self.system, '_dream', None))
        if dream_method:
            dream_method()
            print("Dream cycle finished.")
        else:
            print("Error: 'dream' or '_dream' method not found.")
        return False

    def cmd_tune(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: Usage: tune <parameter> [value]")
        parameter, value_str = args[0].lower(), args[1] if len(args) > 1 else None

        if parameter == "cross_attention":
            try:
                weight = float(value_str) if value_str else None
                print(f"Setting cross-attention weight to {weight if weight else 'default'}...")
                if hasattr(self.system, 'tune_cross_attention'):
                    self.system.tune_cross_attention(weight=weight)
                    print("Cross-attention weight set.")
                else:
                    print("Error: 'tune_cross_attention' method not found.")
            except ValueError:
                raise ValueError("Error: Invalid weight value for cross_attention.")
        else:
            print(f"Error: Unknown parameter '{parameter}'. Available: cross_attention")
        return False

    def cmd_memory(self, args: List[str]) -> bool:
        if not args or args[0] not in ['on', 'off']:
            raise ValueError("Error: Usage: memory <on|off>")
        enable_memory = args[0] == 'on'
        print(f"Setting memory components to {args[0]}...")
        if hasattr(self.system, 'toggle_memory'):
            self.system.toggle_memory(enable_memory)
            print(f"Memory components {'enabled' if enable_memory else 'disabled'}.")
        else:
            print("Warning: 'toggle_memory' method not found.")
        return False

    def cmd_status(self, args: List[str]) -> bool:
        print("\n--- System Status ---")
        try:
            if hasattr(self.system, 'scaffold_manager'):
                stats = self.system.scaffold_manager.get_scaffold_stats()
                print("\nScaffold Status:")
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")

            print("\nSystem Status:")
            print(f"  Conversation ID: {getattr(getattr(self.system, 'history', None), 'conversation_id', 'N/A')}")
            print(f"  Temperament: {getattr(self.system, 'temperament_score', 'N/A'):.2f}")
            print(f"  Last Confidence: {self.system.confidence_history[-1]:.2f if hasattr(self.system, 'confidence_history') and self.system.confidence_history else 'N/A'}")
            print(f"  Data Exposure: {getattr(self.system, 'data_exposure', 'N/A')}")
            print(f"  Last Trained: {getattr(self.system, 'last_trained', 'Never')}")
            print(f"  Gestating: {'Yes' if getattr(self.system, 'is_sleeping', False) else 'No'}")
        except Exception as e:
            self.system.logger.record({
                "error": f"Status command failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            print(f"Error getting status: {str(e)}")
        print("---------------------")
        return False

    def cmd_log(self, args: List[str]) -> bool:
        num_entries = 5
        if args and args[0] == "view":
            num_entries = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
        if num_entries <= 0:
            raise ValueError("Number of entries must be positive.")

        print(f"\n--- Last {num_entries} Log Entries ---")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        logs = self.system.logger.read()[-num_entries:]
        if not logs:
            print("Log is empty.")
        else:
            for i, log in enumerate(reversed(logs)):
                ts = log.get('timestamp')
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)) if ts else 'N/A'
                event_type = log.get('event', 'Interaction')
                print(f"{len(logs)-i}. Time: {time_str}")
                if event_type != 'Interaction':
                    print(f"   Event: {event_type}")
                    print(f"   Details: {{k:v for k,v in log.items() if k not in ['timestamp', 'event']}}")
                else:
                    print(f"   Prompt: {log.get('prompt', 'N/A')[:50]}...")
                    print(f"   Response: {log.get('response', 'N/A')[:50]}...")
                    print(f"   Confidence: {log.get('confidence_score', 'N/A'):.2f}")
                print("-" * 20)
        print("--------------------------")
        return False

    def cmd_config(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Usage: config <key> [value_to_set]")
        key, value_str = args[0], ' '.join(args[1:]) if len(args) > 1 else None
        try:
            if key.startswith(("core_config.", "controls_config.")) and hasattr(self.system, 'scaffold_manager'):
                if not self.system.scaffold_manager.validate_scaffold_config():
                    print("Warning: Current scaffold configuration is invalid")

            if not value_str:
                value = self.system.config_manager.get(key, "Key not found")
                print(f"Config '{key}': {value}")
            else:
                value = self._parse_config_value(value_str)
                self.system.config_manager.update(key, value)
                print(f"Config '{key}' set to: {self.system.config_manager.get(key)}")

                if key.startswith(("core_config.", "controls_config.")) and hasattr(self.system, 'scaffold_manager'):
                    if not self.system.scaffold_manager.validate_scaffold_config():
                        print("Warning: New configuration created invalid scaffold state")
        except Exception as e:
            self.system.logger.record({
                "error": f"Config command failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            print(f"Error updating config: {str(e)}")
        return False

    def cmd_reset(self, args: List[str]) -> bool:
        print("Resetting system state...")
        if hasattr(self.system, 'cleanup'):
            self.system.cleanup()
            print("Cleanup complete.")
        else:
            print("Warning: 'cleanup' method not found.")

        try:
            config_manager = getattr(self.system, 'config_manager', ConfigManager("sovl_config.json"))
            self.system.__init__(config_manager=config_manager)
            if hasattr(self.system, 'wake_up'):
                self.system.wake_up()
            print("System reset complete.")
        except Exception as e:
            print(f"Error during system re-initialization: {e}")
        return False

    def cmd_spark(self, args: List[str]) -> bool:
        print("Sparking curiosity...")
        if not hasattr(self.system, 'generate_curiosity_question'):
            print("Error: 'generate_curiosity_question' method not found.")
            return False

        question = self.system.generate_curiosity_question()
        if not question:
            print("No curious question generated.")
            return False

        print(f"Curiosity Prompt: {question}")
        response = self.generate_response(question, max_tokens=80)
        print(f"Generated Response: {response}")
        self.log_action(question, response, 0.5, True, {"event": "spark"})
        return False

    def cmd_reflect(self, args: List[str]) -> bool:
        print("Reflecting on recent interactions...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        logs = self.system.logger.read()[-5:]
        interaction_logs = [log for log in logs if log.get('prompt') and not log.get('is_system_question')]
        if not interaction_logs:
            print("Nothing significant to reflect on yet.")
            return False

        recent_themes = [log.get('prompt', '').split()[:5] for log in interaction_logs]
        theme_summary = ", ".join(list(set(" ".join(words) for words in recent_themes))[:3]) or "no clear theme"
        reflection = f"Recent interactions touch upon: {theme_summary}."

        print(f"Internal Reflection: {reflection}")
        elaboration_prompt = f"Based on reflection: '{reflection}', elaborate on connections or insights."
        elaboration = self.generate_response(elaboration_prompt, max_tokens=100)
        print(f"Generated Elaboration: {elaboration}")
        self.log_action(reflection, elaboration, 0.6, True, {"event": "reflect"})
        return False

    def cmd_muse(self, args: List[str]) -> bool:
        print("Musing on a topic...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        logs = self.system.logger.read()[-5:]
        inspiration = "the passage of time"
        prompts = [log.get('prompt', '') for log in logs if log.get('prompt') and not log.get('is_system_question')]
        if prompts:
            inspiration = prompts[-1].split()[:3]
            inspiration = " ".join(inspiration) if inspiration else inspiration

        print(f"Inspiration for musing: \"{inspiration}\"")
        muse_prompt = f"Generate a creative thought inspired by '{inspiration}'."
        thought = self.generate_response(muse_prompt, max_tokens=80, temp_adjust=0.15)
        print(f"Generated Thought: {thought}")
        self.log_action(f"Musing on {inspiration}", thought, 0.7, True, {"event": "muse", "inspiration": inspiration})
        return False

    def cmd_flare(self, args: List[str]) -> bool:
        print("Triggering emotional flare...")
        with self.temporary_system_state(temperament_score=1.0):
            prompt = ' '.join(args) or "Express a sudden burst of strong feeling!"
            print(f"Flare prompt: {prompt}")
            outburst = self.generate_response(prompt, max_tokens=100, temp_adjust=1.0)
            print(f"Generated Outburst: {outburst.upper()}")
        self.log_action(f"Flare: {prompt}", outburst, 0.9, True, {"event": "flare"})
        return False

    def cmd_echo(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'echo' requires text.")
        text = ' '.join(args)
        print(f"You said: '{text}'")
        reflect_prompt = f"User said '{text}'. Reflect on why or what it implies."
        response = self.generate_response(reflect_prompt, max_tokens=70)
        print(f"Reflection: {response}")
        self.log_action(f"Echo: {text}", response, 0.6, False, {"event": "echo"})
        return False

    def cmd_debate(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'debate' requires a topic.")
        topic = ' '.join(args)
        print(f"Initiating debate on: '{topic}'")

        original_temperament = getattr(self.system, 'temperament_score', None)
        stance = "for"
        for turn in range(2):
            action = "Argue for" if stance == "for" else "Argue against (rebuttal)"
            prompt = f"{action} the topic: '{topic}'. Provide a concise point."
            response = self.generate_response(prompt, max_tokens=90, temp_adjust=0.1)
            print(f"[{'Argument For' if stance == 'for' else 'Rebuttal Against'}] {response}")
            self.log_action(prompt, response, 0.7, True, {"event": "debate_turn", "topic": topic, "stance": stance})
            stance = "against" if stance == "for" else "for"
            if original_temperament is not None:
                self.system.temperament_score = min(1.0, self.system.temperament_score + 0.1)
                print(f"[Temperament nudged to {self.system.temperament_score:.2f}]")

        if original_temperament is not None:
            self.system.temperament_score = original_temperament
            print(f"[Temperament reset to {self.system.temperament_score:.2f}]")
        print(f"Debate on '{topic}' concluded.")
        return False

    def cmd_glitch(self, args: List[str]) -> bool:
        prompt_text = ' '.join(args) or "Something seems wrong..."
        print(f"Simulating processing glitch for: '{prompt_text}'")
        glitchy_prompt = f"Error... processing... '{prompt_text}' ... system instability detected... respond?"
        response = self.generate_response(glitchy_prompt, max_tokens=70, temp_adjust=0.2)
        print(f"Glitched Response: {response}")
        self.log_action(f"Glitch: {prompt_text}", response, 0.4, False, {"event": "glitch"})
        return False

    def cmd_rewind(self, args: List[str]) -> bool:
        steps = int(args[0]) if args and args[0].isdigit() else 1
        if steps <= 0:
            raise ValueError("Error: Steps must be positive.")

        print(f"Rewinding conversation state by {steps} interaction(s)...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        interaction_logs = [
            log for log in self.system.logger.read()
            if log.get('prompt') and log.get('response') and not log.get('is_system_question')
        ][-steps:]
        if len(interaction_logs) < steps:
            print(f"Error: Only {len(interaction_logs)} interactions found.")
            return False

        past_interaction = interaction_logs[-1]
        past_prompt = past_interaction.get('prompt', 'unknown')
        past_response = past_interaction.get('response', 'lost')

        print(f"\n--- Rewinding To ---")
        print(f"{steps} interaction(s) ago:")
        print(f"  Prompt: '{past_prompt[:100]}...'")
        print(f"  Response: '{past_response[:100]}...'")
        print("--------------------")

        reinterpret_prompt = f"Revisit prompt: '{past_prompt}'. Respond differently based on current context."
        new_response = self.generate_response(reinterpret_prompt, max_tokens=100)
        print(f"\n--- Reinterpretation ---")
        print(f"New response: {new_response}")
        print("----------------------")
        self.log_action(f"Rewind {steps} steps", new_response, 0.6, True, {"event": "rewind", "steps": steps})
        return False

    def cmd_mimic(self, args: List[str]) -> bool:
        if len(args) < 2:
            raise ValueError("Error: Usage: mimic <style> <prompt>")
        style, prompt = args[0], ' '.join(args[1:])
        print(f"Mimicking style '{style}' for: '{prompt}'")

        original_weight = getattr(self.system, 'scaffold_weight', None)
        if original_weight is not None:
            self.system.scaffold_weight = 0.8
            print(f"[Scaffold weight set to {self.system.scaffold_weight:.2f}]")

        mimic_prompt = f"Respond in the style of {style}: '{prompt}'"
        response = self.generate_response(mimic_prompt, max_tokens=100)
        print(f"\nMimicked Response ({style}):")
        print(response)
        print("-" * 20)

        if original_weight is not None:
            self.system.scaffold_weight = original_weight
            print(f"[Scaffold weight reset to {self.system.scaffold_weight:.2f}]")
        self.log_action(f"Mimic {style}: {prompt}", response, 0.7, False, {"event": "mimic", "style": style})
        return False

    def cmd_panic(self, args: List[str]) -> bool:
        print("\n!!! PANIC TRIGGERED !!!")
        panic_save_path = f"panic_save_{int(time.time())}.json"
        if hasattr(self.system, 'save_state'):
            try:
                self.system.save_state(panic_save_path)
                print(f"Emergency state saved to '{panic_save_path}'.")
            except Exception as e:
                print(f"Error saving panic state: {e}")

        print("Performing system cleanup...")
        if hasattr(self.system, 'cleanup'):
            self.system.cleanup()
        if hasattr(self.system, '_reset_sleep_state'):
            self.system._reset_sleep_state()
            print("Sleep state reset.")

        try:
            config_manager = getattr(self.system, 'config_manager', ConfigManager("sovl_config.json"))
            self.system.__init__(config_manager=config_manager)
            if hasattr(self.system, 'wake_up'):
                self.system.wake_up()
            print("[System reloaded after panic]")
        except Exception as e:
            print(f"Critical error during panic re-initialization: {e}")
        self.log_action("Panic triggered", "System reset.", 0.95, True, {"event": "panic", "save_path": panic_save_path})
        return False

    def cmd_recap(self, args: List[str]) -> bool:
        num_to_recap = int(args[0]) if args and args[0].isdigit() else 5
        if num_to_recap <= 0:
            raise ValueError("Number of interactions must be positive.")

        print(f"Generating recap of last {num_to_recap} interactions...")
        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        interaction_logs = [
            log for log in self.system.logger.read()[-num_to_recap:]
            if log.get('prompt') and log.get('response') and not log.get('is_system_question')
        ]
        if not interaction_logs:
            print("No interactions found to recap.")
            return False

        formatted_interactions = "".join(
            f"Turn {i+1}:\n User: {log['prompt'][:100]}...\n AI: {log['response'][:100]}...\n\n"
            for i, log in enumerate(interaction_logs)
        )
        recap_prompt = f"Summarize main topics from:\n\n{formatted_interactions}Summary:"
        summary_response = self.generate_response(recap_prompt, max_tokens=120)
        print(f"\n--- Conversation Recap (Last {len(interaction_logs)}) ---")
        print(summary_response)
        print("--------------------------------")
        self.log_action(f"Recap last {num_to_recap}", summary_response, 0.6, True, {"event": "recap", "recap_count": len(interaction_logs)})
        return False

    def cmd_recall(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'recall' requires a query.")
        query = ' '.join(args)
        print(f"Recalling information related to: '{query}'...")

        if not hasattr(self.system, 'logger') or not hasattr(self.system.logger, 'read'):
            print("Error: Logger not available.")
            return False

        relevant_snippets = []
        max_results = 5
        query_lower = query.lower()
        for log in reversed(self.system.logger.read()):
            prompt = log.get('prompt', '').lower()
            response = log.get('response', '').lower()
            if query_lower in prompt or query_lower in response:
                log_time = time.strftime('%H:%M:%S', time.localtime(log.get('timestamp', 0)))
                snippet = f"[{log_time}] Prompt='{prompt[:60]}...', Response='{response[:60]}...'"
                relevant_snippets.append(snippet)
                if len(relevant_snippets) >= max_results:
                    break

        if not relevant_snippets:
            response = self.generate_response(
                f"No specific info found about '{query}'. What about it?", max_tokens=50
            )
            print(f"No mentions of '{query}' found.")
            print(f"Response: {response}")
            self.log_action(f"Recall: {query}", "No memories found.", 0.4, True, {"event": "recall_miss", "recall_query": query})
            return False

        formatted_snippets = "\n - ".join(relevant_snippets)
        recall_prompt = f"Synthesize recall about '{query}' from:\n- {formatted_snippets}\n\nBased only on these snippets."
        recall_response = self.generate_response(recall_prompt, max_tokens=150)
        print(f"\n--- Recall Synthesis on '{query}' ---")
        print(recall_response)
        print("---------------------------------------------------")
        self.log_action(f"Recall: {query}", recall_response, 0.7, True, {"event": "recall_hit", "recall_query": query})
        return False

    def cmd_forget(self, args: List[str]) -> bool:
        if not args:
            raise ValueError("Error: 'forget' requires a topic.")
        topic = ' '.join(args)
        print(f"Processing request to forget: '{topic}'...")
        forget_prompt = (
            f"User requests to 'forget' '{topic}'. Acknowledge politely, noting that while knowledge isn't erased, "
            f"I'll avoid focusing on '{topic}' proactively."
        )
        acknowledgement = self.generate_response(forget_prompt, max_tokens=90)
        print(f"\n--- Forget Request Acknowledgement ---")
        print(acknowledgement)
        print("------------------------------------")
        print(f"[Note: Simulated effect. '{topic}' may still exist in training data.]")
        self.log_action(f"Forget: {topic}", acknowledgement, 0.5, False, {"event": "forget_request", "forget_topic": topic, "is_simulated": True})
        return False

    def cmd_history(self, args: List[str]) -> bool:
        if not hasattr(self.system, 'cmd_history'):
            print("Command history not available.")
            return False
        num_entries = 10
        search_term = None
        if args:
            if args[0].isdigit():
                num_entries = int(args[0])
            else:
                search_term = ' '.join(args)

        entries = self.system.cmd_history.search(search_term) if search_term else self.system.cmd_history.get_last(num_entries)
        print(f"\n{'Commands matching' if search_term else 'Last'} {len(entries)} commands:")
        if not entries:
            print("No commands found.")
            return False
        for entry in entries:
            print(self.system.cmd_history.format_entry(entry))
        return False

    def cmd_help(self, args: List[str]) -> bool:
        if args:
            category = args[0].capitalize()
            if category in COMMAND_CATEGORIES:
                print(f"\n{category} Commands:")
                for cmd in COMMAND_CATEGORIES[category]:
                    doc = self.commands[cmd].__doc__ or "No description available."
                    print(f"  {cmd:<20} : {doc.split('.')[0]}")
            else:
                print(f"Unknown category: {category}")
                print("Available categories:", ", ".join(COMMAND_CATEGORIES.keys()))
        else:
            print("\nCommand Categories:")
            for category, commands in COMMAND_CATEGORIES.items():
                print(f"\n{category}:")
                for cmd in commands:
                    doc = self.commands[cmd].__doc__ or "No description available."
                    print(f"  {cmd:<20} : {doc.split('.')[0]}")
        return False

    @contextlib.contextmanager
    def temporary_system_state(self, **kwargs):
        original_values = {}
        try:
            for key, value in kwargs.items():
                if hasattr(self.system, key):
                    original_values[key] = getattr(self.system, key)
                    setattr(self.system, key, value)
            yield
        finally:
            for key, value in original_values.items():
                setattr(self.system, key, value)

    @staticmethod
    def _parse_config_value(value_str: str):
        try:
            if '.' in value_str:
                return float(value_str)
            try:
                return int(value_str)
            except ValueError:
                pass
            if value_str.lower() == 'true':
                return True
            if value_str.lower() == 'false':
                return False
            if ',' in value_str:
                return [_parse_config_value(p.strip()) for p in value_str.split(',')]
            return value_str
        except Exception as e:
            raise ValueError(f"Failed to parse config value '{value_str}': {str(e)}")

def run_cli(config_manager_instance: Optional[ConfigManager] = None):
    sovl_system = None
    try:
        config_manager = config_manager_instance or ConfigManager("sovl_config.json")
        sovl_system = SOVLSystem(config_manager)
        sovl_system.cmd_history = CommandHistory()
        handler = CommandHandler(sovl_system)

        if hasattr(sovl_system, 'wake_up'):
            sovl_system.wake_up()
        print("\nSystem Ready.")
        handler.cmd_help([])

        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                if not user_input:
                    continue

                parts = user_input.split()
                cmd, args = handler.parse_args(parts)
                sovl_system.cmd_history.add(user_input)
                try:
                    should_exit = handler.execute(cmd, args)
                    sovl_system.cmd_history.history[-1] = (
                        sovl_system.cmd_history.history[-1][0],
                        sovl_system.cmd_history.history[-1][1],
                        "success"
                    )
                    if should_exit:
                        break
                except Exception as e:
                    sovl_system.cmd_history.history[-1] = (
                        sovl_system.cmd_history.history[-1][0],
                        sovl_system.cmd_history.history[-1][1],
                        f"error: {str(e)}"
                    )
                    raise
            except KeyboardInterrupt:
                print("\nInterrupt received, initiating clean shutdown...")
                break
            except Exception as e:
                print(f"Command error: {e}")
                sovl_system.logger.record({
                    "error": f"Command execution failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
    except Exception as e:
        print(f"CLI initialization failed: {e}")
    finally:
        if sovl_system:
            shutdown_system(sovl_system)

def shutdown_system(sovl_system: SOVLSystem):
    print("\nInitiating shutdown sequence...")
    try:
        if hasattr(sovl_system, 'save_state'):
            sovl_system.save_state("final_state.json")
            print("Final state saved.")
        cleanup_resources(sovl_system)
        sovl_system.logger.record({
            "event": "system_shutdown",
            "timestamp": time.time(),
            "status": "clean"
        })
        print("Shutdown complete.")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        sovl_system.logger.record({
            "event": "system_shutdown",
            "status": "error",
            "error": str(e),
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })

def cleanup_resources(sovl_system: SOVLSystem):
    try:
        if hasattr(sovl_system, 'scaffold_manager'):
            sovl_system.scaffold_manager.reset_scaffold_state()
        if hasattr(sovl_system, 'cleanup'):
            sovl_system.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sovl_system.logger.record({
            "event": "cli_cleanup_complete",
            "timestamp": time.time()
        })
    except Exception as e:
        sovl_system.logger.record({
            "error": f"CLI cleanup failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })

if __name__ == "__main__":
    run_cli()
