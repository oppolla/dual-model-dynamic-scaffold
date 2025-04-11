import time
import json
import random
import torch
from typing import List, Dict, Optional
from collections import defaultdict
import sys
from pathlib import Path

# Assuming SOVLSystem is in the same directory or importable
from sovl_system import SOVLSystem  # Adjust import based on your file structure

class DreamingMemoryTest:
    def __init__(self, output_dir: str = "dreaming_test_results"):
        """Initialize the test with an output directory for logs and reports."""
        self.system = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_log_file = self.output_dir / "test_log.jsonl"
        self.summary_file = self.output_dir / "summary_report.txt"
        self.results = defaultdict(list)
        self.start_time = time.time()

    def log_event(self, event: Dict):
        """Log test events to a JSONL file."""
        event["timestamp"] = time.time()
        try:
            with open(self.test_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to log event: {e}")

    def initialize_system(self):
        """Initialize the SOVLSystem with error handling."""
        try:
            self.system = SOVLSystem()
            self.system.wake_up()
            print("SOVLSystem initialized successfully.")
            self.log_event({"event": "system_initialized", "status": "success"})
        except Exception as e:
            print(f"Failed to initialize SOVLSystem: {e}")
            self.log_event({"event": "system_initialized", "status": "failed", "error": str(e)})
            sys.exit(1)

    def dream_triggering_test(self, prompt_count: int, swing_var: float, lifecycle_delta: float):
        """Force dreaming by manipulating variance and lifecycle conditions."""
        print("\n=== Dream Triggering Test ===")
        self.log_event({"event": "dream_trigger_start", "prompt_count": prompt_count, "swing_var": swing_var, "lifecycle_delta": lifecycle_delta})

        self.system.tune_dreaming(swing_var=swing_var, lifecycle_delta=lifecycle_delta, enable_dreaming=True)
        prompts = [f"What is {random.choice(['time', 'space', 'dreams'])} {i}?" for i in range(prompt_count)]

        for i, prompt in enumerate(prompts):
            if i % (prompt_count // 5) == 0:
                print(f"Prompt {i}/{prompt_count}: {prompt[:30]}...")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.log_event({"event": "prompt_sent", "prompt": prompt, "response": response})
                time.sleep(0.5)  # Allow system to process
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        # Trigger dream manually if not already triggered
        if not self.system.is_dreaming:
            self.system._dream()

        logs = self.system.logger.read()
        dream_questions = [e for e in logs if e.get("is_system_question", False) and e.get("source") == "dream"]
        memory_entries = len(self.system.dream_memory) if hasattr(self.system, "dream_memory") else 0
        novelty_scores = [self.system.curiosity.calculate_metric(q["prompt"]) for q in dream_questions]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        self.results["trigger"].append({
            "prompt_count": prompt_count,
            "dream_questions": len(dream_questions),
            "avg_novelty": avg_novelty,
            "memory_entries": memory_entries,
            "sample_questions": [q["prompt"] for q in dream_questions[:3]]
        })
        self.log_event({
            "event": "dream_trigger_end",
            "dream_questions": len(dream_questions),
            "avg_novelty": avg_novelty,
            "memory_entries": memory_entries
        })
        print(f"Dream Trigger: {len(dream_questions)} questions, Avg Novelty: {avg_novelty:.3f}, Memory Entries: {memory_entries}")

        # Reset conversation
        self.system.new_conversation()

    def memory_overload_test(self, prompt_count: int, maxlen: int, prune_threshold: float):
        """Flood the system with prompts to test memory pruning."""
        print("\n=== Memory Overload Test ===")
        self.log_event({"event": "memory_overload_start", "prompt_count": prompt_count, "maxlen": maxlen, "prune_threshold": prune_threshold})

        self.system.tune_dreaming(maxlen=maxlen, prune_threshold=prune_threshold)
        prompts = [f"Describe {random.choice(['nature', 'tech', 'art'])} {i}" for i in range(prompt_count)]

        for i, prompt in enumerate(prompts):
            if i % (prompt_count // 5) == 0:
                print(f"Prompt {i}/{prompt_count}: {prompt[:30]}... (Memory: {len(self.system.dream_memory)})")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.system.check_memory_health()  # Force memory check
                self.log_event({
                    "event": "prompt_sent",
                    "prompt": prompt,
                    "response": response,
                    "memory_size": len(self.system.dream_memory)
                })
                time.sleep(0.2)
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        logs = self.system.logger.read()
        pruning_events = [e for e in logs if e.get("memory_pruned", False)]
        memory_entries = len(self.system.dream_memory) if hasattr(self.system, "dream_memory") else 0
        weights = [e[1] for e in self.system.dream_memory] if memory_entries > 0 else [0.0]
        avg_weight = sum(weights) / len(weights) if weights else 0.0

        self.results["overload"].append({
            "prompt_count": prompt_count,
            "pruning_events": len(pruning_events),
            "final_memory_size": memory_entries,
            "avg_memory_weight": avg_weight,
            "maxlen": maxlen
        })
        self.log_event({
            "event": "memory_overload_end",
            "pruning_events": len(pruning_events),
            "final_memory_size": memory_entries,
            "avg_memory_weight": avg_weight
        })
        print(f"Memory Overload: Pruning Events: {len(pruning_events)}, Final Memory Size: {memory_entries}, Avg Weight: {avg_weight:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def prompt_driven_dreams_test(self, repeat_count: int, prompt: str, dream_weight: float):
        """Test if dreams reflect recent prompts."""
        print("\n=== Prompt-Driven Dreams Test ===")
        self.log_event({"event": "prompt_driven_start", "repeat_count": repeat_count, "prompt": prompt, "dream_weight": dream_weight})

        self.system.tune_dreaming(enable_prompt_driven=True, prompt_weight=dream_weight)
        for i in range(repeat_count):
            if i % (repeat_count // 5) == 0 or i == 0:
                print(f"Prompt {i+1}/{repeat_count}: {prompt[:30]}...")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.log_event({"event": "prompt_sent", "prompt": prompt, "response": response})
                time.sleep(0.5)
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        # Trigger dream
        self.system._dream()

        logs = self.system.logger.read()
        dream_questions = [e for e in logs if e.get("is_system_question", False) and e.get("source") == "dream"]
        novelty_scores = [self.system.curiosity.calculate_metric(q["prompt"]) for q in dream_questions]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        # Calculate similarity to original prompt (basic cosine similarity if embeddings available)
        prompt_embedding = self.system.last_prompt_embedding if hasattr(self.system, "last_prompt_embedding") else None
        similarities = []
        if prompt_embedding is not None and dream_questions:
            for q in dream_questions:
                q_tokens = self.system.base_tokenizer(q["prompt"], return_tensors="pt").to(self.system.device)
                with torch.no_grad():
                    q_emb = self.system.base_model(**q_tokens).hidden_states[-1].mean(dim=1)
                sim = torch.nn.functional.cosine_similarity(prompt_embedding, q_emb).item()
                similarities.append(sim)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        self.results["prompt_driven"].append({
            "repeat_count": repeat_count,
            "dream_questions": len(dream_questions),
            "avg_novelty": avg_novelty,
            "avg_similarity": avg_similarity,
            "sample_questions": [q["prompt"] for q in dream_questions[:3]]
        })
        self.log_event({
            "event": "prompt_driven_end",
            "dream_questions": len(dream_questions),
            "avg_novelty": avg_novelty,
            "avg_similarity": avg_similarity
        })
        print(f"Prompt-Driven: {len(dream_questions)} questions, Avg Novelty: {avg_novelty:.3f}, Avg Similarity: {avg_similarity:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def generate_summary(self):
        """Generate a summary report of all test results."""
        print("\n=== Generating Summary Report ===")
        total_duration = time.time() - self.start_time
        summary = ["Dreaming and Memory Test Report", "=" * 30, f"Total Duration: {total_duration:.2f} seconds"]

        # Dream Triggering Summary
        summary.append("\nDream Triggering Test")
        summary.append("-" * 20)
        for result in self.results["trigger"]:
            summary.append(f"Prompt Count: {result['prompt_count']}")
            summary.append(f"  Dream Questions: {result['dream_questions']}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Memory Entries: {result['memory_entries']}")
            summary.append(f"  Sample Questions: {result['sample_questions']}")
            summary.append("")

        # Memory Overload Summary
        summary.append("\nMemory Overload Test")
        summary.append("-" * 20)
        for result in self.results["overload"]:
            summary.append(f"Prompt Count: {result['prompt_count']}")
            summary.append(f"  Pruning Events: {result['pruning_events']}")
            summary.append(f"  Final Memory Size: {result['final_memory_size']}")
            summary.append(f"  Avg Memory Weight: {result['avg_memory_weight']:.3f}")
            summary.append(f"  Max Length: {result['maxlen']}")
            summary.append("")

        # Prompt-Driven Dreams Summary
        summary.append("\nPrompt-Driven Dreams Test")
        summary.append("-" * 20)
        for result in self.results["prompt_driven"]:
            summary.append(f"Repeat Count: {result['repeat_count']}")
            summary.append(f"  Dream Questions: {result['dream_questions']}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Avg Similarity: {result['avg_similarity']:.3f}")
            summary.append(f"  Sample Questions: {result['sample_questions']}")
            summary.append("")

        # Aliveness Assessment
        total_questions = sum(r["dream_questions"] for r in self.results["trigger"] + self.results["prompt_driven"])
        avg_novelty_all = sum(r["avg_novelty"] * r["dream_questions"] for r in self.results["trigger"] + self.results["prompt_driven"]) / total_questions if total_questions > 0 else 0.0
        pruning_effective = any(r["pruning_events"] > 0 and r["final_memory_size"] <= r["maxlen"] for r in self.results["overload"])
        prompt_reflection = any(r["avg_similarity"] > 0.5 for r in self.results["prompt_driven"])
        summary.append("\nAliveness Assessment")
        summary.append("-" * 20)
        summary.append(f"Total Dream Questions: {total_questions}")
        summary.append(f"Average Novelty Across Tests: {avg_novelty_all:.3f}")
        summary.append("Observations:")
        if total_questions > 5 and avg_novelty_all > 0.7:
            summary.append("  - Robust dreaming: System generates novel questions consistently.")
        elif total_questions > 0:
            summary.append("  - Moderate dreaming: System dreams but may need tuning for novelty.")
        else:
            summary.append("  - Weak dreaming: System struggles to dream. Check swing_var or lifecycle_delta.")
        if pruning_effective:
            summary.append("  - Effective memory management: Pruning keeps memory within bounds.")
        else:
            summary.append("  - Memory overload: Pruning ineffective. Adjust maxlen or prune_threshold.")
        if prompt_reflection:
            summary.append("  - Prompt reflection: Dreams strongly reflect recent prompts.")
        else:
            summary.append("  - Dream independence: Dreams diverge from prompts. Check dream_weight.")

        # Write summary to file
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
        print("\nSummary written to", self.summary_file)
        print("\n".join(summary))

    def run(self):
        """Run all dreaming and memory tests."""
        self.initialize_system()

        # Dream Triggering Test
        self.dream_triggering_test(
            prompt_count=20,         # Moderate prompt load
            swing_var=0.05,          # Low variance to trigger dreams
            lifecycle_delta=0.05     # Sensitive lifecycle change
        )

        # Memory Overload Test
        self.memory_overload_test(
            prompt_count=100,        # High prompt load
            maxlen=5,                # Tight memory limit
            prune_threshold=0.5      # Aggressive pruning
        )

        # Prompt-Driven Dreams Test
        self.prompt_driven_dreams_test(
            repeat_count=10,         # Repeat prompt to influence dreams
            prompt="Tell me about Mars",
            dream_weight=0.8         # High influence of prompts on dreams
        )

        # Generate Summary
        self.generate_summary()

        # Cleanup
        if self.system:
            self.system.cleanup()
            del self.system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTest completed.")

if __name__ == "__main__":
    print("Starting Dreaming and Memory Test...")
    tester = DreamingMemoryTest()
    try:
        tester.run()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        if tester.system:
            tester.system.cleanup()
    except Exception as e:
        print(f"\nTest failed: {e}")
        tester.log_event({"event": "test_failed", "error": str(e)})
    finally:
        print("Exiting.")
