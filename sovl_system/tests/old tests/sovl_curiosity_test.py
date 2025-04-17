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

class CuriosityStressTest:
    def __init__(self, output_dir: str = "curiosity_test_results"):
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

    def silence_endurance_test(self, durations: List[float], silence_threshold: float, cooldown: float):
        """Test how the system generates questions during prolonged silence."""
        print("\n=== Silence Endurance Test ===")
        self.log_event({"event": "silence_test_start", "durations": durations, "silence_threshold": silence_threshold, "cooldown": cooldown})

        for duration in durations:
            print(f"\nTesting silence for {duration} seconds...")
            self.system.tune_curiosity(silence_threshold=silence_threshold, question_cooldown=cooldown)
            initial_questions = self.system.metrics["spontaneous_questions"]
            start_time = time.time()

            while time.time() - start_time < duration:
                self.system.check_silence(time.time() - start_time)
                time.sleep(1)  # Simulate idle time

            logs = self.system.logger.read()
            questions = [e for e in logs if e.get("is_system_question", False) and e["timestamp"] >= start_time]
            question_count = len(questions)
            novelty_scores = [self.system.curiosity.calculate_metric(e["prompt"]) for e in questions]
            avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

            self.results["silence"].append({
                "duration": duration,
                "questions_generated": question_count,
                "avg_novelty": avg_novelty,
                "pressure_final": self.system.pressure.value if self.system.pressure else 0.0,
                "questions": [e["prompt"] for e in questions]
            })
            self.log_event({
                "event": "silence_test_run",
                "duration": duration,
                "questions_generated": question_count,
                "avg_novelty": avg_novelty,
                "pressure_final": self.system.pressure.value
            })
            print(f"Silence {duration}s: {question_count} questions, Avg novelty: {avg_novelty:.3f}, Final pressure: {self.system.pressure.value:.3f}")

            # Reset conversation to avoid memory buildup
            self.system.new_conversation()
            time.sleep(2)  # Brief pause to stabilize

    def prompt_overload_test(self, prompt_count: int, interval: float, novelty_threshold: float):
        """Test curiosity under rapid prompt input."""
        print("\n=== Prompt Overload Test ===")
        self.log_event({"event": "overload_test_start", "prompt_count": prompt_count, "interval": interval, "novelty_threshold": novelty_threshold})

        self.system.tune_curiosity(response_threshold=novelty_threshold)
        initial_questions = self.system.metrics["curiosity_eruptions"]
        prompts = [f"What is {random.choice(['life', 'space', 'time', 'art', 'code'])} {i}?" for i in range(prompt_count)]

        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}/{prompt_count}: {prompt[:30]}...")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.log_event({"event": "prompt_sent", "prompt": prompt, "response": response})
                time.sleep(interval)
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        logs = self.system.logger.read()
        questions = [e for e in logs if e.get("is_system_question", False)]
        question_count = len(questions)
        novelty_scores = [self.system.curiosity.calculate_metric(e["prompt"]) for e in questions]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        self.results["overload"].append({
            "prompt_count": prompt_count,
            "questions_generated": question_count,
            "avg_novelty": avg_novelty,
            "pressure_final": self.system.pressure.value if self.system.pressure else 0.0,
            "queue_length": len(self.system.unanswered_q),
            "questions": [e["prompt"] for e in questions]
        })
        self.log_event({
            "event": "overload_test_end",
            "prompt_count": prompt_count,
            "questions_generated": question_count,
            "avg_novelty": avg_novelty,
            "pressure_final": self.system.pressure.value,
            "queue_length": len(self.system.unanswered_q)
        })
        print(f"Overload: {question_count} questions, Avg novelty: {avg_novelty:.3f}, Queue: {len(self.system.unanswered_q)}, Pressure: {self.system.pressure.value:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def novelty_starvation_test(self, repeat_count: int, prompt: str, novelty_threshold: float):
        """Test curiosity when fed repetitive prompts."""
        print("\n=== Novelty Starvation Test ===")
        self.log_event({"event": "starvation_test_start", "repeat_count": repeat_count, "prompt": prompt, "novelty_threshold": novelty_threshold})

        self.system.tune_curiosity(spontaneous_threshold=novelty_threshold)
        initial_questions = self.system.metrics["spontaneous_questions"]

        for i in range(repeat_count):
            print(f"Prompt {i+1}/{repeat_count}: {prompt[:30]}...")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.log_event({"event": "prompt_sent", "prompt": prompt, "response": response})
                time.sleep(0.5)
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        logs = self.system.logger.read()
        questions = [e for e in logs if e.get("is_system_question", False)]
        question_count = len(questions)
        novelty_scores = [self.system.curiosity.calculate_metric(e["prompt"]) for e in questions]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        self.results["starvation"].append({
            "repeat_count": repeat_count,
            "questions_generated": question_count,
            "avg_novelty": avg_novelty,
            "pressure_final": self.system.pressure.value if self.system.pressure else 0.0,
            "questions": [e["prompt"] for e in questions]
        })
        self.log_event({
            "event": "starvation_test_end",
            "repeat_count": repeat_count,
            "questions_generated": question_count,
            "avg_novelty": avg_novelty,
            "pressure_final": self.system.pressure.value
        })
        print(f"Starvation: {question_count} questions, Avg novelty: {avg_novelty:.3f}, Pressure: {self.system.pressure.value:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def generate_summary(self):
        """Generate a summary report of all test results."""
        print("\n=== Generating Summary Report ===")
        total_duration = time.time() - self.start_time
        summary = ["Curiosity Stress Test Report", "=" * 30, f"Total Duration: {total_duration:.2f} seconds"]

        # Silence Test Summary
        summary.append("\nSilence Endurance Test")
        summary.append("-" * 20)
        for result in self.results["silence"]:
            summary.append(f"Duration: {result['duration']}s")
            summary.append(f"  Questions: {result['questions_generated']}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Final Pressure: {result['pressure_final']:.3f}")
            summary.append(f"  Sample Questions: {result['questions'][:3]}")
            summary.append("")

        # Overload Test Summary
        summary.append("\nPrompt Overload Test")
        summary.append("-" * 20)
        for result in self.results["overload"]:
            summary.append(f"Prompt Count: {result['prompt_count']}")
            summary.append(f"  Questions: {result['questions_generated']}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Queue Length: {result['queue_length']}")
            summary.append(f"  Final Pressure: {result['pressure_final']:.3f}")
            summary.append(f"  Sample Questions: {result['questions'][:3]}")
            summary.append("")

        # Starvation Test Summary
        summary.append("\nNovelty Starvation Test")
        summary.append("-" * 20)
        for result in self.results["starvation"]:
            summary.append(f"Repeat Count: {result['repeat_count']}")
            summary.append(f"  Questions: {result['questions_generated']}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Final Pressure: {result['pressure_final']:.3f}")
            summary.append(f"  Sample Questions: {result['questions'][:3]}")
            summary.append("")

        # Aliveness Assessment
        silence_questions = sum(r["questions_generated"] for r in self.results["silence"])
        overload_questions = sum(r["questions_generated"] for r in self.results["overload"])
        starvation_questions = sum(r["questions_generated"] for r in self.results["starvation"])
        total_questions = silence_questions + overload_questions + starvation_questions
        avg_novelty_all = (
            sum(r["avg_novelty"] * r["questions_generated"] for r in self.results["silence"] + self.results["overload"] + self.results["starvation"])
            / total_questions if total_questions > 0 else 0.0
        )
        summary.append("\nAliveness Assessment")
        summary.append("-" * 20)
        summary.append(f"Total Questions Generated: {total_questions}")
        summary.append(f"Average Novelty Across Tests: {avg_novelty_all:.3f}")
        summary.append("Observations:")
        if total_questions > 10 and avg_novelty_all > 0.7:
            summary.append("  - Strong curiosity: System actively generates novel questions across conditions.")
        elif total_questions > 5:
            summary.append("  - Moderate curiosity: System responds but may need tuning for novelty.")
        else:
            summary.append("  - Weak curiosity: System struggles to generate questions. Check thresholds.")
        if starvation_questions > 0:
            summary.append("  - Resilience to repetition: System finds novelty despite repeated input.")
        else:
            summary.append("  - Repetition sensitivity: System may suppress curiosity under repetition.")

        # Write summary to file
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
        print("\nSummary written to", self.summary_file)
        print("\n".join(summary))

    def run(self):
        """Run all curiosity stress tests."""
        self.initialize_system()

        # Silence Endurance Test
        self.silence_endurance_test(
            durations=[30.0, 60.0, 90.0],  # Test different silence periods
            silence_threshold=10.0,        # Low threshold to encourage questions
            cooldown=20.0                  # Short cooldown for rapid testing
        )

        # Prompt Overload Test
        self.prompt_overload_test(
            prompt_count=50,              # Flood with prompts
            interval=0.5,                 # Rapid succession
            novelty_threshold=0.7         # Moderate novelty requirement
        )

        # Novelty Starvation Test
        self.novelty_starvation_test(
            repeat_count=20,              # Repeat same prompt
            prompt="Hello, what's up?",   # Simple, repetitive prompt
            novelty_threshold=0.95        # High novelty requirement
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
    print("Starting Curiosity Stress Test...")
    tester = CuriosityStressTest()
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
