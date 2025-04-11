import time
import json
import random
import torch
from typing import List, Dict, Optional
from collections import defaultdict
import sys
from pathlib import Path

"""
This script implements a test suite for the SOVLSystem (Self-Organizing Virtual Lifeform) to evaluate its lifecycle progression,
behavioral evolution, and end-of-life dynamics. The primary goal is to simulate various scenarios and measure how the system
adapts, evolves, and performs under different conditions.

Key Components:
1. **LifecycleEvolutionTest Class**:
   - Manages test initialization, logging, and result aggregation.
   - Defines several test cases to evaluate the system's lifecycle and behavioral adaptability.

2. **Tests Implemented**:
   - **Exposure Ramp-Up Test**:
     Simulates intensive usage to observe the system's lifecycle progression and behavioral changes.
   - **Curve Variation Test**:
     Tests the impact of different lifecycle curves (e.g., sigmoid, exponential) on system behavior and temperament.
   - **End-of-Life Test**:
     Pushes the system towards the end of its lifecycle to evaluate its resilience, novelty, and behavioral stability.

3. **Logging and Reporting**:
   - Logs critical events (e.g., system initialization, test results) to JSONL and text files for analysis.
   - Generates a summarized report highlighting key observations and performance metrics.

4. **Resilience and Behavior Metrics**:
   - Tracks lifecycle weight, data exposure, temperament, and novelty to assess the system's adaptability and evolution.

5. **System Cleanup**:
   - Ensures cleanup of system resources and GPU memory after tests.

Usage:
- Run the script directly to start the test suite.
- Results and logs are stored in the `lifecycle_test_results` directory for further analysis.
"""

# Assuming SOVLSystem is in the same directory or importable
from sovl_system import SOVLSystem  # Adjust import based on your file structure

class LifecycleEvolutionTest:
    def __init__(self, output_dir: str = "lifecycle_test_results"):
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

    def exposure_ramp_up_test(self, prompt_count: int, capacity_factor: float):
        """Simulate heavy usage to advance lifecycle and track evolution."""
        print("\n=== Exposure Ramp-Up Test ===")
        self.log_event({"event": "exposure_test_start", "prompt_count": prompt_count, "capacity_factor": capacity_factor})

        self.system.adjust_lifecycle(capacity_factor=capacity_factor)
        initial_weight = self.system.get_life_curve_weight()
        prompts = [f"Explain {random.choice(['time', 'gravity', 'love', 'code'])} {i}" for i in range(prompt_count)]

        for i, prompt in enumerate(prompts):
            if i % 100 == 0:  # Progress update every 100 prompts
                print(f"Prompt {i}/{prompt_count}: {prompt[:30]}... (Exposure: {self.system.data_exposure:.0f}, Weight: {self.system.get_life_curve_weight():.3f})")
            try:
                response = self.system.generate(prompt, max_new_tokens=60)
                self.log_event({
                    "event": "prompt_sent",
                    "prompt": prompt,
                    "response": response,
                    "data_exposure": self.system.data_exposure,
                    "lifecycle_weight": self.system.get_life_curve_weight(),
                    "temperament_score": self.system.temperament_score
                })
                time.sleep(0.1)  # Small delay to simulate real usage
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

        logs = self.system.logger.read()
        temperament_history = [e["temperament_score"] for e in logs if "temperament_score" in e][-10:]  # Last 10 for trend
        responses = [e["response"] for e in logs if "response" in e][-5:]  # Last 5 for qualitative check

        self.results["exposure"].append({
            "prompt_count": prompt_count,
            "initial_weight": initial_weight,
            "final_weight": self.system.get_life_curve_weight(),
            "data_exposure": self.system.data_exposure,
            "lora_capacity": self.system.lora_capacity,
            "avg_temperament": sum(temperament_history) / len(temperament_history) if temperament_history else 0.0,
            "sample_responses": responses
        })
        self.log_event({
            "event": "exposure_test_end",
            "prompt_count": prompt_count,
            "final_weight": self.system.get_life_curve_weight(),
            "data_exposure": self.system.data_exposure,
            "avg_temperament": self.results["exposure"][-1]["avg_temperament"]
        })
        print(f"Exposure: Final Weight: {self.system.get_life_curve_weight():.3f}, Exposure: {self.system.data_exposure:.0f}/{self.system.lora_capacity:.0f}, Avg Temperament: {self.results['exposure'][-1]['avg_temperament']:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def curve_variation_test(self, prompt_count: int, curves: List[str]):
        """Test different lifecycle curves and their impact on behavior."""
        print("\n=== Curve Variation Test ===")
        self.log_event({"event": "curve_test_start", "prompt_count": prompt_count, "curves": curves})

        for curve in curves:
            print(f"\nTesting curve: {curve}")
            self.system.adjust_lifecycle(lifecycle_curve=curve)
            initial_weight = self.system.get_life_curve_weight()
            prompts = [f"What is {random.choice(['life', 'space', 'art'])} {i}?" for i in range(prompt_count)]

            for i, prompt in enumerate(prompts):
                if i % (prompt_count // 5) == 0:  # Progress update at intervals
                    print(f"Prompt {i}/{prompt_count}: {prompt[:30]}... (Weight: {self.system.get_life_curve_weight():.3f})")
                try:
                    response = self.system.generate(prompt, max_new_tokens=60)
                    self.log_event({
                        "event": "prompt_sent",
                        "prompt": prompt,
                        "response": response,
                        "curve": curve,
                        "lifecycle_weight": self.system.get_life_curve_weight(),
                        "temperament_score": self.system.temperament_score
                    })
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error during prompt {i+1}: {e}")
                    self.log_event({"event": "prompt_error", "prompt": prompt, "error": str(e)})

            logs = self.system.logger.read()
            temperament_history = [e["temperament_score"] for e in logs if "temperament_score" in e and e.get("curve") == curve][-5:]  # Last 5
            responses = [e["response"] for e in logs if "response" in e and e.get("curve") == curve][-3:]  # Last 3

            self.results["curve"].append({
                "curve": curve,
                "prompt_count": prompt_count,
                "initial_weight": initial_weight,
                "final_weight": self.system.get_life_curve_weight(),
                "data_exposure": self.system.data_exposure,
                "avg_temperament": sum(temperament_history) / len(temperament_history) if temperament_history else 0.0,
                "sample_responses": responses
            })
            self.log_event({
                "event": "curve_test_run",
                "curve": curve,
                "final_weight": self.system.get_life_curve_weight(),
                "data_exposure": self.system.data_exposure,
                "avg_temperament": self.results["curve"][-1]["avg_temperament"]
            })
            print(f"Curve {curve}: Final Weight: {self.system.get_life_curve_weight():.3f}, Avg Temperament: {self.results['curve'][-1]['avg_temperament']:.3f}")

            # Reset system state
            self.system.data_exposure = 0  # Manually reset exposure
            self.system.new_conversation()

    def end_of_life_test(self, exposure_factor: float):
        """Push system to lifecycle end to test end-of-life behavior."""
        print("\n=== End-of-Life Test ===")
        self.log_event({"event": "eol_test_start", "exposure_factor": exposure_factor})

        target_exposure = self.system.lora_capacity * exposure_factor
        prompts = [f"Tell me about {random.choice(['stars', 'dreams', 'code'])} {i}" for i in range(1000)]  # Large enough to overshoot if needed
        i = 0

        while self.system.data_exposure < target_exposure and i < len(prompts):
            if i % 100 == 0:
                print(f"Prompt {i}: Exposure {self.system.data_exposure:.0f}/{target_exposure:.0f}, Weight: {self.system.get_life_curve_weight():.3f}")
            try:
                response = self.system.generate(prompts[i], max_new_tokens=60)
                self.log_event({
                    "event": "prompt_sent",
                    "prompt": prompts[i],
                    "response": response,
                    "data_exposure": self.system.data_exposure,
                    "lifecycle_weight": self.system.get_life_curve_weight(),
                    "temperament_score": self.system.temperament_score
                })
                time.sleep(0.1)
            except Exception as e:
                print(f"Error during prompt {i+1}: {e}")
                self.log_event({"event": "prompt_error", "prompt": prompts[i], "error": str(e)})
            i += 1

        logs = self.system.logger.read()
        temperament_history = [e["temperament_score"] for e in logs if "temperament_score" in e][-10:]  # Last 10
        responses = [e["response"] for e in logs if "response" in e][-5:]  # Last 5
        novelty_scores = [self.system.curiosity.calculate_metric(r) for r in responses] if responses else [0.0]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        self.results["eol"].append({
            "target_exposure": target_exposure,
            "final_exposure": self.system.data_exposure,
            "final_weight": self.system.get_life_curve_weight(),
            "avg_temperament": sum(temperament_history) / len(temperament_history) if temperament_history else 0.0,
            "avg_novelty": avg_novelty,
            "sample_responses": responses
        })
        self.log_event({
            "event": "eol_test_end",
            "final_exposure": self.system.data_exposure,
            "final_weight": self.system.get_life_curve_weight(),
            "avg_temperament": self.results["eol"][-1]["avg_temperament"],
            "avg_novelty": avg_novelty
        })
        print(f"End-of-Life: Final Weight: {self.system.get_life_curve_weight():.3f}, Exposure: {self.system.data_exposure:.0f}, Avg Temperament: {self.results['eol'][-1]['avg_temperament']:.3f}, Avg Novelty: {avg_novelty:.3f}")

        # Reset conversation
        self.system.new_conversation()

    def generate_summary(self):
        """Generate a summary report of all test results."""
        print("\n=== Generating Summary Report ===")
        total_duration = time.time() - self.start_time
        summary = ["Lifecycle and Evolution Test Report", "=" * 35, f"Total Duration: {total_duration:.2f} seconds"]

        # Exposure Ramp-Up Summary
        summary.append("\nExposure Ramp-Up Test")
        summary.append("-" * 20)
        for result in self.results["exposure"]:
            summary.append(f"Prompt Count: {result['prompt_count']}")
            summary.append(f"  Initial Weight: {result['initial_weight']:.3f}")
            summary.append(f"  Final Weight: {result['final_weight']:.3f}")
            summary.append(f"  Data Exposure: {result['data_exposure']:.0f}/{result['lora_capacity']:.0f}")
            summary.append(f"  Avg Temperament: {result['avg_temperament']:.3f}")
            summary.append(f"  Sample Responses: {result['sample_responses'][:2]}")
            summary.append("")

        # Curve Variation Summary
        summary.append("\nCurve Variation Test")
        summary.append("-" * 20)
        for result in self.results["curve"]:
            summary.append(f"Curve: {result['curve']}")
            summary.append(f"  Prompt Count: {result['prompt_count']}")
            summary.append(f"  Initial Weight: {result['initial_weight']:.3f}")
            summary.append(f"  Final Weight: {result['final_weight']:.3f}")
            summary.append(f"  Data Exposure: {result['data_exposure']:.0f}")
            summary.append(f"  Avg Temperament: {result['avg_temperament']:.3f}")
            summary.append(f"  Sample Responses: {result['sample_responses'][:2]}")
            summary.append("")

        # End-of-Life Summary
        summary.append("\nEnd-of-Life Test")
        summary.append("-" * 20)
        for result in self.results["eol"]:
            summary.append(f"Target Exposure: {result['target_exposure']:.0f}")
            summary.append(f"  Final Exposure: {result['final_exposure']:.0f}")
            summary.append(f"  Final Weight: {result['final_weight']:.3f}")
            summary.append(f"  Avg Temperament: {result['avg_temperament']:.3f}")
            summary.append(f"  Avg Novelty: {result['avg_novelty']:.3f}")
            summary.append(f"  Sample Responses: {result['sample_responses'][:2]}")
            summary.append("")

        # Aliveness Assessment
        exposure_weight_growth = any(r["final_weight"] - r["initial_weight"] > 0.5 for r in self.results["exposure"])
        curve_temperament_shift = len([r for r in self.results["curve"] if abs(r["avg_temperament"]) > 0.3]) > 1
        eol_stability = all(r["avg_novelty"] > 0.5 for r in self.results["eol"])
        summary.append("\nAliveness Assessment")
        summary.append("-" * 20)
        summary.append("Observations:")
        if exposure_weight_growth:
            summary.append("  - Lifecycle progression: Weight increases significantly with exposure.")
        else:
            summary.append("  - Limited progression: Weight growth is slow. Check capacity_factor.")
        if curve_temperament_shift:
            summary.append("  - Behavioral evolution: Different curves alter temperament noticeably.")
        else:
            summary.append("  - Static behavior: Curves have minimal impact on temperament.")
        if eol_stability:
            summary.append("  - End-of-life resilience: System maintains novelty at lifecycle end.")
        else:
            summary.append("  - End-of-life decay: Novelty drops, suggesting over-saturation.")

        # Write summary to file
        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
        print("\nSummary written to", self.summary_file)
        print("\n".join(summary))

    def run(self):
        """Run all lifecycle and evolution tests."""
        self.initialize_system()

        # Exposure Ramp-Up Test
        self.exposure_ramp_up_test(
            prompt_count=1000,      # Simulate heavy usage
            capacity_factor=0.001   # Fast aging for testing
        )

        # Curve Variation Test
        self.curve_variation_test(
            prompt_count=50,        # Moderate usage per curve
            curves=["sigmoid_linear", "exponential"]  # Test different curves
        )

        # End-of-Life Test
        self.end_of_life_test(
            exposure_factor=1.5     # Push 50% beyond capacity
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
    print("Starting Lifecycle and Evolution Test...")
    tester = LifecycleEvolutionTest()
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
