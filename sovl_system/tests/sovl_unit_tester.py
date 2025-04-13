import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock, call
import time
from pathlib import Path

class TestCuriosityStressTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "test_outputs"
        self.output_dir.mkdir()
        
        # Mock SOVLSystem
        self.mock_system = MagicMock()
        self.mock_system.logger.read.return_value = []
        self.mock_system.metrics = {
            "spontaneous_questions": 0,
            "curiosity_eruptions": 0
        }
        self.mock_system.pressure = MagicMock()
        self.mock_system.pressure.value = 0.0
        self.mock_system.unanswered_q = []
        self.mock_system.curiosity = MagicMock()
        self.mock_system.curiosity.calculate_metric.return_value = 0.8

    def tearDown(self):
        # Clean up temporary files
        for f in self.output_dir.glob("*"):
            f.unlink()
        self.output_dir.rmdir()
        Path(self.test_dir).rmdir()

    @patch('sovl_system.SOVLSystem', return_value=MagicMock())
    def test_initialization(self, mock_sovl):
        """Test the test harness initializes correctly"""
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        self.assertEqual(tester.output_dir, self.output_dir)
        self.assertIsNone(tester.system)
        
        # Test directory should be created
        self.assertTrue(self.output_dir.exists())
        
        # Test log file should be created on first event
        self.assertFalse((self.output_dir / "test_log.jsonl").exists())
        
        # Test initialization
        tester.initialize_system()
        self.assertIsNotNone(tester.system)
        self.assertTrue((self.output_dir / "test_log.jsonl").exists())

    def test_silence_endurance_test(self):
        """Test silence endurance testing"""
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        tester.system = self.mock_system
        
        # Run test with short duration for testing
        durations = [5.0]
        tester.silence_endurance_test(durations, 10.0, 20.0)
        
        # Verify system methods were called
        self.mock_system.tune_curiosity.assert_called_once_with(
            silence_threshold=10.0,
            question_cooldown=20.0
        )
        self.mock_system.check_silence.assert_called()
        self.mock_system.new_conversation.assert_called_once()
        
        # Verify logging
        self.assertEqual(len(tester.results["silence"]), 1)
        self.assertTrue((self.output_dir / "test_log.jsonl").exists())

    def test_prompt_overload_test(self):
        """Test prompt overload scenario"""
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        tester.system = self.mock_system
        
        # Configure mock responses
        self.mock_system.generate.return_value = "Test response"
        
        # Run test with reduced count for testing
        tester.prompt_overload_test(prompt_count=3, interval=0.1, novelty_threshold=0.7)
        
        # Verify system methods were called
        self.mock_system.tune_curiosity.assert_called_once_with(
            response_threshold=0.7
        )
        self.assertEqual(self.mock_system.generate.call_count, 3)
        self.mock_system.new_conversation.assert_called_once()
        
        # Verify results
        self.assertEqual(len(tester.results["overload"]), 1)
        self.assertEqual(tester.results["overload"][0]["prompt_count"], 3)

    def test_novelty_starvation_test(self):
        """Test novelty starvation scenario"""
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        tester.system = self.mock_system
        
        # Configure mock responses
        self.mock_system.generate.return_value = "Test response"
        
        # Run test with reduced count for testing
        tester.novelty_starvation_test(repeat_count=3, prompt="Test prompt", novelty_threshold=0.95)
        
        # Verify system methods were called
        self.mock_system.tune_curiosity.assert_called_once_with(
            spontaneous_threshold=0.95
        )
        self.assertEqual(self.mock_system.generate.call_count, 3)
        self.mock_system.new_conversation.assert_called_once()
        
        # Verify results
        self.assertEqual(len(tester.results["starvation"]), 1)
        self.assertEqual(tester.results["starvation"][0]["repeat_count"], 3)

    def test_generate_summary(self):
        """Test report generation"""
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        
        # Populate with test data
        tester.results = {
            "silence": [{
                "duration": 30.0,
                "questions_generated": 2,
                "avg_novelty": 0.8,
                "pressure_final": 0.5,
                "questions": ["Q1", "Q2"]
            }],
            "overload": [{
                "prompt_count": 10,
                "questions_generated": 3,
                "avg_novelty": 0.7,
                "pressure_final": 0.6,
                "queue_length": 1,
                "questions": ["Q3", "Q4"]
            }],
            "starvation": [{
                "repeat_count": 5,
                "questions_generated": 1,
                "avg_novelty": 0.9,
                "pressure_final": 0.4,
                "questions": ["Q5"]
            }]
        }
        
        tester.generate_summary()
        
        # Verify report file was created
        self.assertTrue((self.output_dir / "summary_report.txt").exists())
        
        # Check report content
        with open(self.output_dir / "summary_report.txt", "r") as f:
            content = f.read()
            self.assertIn("Curiosity Stress Test Report", content)
            self.assertIn("Silence Endurance Test", content)
            self.assertIn("Prompt Overload Test", content)
            self.assertIn("Novelty Starvation Test", content)
            self.assertIn("Aliveness Assessment", content)

    @patch.object(CuriosityStressTest, 'silence_endurance_test')
    @patch.object(CuriosityStressTest, 'prompt_overload_test')
    @patch.object(CuriosityStressTest, 'novelty_starvation_test')
    @patch.object(CuriosityStressTest, 'generate_summary')
    def test_run_with_module_selection(self, mock_summary, mock_starvation, mock_overload, mock_silence):
        """Test running with selected modules"""
        # Test running only silence test
        tester = CuriosityStressTest(output_dir=str(self.output_dir))
        tester.run(modules=['silence'])
        mock_silence.assert_called_once()
        mock_overload.assert_not_called()
        mock_starvation.assert_not_called()
        mock_summary.assert_called_once()
        
        # Test running overload and starvation
        mock_silence.reset_mock()
        tester.run(modules=['overload', 'starvation'])
        mock_silence.assert_not_called()
        mock_overload.assert_called_once()
        mock_starvation.assert_called_once()
        mock_summary.assert_called()

class ModuleSelectionTestRunner:
    """Custom test runner that allows module selection"""
    
    def __init__(self, modules=None):
        self.modules = modules or ['all']
        
    def run(self, test_case):
        """Run tests for selected modules"""
        suite = unittest.TestSuite()
        
        # Map module names to test methods
        module_map = {
            'initialization': ['test_initialization'],
            'silence': ['test_silence_endurance_test'],
            'overload': ['test_prompt_overload_test'],
            'starvation': ['test_novelty_starvation_test'],
            'reporting': ['test_generate_summary'],
            'integration': ['test_run_with_module_selection']
        }
        
        # Add all tests if 'all' specified
        if 'all' in self.modules:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))
        else:
            # Add only selected module tests
            for module in self.modules:
                if module in module_map:
                    for test_name in module_map[module]:
                        suite.addTest(test_case(test_name))
        
        # Run the filtered suite
        runner = unittest.TextTestRunner()
        return runner.run(suite)

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run curiosity stress tests')
    parser.add_argument('--modules', nargs='+', default=['all'],
                        choices=['all', 'initialization', 'silence', 'overload', 
                                'starvation', 'reporting', 'integration'],
                        help='Test modules to run')
    
    args = parser.parse_args()
    
    # Run selected tests
    print(f"Running tests for modules: {args.modules}")
    test_runner = ModuleSelectionTestRunner(modules=args.modules)
    test_runner.run(TestCuriosityStressTest)
