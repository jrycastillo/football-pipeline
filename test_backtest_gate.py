#!/usr/bin/env python3
import sys
import types
import os
import json
import tempfile
import unittest
import subprocess

# Rule constraint: Stub cv2 before importing project modules to support local machines
sys.modules["cv2"] = types.ModuleType("cv2")

from tools.backtest_gate import run_gate

class TestBacktestGate(unittest.TestCase):

    def setUp(self):
        # Create temp dir for output and temp JSON files
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name
        
        # 1. Create a dummy baselines file
        self.baselines_path = os.path.join(self.output_dir, "test_baselines.json")
        self.baselines = {
            "count_accuracy": {
                "passes": {"low": 40.0, "high": 160.0, "optional": False},
                "shots": {"low": 50.0, "high": 150.0, "optional": False},
                "goals": {"low": 0.0, "high": 200.0, "optional": True}
            },
            "event_f1": {
                "pass": {"min": 0.5, "optional": False},
                "shot": {"min": 0.5, "optional": False},
                "goal": {"min": 0.0, "optional": True}
            }
        }
        with open(self.baselines_path, "w") as f:
            json.dump(self.baselines, f)

        # 2. Create a passing accuracy report
        self.passing_acc_path = os.path.join(self.output_dir, "passing_acc.json")
        self.passing_acc = [
            {"metric": "passes", "pipeline": 100, "ground_truth": 100, "accuracy_pct": 100.0},
            {"metric": "shots", "pipeline": 100, "ground_truth": 100, "accuracy_pct": 100.0},
            {"metric": "goals", "pipeline": 300, "ground_truth": 100, "accuracy_pct": 300.0} # optional fails band, but optional!
        ]
        with open(self.passing_acc_path, "w") as f:
            json.dump(self.passing_acc, f)

        # 3. Create a failing accuracy report
        self.failing_acc_path = os.path.join(self.output_dir, "failing_acc.json")
        self.failing_acc = [
            {"metric": "passes", "pipeline": 30, "ground_truth": 100, "accuracy_pct": 30.0}, # Out of band (low=40)
            {"metric": "shots", "pipeline": 100, "ground_truth": 100, "accuracy_pct": 100.0}
        ]
        with open(self.failing_acc_path, "w") as f:
            json.dump(self.failing_acc, f)

        # 4. Create a passing eval events report
        self.passing_eval_path = os.path.join(self.output_dir, "passing_eval.json")
        self.passing_eval = {
            "metrics": {
                "pass": {"f1": 0.8},
                "shot": {"f1": 0.75},
                "goal": {"f1": 0.4}
            }
        }
        with open(self.passing_eval_path, "w") as f:
            json.dump(self.passing_eval, f)

        # 5. Create a failing eval events report
        self.failing_eval_path = os.path.join(self.output_dir, "failing_eval.json")
        self.failing_eval = {
            "metrics": {
                "pass": {"f1": 0.4}, # Below min=0.5 floor
                "shot": {"f1": 0.75}
            }
        }
        with open(self.failing_eval_path, "w") as f:
            json.dump(self.failing_eval, f)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_pass_path(self):
        """
        Verify that correct counts and F1 scores result in a PASS verdict.
        """
        exit_code = run_gate(
            accuracy_report_path=self.passing_acc_path,
            eval_events_path=self.passing_eval_path,
            baselines_path=self.baselines_path,
            output_dir=self.output_dir
        )
        self.assertEqual(exit_code, 0)
        
        # Check verdict markdown
        verdict_file = os.path.join(self.output_dir, "backtest_verdict.md")
        self.assertTrue(os.path.exists(verdict_file))
        with open(verdict_file, "r") as f:
            content = f.read()
            self.assertIn("# Back-test Verdict: PASS", content)
            self.assertIn("passes | Count Accuracy | 100.0% | 40.0% - 160.0% | Yes | **PASS**", content)

    def test_fail_count_out_of_band(self):
        """
        Verify that a count accuracy metric outside the baseline band results in a FAIL verdict.
        """
        exit_code = run_gate(
            accuracy_report_path=self.failing_acc_path,
            eval_events_path=self.passing_eval_path,
            baselines_path=self.baselines_path,
            output_dir=self.output_dir
        )
        self.assertEqual(exit_code, 1)

        # Check verdict markdown
        verdict_file = os.path.join(self.output_dir, "backtest_verdict.md")
        self.assertTrue(os.path.exists(verdict_file))
        with open(verdict_file, "r") as f:
            content = f.read()
            self.assertIn("# Back-test Verdict: FAIL", content)
            self.assertIn("passes | Count Accuracy | 30.0% | 40.0% - 160.0% | Yes | **FAIL**", content)

    def test_fail_f1_below_floor(self):
        """
        Verify that an event F1 score below the baseline floor results in a FAIL verdict.
        """
        exit_code = run_gate(
            accuracy_report_path=self.passing_acc_path,
            eval_events_path=self.failing_eval_path,
            baselines_path=self.baselines_path,
            output_dir=self.output_dir
        )
        self.assertEqual(exit_code, 1)

        # Check verdict markdown
        verdict_file = os.path.join(self.output_dir, "backtest_verdict.md")
        self.assertTrue(os.path.exists(verdict_file))
        with open(verdict_file, "r") as f:
            content = f.read()
            self.assertIn("# Back-test Verdict: FAIL", content)
            self.assertIn("pass | Event F1 Score | 0.4000 | >= 0.5000 | Yes | **FAIL**", content)

    def test_missing_event_input(self):
        """
        Verify that running without eval_events gates counts only and includes a clear skipped notice.
        """
        exit_code = run_gate(
            accuracy_report_path=self.passing_acc_path,
            eval_events_path=None,
            baselines_path=self.baselines_path,
            output_dir=self.output_dir
        )
        self.assertEqual(exit_code, 0)

        # Check verdict markdown
        verdict_file = os.path.join(self.output_dir, "backtest_verdict.md")
        self.assertTrue(os.path.exists(verdict_file))
        with open(verdict_file, "r") as f:
            content = f.read()
            self.assertIn("# Back-test Verdict: PASS", content)
            self.assertIn("*Event-level F1 evaluation was skipped (no --eval_events input provided).*", content)

    def test_cli_execution_and_exit_codes(self):
        """
        Verify CLI invocation and exit code reporting via subprocess.
        """
        # Test CLI PASS
        cmd_pass = [
            sys.executable,
            "tools/backtest_gate.py",
            "--accuracy_report", self.passing_acc_path,
            "--eval_events", self.passing_eval_path,
            "--baselines", self.baselines_path,
            "--output_dir", self.output_dir
        ]
        res_pass = subprocess.run(cmd_pass, capture_output=True, text=True)
        self.assertEqual(res_pass.returncode, 0)
        self.assertIn("Verdict: PASS", res_pass.stdout)

        # Test CLI FAIL
        cmd_fail = [
            sys.executable,
            "tools/backtest_gate.py",
            "--accuracy_report", self.failing_acc_path,
            "--eval_events", self.passing_eval_path,
            "--baselines", self.baselines_path,
            "--output_dir", self.output_dir
        ]
        res_fail = subprocess.run(cmd_fail, capture_output=True, text=True)
        self.assertEqual(res_fail.returncode, 1)
        self.assertIn("Verdict: FAIL", res_fail.stdout)

if __name__ == "__main__":
    unittest.main()
