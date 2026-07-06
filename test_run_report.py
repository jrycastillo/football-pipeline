#!/usr/bin/env python3
import sys
import types
import os
import json
import tempfile
import unittest
import subprocess

# Rule constraint: Stub cv2 before importing project modules to support local environment validation
sys.modules["cv2"] = types.ModuleType("cv2")

from tools.run_report import build_markdown_report

class TestRunReport(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.run_dir = self.test_dir.name

    def tearDown(self):
        self.test_dir.cleanup()

    def test_empty_dir(self):
        """
        Verify that generating a report in an empty directory degrades gracefully and never crashes.
        """
        report = build_markdown_report(self.run_dir, title="Empty Run Test")
        
        self.assertIn("# Empty Run Test", report)
        self.assertIn("**Run Directory:**", report)
        
        # Verify graceful degradation in each section
        self.assertIn("## Accuracy vs Ground Truth\nNot available", report)
        self.assertIn("## Back-test Verdict\nNot available", report)
        self.assertIn("## Event Summary\nNot available", report)
        self.assertIn("## Top Players by Observations\nNot available", report)
        self.assertIn("## Clips Summary\nNot available", report)
        self.assertIn("## Diagnostics\nNot available", report)
        
        # Ensure no emojis exist in the generated report
        self.assertNotIn("🔴", report)
        self.assertNotIn("🟢", report)
        self.assertNotIn("⚠️", report)

    def test_minimal_dir(self):
        """
        Verify report generation with only player_stats.json present.
        """
        player_stats_data = {
            "10": {
                "player_name": "Lionel Messi",
                "jersey_number": 10,
                "team": "Inter Miami",
                "observations": 1500,
                "verification_status": "verified",
                "stats": {
                    "passes_total": 45,
                    "shots_on_target_total": 3,
                    "tackles_total": 1,
                    "total_distance": 9200.5
                }
            }
        }
        with open(os.path.join(self.run_dir, "player_stats.json"), "w") as f:
            json.dump(player_stats_data, f)
            
        report = build_markdown_report(self.run_dir, title="Minimal Run Test")
        
        self.assertIn("# Minimal Run Test", report)
        self.assertIn("Lionel Messi", report)
        self.assertIn("Inter Miami", report)
        self.assertIn("1500", report)
        self.assertIn("verified", report)
        self.assertIn("9200.50", report)
        
        # Other sections must still degrade gracefully
        self.assertIn("## Accuracy vs Ground Truth\nNot available", report)
        self.assertIn("## Back-test Verdict\nNot available", report)
        self.assertIn("## Event Summary\nNot available", report)
        self.assertIn("## Clips Summary\nNot available", report)
        self.assertIn("## Diagnostics\nNot available", report)

    def test_full_dir(self):
        """
        Verify report generation when all files are present in the run directory.
        """
        # 1. accuracy_report.json
        acc_data = [
            {"metric": "passes", "pipeline": 80.0, "ground_truth": 82.0, "accuracy_pct": 97.6},
            {"metric": "shots", "pipeline": 5.0, "ground_truth": 4.0, "accuracy_pct": 80.0}
        ]
        with open(os.path.join(self.run_dir, "accuracy_report.json"), "w") as f:
            json.dump(acc_data, f)
            
        # 2. backtest_verdict.md
        with open(os.path.join(self.run_dir, "backtest_verdict.md"), "w") as f:
            f.write("# Back-test Verdict: PASS\nThis run has met the required thresholds.")
            
        # 3. verification_summary.json
        verification_data = {
            "verified_events": 10,
            "unverified_events": 90,
            "confidence_bands": {
                "high_0.75+": 10,
                "mid_0.50-0.75": 70,
                "low_<0.50": 20
            },
            "by_event_type": {
                "pass": {
                    "count": 80,
                    "avg_confidence": 0.6854,
                    "avg_identity_confidence": 0.725
                },
                "shot": {
                    "count": 20,
                    "avg_confidence": 0.8123,
                    "avg_identity_confidence": None
                }
            }
        }
        with open(os.path.join(self.run_dir, "verification_summary.json"), "w") as f:
            json.dump(verification_data, f)
            
        # 4. player_stats.json
        player_stats_data = {}
        for i in range(1, 13):
            player_stats_data[str(i)] = {
                "player_name": f"Player {i}",
                "jersey_number": i,
                "team": "RedTeam" if i % 2 == 0 else "BlueTeam",
                "observations": i * 100, # Sorted by observations descending
                "verification_status": "unverified",
                "stats": {
                    "passes_total": i * 2,
                    "shots_on_target_total": i // 3,
                    "tackles_total": i // 2,
                    "total_distance": i * 100.0
                }
            }
        with open(os.path.join(self.run_dir, "player_stats.json"), "w") as f:
            json.dump(player_stats_data, f)
            
        # 5. clips_manifest.json
        clips_data = [
            {"clip": "clips/001_pass.mp4", "type": "pass", "confidence": 0.82, "size_bytes": 1024 * 1024},
            {"clip": "clips/002_shot.mp4", "type": "shot", "confidence": 0.68, "size_bytes": 2 * 1024 * 1024}
        ]
        with open(os.path.join(self.run_dir, "clips_manifest.json"), "w") as f:
            json.dump(clips_data, f)
            
        # 6. pipeline.log
        log_lines = [
            "2026-07-05 12:00:00 - Starting analysis pipeline",
            "2026-07-05 12:00:05 - [PitchHomography] fit stats: error=0.045",
            "2026-07-05 12:00:10 - [FoulDebug] fouls detected: 3",
            "2026-07-05 12:00:15 - Fragment dump: 12 tracks active",
            "2026-07-05 12:00:20 - [StatsEngine] Ball track smoothed"
        ]
        with open(os.path.join(self.run_dir, "pipeline.log"), "w") as f:
            f.write("\n".join(log_lines) + "\n")
            
        # Build report
        report = build_markdown_report(self.run_dir, title="Full Run Test")
        
        # Check Header
        self.assertIn("# Full Run Test", report)
        
        # Check Accuracy vs Ground Truth
        self.assertIn("passes | 80.0 | 82.0 | 97.6%", report)
        self.assertIn("shots | 5.0 | 4.0 | 80.0%", report)
        
        # Check Back-test Verdict
        self.assertIn("# Back-test Verdict: PASS", report)
        self.assertIn("[backtest_verdict.md](backtest_verdict.md)", report)
        
        # Check Event Summary
        self.assertIn("- **Total Events:** 100", report)
        self.assertIn("- **Verified Events:** 10", report)
        self.assertIn("High (>= 0.75): 10", report)
        self.assertIn("pass | 80 | 0.6854 | 0.7250 |", report)
        self.assertIn("shot | 20 | 0.8123 | N/A |", report)
        
        # Check Players (Top 10 observations descending - should include Player 12 down to Player 3)
        self.assertIn("Player 12", report)
        self.assertIn("Player 3", report)
        self.assertNotIn("Player 2", report) # Only top 10 players
        self.assertNotIn("Player 1 ", report)
        
        # Check Clips
        self.assertIn("- **Total Clips:** 2", report)
        self.assertIn("- **Total Size:** 3.00 MB", report)
        self.assertIn("- **Admin Queue Size (Conf < 0.75):** 1", report)
        self.assertIn("- pass: 1", report)
        self.assertIn("- shot: 1", report)
        
        # Check Diagnostics
        self.assertIn("[PitchHomography] fit stats: error=0.045", report)
        self.assertIn("[FoulDebug] fouls detected: 3", report)
        self.assertIn("Fragment dump: 12 tracks active", report)
        self.assertIn("[StatsEngine] Ball track smoothed", report)
        
    def test_cli_execution(self):
        """
        Verify CLI invocation via subprocess.
        """
        out_report = os.path.join(self.run_dir, "custom_report.md")
        cmd = [
            sys.executable,
            "tools/run_report.py",
            "--run_dir", self.run_dir,
            "--output", out_report,
            "--title", "CLI Test Report"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertTrue(os.path.exists(out_report))
        
        with open(out_report, "r") as f:
            content = f.read()
            self.assertIn("# CLI Test Report", content)

if __name__ == "__main__":
    unittest.main()
