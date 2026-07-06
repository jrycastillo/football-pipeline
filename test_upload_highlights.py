"""Tests for tools/upload_highlights.py field building and selection logic.
No network, no cv2 — pure logic. Run: python3 test_upload_highlights.py
"""
import sys
import types

sys.path.insert(0, ".")
from tools.upload_highlights import build_fields, select_entries  # noqa: E402


class Args:
    user_id = "u123"
    matches_video_id = None
    analysis_id = None
    types = None
    max_conf = None
    min_conf = None
    limit = None


# 1. Percent conversion + full fields
args = Args()
args.matches_video_id = "mv9"
args.analysis_id = "an7"
entry = {"type": "shot", "confidence": 0.78, "time_s": 51.45, "player": 7,
         "clip": "clips/000_shot_p7_f1029.mp4"}
f = build_fields(entry, args)
assert f["confidence_level"] == "78", f
assert f["event_type"] == "shot"
assert f["timestamp_sec"] == "51.45"
assert f["jersey_number"] == "7"
assert f["matches_video_id"] == "mv9" and f["analysis_id"] == "an7"
print("PASS field building + percent conversion")

# 2. Optional fields omitted, non-numeric player omitted
args2 = Args()
entry2 = {"type": "goal", "confidence": None, "player": "Unknown_12"}
f2 = build_fields(entry2, args2)
assert "confidence_level" not in f2
assert "jersey_number" not in f2
assert "matches_video_id" not in f2 and "analysis_id" not in f2
assert "timestamp_sec" not in f2
print("PASS optional/None fields omitted")

# 3. Selection: types + max_conf (admin-queue mode) + limit
manifest = [
    {"type": "shot", "confidence": 0.9},
    {"type": "shot", "confidence": 0.6},
    {"type": "goal", "confidence": 0.5},
    {"type": "touch", "confidence": 0.4},
]
args3 = Args()
args3.types = "shot,goal"
args3.max_conf = 0.75
sel = select_entries(manifest, args3)
assert len(sel) == 2 and all(e["type"] in ("shot", "goal") for e in sel), sel
args3.limit = 1
assert len(select_entries(manifest, args3)) == 1
print("PASS selection filters (types, max_conf, limit)")

print("ALL PASS")
