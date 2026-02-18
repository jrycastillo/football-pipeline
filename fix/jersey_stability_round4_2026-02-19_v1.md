# Jersey Number Stability Fix (Round 4) — 2026-02-19 v1

## Context

Across four H100 runs (Feb 12, 14, 17, 18), the same videos produce **different jersey
number assignments and different player counts** each time. For example:

| Video | Feb 12 Players | Feb 14 Players | Feb 17 Players | Feb 18 Players |
|-------|---------------|----------------|----------------|----------------|
| V1    | 23            | 17             | 20             | 20             |
| V2    | 25            | 18             | 14             | 24             |
| V3    | 23            | 19             | 19             | 21             |
| Total | 71            | 54             | 53             | 65             |

Video 2 team colors flipped between runs (Green→Yellow→Green). Same physical players
get assigned different jersey numbers across runs.

**Root cause:** Zero deterministic seed control in the pipeline. GPU non-determinism in
YOLO/ResNet/ByteTrack causes different detections → different track IDs → different JNR
votes → different jersey locks → different final player sets.

**Files changed:**
- `pipeline_consolidated.py` — Seed control, vote tie-breaking, iteration ordering

**Stats code NOT changed:** `stats/event_logic.py` and `stats/metrics.py` are untouched.

---

## Fixes Applied

### P0-1: Deterministic Seed Control (pipeline_consolidated.py)

**Problem:** No random seeds are set anywhere in the pipeline. PyTorch, CUDA, numpy, and
Python's `random` module all use different seeds each run, causing:
- Different YOLO detection confidence scores → different ByteTrack assignments
- Different ResNet34 JNR predictions → different jersey number votes
- Different cuDNN kernel selections → cascading numerical divergence

**Fix:** Added seed block immediately before model initialization:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Impact:** Forces identical GPU kernel selection across runs. Same video input → same
detection outputs → same track IDs → same jersey assignments.

**Note:** `cudnn.benchmark = False` may cause a slight performance decrease (~5-10%) as
cuDNN won't auto-tune kernel selection. This is the standard trade-off for reproducibility.

### P0-2: Deterministic Vote Tie-Breaking (pipeline_consolidated.py:615)

**Problem:** In `process_detection()` Mode 2 (strict voting), when two jersey number
candidates have identical accumulated scores, the winner depends on dict insertion order:
```python
sorted_candidates = sorted(votes.items(), key=lambda x: x[1], reverse=True)
```
If JNR produces results in a different order across runs (e.g., due to batch processing
timing), the same track can lock to different jersey numbers.

**Fix:** Added secondary sort key (jersey number as string) for deterministic tie-breaking:
```python
sorted_candidates = sorted(votes.items(), key=lambda x: (x[1], str(x[0])), reverse=True)
```

### P1-1: Deterministic finalize_bindings() (pipeline_consolidated.py:1031)

**Problem:** `finalize_bindings()` iterates `self.alpha.keys()` (a defaultdict). The
iteration order depends on insertion order, which varies with non-deterministic JNR timing.
Different tracks get finalized first, potentially claiming jersey numbers before others.

**Fix:** Sort keys before iterating:
```python
for tid in sorted(self.alpha.keys(), key=lambda x: str(x)):
```

### P1-2: Deterministic suppress_conflicts() (pipeline_consolidated.py:793)

**Problem:** `suppress_conflicts()` iterates `jersey_map.items()` in insertion order.
When multiple collisions exist, resolution order can affect which track wins.

**Fix:** Sort jersey_map keys before iterating:
```python
for jnum in sorted(jersey_map.keys(), key=lambda x: str(x)):
```

### P1-3: Deterministic Team Color Detection (pipeline_consolidated.py:933)

**Problem:** `detect_team_colors()` sorts colors by frequency. When two colors have the
same count (common early in the video), the winner is insertion-order dependent.

**Fix:** Added color name as secondary sort key:
```python
sorted_colors = sorted(valid_colors.items(), key=lambda x: (x[1], x[0]), reverse=True)
```

---

## Known Bug: Double `all_frames.append` (NOT FIXED)

**Location:** `pipeline_consolidated.py` lines 2087 and 2111

**Problem:** Every frame is appended to `all_frames` TWICE:
1. Line 2087: Inside the inner `try` block (after ID resolution)
2. Line 2111: Outside the inner `try` block (after suppress_conflicts + visualization)

Both entries reference the same `frame_data` dict object, so they contain identical data.
This doubles all observation counts reported by the stats engine.

**Why not fixed:** Removing one append would halve all observation counts, distance values,
and touch frame counts — a massive change to all stats. Since this bug has been present
in all runs (Feb 12-18), removing it now would break consistency. This should be fixed
in a future release with full stats recalibration.

**Observation count impact:** If a player appears in 15,000 actual frames, the stats
engine sees 30,000 frames (doubled). All per-player observation counts in reports are
~2x inflated.

---

## Expected Impact

| Issue | Before Fix | After Fix |
|-------|-----------|-----------|
| Jersey number stability | Different each run | Same for same video |
| Player count stability | Varies ±20% | Consistent |
| Team color assignment | Flips (Green↔Yellow) | Stable |
| Stats values | May shift slightly* | Reproducible |

*Stats may differ slightly from Feb 18 because `cudnn.deterministic` changes GPU kernel
selection, producing marginally different detection confidence scores. The overall
distribution and ranges should be equivalent.

---

## Verification

```bash
python3 -c "import py_compile; py_compile.compile('pipeline_consolidated.py', doraise=True)"
```

### Reproducibility Test
Run the same video twice with identical config:
```bash
python3 pipeline_consolidated.py --video <same_video> --output_dir run_a --vid_stride 3
python3 pipeline_consolidated.py --video <same_video> --output_dir run_b --vid_stride 3
diff run_a/player_stats.json run_b/player_stats.json  # Should be identical
```

---

## Files Changed

- `pipeline_consolidated.py` — P0-1 (seeds), P0-2 (vote sort), P1-1 (finalize sort),
  P1-2 (conflict sort), P1-3 (team color sort)
