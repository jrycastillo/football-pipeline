# Onboarding — Football Match Analysis AI Pipeline

Welcome to the team! This document gets you oriented on what we're building, how the pipeline works, and what to start with. Take the first few days to read and run things before writing any code.

---

## 1. What We're Building

An end-to-end computer vision pipeline that takes a football match video and produces **per-player statistics**: passes, tackles, shots, dribbles, xG, goals, interceptions, possession, distance covered, and more.

- **Input:** a match video (MP4/WebM/AVI)
- **Output:** `player_stats.json` — per-player stats, team assignments, jersey numbers, and match events

We benchmark accuracy against **Wyscout** professional ground-truth data (a real Hamburg vs Bayern match: 945 passes, 4 goals, 24 shots, xG 3.75, etc.).

---

## 2. The Pipeline (read this first)

```
Video → YOLO Detection → ByteTrack Tracking → ResNet34 Jersey Recognition
      → HSV Color/Team Clustering → OSNet ReID → Stats Engine → player_stats.json
```

Stage by stage:

1. **YOLO detection** — detects players, goalkeepers, referees, and the ball per frame. Classes: `0=ball, 1=GK, 2=player, 3=referee`.
2. **ByteTrack tracking** — assigns track IDs across frames. Important: at frame stride > 1 it fragments one player into many short track IDs (a core challenge — see §5).
3. **Jersey Number Recognition (JNR)** — a ResNet34 model reads jersey numbers from player crops; vote-based locking assigns a number to a track.
4. **Color / team clustering** — HSV color classification + K-means discovers the two team kit colors and assigns players to teams.
5. **OSNet ReID** — appearance-based re-identification (adapted from a prior prototype) that builds an appearance embedding per player. Currently stores embeddings; not yet fully used to re-link lost tracks.
6. **Stats engine** — computes all events (passes, shots, tackles, etc.) from ball ownership over time.

### Key files to read (in order)
| File | What it does |
|------|--------------|
| `CLAUDE.md` | Project overview and architecture (start here) |
| `pipeline_consolidated.py` | Main pipeline — detection, tracking, JNR, color, orchestration |
| `stats/event_logic.py` | Pass / shot / tackle / dribble / interception detection logic |
| `stats/metrics.py` | StatsEngine + identity resolution (track IDs → jersey numbers) |
| `vision/camera.py` | Image-pixel → pitch-meter projection (currently a simple scale, not a true homography) |
| `vision/color_classifier.py` | HSV color classification with grass masking |

---

## 3. Environment Setup

```bash
# Clone / pull the repo, then:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on a short test clip
python pipeline_consolidated.py \
    --video path/to/clip.mp4 \
    --tracking_mode bytetrack \
    --output_dir output/my_first_run \
    --vid_stride 3 \
    --enable_reid true

# Inspect the result
cat output/my_first_run/player_stats.json | python -m json.tool
```

- **GPU:** NVIDIA (CUDA) is fastest. The pipeline auto-detects CUDA/MPS/CPU.
- Models live in `models/` (YOLO player, YOLO ball, ResNet34 JNR, OSNet ReID).
- Config precedence: **CLI args > environment vars > config.yaml**.

---

## 4. Current State (as of now)

**Working well:**
- End-to-end pipeline runs reliably; team-color discovery is solid.
- On the 30-min test match: goals, interceptions, and xG are accurate vs Wyscout.
- Five recent structural bugs fixed (frame double-counting, phantom players, goal detection, fragment mis-merge, fragment over-count).

**Known open problems (where you can help):**
1. **Jersey "magnet" over-assignment** — sometimes multiple different players' tracks resolve to the same jersey number, inflating that player's stats. Root cause is in the identity-locking / resolution layer.
2. **Tackles / passes still need calibration** vs ground truth.
3. **No real homography** — distance and shot geometry suffer because `camera.py` uses a flat linear scale instead of pitch-keypoint homography.
4. **Detection on non-broadcast footage** — under-detects on elevated/training camera angles.
5. **No automated accuracy validation harness** — we eyeball results against Wyscout manually.

---

## 5. First Tasks (start here, in order)

**Week 1 — Learn by running (no code changes yet):**
- [ ] Read `CLAUDE.md` and skim the key files in §2.
- [ ] Set up the environment and run the pipeline on a short clip end-to-end.
- [ ] Open `player_stats.json` and `debug_all_frames.json` and understand the output format.
- [ ] Write a short note (½ page) on what you understood and any questions — this doubles as a comprehension check.

**Week 1–2 — First useful contribution (pick one):**
- [ ] **Validation harness (high value, self-contained):** write a script that loads a `player_stats.json`, sums team totals (passes/shots/tackles/etc.), and compares them against a Wyscout ground-truth JSON, printing a per-metric accuracy table. This gives the whole team a fast "is this run good?" check.
- [ ] **JNR accuracy measurement:** build a script that measures in-video jersey-number lock accuracy against a small set of hand-labeled crops (ground-truth labeling is being arranged).

**Later (bigger items, once oriented):**
- [ ] Investigate the jersey "magnet" bug in the identity layer (pair with Ronan on this).
- [ ] Prototype pitch-keypoint homography in `camera.py`.

---

## 6. How We Work

- **Branches:** `production-v1.0` (deployment) and `football-pipeline-fixes` (dev / PR target). Branch off, don't commit straight to main.
- **Validate every change against the 30-min test match** before claiming a fix — we compare team totals to Wyscout.
- **Be honest about results.** If a stat is off, report the number. We'd rather know a metric is 5× too high than ship a wrong number.
- Ask early. Ping Ronan with questions any time — initial readings + questions are exactly the right start.

---

## 7. Glossary

- **JNR** — Jersey Number Recognition (the ResNet34 model reading shirt numbers).
- **ReID** — Re-identification (matching the same player across frames by appearance).
- **Homography** — a 3×3 matrix mapping image pixels to real pitch coordinates (meters).
- **xG** — expected goals (probability a shot becomes a goal).
- **Fragment** — one of the many short track IDs ByteTrack creates for a single player.
- **Wyscout** — professional match-data provider we use as accuracy ground truth.

Welcome aboard — start with §5, and don't hesitate to ask.
