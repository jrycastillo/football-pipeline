# Round 10 Processing Status Report
**Generated:** 2026-03-11 14:04 UTC  
**Pipeline:** Phase 216 Fix `b04d368` (Pick-Primary) | S=70 | VID_STRIDE=3  
**User:** `f4dbcb0a` (3 videos) + 1 cached video (`3235879b`)
**Status:** ✅ All 4 videos complete

---

## Summary

| # | Video ID | File | Size | Started | Saved | Duration | Status |
|---|----------|------|------|---------|-------|----------|--------|
| 1 | `1e942fd8a6344bd` | `f4dbcb0a_2026-03-09-22-46-45337194.webm` | 1.3 GB | 2026-03-10 09:23 UTC | 2026-03-11 00:28 UTC | **~15.1 hrs** | ✅ Done |
| 2 | `82d0374d58a8433` | `f4dbcb0a_2026-03-09-23-30-32696103.mp4` | 1.4 GB | 2026-03-11 02:19 UTC | 2026-03-11 09:24 UTC | **~7.1 hrs** | ✅ Done |
| 3 | `e4d860cc1e44468` | `f4dbcb0a_2026-03-09-23-01-11456234.webm` | 3.3 GB | 2026-03-10 09:23 UTC | 2026-03-11 02:09 UTC | **~16.8 hrs** | ✅ Done |
| 4 | `f561510bde5e4ca` | `3235879b_2026-03-10-02-10-20767443.mp4` | 2.2 GB | 2026-03-10 10:32 UTC | 2026-03-11 13:30 UTC | **~27 hrs** | ✅ Done |

> **Note:** Videos 1–3 are from user `f4dbcb0a` (Round 10 target). Video 4 is the previously killed 2.2GB clip from user `3235879b`, reprocessed using the locally cached file to avoid re-downloading.
>
> Processing times are long due to 4 parallel workers competing for the same H100 GPU and CPU (load avg 44+). Videos 2 and 4 are now running solo with full GPU/CPU.

---

## Completed Video Stats

| Video | Score | Players | Shots | xG | Passes | Tackles |
|-------|-------|---------|-------|----|--------|---------|
| `1e942fd8a6344bd` (1.3 GB) | Green 0 – White 0 | 30 (19G/11W) | 10 | 0.10 | 96 | 10 |
| `82d0374d58a8433` (1.4 GB) | White 1 – Red 0 | 34 (22W/12R) | 2 | 0.19 | 60 | 5 |
| `e4d860cc1e44468` (3.3 GB) | Red 0 – White 1 | 27 (18R/9W) | 11 | 0.22 | 283 | 21 |
| `f561510bde5e4ca` (2.2 GB) | White 7 – Red 0 | 37 (30W/7R) | 85 | 0.98 | 1,582 | 302 |

---

## Processing Timeline

```
2026-03-10 02:08  3 videos (wrong users: 17f137b0, 3235879b)
2026-03-10 02:40  Orchestrator launched → picks up wrong videos
2026-03-10 03:21  2x 314MB clips (e779f2bec90b474, f77fd5193d594ce) complete ✅
2026-03-10 09:17  Reports: f4dbcb0a videos instead
2026-03-10 09:20  Current pipeline killed (2.2GB video cached in /tmp/)
2026-03-10 09:23  New orchestrator launched → f4dbcb0a V1, V2, V3 start
2026-03-10 10:32  Cached 2.2GB video launched in parallel (local file, no download)
2026-03-10 23:28  V2 killed during a pipeline restart → output empty
2026-03-11 00:28  V1 (1.3GB) completes ✅ (~15.1 hrs)
2026-03-11 02:09  V3 (3.3GB) completes ✅ (~16.8 hrs)
2026-03-11 02:19  V2 auto-requeued and restarted (queue watcher triggered)
2026-03-11 09:24  V2 (1.4GB) completes ✅ (~7.1 hrs)
2026-03-11 13:30  Cached 2.2GB video completes ✅ (~27 hrs, ~482k frames)
```

---

## Why Processing Took Longer Than Usual

Round 10 took significantly longer than previous rounds (~17 hrs typical for Round 9). Four compounding factors:

**1. 4 parallel workers on a single H100 (main culprit)**  
Previous rounds used `--parallel 3`. This round we added a 4th worker (the cached 2.2GB video). At peak load, CPU was at **99.6% (load avg 44)** while the GPU sat at **1% utilisation** — the workers were starving each other of CPU time for frame decoding and tracking. Each pipeline ran at ~25% speed instead of the usual ~33%.

**2. Videos were much longer than previous rounds**  
The `f561510bde5e4ca` `.mp4` file had **~482k frames** — nearly 3× the largest Round 9 video (~180k frames). `.mp4` codec is also more CPU-intensive to decode than `.webm`.

**3. V2 was killed mid-run and had to restart from zero**  
A broad `pkill -f pipeline_consolidated.py` during an orchestrator restart also killed V2 at ~2 hrs in. It restarted fresh at 02:19 UTC, wasting ~2 hours of compute and adding ~7 hrs to the total wall time.

**4. Wrong videos processed first**  
The first orchestrator pickup processed 3 videos from the wrong users (`17f137b0`, `3235879b`) for ~7 hours before pivoting to `f4dbcb0a`. The 2.2GB video from that batch was then reprocessed locally, contributing to the overall extended timeline.

> **If run correctly from the start** (`--parallel 3`, correct video IDs only), total processing time would have been ~17 hours — matching Round 9.
