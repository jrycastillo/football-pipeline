# Round 10 Processing Status Report
**Generated:** 2026-03-11 08:31 UTC  
**Pipeline:** Phase 216 Fix `b04d368` (Pick-Primary) | S=70 | VID_STRIDE=3  
**User:** `f4dbcb0a` (3 videos) + 1 cached video (`3235879b`)

---

## Summary

| # | Video ID | File | Size | Started | Saved | Duration | Status |
|---|----------|------|------|---------|-------|----------|--------|
| 1 | `1e942fd8a6344bd` | `f4dbcb0a_2026-03-09-22-46-45337194.webm` | 1.3 GB | 2026-03-10 09:23 UTC | 2026-03-11 00:28 UTC | **~15.1 hrs** | ✅ Done |
| 2 | `82d0374d58a8433` | `f4dbcb0a_2026-03-09-23-30-32696103.mp4` | 1.4 GB | 2026-03-11 02:19 UTC | — | ~6 hrs so far | 🔄 Running (~Frame 149k) |
| 3 | `e4d860cc1e44468` | `f4dbcb0a_2026-03-09-23-01-11456234.webm` | 3.3 GB | 2026-03-10 09:23 UTC | 2026-03-11 02:09 UTC | **~16.8 hrs** | ✅ Done |
| 4 | `f561510bde5e4ca` | `3235879b_2026-03-10-02-10-20767443.mp4` | 2.2 GB | 2026-03-10 10:32 UTC | — | ~22 hrs so far | 🔄 Running (~Frame 239k) |

> **Note:** Videos 1–3 are from user `f4dbcb0a` (Round 10 target). Video 4 is the previously killed 2.2GB clip from user `3235879b`, reprocessed using the locally cached file to avoid re-downloading.
>
> Processing times are long due to 4 parallel workers competing for the same H100 GPU and CPU (load avg 44+). Videos 2 and 4 are now running solo with full GPU/CPU.

---

## Completed Video Stats

### Video 1 — `1e942fd8a6344bd` (1.3 GB)
**Score:** Green 0  White 0  
**Match Totals:** Shots: 10 | xG: 0.10 | Passes: 96 | Players: 30

| Team | Players | Goals |
|------|---------|-------|
| Green | 19 | 0 |
| White | 11 | 0 |

---

### Video 3 — `e4d860cc1e44468` (3.3 GB)
**Score:** Red 0  White 1  
**Match Totals:** Shots: 11 | xG: 0.22 | Passes: 283 | Players: 27

| Team | Players | Goals |
|------|---------|-------|
| Red | 18 | 0 |
| White | 9 | 1 |

---

## In-Progress Videos

### Video 2 — `82d0374d58a8433` (1.4 GB)
- **Started:** 2026-03-11 02:19 UTC (after queue watcher auto-triggered)
- **Current frame:** ~149,380 (as of 08:32 UTC)
- **Note:** Previously killed during orchestrator restart on 2026-03-10. Successfully re-queued and restarted automatically.

### Video 4 — `f561510bde5e4ca` (2.2 GB cached)
- **Started:** 2026-03-10 10:32 UTC (local file, no re-download)
- **Current frame:** ~239,360 (as of 08:32 UTC)
- **Note:** Original run killed at ~frame 320k on 2026-03-10. Video file was still cached in `/tmp/`, so reprocessed without downloading.

---

## Processing Timeline

```
2026-03-10 02:08  Jan uploads 3 videos (wrong users: 17f137b0, 3235879b)
2026-03-10 02:40  Orchestrator launched → picks up wrong videos
2026-03-10 03:21  2x 314MB clips (e779f2bec90b474, f77fd5193d594ce) complete ✅
2026-03-10 09:17  User reports: we need f4dbcb0a videos instead
2026-03-10 09:20  Current pipeline killed (2.2GB video cached in /tmp/)
2026-03-10 09:23  New orchestrator launched → f4dbcb0a V1, V2, V3 start
2026-03-10 10:32  Cached 2.2GB video launched in parallel (local file, no download)
2026-03-10 23:28  V2 killed during a pipeline restart → output empty
2026-03-11 00:28  V1 (1.3GB) completes ✅ (~15.1 hrs)
2026-03-11 02:09  V3 (3.3GB) completes ✅ (~16.8 hrs)
2026-03-11 02:19  V2 auto-requeued and restarted (queue watcher triggered)
2026-03-11 08:32  V2 (~frame 149k) and cached video (~frame 239k) still running
```
