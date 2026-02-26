# Full Player Stats Report — 2026-02-26 (Round 9)

**Pipeline:** Phase 216 Fix `b04d368` (Pick-Primary) | S=70 | VID_STRIDE=3
**Generated:** 2026-02-26 17:45:00
**Status:** ✅ All 3 Videos Completed successfully with realistic stats.

---

## Video 1 — `162b6abe208946b` (314 MB)
**Teams:** Red, White (27 players total)
**Score:** Red 0–2 White
**Observations:** Realistic goal counts (2 total) and passes (max 12 per player).

| Team | Players | Goals | Max Passes | Max Obs |
|------|---------|-------|------------|---------|
| Red | 13 | 0 | 12 | 2362 |
| White| 14 | 2 | 8 | 4472 |

---

## Video 2 — `14c0f4e8c4af40d` (1.3 GB)
**Teams:** Green, White (31 players total)
**Score:** Green 1–0 White
**Verification:** Inflation resolved. Round 8 had 17 goals; Round 9 has **1 goal**. Max passes dropped from ~100 to **16**.

| Team | Players | Goals | Max Passes | Max Obs |
|------|---------|-------|------------|---------|
| Green | 20 | 1 | 16 | 5930 |
| White | 11 | 0 | 14 | 3530 |

---

## Video 3 — `69a33466fc234db` (3.3 GB)
**Teams:** Red, White (32 players total)
**Score:** Red 0–1 White
**Verification:** Inflation resolved. Round 8 had 26 goals; Round 9 has **1 goal**. Max passes dropped from >150 to **59**.

| Team | Players | Goals | Max Passes | Max Obs |
|------|---------|-------|------------|---------|
| Red | 20 | 0 | 28 | 13322 |
| White | 12 | 1 | 59 | 6144 |

---

## Technical Note: Phase 216 Fix
The "Summing Duplicates" bug in `metrics.py` was causing stats to inflate by up to 26x when multiple tracks mapped to the same jersey. Commit `b04d368` introduced the **Pick-Primary** logic which selects the best track (most weight) instead of summing all of them.

### Log Confirmation (V3 example):
```
[Phase 216] Jersey #14: picked track 1372 (weight=322), dropped 8 duplicate track(s)
[Phase 216] Jersey #18: picked track 1251 (weight=483), dropped 12 duplicate track(s)
[Phase 216] Remapped 275 track IDs to jersey numbers (pick-primary, no summing)
```

The reprocessing is complete and results are verified as accurate.
