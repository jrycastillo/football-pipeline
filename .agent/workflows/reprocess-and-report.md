---
description: Reprocess 3 videos and generate dated stats report
---

# Reprocess Videos & Generate Report

## Steps

// turbo-all

1. Pull latest changes from the repo:
```bash
cd /home/ubuntu/football && git stash && git pull origin production-v1.0 && git stash pop
```

2. Reprocess the 3 videos:
```bash
cd /home/ubuntu/football && NO_DB=1 python3 orchestrator.py --poll --no_db --vid_stride 5 --video_ids 162b6abe208946b,14c0f4e8c4af40d,69a33466fc234db
```

3. Wait for all 3 videos to complete (check for "Stats Saved" messages in the log).

4. Generate the report and copy raw JSONs to a dated directory:
```bash
cd /home/ubuntu/football && python3 generate_report.py
```

This creates:
- `reports/YYYY-MM-DD/full_stats_report_YYYY-MM-DD.md` — formatted report
- `reports/YYYY-MM-DD/player_stats_<video_id>.json` — raw JSON for each video
- `reports/YYYY-MM-DD/match_kits_<video_id>.json` — team kit colors for each video
- `full_stats_report_MMDD.md` — copy in project root for quick access

## Custom date or pipeline label

```bash
python3 generate_report.py --date 2026-02-14 --pipeline "H100 Optimized | VID_STRIDE=5"
```
