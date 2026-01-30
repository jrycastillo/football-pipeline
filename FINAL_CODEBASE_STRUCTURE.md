# Final Production Codebase Structure

**After running `cleanup_for_production.sh`**

---

## Root Directory (Production Files Only)

### ✅ Core Pipeline Files

```
orchestrator.py                  # Main orchestrator (polling + single video)
pipeline_consolidated.py         # Core processing pipeline
config.yaml                      # Default configuration
requirements.txt                 # Python dependencies
.env.example                     # Environment template (secrets excluded)
.gitignore                       # Git exclusions
```

### ✅ Utilities & Scripts

```
check_setup.py                   # Verify installation and dependencies
check_db_status.py               # Database status monitoring
test_db_connection.py            # Database connection diagnostics
test_color_classifier.py         # Color detection testing
test_stats_computation.py        # Statistics diagnostics
monitor_processing.sh            # Live processing dashboard
restart_orchestrator.sh          # Orchestrator restart helper
cleanup_for_production.sh        # Production cleanup script (this file)
```

### ✅ Documentation

```
README.md                        # Project overview
SETUP_STAGING.md                 # Setup guide
MODEL_SETUP.md                   # Model download instructions
IMPROVEMENTS_SUMMARY.md          # Feature summary
PIPELINE_ISSUES_FIXED.md         # Bug fixes log
color_classifier_review.md       # Color detection analysis
PRODUCTION_CHECKLIST.md          # Deployment checklist
H100_MIGRATION_GUIDE.md          # H100 optimization guide
PRODUCTION_READY.md              # Production readiness summary
FINAL_CODEBASE_STRUCTURE.md      # This file
```

---

## Directory Structure

### ✅ vision/ (Vision Components)

```
vision/
├── __init__.py
├── ball_tracking.py             # Ball detection and tracking
├── camera.py                    # Camera calibration and pitch homography
├── color_classifier.py          # HSV color classification (17 colors)
├── custom_botsort.py            # BoT-SORT tracker implementation
├── custom_bytetrack.py          # ByteTrack tracker implementation
├── hybrid_recognition.py        # Hybrid ResNet + Qwen2.5-VL JNR
├── identity.py                  # Identity management (legacy)
├── identity_manager.py          # Current identity management
├── lazy_identity.py             # Lazy identity loading (legacy)
├── ocr.py                       # OCR utilities (legacy)
├── qwen_recognition.py          # Qwen VLM JNR
├── resnet_recognition.py        # ResNet34 JNR
├── stitching.py                 # Video stitching utilities
└── sam2/                        # SAM2 tracking (if used)
```

### ✅ stats/ (Statistics Engine)

```
stats/
├── __init__.py
├── event_logic.py               # Event detection (passes, shots, tackles, etc.)
└── metrics.py                   # Statistics computation
```

### ✅ utils/ (Utilities)

```
utils/
├── __init__.py
├── device_utils.py              # CUDA/MPS/CPU device management
└── health_monitor.py            # Health metrics tracking
```

### ✅ deployment/ (Production Deployment)

```
deployment/
├── football-pipeline.service    # Systemd service configuration
├── install-service.sh           # Service installation script
└── README.md                    # Deployment instructions
```

### ✅ docs/ (Documentation)

```
docs/
├── README.md                    # Documentation index
├── PIPELINE_USAGE.md            # Complete usage guide
├── QUICK_REFERENCE.md           # Command cheat sheet
├── PIPELINE_GUIDE.md            # Architecture documentation
└── TRACKER_SELECTION_GUIDE.md   # Tracking options guide
```

### ✅ models/ (Model Weights - Downloaded Separately)

```
models/
├── yolo_player.pt               # Player/GK/Referee detection (~50MB)
├── yolo_ball.pt                 # Ball detection (~20MB)
├── yolo_pitch.pt                # Pitch keypoint detection (~45MB)
└── resnet34_rgb_jnr.pt          # Jersey number recognition (~85MB)

Note: Models NOT in git (too large), download separately
```

### ✅ test_videos/ (Optional Test Data)

```
test_videos/
└── sample.mp4                   # Small test video for verification

Note: Test videos NOT in git, add as needed
```

---

## Files Archived (Moved to archive_dev_YYYYMMDD_HHMMSS/)

### 🗃️ Training Scripts (~10 files)

```
train_resnet34_jnr.py
train_resnet34_jnr_rgb.py
finetune_*.py (6 files)
```

### 🗃️ Development Test Scripts (~15 files)

```
test_download.py
test_goal_logic.py
test_jnr.py
test_jnr_integration.py
test_jnr_v26.py
test_loader.py
test_pipeline_logic.py
test_sam2_full.py
test_sam2_init.py
test_siglip.py
test_specific_crop.py
test_advanced_stats.py
```

### 🗃️ Batch/Run Scripts (~7 files)

```
batch_processor.py
spaces_processor.py
process_single_video.py
run_batch_all.py
run_clean_test.py
run_full_match.py
run_local_clip.py
run_parallel_test.py
run_spaces_largest.py
```

### 🗃️ Analysis/Debug Tools (~10 files)

```
analyze_pipeline_output.py
audit_stats.py
debug_sbg_api.py
inspect_api_schema.py
inspect_db_stats.py
inspect_model.py
measure_tracking_metrics.py
parse_log_count.py
summarize_batch.py
```

### 🗃️ Cleanup/Fix Scripts (~7 files)

```
clean_ikorodu_stats.py
fix_json.py
fix_teams_phase81.py
sanitize_stats.py
rerun_stats.py
rerun_stats_only.py
reset_db_status.py
```

### 🗃️ Rendering/Visualization (~2 files)

```
render_clips.py
render_validation_video.py
```

### 🗃️ Utility Scripts (~8 files)

```
get_frames.py
get_player_stats.py
json_to_csv.py
convert_py_to_ipynb.py
check_db_results.py
check_jwt.py
check_yolo_classes.py
fetch_sbg_url.py
schemas.py
status_server.py
label_crops_gemini.py
label_crops_qwen.py
```

### 🗃️ Legacy/Backup Files (~5 files)

```
legacy_reference.py
pipeline_consolidated.py.bak
pipeline_consolidated.ipynb
start_pipeline.sh
chain_batch.sh
```

### 🗃️ Config Files (~3 files)

```
botsort.yaml
custom_bytetrack.yaml
football_bytetrack.yaml
classes.txt
```

### 🗃️ Log Files (~10 files)

```
compare.log
eval_base.log
export.log
inference_video.log
polling_service.log
run.log
train.err
train.log
run_log.txt
```

### 🗃️ Text/Markdown Files (~5 files)

```
debug_lowlevel.txt
debug_texts.txt
fresh_url.txt
ANALYSIS.md
TESTING_IN_NOTEBOOK.md
Football (file)
Improving (file)
```

**Total Archived:** ~70+ development files

---

## Excluded from Git (.gitignore)

```gitignore
# System & Python
__pycache__/
*.py[cod]
.DS_Store
.env                    # ← Contains secrets
venv/

# Logs
*.log
*.err
*.txt (except requirements.txt)

# Models & Weights
models/                 # ← Too large (download separately)
*.pt
*.pth
*.onnx

# Output
output/
output_*/
runs/
*.mp4
*.csv
*.json (except specific configs)
*.png

# Archives & Backups
archive/
archive_*/
backups/
backup_*/
*.bak
*.ipynb

# IDE
.vscode/
.idea/
```

---

## Total File Count

### Before Cleanup

```
~120 Python files in root
~150+ total files in root
```

### After Cleanup (Production)

```
~12 Python files in root (utilities + core)
~20 total essential files in root
+ vision/ directory (15 files)
+ stats/ directory (2 files)
+ utils/ directory (2 files)
+ docs/ directory (5 files)
+ deployment/ directory (3 files)
```

**Reduction:** ~85% fewer files in root directory

---

## Repository Size

### Without Models

```
Code only: ~15MB
With docs: ~18MB
```

### With Models (Downloaded Separately)

```
Total: ~200MB
- yolo_player.pt: ~50MB
- yolo_ball.pt: ~20MB
- yolo_pitch.pt: ~45MB
- resnet34_rgb_jnr.pt: ~85MB
```

---

## Production Deployment Files

### Minimal Required for Deployment

```
1. Code (from git)
2. Models (download separately)
3. .env file (create from .env.example)
4. Python environment (pip install -r requirements.txt)
```

### Total Deployment Size

```
~220MB total
- Code: 18MB
- Models: 200MB
- Venv: Variable (depends on GPU, ~2-4GB with PyTorch)
```

---

## Verification Commands

### Check Final Structure

```bash
# After running cleanup
ls -1 *.py | wc -l
# Should show ~12 files

# Check directories
ls -d */
# Should show: deployment/ docs/ models/ stats/ utils/ vision/

# Verify archive created
ls -la archive_dev_*/
# Should show ~70 archived files
```

### Verify Git Status

```bash
# Check what's tracked
git ls-files | head -20

# Check what's ignored
git status --ignored
```

### Verify Models NOT in Git

```bash
# Should show models are ignored
git check-ignore models/*.pt
# Output: models/yolo_player.pt, models/yolo_ball.pt, etc.
```

---

## Next Steps

1. **Run cleanup:**
   ```bash
   ./cleanup_for_production.sh
   ```

2. **Review archived files:**
   ```bash
   ls -la archive_dev_*/
   ```

3. **Verify structure:**
   ```bash
   ls -1 *.py
   tree -L 2
   ```

4. **Commit to repo:**
   ```bash
   git add .
   git commit -m "Production ready: Cleaned codebase"
   git tag v1.0-production
   git push origin main --tags
   ```

5. **Deploy:**
   Follow [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)

---

**Status:** Ready for production deployment ✅

**Last Updated:** 2026-01-30
