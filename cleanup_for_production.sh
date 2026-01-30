#!/bin/bash
# Cleanup script for production deployment
# This moves unnecessary development files to archive/

set -e

echo "=================================================="
echo "  Football Pipeline - Production Cleanup"
echo "=================================================="
echo ""

# Create archive directory
ARCHIVE_DIR="archive_dev_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

echo "📦 Archiving development files to: $ARCHIVE_DIR"
echo ""

# Function to safely move files
safe_move() {
    if [ -f "$1" ]; then
        echo "  Moving: $1"
        mv "$1" "$ARCHIVE_DIR/"
    fi
}

# Remove or archive development scripts
echo "🗑️  Archiving development/testing scripts..."

# Training scripts
safe_move "train_resnet34_jnr.py"
safe_move "train_resnet34_jnr_rgb.py"
safe_move "finetune_jnr.py"
safe_move "finetune_qwen_jnr.py"
safe_move "finetune_qwen_realcrops.py"
safe_move "finetune_resnet34_manual.py"
safe_move "finetune_resnet34_rgb_strict.py"
safe_move "finetune_resnet34_strict.py"

# Labeling scripts
safe_move "label_crops_gemini.py"
safe_move "label_crops_qwen.py"

# Old test scripts (keep only the new ones we created)
safe_move "test_download.py"
safe_move "test_goal_logic.py"
safe_move "test_jnr.py"
safe_move "test_jnr_integration.py"
safe_move "test_jnr_v26.py"
safe_move "test_loader.py"
safe_move "test_pipeline_logic.py"
safe_move "test_sam2_full.py"
safe_move "test_sam2_init.py"
safe_move "test_siglip.py"
safe_move "test_specific_crop.py"
safe_move "test_advanced_stats.py"

# Old run scripts
safe_move "run_batch_all.py"
safe_move "run_clean_test.py"
safe_move "run_full_match.py"
safe_move "run_local_clip.py"
safe_move "run_parallel_test.py"
safe_move "run_spaces_largest.py"

# Debug scripts
safe_move "debug_sbg_api.py"

# Batch processing (replaced by orchestrator polling)
safe_move "batch_processor.py"
safe_move "spaces_processor.py"
safe_move "process_single_video.py"

# Analysis/inspection tools (development only)
safe_move "analyze_pipeline_output.py"
safe_move "audit_stats.py"
safe_move "inspect_api_schema.py"
safe_move "inspect_db_stats.py"
safe_move "inspect_model.py"
safe_move "measure_tracking_metrics.py"
safe_move "parse_log_count.py"
safe_move "summarize_batch.py"

# Cleanup/fix scripts (one-time use)
safe_move "clean_ikorodu_stats.py"
safe_move "fix_json.py"
safe_move "fix_teams_phase81.py"
safe_move "sanitize_stats.py"

# Legacy/reference
safe_move "legacy_reference.py"
safe_move "rerun_stats.py"
safe_move "rerun_stats_only.py"

# Rendering scripts (optional, can keep if needed)
safe_move "render_clips.py"
safe_move "render_validation_video.py"

# Utility scripts that might be redundant
safe_move "get_frames.py"
safe_move "get_player_stats.py"
safe_move "json_to_csv.py"
safe_move "convert_py_to_ipynb.py"

# API/DB check scripts (mostly redundant with our new ones)
safe_move "check_db_results.py"
safe_move "check_jwt.py"
safe_move "check_yolo_classes.py"
safe_move "fetch_sbg_url.py"
safe_move "reset_db_status.py"

# Other utilities
safe_move "schemas.py"
safe_move "status_server.py"

# Backup files
safe_move "pipeline_consolidated.py.bak"
safe_move "pipeline_consolidated.ipynb"

# Shell scripts (check if needed)
safe_move "chain_batch.sh"
safe_move "start_pipeline.sh"

# Log files
safe_move "compare.log"
safe_move "eval_base.log"
safe_move "export.log"
safe_move "inference_video.log"
safe_move "polling_service.log"
safe_move "run.log"
safe_move "train.err"
safe_move "train.log"

# Text files
safe_move "debug_lowlevel.txt"
safe_move "debug_texts.txt"
safe_move "fresh_url.txt"
safe_move "run_log.txt"

# Markdown files that are superseded
safe_move "ANALYSIS.md"
safe_move "TESTING_IN_NOTEBOOK.md"
safe_move "Football"
safe_move "Improving"

# Config files that might be outdated
safe_move "botsort.yaml"
safe_move "custom_bytetrack.yaml"
safe_move "football_bytetrack.yaml"

# Classes file (if redundant)
safe_move "classes.txt"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Files moved to: $ARCHIVE_DIR"
echo ""

# Count archived files
ARCHIVED_COUNT=$(ls -1 "$ARCHIVE_DIR" 2>/dev/null | wc -l | tr -d ' ')
echo "📊 Archived $ARCHIVED_COUNT files"
echo ""

# List remaining Python files
echo "📝 Remaining Python scripts in root:"
ls -1 *.py 2>/dev/null | sort | while read file; do
    echo "  ✓ $file"
done

echo ""
echo "=================================================="
echo "  Production-ready codebase prepared!"
echo "=================================================="
