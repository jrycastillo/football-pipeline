# Football Pipeline Documentation

**Complete documentation index for the Football Analysis Pipeline**

---

## 📖 Documentation Overview

This directory contains comprehensive documentation for the Football Analysis Pipeline system.

---

## 🚀 Getting Started

**New to the pipeline?** Start here:

1. **[Pipeline Usage Guide](PIPELINE_USAGE.md)** ⭐
   - Complete command-line reference
   - All arguments explained
   - Common workflows and examples
   - Troubleshooting guide

2. **[Quick Reference](QUICK_REFERENCE.md)**
   - Cheat sheet for common commands
   - Quick troubleshooting tips
   - Performance mode selection

3. **[Setup Guide](../SETUP_STAGING.md)**
   - Installation instructions
   - Environment configuration
   - Model download and setup

---

## 📚 Reference Documentation

### Architecture & Design

- **[Pipeline Guide](PIPELINE_GUIDE.md)**
  - System architecture
  - Component breakdown
  - Data flow
  - Configuration options

- **[Tracker Selection Guide](TRACKER_SELECTION_GUIDE.md)**
  - ByteTrack vs BoT-SORT vs SAM2
  - Performance comparisons
  - When to use each tracker
  - Troubleshooting tracking issues

### Deployment

- **[Deployment Guide](../deployment/README.md)**
  - Systemd service setup
  - Production configuration
  - Auto-restart configuration
  - Log management

- **[Setup Staging](../SETUP_STAGING.md)**
  - Staging server setup
  - Environment variables
  - Database configuration

### Technical Details

- **[Color Classifier Review](../color_classifier_review.md)**
  - Color detection HSV ranges
  - Boundary overlap analysis
  - Test results

- **[Pipeline Issues Fixed](../PIPELINE_ISSUES_FIXED.md)**
  - Recent bug fixes
  - Resource leak fixes
  - Performance improvements

- **[Improvements Summary](../IMPROVEMENTS_SUMMARY.md)**
  - Feature additions
  - Architecture changes
  - Optimization history

---

## 🎯 Quick Links by Task

### I want to...

**Process a single video**
→ [Pipeline Usage: Single Video Mode](PIPELINE_USAGE.md#mode-1-single-video-processing-debugtesting)

**Set up production polling**
→ [Pipeline Usage: Polling Mode](PIPELINE_USAGE.md#mode-2-polling-mode-production)

**Choose a tracking backend**
→ [Tracker Selection Guide](TRACKER_SELECTION_GUIDE.md)

**Deploy to production**
→ [Deployment Guide](../deployment/README.md)

**Troubleshoot an issue**
→ [Pipeline Usage: Troubleshooting](PIPELINE_USAGE.md#troubleshooting)

**Understand the architecture**
→ [Pipeline Guide](PIPELINE_GUIDE.md)

**Configure settings**
→ [Pipeline Usage: Configuration](PIPELINE_USAGE.md#configuration)

**Monitor processing**
→ [Pipeline Usage: Monitoring](PIPELINE_USAGE.md#monitoring)

---

## 📋 Command Reference

### Main Commands

```bash
# Process single video
python orchestrator.py --local_video video.mp4 --save_local

# Production polling
python orchestrator.py --poll --parallel 3

# Get help
python orchestrator.py --help
python pipeline_consolidated.py --help
```

See [Quick Reference](QUICK_REFERENCE.md) for more commands.

---

## 🧪 Testing & Validation

### Test Scripts

| Script | Purpose |
|--------|---------|
| `check_setup.py` | Verify installation and dependencies |
| `test_db_connection.py` | Test database connectivity |
| `check_db_status.py` | Check current processing status |
| `test_color_classifier.py` | Validate color detection |
| `test_stats_computation.py` | Diagnose statistics issues |
| `monitor_processing.sh` | Live processing dashboard |

### Running Tests

```bash
# Verify installation
python check_setup.py

# Test database
python test_db_connection.py

# Check color classifier
python test_color_classifier.py

# Diagnose stats
python test_stats_computation.py output/player_stats.json
```

---

## 🛠️ Configuration Files

| File | Purpose | Documentation |
|------|---------|---------------|
| `.env` | Credentials & secrets | [Setup Guide](../SETUP_STAGING.md) |
| `config.yaml` | Pipeline defaults | [Pipeline Guide](PIPELINE_GUIDE.md) |
| `deployment/football-pipeline.service` | Systemd service | [Deployment Guide](../deployment/README.md) |

---

## 📊 Output Files

After processing, the pipeline generates:

```
output/
├── player_stats.json       # Main statistics output
├── player_stats.csv        # CSV format
├── match_kits.json         # Detected team colors
├── debug_video.mp4         # Annotated video (optional)
├── pipeline.log            # Processing logs
└── health_snapshot.json    # Health metrics
```

See [Pipeline Usage: Output Files](PIPELINE_USAGE.md#output-files) for details.

---

## 🔧 Common Workflows

### Development

1. **Test on short clip:**
   ```bash
   python orchestrator.py \
       --local_video test.mp4 \
       --save_local \
       --max_frames 500 \
       --no_db
   ```

2. **Full match with debug video:**
   ```bash
   python orchestrator.py \
       --local_video match.mp4 \
       --save_local \
       --make_video
   ```

### Production

1. **Deploy service:**
   ```bash
   sudo ./deployment/install-service.sh
   sudo systemctl start football-pipeline
   ```

2. **Monitor:**
   ```bash
   sudo journalctl -u football-pipeline -f
   python check_db_status.py
   ```

---

## 🆘 Troubleshooting

### Common Issues

| Issue | Quick Fix | Documentation |
|-------|-----------|---------------|
| CUDA Out of Memory | `--vid_stride 2` | [Troubleshooting](PIPELINE_USAGE.md#1-cuda-out-of-memory) |
| Database Connection Failed | Check `.env` | [Troubleshooting](PIPELINE_USAGE.md#2-database-connection-failed) |
| No Statistics | Check logs | [Troubleshooting](PIPELINE_USAGE.md#3-no-statistics-generated) |
| Slow Processing | Adjust tracker | [Tracker Guide](TRACKER_SELECTION_GUIDE.md) |

See [Full Troubleshooting Guide](PIPELINE_USAGE.md#troubleshooting).

---

## 📈 Performance Optimization

### Speed vs Accuracy

| Priority | Settings | Documentation |
|----------|----------|---------------|
| **Speed** | `--vid_stride 2 --tracking_mode bytetrack` | [Quick Ref](QUICK_REFERENCE.md#performance-modes) |
| **Balanced** | `--locking_mode 2 --tracking_mode bytetrack` | [Pipeline Usage](PIPELINE_USAGE.md#locking-modes) |
| **Accuracy** | `--locking_mode 3 --tracking_mode botsort` | [Tracker Guide](TRACKER_SELECTION_GUIDE.md) |

---

## 🔄 Recent Updates

### Latest Changes (2026-01-30)

- ✅ Fixed resource leak in thread pool executor
- ✅ Implemented graceful shutdown
- ✅ Fixed color classifier boundary overlaps
- ✅ Improved error handling (removed bare except clauses)
- ✅ Added comprehensive documentation

See [Pipeline Issues Fixed](../PIPELINE_ISSUES_FIXED.md) for details.

---

## 📞 Support

### Self-Service

1. Check [Quick Reference](QUICK_REFERENCE.md)
2. Review [Troubleshooting Guide](PIPELINE_USAGE.md#troubleshooting)
3. Run diagnostic scripts: `check_setup.py`, `test_db_connection.py`
4. Check logs: `tail -f output/pipeline.log`

### Documentation Navigation

- **Getting started?** → [Pipeline Usage Guide](PIPELINE_USAGE.md)
- **Need quick command?** → [Quick Reference](QUICK_REFERENCE.md)
- **Production deployment?** → [Deployment Guide](../deployment/README.md)
- **Choosing tracker?** → [Tracker Selection Guide](TRACKER_SELECTION_GUIDE.md)
- **Understanding architecture?** → [Pipeline Guide](PIPELINE_GUIDE.md)

---

## 📝 Contributing

When updating documentation:

1. Keep [Pipeline Usage Guide](PIPELINE_USAGE.md) comprehensive
2. Update [Quick Reference](QUICK_REFERENCE.md) for new commands
3. Document breaking changes in main [README](../README.md)
4. Add examples to relevant sections

---

## 📄 License

See [LICENSE](../LICENSE) for details.

---

**Last Updated:** 2026-01-30
**Pipeline Version:** Production v1.0
