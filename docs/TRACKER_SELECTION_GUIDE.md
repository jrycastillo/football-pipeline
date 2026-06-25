# Tracker Selection Guide

This guide helps you choose the right tracking backend for your football video analysis use case.

## Available Trackers

The pipeline supports three tracking backends:

### 1. **ByteTrack** (Default)
- **Algorithm**: Pure motion-based tracking using Kalman filter + Hungarian matching
- **ReID**: No appearance-based re-identification
- **Speed**: Fastest (minimal overhead)
- **Accuracy**: Good for continuous tracking, struggles with long occlusions

#### When to Use ByteTrack
✅ **Best for:**
- High frame rate videos (>25 fps)
- Continuous camera view (minimal cuts/panning)
- Speed-critical applications
- Videos with few occlusions
- Standard broadcast football matches

❌ **Avoid when:**
- Video has frequent camera cuts
- Players are heavily occluded for extended periods
- Jersey colors are very similar
- You need highest possible ID consistency

#### Configuration
```bash
python orchestrator.py --local_video video.mp4 --tracking_mode bytetrack
```

**Performance**: ~30-40 FPS on CPU, ~80-100 FPS on GPU

---

### 2. **BoT-SORT** (Best Accuracy)
- **Algorithm**: ByteTrack + ReID (appearance embedding) + camera motion compensation
- **ReID**: Uses appearance features from player crops
- **Speed**: Moderate (adds ReID overhead)
- **Accuracy**: Best ID consistency, robust to occlusions

#### When to Use BoT-SORT
✅ **Best for:**
- Long videos where ID consistency is critical
- Videos with frequent occlusions (players crossing paths)
- Camera panning or zoom changes
- Production environments requiring highest accuracy
- Post-match analysis where precision matters

❌ **Avoid when:**
- Real-time processing required
- Limited compute resources
- Very short clips (<30 seconds)

#### Configuration
```bash
python orchestrator.py --local_video video.mp4 --tracking_mode botsort
```

**Performance**: ~15-25 FPS on CPU, ~40-60 FPS on GPU

**Note**: BoT-SORT requires more memory for storing appearance embeddings.

---

### 3. **SAM2** (Experimental)
- **Algorithm**: Segment Anything Model 2 for object segmentation + tracking
- **ReID**: Implicit through segmentation masks
- **Speed**: Slowest (deep learning segmentation)
- **Accuracy**: Excellent segmentation, experimental for tracking

#### When to Use SAM2
✅ **Best for:**
- Research and experimentation
- Cases requiring pixel-perfect player masks
- Videos with extreme occlusions or unusual angles
- Scenarios where traditional trackers fail

❌ **Avoid when:**
- Production use (not battle-tested)
- Limited GPU memory (<16GB VRAM)
- Real-time processing needed

#### Configuration
```bash
python orchestrator.py --local_video video.mp4 --tracking_mode sam2 --sam2_model large
```

**Performance**: ~5-10 FPS on high-end GPU

**Models Available**:
- `tiny`: Fastest, lowest accuracy
- `small`: Balanced
- `base`: Good accuracy
- `large`: Best accuracy, slowest

---

## Quick Decision Tree

```
Do you need real-time processing?
├─ Yes → ByteTrack
└─ No
   └─ Is ID consistency critical?
      ├─ Yes → BoT-SORT
      └─ No → ByteTrack

Are there heavy occlusions?
├─ Yes → BoT-SORT
└─ No → ByteTrack

Experimental features needed?
└─ Yes → SAM2 (with caution)
```

---

## Performance Comparison

| Tracker   | Speed (GPU) | ID Switches | Memory | Best For |
|-----------|-------------|-------------|--------|----------|
| ByteTrack | ⚡⚡⚡ 100 FPS | ~5-10% | 2GB | Standard matches |
| BoT-SORT  | ⚡⚡ 50 FPS | ~1-3% | 4GB | High accuracy |
| SAM2      | ⚡ 10 FPS | ~2-5% | 16GB | Research |

*Benchmarks based on 1080p video, YOLO detection model, RTX 3090*

---

## Advanced Configuration

### Tuning ByteTrack
```python
# In config.yaml or via environment
TRACK_THRESH: 0.5       # Detection confidence threshold
TRACK_BUFFER: 30        # Frames to keep lost tracks
MATCH_THRESH: 0.8       # IoU threshold for matching
```

### Tuning BoT-SORT
```python
TRACK_THRESH: 0.5
TRACK_BUFFER: 30
MATCH_THRESH: 0.8
PROXIMITY_THRESH: 0.5   # ReID appearance similarity
APPEARANCE_THRESH: 0.25 # ReID confidence threshold
```

### Tuning SAM2
```python
SAM2_MODEL: "large"     # Model variant
SAM2_CONF: 0.4          # Segmentation confidence
```

---

## Troubleshooting

### Problem: Too many ID switches
**Solution**: Switch from ByteTrack → BoT-SORT

### Problem: Slow processing
**Solution**:
1. Try ByteTrack instead of BoT-SORT
2. Reduce video resolution
3. Increase `--vid_stride` to skip frames

### Problem: Ghost IDs (duplicate players)
**Solution**:
1. Check edge filtering is enabled (should be default)
2. Verify detection confidence threshold (`DET_CONF` in config.yaml)
3. Use BoT-SORT for better disambiguation

### Problem: Lost tracks after occlusion
**Solution**:
1. Increase `TRACK_BUFFER` to 45-60 frames
2. Switch to BoT-SORT for ReID recovery
3. Reduce `--vid_stride` to process more frames

---

## Recommendations by Video Type

### Professional Broadcast (1080p+, steady camera)
→ **ByteTrack** (speed) or **BoT-SORT** (accuracy)

### Amateur/Phone Recording (shaky, variable quality)
→ **BoT-SORT** (handles motion better)

### Tactical Cam (bird's eye view)
→ **ByteTrack** (minimal occlusions)

### Highlight Clips (many cuts)
→ **BoT-SORT** (better across-cut matching)

### Live Streaming
→ **ByteTrack** (lowest latency)

---

## Future Trackers

Planned additions:
- **DeepSORT**: Classic ReID tracker
- **StrongSORT**: Enhanced BoT-SORT variant
- **OC-SORT**: Observation-centric SORT

---

## Support

For issues or questions about tracker selection:
1. Check logs in `output/pipeline.log`
2. Review health metrics: `output/health_snapshot.json`
3. Report issues with video characteristics and chosen tracker

**Last Updated**: 2026-01-29
