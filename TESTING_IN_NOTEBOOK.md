# How to Test the Pipeline in Jupyter Notebook

Since the pipeline logic has been consolidated into `football_pipeline.py`, you can easily import and run it within your Jupyter Notebook. This ensures your notebook stays clean while using the robust, fixed logic.

## Option 1: Import and Run (Recommended)

Place `football_pipeline.py` in the same directory as your notebook.

```python
import football_pipeline as fp
import pandas as pd

# 1. Define your paths
VIDEO_PATH = "/path/to/your/video.mp4"
YOLO_WEIGHTS = "/path/to/yolov8x.pt"  # or your custom weights

# 2. Run the pipeline
# You can override any default parameters here
payload = fp.run_pipeline_to_players_json(
    video_path=VIDEO_PATH,
    weights_path=YOLO_WEIGHTS,
    out_json="results.json",
    out_csv="results.csv",
    device=0,              # 0 for GPU, 'cpu' for CPU
    conf=0.25,
    store_images=True,     # Needed for Jersey/Pitch features
    store_images_up_to=1500,
    vid_stride=1           # Increase to 2 or 4 for faster testing
)

# 3. View the results
df = pd.read_csv("results.csv")
print(f"Total Players: {len(df)}")
print(f"Total Goals Detected: {df['goals'].sum()}")

# Display first few rows
display(df.head())
```

## Option 2: Analyzing the Output

The pipeline returns a `payload` dictionary containing the raw data. You can explore it directly:

```python
# Check goal events specifically
import json

# Load if not in memory
# with open("results.json") as f:
#     payload = json.load(f)

players = payload["players_flat"]

# Filter players who scored
scorers = [p for p in players if p["goals"] > 0]

for p in scorers:
    print(f"Player {p['player_id']} (Team {p['team']}): {p['goals']} Goals")
```

## Option 3: Copy-Paste (If you want to edit logic in-place)

If you prefer to have all code in the notebook cells (e.g., for debugging), you can copy the entire content of `football_pipeline.py` into a cell.

**Note on Dependencies:**
Make sure you have the required libraries installed in your notebook environment:
```bash
!pip install ultralytics opencv-python numpy torch scipy
```
*(Note: `scipy` was replaced by `opencv` for clustering in the latest fix, but `scipy` might still be useful for other tasks).*
