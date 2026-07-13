# Pitch Landmark Annotation Guide

This document guides annotators in marking the 32 standard pitch landmarks for retraining the pitch keypoint detection model.

## Annotation Rules

1. **Index order is strict and fixed**: Always map the correct landmark to its exact index. Do not swap indexes. This order is canonical and matches the `PITCH_VERTICES_M` schema in `vision/pitch_homography.py` and the conversion logic in `tools/landmarks_to_yolo_pose.py` (Index 0 = left-top corner ... Index 31 = centre-circle-right).
2. **Only annotate clearly visible landmarks**: If a landmark is out of frame or occluded by players/referees/objects, omit it from the annotation list entirely. Do not guess the position of occluded points.
3. **Pixel coordinates**: Specify point coordinates in pixels, starting from `(0, 0)` at the top-left of the image.

## Pitch Landmarks Schema

| Index | Pitch Position (X, Y) in Meters | Plain-Language Description |
|-------|---------------------------------|----------------------------|
| 0 | (0.00, 0.00) | Left-top corner of the pitch (touchline / goal line intersection) |
| 1 | (0.00, 13.50) | Left penalty box top corner (intersection with goal line) |
| 2 | (0.00, 24.84) | Left goal box top corner (intersection with goal line) |
| 3 | (0.00, 43.16) | Left goal box bottom corner (intersection with goal line) |
| 4 | (0.00, 54.50) | Left penalty box bottom corner (intersection with goal line) |
| 5 | (0.00, 68.00) | Left-bottom corner of the pitch (touchline / goal line intersection) |
| 6 | (5.50, 24.84) | Left goal box front-top corner |
| 7 | (5.50, 43.16) | Left goal box front-bottom corner |
| 8 | (11.00, 34.00) | Left penalty spot |
| 9 | (20.15, 13.50) | Left penalty box front-top corner |
| 10 | (20.15, 24.84) | Left penalty box front (goal-box top line projection) |
| 11 | (20.15, 43.16) | Left penalty box front (goal-box bottom line projection) |
| 12 | (20.15, 54.50) | Left penalty box front-bottom corner |
| 13 | (52.50, 0.00) | Halfway line intersection with top touchline |
| 14 | (52.50, 24.85) | Center circle intersection with halfway line (top) |
| 15 | (52.50, 43.15) | Center circle intersection with halfway line (bottom) |
| 16 | (52.50, 68.00) | Halfway line intersection with bottom touchline |
| 17 | (84.85, 13.50) | Right penalty box front-top corner |
| 18 | (84.85, 24.84) | Right penalty box front (goal-box top line projection) |
| 19 | (84.85, 43.16) | Right penalty box front (goal-box bottom line projection) |
| 20 | (84.85, 54.50) | Right penalty box front-bottom corner |
| 21 | (94.00, 34.00) | Right penalty spot |
| 22 | (99.50, 24.84) | Right goal box front-top corner |
| 23 | (99.50, 43.16) | Right goal box front-bottom corner |
| 24 | (105.00, 0.00) | Right-top corner of the pitch (touchline / goal line intersection) |
| 25 | (105.00, 13.50) | Right penalty box top corner (intersection with goal line) |
| 26 | (105.00, 24.84) | Right goal box top corner (intersection with goal line) |
| 27 | (105.00, 43.16) | Right goal box bottom corner (intersection with goal line) |
| 28 | (105.00, 54.50) | Right penalty box bottom corner (intersection with goal line) |
| 29 | (105.00, 68.00) | Right-bottom corner of the pitch (touchline / goal line intersection) |
| 30 | (43.35, 34.00) | Center circle leftmost point (horizontal axis) |
| 31 | (61.65, 34.00) | Center circle rightmost point (horizontal axis) |

## Annotation JSON Format

Annotations must be provided in a JSON file containing a list of objects. Each object represents an image and its corresponding annotated keypoints:

```json
[
  {
    "image": "clip1_f000120.jpg",
    "points": {
      "0": [120.5, 450.2],
      "8": [960.0, 540.0],
      "13": [1800.1, 120.4]
    }
  }
]
```

- The `image` field should match the filename of the extracted frame.
- The `points` dictionary maps the landmark index (as a string key) to its `[x, y]` pixel coordinates.
- Unannotated / invisible keypoints must be omitted from the `points` dictionary.

