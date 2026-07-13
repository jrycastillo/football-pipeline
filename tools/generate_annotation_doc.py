#!/usr/bin/env python3
"""
Generate the PITCH_LANDMARK_ANNOTATION.md guide.
Imports coordinates directly from vision.pitch_homography to prevent manual entry errors.
"""

import os
import sys
import types

# Stub cv2 so we can import pitch_homography on local machine without issues
sys.modules["cv2"] = types.ModuleType("cv2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.pitch_homography import PITCH_VERTICES_M

DESCRIPTIONS = {
    0: "Left-top corner of the pitch (touchline / goal line intersection)",
    1: "Left penalty box top corner (intersection with goal line)",
    2: "Left goal box top corner (intersection with goal line)",
    3: "Left goal box bottom corner (intersection with goal line)",
    4: "Left penalty box bottom corner (intersection with goal line)",
    5: "Left-bottom corner of the pitch (touchline / goal line intersection)",
    6: "Left goal box front-top corner",
    7: "Left goal box front-bottom corner",
    8: "Left penalty spot",
    9: "Left penalty box front-top corner",
    10: "Left penalty box front (goal-box top line projection)",
    11: "Left penalty box front (goal-box bottom line projection)",
    12: "Left penalty box front-bottom corner",
    13: "Halfway line intersection with top touchline",
    14: "Center circle intersection with halfway line (top)",
    15: "Center circle intersection with halfway line (bottom)",
    16: "Halfway line intersection with bottom touchline",
    17: "Right penalty box front-top corner",
    18: "Right penalty box front (goal-box top line projection)",
    19: "Right penalty box front (goal-box bottom line projection)",
    20: "Right penalty box front-bottom corner",
    21: "Right penalty spot",
    22: "Right goal box front-top corner",
    23: "Right goal box front-bottom corner",
    24: "Right-top corner of the pitch (touchline / goal line intersection)",
    25: "Right penalty box top corner (intersection with goal line)",
    26: "Right goal box top corner (intersection with goal line)",
    27: "Right goal box bottom corner (intersection with goal line)",
    28: "Right penalty box bottom corner (intersection with goal line)",
    29: "Right-bottom corner of the pitch (touchline / goal line intersection)",
    30: "Center circle leftmost point (horizontal axis)",
    31: "Center circle rightmost point (horizontal axis)"
}

def main():
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PITCH_LANDMARK_ANNOTATION.md")
    
    lines = []
    lines.append("# Pitch Landmark Annotation Guide")
    lines.append("")
    lines.append("This document guides annotators in marking the 32 standard pitch landmarks for retraining the pitch keypoint detection model.")
    lines.append("")
    lines.append("## Annotation Rules")
    lines.append("")
    lines.append("1. **Index order is strict and fixed**: Always map the correct landmark to its exact index. Do not swap indexes. This order is canonical and matches the `PITCH_VERTICES_M` schema in `vision/pitch_homography.py` and the conversion logic in `tools/landmarks_to_yolo_pose.py` (Index 0 = left-top corner ... Index 31 = centre-circle-right).")
    lines.append("2. **Only annotate clearly visible landmarks**: If a landmark is out of frame or occluded by players/referees/objects, omit it from the annotation list entirely. Do not guess the position of occluded points.")
    lines.append("3. **Pixel coordinates**: Specify point coordinates in pixels, starting from `(0, 0)` at the top-left of the image.")
    lines.append("")
    lines.append("## Pitch Landmarks Schema")
    lines.append("")
    lines.append("| Index | Pitch Position (X, Y) in Meters | Plain-Language Description |")
    lines.append("|-------|---------------------------------|----------------------------|")
    
    for i, pt in enumerate(PITCH_VERTICES_M):
        desc = DESCRIPTIONS.get(i, "")
        lines.append(f"| {i} | ({pt[0]:.2f}, {pt[1]:.2f}) | {desc} |")
        
    lines.append("")
    lines.append("## Annotation JSON Format")
    lines.append("")
    lines.append("Annotations must be provided in a JSON file containing a list of objects. Each object represents an image and its corresponding annotated keypoints:")
    lines.append("")
    lines.append("```json")
    lines.append("[")
    lines.append("  {")
    lines.append('    "image": "clip1_f000120.jpg",')
    lines.append('    "points": {')
    lines.append('      "0": [120.5, 450.2],')
    lines.append('      "8": [960.0, 540.0],')
    lines.append('      "13": [1800.1, 120.4]')
    lines.append("    }")
    lines.append("  }")
    lines.append("]")
    lines.append("```")
    lines.append("")
    lines.append("- The `image` field should match the filename of the extracted frame.")
    lines.append("- The `points` dictionary maps the landmark index (as a string key) to its `[x, y]` pixel coordinates.")
    lines.append("- Unannotated / invisible keypoints must be omitted from the `points` dictionary.")
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"Generated {output_path}")

if __name__ == "__main__":
    main()
