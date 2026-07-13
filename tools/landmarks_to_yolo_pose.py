#!/usr/bin/env python3
"""
Convert landmark annotations JSON to YOLO-pose dataset format.

Splits data into train/val subsets, computes bounding boxes from the convex hull
of annotated keypoints, pads them by 5%, normalizes coordinates, and writes
YOLO pose format labels and a corresponding data.yaml.
"""

import os
import sys
import json
import random
import shutil
import argparse
from PIL import Image

def convert_annotation_to_yolo_line(points_dict, w_img, h_img):
    """
    Convert a points dictionary of pixel coordinates to a YOLO-pose label line.
    
    points_dict: dict mapping landmark indices (str or int) to [x, y] in pixels.
    w_img: width of the image in pixels.
    h_img: height of the image in pixels.
    
    Returns:
        str: YOLO-pose formatted line starting with class 0.
    """
    if not points_dict:
        raise ValueError("Cannot convert an annotation with zero points.")
        
    # Convert keys to integers for sorted indexing
    points_int = {int(k): v for k, v in points_dict.items()}
    
    xs = [pt[0] for pt in points_int.values()]
    ys = [pt[1] for pt in points_int.values()]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    w_box = x_max - x_min
    h_box = y_max - y_min
    
    # Handle single point or degenerate box case
    if w_box <= 0:
        w_box = 1.0
    if h_box <= 0:
        h_box = 1.0
        
    x_center = x_min + w_box / 2
    y_center = y_min + h_box / 2
    
    # Pad by 5%
    w_padded = w_box * 1.05
    h_padded = h_box * 1.05
    
    # Clamp to image boundary
    x1 = max(0.0, x_center - w_padded / 2)
    y1 = max(0.0, y_center - h_padded / 2)
    x2 = min(float(w_img), x_center + w_padded / 2)
    y2 = min(float(h_img), y_center + h_padded / 2)
    
    w_final = x2 - x1
    h_final = y2 - y1
    xc_final = x1 + w_final / 2
    yc_final = y1 + h_final / 2
    
    # Normalize bbox
    xc_norm = xc_final / w_img
    yc_norm = yc_final / h_img
    w_norm = w_final / w_img
    h_norm = h_final / h_img
    
    # Keypoints processing (32 keypoints)
    kps_str = []
    for i in range(32):
        if i in points_int:
            px, py = points_int[i]
            px_norm = px / w_img
            py_norm = py / h_img
            pv = 2  # Annotated
        else:
            px_norm = 0.0
            py_norm = 0.0
            pv = 0  # Missing
        kps_str.append(f"{px_norm:.6f} {py_norm:.6f} {pv}")
        
    return f"0 {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(kps_str)

def main():
    parser = argparse.ArgumentParser(description="Convert landmark annotations JSON to YOLO pose format.")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations JSON file")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing the source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the YOLO pose dataset")
    parser.add_argument("--train_split", type=float, default=0.9, help="Fraction of dataset to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and splitting")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)
        
    with open(args.annotations, "r") as f:
        try:
            annotations = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON annotations: {e}")
            sys.exit(1)
            
    if not isinstance(annotations, list):
        print("Error: Annotations JSON must be a list of objects.")
        sys.exit(1)
        
    # Set seed for reproducible splitting
    random.seed(args.seed)
    random.shuffle(annotations)
    
    split_idx = int(len(annotations) * args.train_split)
    train_set = annotations[:split_idx]
    val_set = annotations[split_idx:]
    
    subsets = {
        "train": train_set,
        "val": val_set
    }
    
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    print(f"Loaded {len(annotations)} annotations. Splitting: {len(train_set)} train, {len(val_set)} val")
    
    for subset_name, subset_data in subsets.items():
        img_out_dir = os.path.join(abs_output_dir, "images", subset_name)
        lbl_out_dir = os.path.join(abs_output_dir, "labels", subset_name)
        
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for entry in subset_data:
            image_name = entry.get("image")
            points = entry.get("points")
            
            if not image_name or not points:
                print(f"Warning: Skipping invalid annotation entry: {entry}")
                continue
                
            src_img_path = os.path.join(args.images_dir, image_name)
            if not os.path.exists(src_img_path):
                print(f"Warning: Image file not found, skipping: {src_img_path}")
                continue
                
            try:
                with Image.open(src_img_path) as img:
                    w_img, h_img = img.size
            except Exception as e:
                print(f"Warning: Failed to open image {src_img_path} ({e}), skipping.")
                continue
                
            try:
                yolo_line = convert_annotation_to_yolo_line(points, w_img, h_img)
            except Exception as e:
                print(f"Warning: Failed to convert annotation for {image_name} ({e}), skipping.")
                continue
                
            # Copy image
            dest_img_path = os.path.join(img_out_dir, image_name)
            shutil.copy2(src_img_path, dest_img_path)
            
            # Write label file
            image_stem = os.path.splitext(image_name)[0]
            label_path = os.path.join(lbl_out_dir, f"{image_stem}.txt")
            with open(label_path, "w") as f:
                f.write(yolo_line + "\n")
                
    # Write data.yaml
    yaml_path = os.path.join(abs_output_dir, "data.yaml")
    yaml_lines = [
        f"path: {abs_output_dir}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        "  0: pitch",
        "",
        "kpt_shape: [32, 3]"
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")
        
    print(f"Dataset conversion complete. Saved to {abs_output_dir}")

if __name__ == "__main__":
    main()
