# Model Setup Guide

This pipeline requires 4 specific model weight files to function. 

## 1. Download Models
Download the models from the shared Google Drive:
[Google Drive Link](https://drive.google.com/drive/folders/1lePXabD0EbDKVzN5eHTSTxKH6-pq74WL?usp=sharing)

## 2. Model Files
You should have the following 4 files:

| File Name | Description | Size |
|-----------|-------------|------|
| `resnet34_rgb_jnr.pt` | **JNR**: RGB-Fine-Tuned ResNet34 for Jersey Number Recog | ~82MB |
| `yolo_player.pt` | **Detection**: YOLOv8 Player/Person Detection | ~131MB |
| `yolo_ball.pt` | **Ball**: YOLOv8 Ball Detection | ~131MB |
| `yolo_pitch.pt` | **Pitch**: YOLOv8 Pitch Keypoint Detection | ~134MB |

## 3. Installation
1. Create a `models/` directory in the root of the project:
   ```bash
   mkdir -p models
   ```

2. Place the 4 `.pt` files inside the `models/` directory.

   Your structure should look like:
   ```
   football/
   ├── models/
   │   ├── resnet34_rgb_jnr.pt
   │   ├── yolo_player.pt
   │   ├── yolo_ball.pt
   │   └── yolo_pitch.pt
   ├── orchestrator.py
   ├── pipeline_consolidated.py
   └── ...
   ```

## 4. Configuration
If you placed the files in `models/` as above, update your `config.yaml` or pass arguments to the pipeline.

**Recommended:** The `pipeline_consolidated.py` script checks `models/` by default for some, but you may need to specify paths if they differ from legacy hardcoded paths.
