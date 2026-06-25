# Models — Download & Setup

The model weights are **not** stored in this repo (too large). Download them from
the shared Google Drive folder and place them in a local `models/` directory.

**Google Drive folder:** https://drive.google.com/drive/folders/1lePXabD0EbDKVzN5eHTSTxKH6-pq74WL

## Active models the pipeline loads (required — ~395 MB)

| File | Size | Purpose | md5 |
|------|------|---------|-----|
| `yolo_player.pt` | 131 MB | YOLO player/GK/referee detection | `32ef0188b312e3fe85c7641eeee98f6d` |
| `yolo_ball.pt` | 131 MB | YOLO ball detection | `b1facb5fbcdb8b5effe393f134158c50` |
| `resnet34_clean.pt` | 82 MB | Jersey number recognition (ResNet34, 54 classes) | `071b78249b9dbe6d2586e04f24c02ed7` |
| `legibility_resnet18.pt` | 43 MB | Legibility gate (is a number readable?) | `af763e08023b9d1400bdb5106e90cbb3` |
| `osnet_x0_25.pth` | 9 MB | OSNet appearance ReID | `832c49169ef6733c43f9c9941ff278f3` |

## Setup

```bash
mkdir -p models
# Download the 5 files above into models/ from the Drive folder, then verify:
cd models && md5sum -c ../models.md5   # (optional integrity check)
```

After placing the files, the default `config.yaml` paths resolve correctly:
`models/yolo_player.pt`, `models/yolo_ball.pt`, `models/resnet34_clean.pt`, etc.

## Note on other weights
The Drive folder may also contain experimental iterations (PARSeq variants,
older ResNet checkpoints). These are **not needed** to run the pipeline — only
the 5 files above are loaded by default.
