
import os
import sys
import argparse
from orchestrator import process_spaces_video

def main():
    parser = argparse.ArgumentParser(description="Run Local Clip with Orchestrator Wrapper")
    parser.add_argument("--tracking_mode", type=str, default="bytetrack", choices=["bytetrack", "sam2", "botsort"])
    parser.add_argument("--sam2_model", type=str, default="large", choices=["large", "base", "small", "tiny"])
    parser.add_argument("--locking_mode", type=int, default=3)
    parser.add_argument("--jnr_stride", type=int, default=15)
    args = parser.parse_args()

    # Define the local video item
    video_item = {
        "id": "clipped_ikorudo_tornadoes",
        "filename": "clipped_ikorudo_tornadoes.mp4",
        "spacesURL": "/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4", 
        "fileLocation": "matches_upload/manual/clipped_ikorudo_tornadoes.mp4"
    }

    print(f"🚀 Launching Manual Run for: {video_item['filename']}")
    print(f"📡 Mode: {args.tracking_mode} (Model: {args.sam2_model})")

    # Run Orchestrator Wrapper
    result = process_spaces_video(
        video_item, 
        save_local=True, 
        no_db=True, 
        max_frames=None,
        locking_mode=args.locking_mode,
        jnr_stride=args.jnr_stride,
        tracking_mode=args.tracking_mode,
        sam2_model=args.sam2_model,
        make_video=True
    )

    print("✅ Run Complete!")
    print(result)

if __name__ == "__main__":
    main()
