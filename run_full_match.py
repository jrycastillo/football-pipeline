
import os
import sys
from orchestrator import process_spaces_video

# Define the full match video
video_item = {
    "id": "full_match_ikorudo_tornadoes",
    "filename": "full-match-ikorudo-niger-tornadoes.mp4",
    "spacesURL": "/home/ubuntu/videoforprocessing_link/full-match-ikorudo-niger-tornadoes.mp4",
    "fileLocation": "matches_upload/manual/full-match-ikorudo-niger-tornadoes.mp4"
}

print(f"🚀 Processing Full Match: {video_item['filename']} (3.4GB)")
print("📊 Settings: jnr_stride=60, no video output, DB dump enabled")

# Run Orchestrator Wrapper
result = process_spaces_video(
    video_item, 
    save_local=True, 
    no_db=False,      # ENABLE DB dump
    max_frames=None,  # Full video
    locking_mode=2,
    jnr_stride=60,    # User requested
    make_video=False  # NO video output for speed
)

print("✅ Run Complete!")
print(result)
