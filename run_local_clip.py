
import os
import sys
from orchestrator import process_spaces_video

# Define the local video item
video_item = {
    "id": "clipped_ikorudo_tornadoes",
    "filename": "clipped_ikorudo_tornadoes.mp4",
    "spacesURL": "/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4", # Use local absolute path
    "fileLocation": "matches_upload/manual/clipped_ikorudo_tornadoes.mp4"
}

print(f"🚀 Launching Manual Run for: {video_item['filename']}")

# Run Orchestrator Wrapper
result = process_spaces_video(
    video_item, 
    save_local=True, 
    no_db=True,   # Don't try to update DB
    max_frames=None, # Full clip
    locking_mode=2,
    jnr_stride=60,   # User requested
    # vid_stride removed - sparse sampling breaks stats attribution
    make_video=True  # Make debug video if needed, or False for speed
)

print("✅ Run Complete!")
print(result)
