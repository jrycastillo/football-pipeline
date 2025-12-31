#!/usr/bin/env python3
"""
Process the largest video from SBG Spaces.
- Download from SBG API
- Process with Qwen JNR (jnr_stride=60)
- No video output (speed)
- Dump results to DB
"""

import os
import sys
from orchestrator import process_spaces_video

# The largest video from SBG Spaces
video_item = {
    "id": "605cb6dfa5784f0",
    "filename": "2e5f877b_2025-11-25-05-59-50838476.webm",
    "fileLocation": "matches_upload/605cb6dfa5784f0/2e5f877b_2025-11-25-05-59-50838476.webm"
}

# The spacesURL needs to be fetched from the API
import requests
import yaml

config = yaml.safe_load(open('config.yaml'))
sbg_base = config['env']['SBG_BASE'].rstrip('/')
sbg_token = config['env']['SBG_TOKEN']

# Get video details to fetch spacesURL
url = f'{sbg_base}/v2/files/list/video/for-match-analysis'
headers = {
    'Authorization': f'Bearer {sbg_token}',
    'Content-Type': 'application/json'
}

print("🔍 Fetching video list from SBG Spaces...")
resp = requests.get(url, headers=headers, timeout=30)
if resp.status_code != 200:
    print(f"❌ Failed to fetch video list: {resp.status_code}")
    sys.exit(1)

# Find our video
items = resp.json().get('items', [])
target_video = None
for v in items:
    if v.get('id') == '605cb6dfa5784f0':
        target_video = v
        break

if not target_video:
    print("❌ Could not find video 605cb6dfa5784f0")
    sys.exit(1)

# Update video_item with spacesURL
video_item['spacesURL'] = target_video.get('spacesURL')
size_mb = target_video.get('fileSize', 0) / 1024 / 1024

print(f"🚀 Processing: {video_item['filename']}")
print(f"📊 Size: {size_mb:.1f}MB")
print(f"📊 Settings: jnr_stride=60, no video output, DB dump enabled")

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
