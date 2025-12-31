#!/usr/bin/env python3
"""
Batch Process ALL Videos from SBG Spaces
- Fetches all videos from SBG API
- Sorts by size (smallest first)
- Processes each sequentially
- Dumps results to DB after each
"""

import os
import sys
import time
import requests
import yaml
from orchestrator import process_spaces_video

def main():
    # Load config
    config = yaml.safe_load(open('config.yaml'))
    sbg_base = config['env']['SBG_BASE'].rstrip('/')
    sbg_token = config['env']['SBG_TOKEN']
    
    # Fetch video list
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
    
    items = resp.json().get('items', [])
    print(f"📊 Found {len(items)} videos")
    
    # Sort by size (smallest first)
    items_sorted = sorted(items, key=lambda x: x.get('fileSize', 0))
    
    print("\n📋 Processing order (smallest first):")
    for i, v in enumerate(items_sorted[:10]):
        size_mb = v.get('fileSize', 0) / 1024 / 1024
        print(f"  {i+1}. {v.get('filename', 'unknown')[:40]}... - {size_mb:.1f}MB")
    if len(items_sorted) > 10:
        print(f"  ... and {len(items_sorted) - 10} more videos")
    
    print("\n" + "=" * 60)
    print("🚀 STARTING BATCH PROCESSING")
    print("=" * 60 + "\n")
    
    processed = 0
    failed = 0
    start_time = time.time()
    
    for i, video in enumerate(items_sorted):
        video_id = video.get('id', 'unknown')
        filename = video.get('filename', 'video.mp4')
        spaces_url = video.get('spacesURL')
        size_mb = video.get('fileSize', 0) / 1024 / 1024
        file_location = video.get('fileLocation', '')
        
        print(f"\n{'=' * 60}")
        print(f"📹 VIDEO {i+1}/{len(items_sorted)}: {filename[:50]}")
        print(f"📊 Size: {size_mb:.1f}MB | ID: {video_id}")
        print(f"{'=' * 60}")
        
        if not spaces_url:
            print("⚠️ No spacesURL, skipping...")
            failed += 1
            continue
        
        video_item = {
            "id": video_id,
            "filename": filename,
            "spacesURL": spaces_url,
            "fileLocation": file_location
        }
        
        try:
            result = process_spaces_video(
                video_item, 
                save_local=True, 
                no_db=False,      # ENABLE DB dump
                max_frames=None,  # Full video
                locking_mode=2,
                jnr_stride=60,    # Optimized for speed
                make_video=False  # NO video output for speed
            )
            
            if result.get('status') == 'success':
                processed += 1
                print(f"✅ SUCCESS: {video_id}")
            else:
                failed += 1
                print(f"❌ FAILED: {video_id} - {result}")
                
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {video_id} - {str(e)[:100]}")
        
        # Progress report
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(items_sorted) - i - 1)
        print(f"\n📊 Progress: {processed} done, {failed} failed, {len(items_sorted) - i - 1} remaining")
        print(f"⏱️ Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🏁 BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✅ Processed: {processed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()
