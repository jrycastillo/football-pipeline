import requests
import os
import yaml
import subprocess
import sys

def run():
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    token = os.getenv("SBG_TOKEN", config["env"]["SBG_TOKEN"])
    url = "https://api-staging.scoutbridge.net/football-gallery/api/v2/files/list/video/for-match-analysis"
    headers = {"Authorization": f"Bearer {token}"}

    print("Fetching video list...")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to fetch videos: {resp.text}")
        return

    items = resp.json().get("items", [])
    
    # Find candidate ~80-150MB
    candidate_url = None
    target_video_name = None
    
    print("Searching for candidate video (80MB - 150MB)...")
    
    for item in items:
        fname = item.get("filename", "Unknown")
        url_val = item.get("spacesURL", "")
        size_bytes = item.get("fileSize", 0)
        size_mb = size_bytes / (1024 * 1024)
        
        # Valid range: 80MB to 150MB
        if 80 <= size_mb <= 150:
             print(f"FOUND VIDEO: {fname} ({size_mb:.2f} MB)")
             candidate_url = url_val
             target_video_name = fname
             break
    
    if not candidate_url:
        print("No suitable video ~100MB found. Checking for any video > 50MB...")
        # Fallback
        for item in items:
             size_bytes = item.get("fileSize", 0)
             size_mb = size_bytes / (1024 * 1024)
             if size_mb > 50:
                 candidate_url = item.get("spacesURL", "")
                 print(f"Fallback selection: {item.get('filename')} ({size_mb:.2f} MB)")
                 break
                 
    if not candidate_url:
        print("No videos found.")
        return

    # Check for stream availability
    print(f"Verifying stream: {candidate_url[:60]}...")
    try:
        r = requests.get(candidate_url, stream=True)
        if r.status_code != 200:
             print(f"Selected URL is not accessible (Status {r.status_code}). Skipping.")
             return
        r.close()
    except Exception as e:
        print(f"Stream verification failed: {e}")
        return

    # Construct Command - Full Video (No max_frames)
    cmd = [
        "python", "orchestrator.py",
        "--local_video", candidate_url,
        "--save_local",
        "--make_video",
        "--max_frames", "2000",
        "--jnr_stride", "10",
        "--locking_mode", "1",
        "--no_db"
    ]
    
    print(f"Running full test on: {target_video_name}")
    print(f"Command: {' '.join(cmd)}")
    
    with open("clean_test_v16.log", "w") as log_file:
         subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    run()
