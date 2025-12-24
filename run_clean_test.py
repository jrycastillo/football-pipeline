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
    
    # Find candidate ~50-100MB
    candidate_url = None
    for item in items:
        fname = item.get("filename", "Unknown")
        url_val = item.get("spacesURL", "")
        
        # TARGET SPECIFIC VIDEO
        if "09627712.mp4" in fname:
             print(f"FOUND TARGET VIDEO: {fname}")
             candidate_url = url_val
             break
        
        # Fallback logic (commented out for now to ensure we get the specific one if possible)
        # size_mb = item.get("fileSize", 0) / (1024 * 1024)
        # if 10 <= size_mb <= 500: ...
    
    if not candidate_url:
        print("No suitable video found.")
        return

    # Construct Command
    cmd = [
        "python", "orchestrator.py",
        "--local_video", candidate_url,
        "--save_local",
        "--make_video",
        "--max_frames", "400",
        "--jnr_stride", "10",
        "--locking_mode", "1",
        "--no_db"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    with open("clean_test_v12.log", "w") as log_file:
         subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    run()
