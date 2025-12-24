import os
import time
import subprocess
import shutil
import gc
import requests
import json
from urllib.parse import urlparse

# ======================= CONFIG =======================
SBG_BASE      = os.getenv("SBG_BASE", "https://api-staging.scoutbridge.net/football-gallery/api").rstrip("/")
SBG_LIST_URL  = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN     = os.getenv("SBG_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjY1NzgwNDQsInN1YiI6ImFudG9uaW9qaGFuY2Vkcmljays1QGdtYWlsLmNvbSIsInVzZXJfaWQiOiJlZTIyYWFlNiJ9.YZ4r1Ahp--tX0h2rO-FnV8ZusfOshPZO7sn4Kat_J1E")

def fetch_queue():
    headers = {"Authorization": f"Bearer {SBG_TOKEN}"}
    try:
        print(f"[batch] Fetching queue from {SBG_LIST_URL}...")
        resp = requests.get(SBG_LIST_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle Output Structure
        items = []
        if isinstance(data, list): items = data
        elif isinstance(data, dict):
            if "items" in data: items = data["items"]
            elif "data" in data: items = data["data"]
            elif "files" in data: items = data["files"]
            else: items = [] # Fallback
            
        return items
    except Exception as e:
        print(f"[batch] API Fetch Failed: {e}")
        return []

def process_pipeline_stream(video_url, match_name):
    # 1. Output Dir
    final_out_dir = os.path.join("output", match_name)
    os.makedirs(final_out_dir, exist_ok=True)
    
    # 2. Cleanup
    if os.path.exists("output/players_stats.json"): os.remove("output/players_stats.json")
    if os.path.exists("output/events.json"): os.remove("output/events.json")
    subprocess.run("rm -f output/temp_stats_chunk_*.json", shell=True)
    
    log_file = os.path.join(final_out_dir, "pipeline.log")
    
    try:
        # 3. Run Pipeline (Headless, Streaming)
        # We pass the URL as --local_video. Orchestrator now accepts URLs.
        cmd = [
            "python3", "-u", "orchestrator.py",
            "--local_video", video_url,
            "--save_local"
            # "--no_db" REMOVED for Production Run (DB Enabled)
        ]
        
        with open(log_file, "w") as f:
            print(f"[batch] Starting Pipeline for {match_name} (STREAMING)...")
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            
        print("[batch] Pipeline Success.")
        
        # 4. Move Artifacts
        if os.path.exists("output/players_stats.json"):
            shutil.move("output/players_stats.json", os.path.join(final_out_dir, "players_stats.json"))
        if os.path.exists("output/events.json"):
            shutil.move("output/events.json", os.path.join(final_out_dir, "events.json"))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[batch] ERROR: Pipeline Failed. Check {log_file}")
        return False
        
    except Exception as e:
        print(f"[batch] ERROR: Unexpected Exception: {e}")
        return False
        
    finally:
         # 5. Cleanup
         print("[batch] Cleaning up chunks...")
         subprocess.run("rm -f output/temp_stats_chunk_*.json", shell=True)
         gc.collect()

def main():
    if not os.path.exists("output"): os.makedirs("output")

    # 1. Fetch
    queue = fetch_queue()
    if not queue:
        print("[batch] Queue is empty or fetch failed.")
        return

    print(f"[batch] Found {len(queue)} items from API.")
    
    for i, item in enumerate(queue):
        # Extract Info
        filename = item.get("filename")
        spaces_url = item.get("spacesURL")
        
        if not filename or not spaces_url:
            print(f"[batch] Skipping Item {i}: Missing filename/url. {item.keys()}")
            continue
            
        # Match Name from Filename
        match_name = os.path.splitext(filename)[0].replace(" ", "_")
        print(f"\n[batch] Processing Item {i+1}/{len(queue)}: {match_name}")
        
        # Process (Stream)
        process_pipeline_stream(spaces_url, match_name)

if __name__ == "__main__":
    main()
