
import os
import sys
import subprocess
import requests
import yaml
import time

def get_valid_video_url():
    """Fetch a valid video URL using the ScoutBridge API."""
    try:
        if not os.path.exists("config.yaml"):
            print("Config not found.")
            return None
            
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        token = os.getenv("SBG_TOKEN", config["env"]["SBG_TOKEN"])
        url = "https://api-staging.scoutbridge.net/football-gallery/api/v2/files/list/video/for-match-analysis"
        headers = {"Authorization": f"Bearer {token}"}

        print("Fetching video list from API...")
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"API Failed: {resp.text}")
            return None

        items = resp.json().get("items", [])
        
        # Look for the ~100MB video we know works
        for item in items:
            size_bytes = item.get("fileSize", 0)
            size_mb = size_bytes / (1024 * 1024)
            if 80 <= size_mb <= 150:
                print(f"Selected Video: {item.get('filename')} ({size_mb:.2f} MB)")
                return item.get("spacesURL")
                
        # Fallback
        if items:
            return items[0].get("spacesURL")
            
    except Exception as e:
        print(f"Error fetching video: {e}")
        return None

def run_parallel_jobs():
    url = get_valid_video_url()
    if not url:
        print("Could not get a valid video URL. Exiting.")
        return

    # Simulate 3 different inputs by using the same heavy video
    videos = [
        (url, "Video_Instance_1"),
        (url, "Video_Instance_2"),
        (url, "Video_Instance_3")
    ]
    
    processes = []
    
    print(f"Launching {len(videos)} parallel pipelines on H100 GPU...")
    print("-" * 60)
    
    for idx, (target_url, label) in enumerate(videos):
        job_id = idx + 1
        output_dir = f"output_parallel_{job_id}"
        log_file = f"parallel_job_{job_id}.log"
        
        # Clean previous output
        subprocess.run(["rm", "-rf", output_dir])
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python", "orchestrator.py",
            "--local_video", target_url,
            "--output_dir", output_dir,
            "--save_local",
            "--make_video", 
            "--max_frames", "2000",
            "--jnr_stride", "10",
            "--locking_mode", "1",
            "--no_db"
        ]
        
        print(f"[Job {job_id}] Label: {label}")
        print(f"[Job {job_id}] Log: {log_file}")
        
        # Launch background process
        with open(log_file, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            processes.append((p, job_id, log_file))
            
    print("-" * 60)
    print("All jobs launched. Monitoring...")
    
    try:
        while True:
            all_done = True
            dataset = []
            for p, jid, logf in processes:
                if p.poll() is None:
                    all_done = False
                    # Peek at last line of log
                    try:
                        last_line = subprocess.check_output(["tail", "-n", "1", logf]).decode().strip()
                        if "Processing Frame" in last_line:
                            dataset.append(f"Job {jid}: {last_line[-30:]}")
                        else:
                            dataset.append(f"Job {jid}: Running...")
                    except:
                        dataset.append(f"Job {jid}: Starting...")
                else:
                    dataset.append(f"Job {jid}: DONE (Exit {p.returncode})")
            
            # Print status
            print(f"Status: { ' | '.join(dataset) }")
                
            if all_done:
                break
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nStopping all jobs...")
        for p, _, _ in processes:
            p.terminate()

if __name__ == "__main__":
    run_parallel_jobs()
