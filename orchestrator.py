import os
import time
import json
import uuid
import requests
import pymysql
import yaml
import torch
import traceback
import argparse
import sys
import subprocess
from pymysql.cursors import DictCursor

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Environment Variables (Override Config if set)
MYSQL_HOST = os.getenv("MYSQL_HOST", CONFIG["env"]["MYSQL_HOST"])
MYSQL_PORT = int(os.getenv("MYSQL_PORT", CONFIG["env"]["MYSQL_PORT"]))
MYSQL_USER = os.getenv("MYSQL_USER", CONFIG["env"]["MYSQL_USER"])
MYSQL_PASS = os.getenv("MYSQL_PASSWORD", "***REMOVED_SECRET***") # Hardcoded fallback from legacy
MYSQL_DB = os.getenv("MYSQL_DB", CONFIG["env"]["MYSQL_DB"])
ANALYSIS_TABLE = os.getenv("TABLE_NAME", CONFIG["env"]["TABLE_NAME"])

SBG_BASE = os.getenv("SBG_BASE", CONFIG["env"]["SBG_BASE"]).rstrip("/")
SBG_LIST_URL = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN = os.getenv("SBG_TOKEN", CONFIG["env"]["SBG_TOKEN"])

# Global flag for DB connection
NO_DB = False

# Database Helpers
def _conn():
    if NO_DB: return None
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=DictCursor, autocommit=True
    )

def _sha1(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def upsert_status_row(matches_video_id, user_id, source_url, status, task_id,
                      validation_status_id=None, analysis=None, error=None):
    if NO_DB:
        print(f"[db-mock] Upserting status: {status} for {matches_video_id} (Error: {error})")
        return

    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    mv_id_num = int(matches_video_id) if (matches_video_id is not None and str(matches_video_id).isdigit()) else None
    payload = analysis.copy() if isinstance(analysis, dict) else (analysis or {})
    if isinstance(payload, dict):
        payload.setdefault("matches_video_key", str(matches_video_id) if matches_video_id is not None else None)
        payload.setdefault("source_url", source_url)
        payload.setdefault("user_id", user_id)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",",":")) if payload is not None else None

    try:
        sel_sql = f"SELECT id FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(sel_sql, (unique_id,))
            row = cur.fetchone()
            if row:
                upd_sql = f"""
                UPDATE {ANALYSIS_TABLE}
                   SET matches_video_id=%s, user_id=%s, validation_status_id=%s,
                       source_url=%s, task_id=%s, status=%s, analysis=CAST(%s AS JSON),
                       error=%s, updated_at=NOW()
                 WHERE id=%s
                """
                cur.execute(upd_sql, (mv_id_num, user_id, validation_status_id, source_url,
                                      task_id, status, payload_json, error, row["id"]))
            else:
                ins_sql = f"""
                INSERT INTO {ANALYSIS_TABLE}
                  (matches_video_id, user_id, unique_id, validation_status_id,
                   source_url, task_id, status, analysis, error, created_at, updated_at)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS JSON), %s, NOW(), NOW())
                """
                cur.execute(ins_sql, (mv_id_num, user_id, unique_id, validation_status_id,
                                      source_url, task_id, status, payload_json, error))
    except Exception as e:
        print(f"[db] Failed to upsert status: {e}")

def is_video_processed(matches_video_id, source_url):
    if NO_DB: return False
    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    try:
        with _conn() as conn, conn.cursor() as cur:
            sql = f"SELECT status FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
            cur.execute(sql, (unique_id,))
            row = cur.fetchone()
            if row and row["status"] == "finished":
                return True
    except Exception as e:
        print(f"[db] Error checking status: {e}")
    return False

def run_pipeline(video_path, output_dir, max_frames=None, no_db=False, video_id=None, user_id=None, spaces_url=None, 
                 locking_mode=2, jnr_stride=None, make_video=False):
    """
    Unified metadata-aware pipeline wrapper.
    Delegates to pipeline_consolidated.py and handles DB updates.
    """
    import cv2
    import tempfile
    
    local_video_path = video_path
    use_streaming = True
    temp_video_path = None
    filename = os.path.basename(video_path.split("?")[0]) or "video.mp4"

    # 1. Download/Streaming Hybrid Logic
    if video_path.startswith("http"):
        print(f"[pipeline] Attempting to stream: {video_path[:50]}...")
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            success_count = 0
            for _ in range(3):
                ret, _ = cap.read()
                if ret: success_count += 1
            cap.release()
            if success_count >= 2:
                print(f"[pipeline] Streaming verified.")
                local_video_path = video_path
            else:
                use_streaming = False
        else:
            use_streaming = False

        if not use_streaming:
            print(f"[pipeline] Streaming failed/unstable. Downloading to temp...")
            temp_dir = tempfile.mkdtemp(prefix="pf_")
            temp_video_path = os.path.join(temp_dir, filename)
            try:
                print(f"[pipeline] FULL URL: {video_path}")
                resp = requests.get(video_path, stream=True, timeout=300)
                resp.raise_for_status()
                with open(temp_video_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                local_video_path = temp_video_path
                print(f"[pipeline] Downloaded to {temp_video_path}")
            except Exception as e:
                print(f"[pipeline] Download Failed: {e}")
                if not no_db:
                    upsert_status_row(video_id, user_id, spaces_url or video_path, "error", None, error=f"Download failed: {e}")
                return False

    # 2. Initial status update (In Progress)
    if not no_db:
        print(f"[pipeline] Setting In Progress for {video_id}...")
        upsert_status_row(video_id, user_id, spaces_url or video_path, "in_progress", None)

    try:
        # 3. Execute Subprocess
        cmd = [
            sys.executable, "pipeline_consolidated.py",
            "--video", local_video_path,
            "--output_dir", output_dir,
            "--locking_mode", str(locking_mode)
        ]
        if not make_video:
            cmd.append("--no_video_output")
        
        if max_frames:
            cmd.extend(["--max_frames", str(max_frames)])
        if jnr_stride:
            cmd.extend(["--jnr_stride", str(jnr_stride)])
            
        print(f"[pipeline] Executing Core: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=os.environ, timeout=7200)
        
        if result.returncode != 0:
            print(f"[pipeline] Core failed with code {result.returncode}")
            if not no_db:
                upsert_status_row(video_id, user_id, spaces_url or video_path, "error", None, error=f"Core exit {result.returncode}")
            return False

        # 3. Handle Results & DB
        stats_path = os.path.join(output_dir, "player_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
            
            if not no_db:
                print(f"[pipeline] Updating DB for {video_id}...")
                upsert_status_row(video_id, user_id, spaces_url or video_path, "finished", None, analysis={"stats": stats_data})
            return True
        else:
            print(f"[pipeline] Missing player_stats.json in {output_dir}")
            return False

    except Exception as e:
        print(f"[pipeline] Error: {e}")
        if not no_db:
            upsert_status_row(video_id, user_id, spaces_url or video_path, "error", None, error=str(e))
        return False
    finally:
        # Cleanup temp
        if temp_video_path and os.path.exists(temp_video_path):
            print(f"[pipeline] Cleaning up temp file...")
            try:
                os.remove(temp_video_path)
                os.rmdir(os.path.dirname(temp_video_path))
            except: pass

# Polling Logic
def fetch_pending_videos():
    """Fetch list of pending videos from ScoutBridge API."""
    print(f"[poll] Fetching from {SBG_LIST_URL}...")
    headers = {
        "Authorization": f"Bearer {SBG_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.get(SBG_LIST_URL, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("items", [])
        else:
            print(f"[poll] Error fetching videos: {resp.status_code} - {resp.text}")
            return []
    except Exception as e:
        print(f"[poll] Exception fetching videos: {e}")
        return []


def process_spaces_video(video_item, save_local=True, no_db=True, max_frames=None, locking_mode=2, jnr_stride=None, make_video=False):
    """
    Process a single video from SPACES using the unified run_pipeline wrapper.
    """
    video_id = video_item.get("id", "unknown")
    spaces_url = video_item.get("spacesURL")
    filename = video_item.get("filename", "video.mp4")
    
    # Extract UserID from fileLocation or filename
    # Usually: matches_upload/{userID}/{filename}
    file_loc = video_item.get("fileLocation", "")
    if "/" in file_loc:
        user_id = file_loc.split("/")[1]
    else:
        user_id = filename.split("_")[0] if "_" in filename else "unknown"
    
    print(f"[process] Starting: {filename}")
    print(f"[process] Video ID: {video_id}, User ID: {user_id}")
    
    if not spaces_url:
        print(f"[process] Error: No spacesURL for {video_id}")
        return {"status": "error", "error": "No spacesURL"}
    
    # Create output directory
    out_dir = f"./output/{video_id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Process using the new unified wrapper
    success = run_pipeline(
        video_path=spaces_url,
        output_dir=out_dir,
        max_frames=max_frames,
        no_db=no_db,
        video_id=video_id,
        user_id=user_id,
        spaces_url=spaces_url,
        locking_mode=locking_mode,
        jnr_stride=jnr_stride,
        make_video=make_video
    )
    
    if success:
        return {"status": "success", "video_id": video_id, "stats_path": os.path.join(out_dir, "player_stats.json")}
    else:
        return {"status": "error", "video_id": video_id, "error": "Pipeline failed"}


def start_polling_loop(poll_interval=60, max_videos=None, min_size_mb=0, max_size_mb=float('inf'), 
                       locking_mode=2, jnr_stride=None, make_video=False):
    """
    Continuously poll for pending videos and process them.
    
    Args:
        poll_interval: Seconds between polls (default 60)
        max_videos: Max videos to process before stopping (None = infinite)
        min_size_mb: Minimum file size filter in MB
        max_size_mb: Maximum file size filter in MB
        locking_mode: Mode passed down to pipeline
        jnr_stride: Stride passed down to pipeline
    """
    print(f"[poll] Starting polling loop (interval={poll_interval}s)...")
    print(f"[poll] Size filter: {min_size_mb}MB - {max_size_mb}MB")
    
    processed_count = 0
    processed_ids = set()
    
    while True:
        try:
            videos = fetch_pending_videos()
            
            if not videos:
                print(f"[poll] No pending videos. Waiting {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            
            # Filter by size and already processed
            for video in videos:
                video_id = video.get("id")
                file_size_mb = video.get("fileSize", 0) / 1024 / 1024
                
                # Skip already processed
                if video_id in processed_ids:
                    continue
                
                # Skip if outside size range
                if file_size_mb < min_size_mb or file_size_mb > max_size_mb:
                    continue
                
                # Skip if already in DB
                if is_video_processed(video_id, video.get("spacesURL")):
                    print(f"[poll] Already processed: {video_id}")
                    processed_ids.add(video_id)
                    continue
                
                # Process this video
                result = process_spaces_video(
                    video, save_local=True, no_db=NO_DB,
                    locking_mode=locking_mode,
                    jnr_stride=jnr_stride,
                    make_video=make_video
                )
                processed_ids.add(video_id)
                processed_count += 1
                
                if result.get("status") == "success":
                    print(f"[poll] Successfully processed {processed_count} videos")
                
                # Check max limit
                if max_videos and processed_count >= max_videos:
                    print(f"[poll] Reached max_videos limit ({max_videos}). Stopping.")
                    return
                
                # Small delay between videos
                time.sleep(5)
            
            # Wait before next poll
            print(f"[poll] Cycle complete. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("[poll] Interrupted by user. Stopping.")
            break
        except Exception as e:
            print(f"[poll] Error in polling loop: {e}")
            traceback.print_exc()
            time.sleep(poll_interval)


# ... (process_video remains mostly unchanged, but we could update it if needed, though this request is for debug mode)

def main():
    global NO_DB
    
    parser = argparse.ArgumentParser(description="Football Pipeline Orchestrator")
    parser.add_argument("--local_video", type=str, help="Path to local video file or SPACES URL for debug mode")
    parser.add_argument("--no_db", action="store_true", help="Skip DB connections")
    parser.add_argument("--save_local", action="store_true", help="Save output to ./output folder (or --output_dir)")
    parser.add_argument("--make_video", action="store_true", help="Generate debug video output")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to process")
    parser.add_argument("--resume_frame", type=int, default=0, help="Start processing from this frame index")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    
    # Polling mode arguments
    parser.add_argument("--poll", action="store_true", help="Enable polling mode to fetch from SPACES")
    parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between polls (default 60)")
    parser.add_argument("--max_videos", type=int, help="Max videos to process before stopping")
    parser.add_argument("--min_size_mb", type=float, default=0, help="Minimum video size in MB")
    parser.add_argument("--max_size_mb", type=float, default=float('inf'), help="Maximum video size in MB")
    parser.add_argument("--locking_mode", type=int, choices=[1, 2], default=2, help="Internal pipeline locking mode")
    parser.add_argument("--jnr_stride", type=int, help="Internal pipeline JNR stride (frames)")
    
    args = parser.parse_args()
    
    if args.no_db:
        NO_DB = True
        print("[debug] DB connections disabled.")
        
    if args.local_video:
        # DEBUG PATH - Single video
        if not args.local_video.startswith("http") and not os.path.exists(args.local_video):
            print(f"Error: Video file {args.local_video} not found.")
            sys.exit(1)
            
        print(f"Running Debug Mode on {args.local_video}")
        
        # Determine output directory
        if args.output_dir:
            out_dir = args.output_dir
        elif args.save_local:
            out_dir = "./output"
        else:
            out_dir = "."
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        base_name = os.path.splitext(os.path.basename(args.local_video.split("?")[0]))[0]
        
        try:
            # We use the new unified wrapper
            success = run_pipeline(
                video_path=args.local_video, 
                output_dir=out_dir,
                no_db=args.no_db, 
                max_frames=args.max_frames,
                video_id=os.path.splitext(os.path.basename(args.local_video))[0].split('?')[0],
                user_id="local_user",
                locking_mode=args.locking_mode,
                jnr_stride=args.jnr_stride,
                make_video=args.make_video
            )
            if success:
                print(f"Finished successfully. Output in {out_dir}")
            else:
                print("Pipeline execution failed.")
                sys.exit(1)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        sys.exit(0)
        
    elif args.poll:
        # POLLING MODE - Fetch from SPACES and process
        print("[main] Starting SPACES polling mode...")
        start_polling_loop(
            poll_interval=args.poll_interval,
            max_videos=args.max_videos,
            min_size_mb=args.min_size_mb,
            max_size_mb=args.max_size_mb,
            locking_mode=args.locking_mode,
            jnr_stride=args.jnr_stride,
            make_video=args.make_video
        )
    else:
        # Show help if no mode specified
        parser.print_help()
        print("\n[main] Use --local_video for single video or --poll for SPACES polling mode.")

if __name__ == "__main__":
    main()
