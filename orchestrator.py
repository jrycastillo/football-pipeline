import os
import time
import random
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
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()
from utils.health_monitor import get_health_monitor

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Environment Variables (Override Config if set)
MYSQL_HOST = os.getenv("MYSQL_HOST", CONFIG["env"]["MYSQL_HOST"])
MYSQL_PORT = int(os.getenv("MYSQL_PORT", CONFIG["env"]["MYSQL_PORT"]))
MYSQL_USER = os.getenv("MYSQL_USER", CONFIG["env"]["MYSQL_USER"])
MYSQL_PASS = os.getenv("MYSQL_PASSWORD")  # Required: Set via environment variable
if not MYSQL_PASS:
    print("[WARNING] MYSQL_PASSWORD not set. Database operations will fail.")
MYSQL_DB = os.getenv("MYSQL_DB", CONFIG["env"]["MYSQL_DB"])
ANALYSIS_TABLE = os.getenv("TABLE_NAME", CONFIG["env"]["TABLE_NAME"])

SBG_BASE = os.getenv("SBG_BASE", CONFIG["env"]["SBG_BASE"]).rstrip("/")
SBG_LIST_URL = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN = os.getenv("SBG_TOKEN", CONFIG["env"]["SBG_TOKEN"])

# Global flag for DB connection
NO_DB = os.getenv("NO_DB", "0") == "1"

# Initialize health monitor
health = get_health_monitor()

# Database Helpers
def _conn():
    if NO_DB: return None
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=DictCursor, autocommit=True
    )

def _sha1(s: str) -> str:
    import hashlib
    from urllib.parse import urlparse
    # Strip query parameters from signed URLs (they contain timestamps)
    if s and s.startswith("http"):
        parsed = urlparse(s)
        s = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
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
            # Log success
            has_stats = analysis is not None and 'stats' in (analysis or {})
            if has_stats:
                print(f"[db] ✅ SUCCESS: Stats dumped for {matches_video_id} (status={status})")
            else:
                print(f"[db] ✅ Status updated for {matches_video_id}: {status}")
            health.record_db_query(success=True)
    except Exception as e:
        print(f"[db] ❌ FAILED to upsert: {e}")
        health.record_db_query(success=False)

def is_video_processed(matches_video_id, source_url):
    if NO_DB: return False
    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    try:
        with _conn() as conn, conn.cursor() as cur:
            sql = f"SELECT status FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
            cur.execute(sql, (unique_id,))
            row = cur.fetchone()
            health.record_db_query(success=True)
            if row and row["status"] in ("finished", "running", "failed"):
                return True
    except Exception as e:
        print(f"[db] Error checking status: {e}")
        health.record_db_query(success=False)
    return False

def run_pipeline(video_path, output_dir, max_frames=None, no_db=False, video_id=None, user_id=None, spaces_url=None,
                 locking_mode=2, jnr_stride=None, vid_stride=None, tracking_mode="bytetrack", make_video=False, task_id=0):
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
    # USER REQUEST: Always download to temp for reliable processing
    if video_path.startswith("http"):
        print(f"[pipeline] Downloading to temp: {video_path}...")
        use_streaming = False  # Always download

        if not use_streaming:
            temp_dir = tempfile.mkdtemp(prefix="pf_")
            temp_video_path = os.path.join(temp_dir, filename)
            try:
                resp = requests.get(video_path, stream=True, timeout=3600)
                resp.raise_for_status()
                with open(temp_video_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                local_video_path = temp_video_path
                print(f"[pipeline] Downloaded to {temp_video_path}")
            except Exception as e:
                print(f"[pipeline] Download Failed: {e}")
                if not no_db:
                    upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=f"Download failed: {e}")
                return False

    # 2. Initial status update (In Progress) - Use 'running' (7 chars) instead of 'in_progress' (11 chars)
    health.record_video_start(video_id, user_id)
    if not no_db:
        print(f"[pipeline] Setting status 'running' for {video_id} (Task {task_id})...")
        upsert_status_row(video_id, user_id, spaces_url or video_path, "running", task_id)

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
        # Round 12 fix: Always pass --vid_stride from config if not explicitly provided.
        # Previously, omitting --vid_stride caused pipeline to default to stride=1 (every frame),
        # making processing 3x slower than intended and causing V4 (60fps) to timeout at 24h.
        if vid_stride:
            cmd.extend(["--vid_stride", str(vid_stride)])
        else:
            vid_stride_cfg = CONFIG.get("heuristics", {}).get("VID_STRIDE", 3)
            cmd.extend(["--vid_stride", str(vid_stride_cfg)])
        if tracking_mode:
            cmd.extend(["--tracking_mode", tracking_mode])
            
        print(f"[pipeline] Executing Core: {' '.join(cmd)}")
        # Increased timeout to 24 hours for H100 full matches
        result = subprocess.run(cmd, env=os.environ, timeout=86400)
        
        if result.returncode != 0:
            print(f"[pipeline] Core failed with code {result.returncode}")
            if not no_db:
                upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=f"Core exit {result.returncode}")
            return False

        # 3. Handle Results & DB - Use 'finished' to match DB ENUM
        stats_path = os.path.join(output_dir, "player_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)

            # Extract frame count if available
            frames_processed = stats_data.get('metadata', {}).get('total_frames', 0)
            health.record_video_complete(video_id, frames_processed)

            if not no_db:
                print(f"[pipeline] Updating DB with stats for {video_id}...")
                upsert_status_row(video_id, user_id, spaces_url or video_path, "finished", task_id, analysis={"stats": stats_data})
            return True
        else:
            print(f"[pipeline] Missing player_stats.json in {output_dir}")
            health.record_video_failure(video_id, "Missing output file")
            return False

    except Exception as e:
        print(f"[pipeline] Error: {e}")
        health.record_video_failure(video_id, str(e))
        if not no_db:
            upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=str(e))
        return False
    finally:
        # Cleanup temp
        if temp_video_path and os.path.exists(temp_video_path):
            print(f"[pipeline] Cleaning up temp file...")
            try:
                os.remove(temp_video_path)
                temp_dir = os.path.dirname(temp_video_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except OSError as e:
                print(f"[pipeline] Warning: Could not remove temp files: {e}")

# Polling Logic
def fetch_pending_videos():
    """Fetch list of pending videos from ScoutBridge API (all pages)."""
    print(f"[poll] Fetching from {SBG_LIST_URL}...")
    headers = {
        "Authorization": f"Bearer {SBG_TOKEN}",
        "Content-Type": "application/json"
    }
    
    all_items = []
    page = 1
    
    try:
        while True:
            resp = requests.get(f"{SBG_LIST_URL}?page={page}&size=100", headers=headers, timeout=10)
            health.record_api_call(success=(resp.status_code == 200))
            
            if resp.status_code != 200:
                print(f"[poll] Error fetching videos: {resp.status_code} - {resp.text}")
                break
                
            data = resp.json()
            items = data.get("items", [])
            all_items.extend(items)
            
            total_pages = data.get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        
        print(f"[poll] Found {len(all_items)} pending videos (across {page} page(s)).")
        return all_items
        
    except Exception as e:
        print(f"[poll] Exception fetching videos: {e}")
        health.record_api_call(success=False)
        return all_items  # Return whatever we got so far


def process_spaces_video(video_item, save_local=True, no_db=True, max_frames=None, locking_mode=2, jnr_stride=None, vid_stride=None, tracking_mode="bytetrack", make_video=False):
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
    print(f"[process] Starting: {filename}")
    print(f"[process] Video ID: {video_id}, User ID: {user_id}")
    
    # Generate unique TaskID (pseudo-unique 32-bit int)
    # Avoids 'Duplicate entry 0' error
    import random
    task_id = int(time.time() * 1000) % 1000000000 + random.randint(0, 100000)
    print(f"[process] Generated Task ID: {task_id}")
    
    if not spaces_url:
        print(f"[process] Error: No spacesURL for {video_id}")
        return {"status": "error", "error": "No spacesURL"}
    
    # Create output directory
    out_dir = f"./output/{video_id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Process using the new unified wrapper
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
        vid_stride=vid_stride,
        tracking_mode=tracking_mode,
        make_video=make_video,
        task_id=task_id
    )
    
    if success:
        return {"status": "success", "video_id": video_id, "stats_path": os.path.join(out_dir, "player_stats.json")}
    else:
        return {"status": "error", "video_id": video_id, "error": "Pipeline failed"}


import concurrent.futures

def start_polling_loop(poll_interval=60, max_videos=None, min_size_mb=0, max_size_mb=float('inf'), 
                       locking_mode=2, jnr_stride=None, vid_stride=None, make_video=False, parallel_workers=1, max_frames=None, video_ids_filter=None):
    """
    Continuously poll for pending videos and process them.
    
    Args:
        poll_interval: Seconds between polls (default 60)
        max_videos: Max videos to PROCESS (submit) before stopping
        min_size_mb: Minimum file size filter in MB
        max_size_mb: Maximum file size filter in MB
        locking_mode: Mode passed down to pipeline
        jnr_stride: Stride passed down to pipeline
        parallel_workers: Number of concurrent pipeline jobs
    """
    print(f"[poll] Starting polling loop (interval={poll_interval}s, workers={parallel_workers})...")
    print(f"[poll] Size filter: {min_size_mb}MB - {max_size_mb}MB")
    if video_ids_filter:
        print(f"[poll] Video ID filter: {video_ids_filter}")
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers)

    processed_count = 0
    processed_ids = set()
    futures = []  # Track submitted futures for error handling
    
    while True:
        try:
            videos = fetch_pending_videos()
            
            if not videos:
                print(f"[poll] No pending videos. Waiting {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            
            # Filter by size and already processed
            submitted_in_this_cycle = 0
            
            for video in videos:
                video_id = video.get("id")
                file_size_mb = video.get("fileSize", 0) / 1024 / 1024
                
                # Skip already processed (local cache)
                if video_id in processed_ids:
                    continue
                
                # Skip if not in whitelist (when --video_ids is used)
                if video_ids_filter and video_id not in video_ids_filter:
                    continue
                
                # Skip if outside size range
                if file_size_mb < min_size_mb or file_size_mb > max_size_mb:
                    continue
                
                # Skip if already in DB (Persistent check)
                if is_video_processed(video_id, video.get("spacesURL")):
                   print(f"[poll] Already processed (DB): {video_id}")
                   processed_ids.add(video_id)
                   continue

                # No filename filter - process all videos
                
                # Submit to worker pool
                print(f"[poll] Submitting {video_id} ({file_size_mb:.1f}MB) to worker pool...")

                # Frame Limit (Debug Override)
                limit_frames = max_frames

                future = executor.submit(
                    process_spaces_video,
                    video, save_local=True, no_db=NO_DB,
                    locking_mode=locking_mode,
                    jnr_stride=jnr_stride,
                    vid_stride=vid_stride,
                    make_video=make_video,
                    max_frames=limit_frames
                )
                futures.append(future)

                processed_ids.add(video_id)
                processed_count += 1
                submitted_in_this_cycle += 1
                
                # Check max limit
                if max_videos and processed_count >= max_videos:
                    print(f"[poll] Reached max_videos limit ({max_videos}). Stopping submissions.")
                    print(f"[poll] Waiting for {len(futures)} running tasks to complete...")
                    executor.shutdown(wait=True)  # Wait for all tasks to complete
                    return
            
            if submitted_in_this_cycle == 0:
                 print(f"[poll] No new actionable videos found this cycle.")
            else:
                 print(f"[poll] Submitted {submitted_in_this_cycle} new jobs.")

            # Clean up completed futures and check for errors
            completed_futures = [f for f in futures if f.done()]
            for future in completed_futures:
                try:
                    future.result()  # Re-raise any exceptions from worker
                except Exception as e:
                    print(f"[poll] Worker task failed with error: {e}")
                    traceback.print_exc()
                    health.record_error("worker_task_failure", str(e))
            futures = [f for f in futures if not f.done()]  # Keep only running futures

            # Print health summary every 10 cycles
            if processed_count % 10 == 0 and processed_count > 0:
                health.print_summary()
                health.save_snapshot()

            # Wait before next poll
            print(f"[poll] Active workers: {len([f for f in futures if not f.done()])}")
            print(f"[poll] Cycle complete. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("[poll] Interrupted by user. Stopping...")
            print(f"[poll] Waiting for {len([f for f in futures if not f.done()])} running tasks to complete...")
            executor.shutdown(wait=True)  # Wait for graceful shutdown
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
    parser.add_argument("--locking_mode", type=int, choices=[1, 2, 3], default=2, help="Internal pipeline locking mode")
    parser.add_argument("--jnr_stride", type=int, help="Internal pipeline JNR stride (frames)")
    parser.add_argument("--vid_stride", type=int, help="Internal pipeline VIDEO stride (skip frames)")
    parser.add_argument("--tracking_mode", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracking backend")
    
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent pipelines (default 1)")
    parser.add_argument("--video_ids", type=str, help="Comma-separated list of video IDs to process (whitelist filter)")
    
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
                vid_stride=args.vid_stride,
                tracking_mode=args.tracking_mode,
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
        print(f"[main] Starting SPACES polling mode (Parallel Workers: {args.parallel})...")
        video_ids_filter = set(args.video_ids.split(',')) if args.video_ids else None
        start_polling_loop(
            poll_interval=args.poll_interval,
            max_videos=args.max_videos,
            min_size_mb=args.min_size_mb,
            max_size_mb=args.max_size_mb,
            locking_mode=args.locking_mode,
            jnr_stride=args.jnr_stride,
            vid_stride=args.vid_stride,
            make_video=args.make_video,
            parallel_workers=args.parallel,
            max_frames=args.max_frames,
            video_ids_filter=video_ids_filter
        )
    else:
        # Show help if no mode specified
        parser.print_help()
        print("\n[main] Use --local_video for single video or --poll for SPACES polling mode.")

if __name__ == "__main__":
    main()
