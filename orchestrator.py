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

# Polling Logic
def fetch_pending_videos():
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

def run_pipeline(video_path, output_json, output_csv, make_video=False, max_frames=None, viz_dir=None):
    # --- PIPELINE START ---
    from vision.tracking import track_video
    from vision.identity import assign_teams, get_jersey_numbers
    from stats.logic import calculate_ownership, detect_passes_and_events, detect_shots_and_xg
    from stats.formatting import format_and_save
    
    # 1. Tracking
    print(f"[pipeline] Step 1: Tracking (Max Frames: {max_frames})...")
    frames, model = track_video(video_path, CONFIG["env"]["DET_WEIGHTS"], max_frames=max_frames)
    
    # 2. Team Assignment
    print("[pipeline] Step 2: Team Assignment...")
    team_map, team_labels = assign_teams(frames)
    
    # 3. Jersey Numbers
    print("[pipeline] Step 3: Jersey Numbers...")
    jersey_map = get_jersey_numbers(frames, team_map)
    
    # 3.5 Global Identity Registry
    print("[pipeline] Step 3.5: Global Identity Registry...")
    from vision.identity_manager import GlobalRegistry
    registry = GlobalRegistry()
    
    track_to_global = {}
    
    # Register all tracks
    # We iterate over all tracks found in frames to ensure everyone gets a Global ID
    all_track_ids = set()
    for f in frames:
        for b in f["boxes"]:
            if b["id"] is not None:
                all_track_ids.add(b["id"])
                
    for tid in all_track_ids:
        # Get info from jersey_map if available
        j_info = jersey_map.get(tid, {"number": "Unknown"})
        number = j_info.get("number", "Unknown")
        team = team_map.get(tid, "Unknown")
        
        # Get or Create Global ID
        gid = registry.get_or_create_global_id(team, number, tid)
        track_to_global[tid] = gid
        
    # Remap Data Structures to Global IDs
    print(f"[pipeline] Remapping {len(track_to_global)} tracks to Global IDs...")
    
    # 1. Remap Frames (Mutate in place)
    for f in frames:
        for b in f["boxes"]:
            if b["id"] is not None:
                b["id"] = track_to_global.get(b["id"], b["id"]) # Should always be in map
                
    # 2. Remap Jersey Map
    global_jersey_map = {}
    for tid, info in jersey_map.items():
        gid = track_to_global.get(tid)
        if gid:
            # If multiple tracks map to same GID, we keep the one with info (or merge?)
            # Since GID is based on (Team, Number), the info should be consistent.
            global_jersey_map[gid] = info
            
    # 3. Remap Team Map
    global_team_map = {}
    for tid, team in team_map.items():
        gid = track_to_global.get(tid)
        if gid:
            global_team_map[gid] = team
            
    # Update references for next steps
    jersey_map = global_jersey_map
    team_map = global_team_map
    
    # 4. Stats
    print("[pipeline] Step 4: Stats Calculation...")
    ownership = calculate_ownership(frames, team_map)
    events, player_stats, team_stats = detect_passes_and_events(frames, ownership, team_map)
    shots, xg_player = detect_shots_and_xg(frames, ownership, team_map)
    
    # 5. Formatting & Saving
    print("[pipeline] Step 5: Formatting...")
    
    # Collect all tracked IDs (Now Global IDs)
    all_tracked_ids = set()
    for f in frames:
        for b in f["boxes"]:
            if b["id"] is not None and b["cls"] in (CONFIG["classes"]["player"], CONFIG["classes"]["goalkeeper"]):
                all_tracked_ids.add(b["id"])
                
    payload = format_and_save(
        events, player_stats, team_stats, shots, xg_player, 
        team_map, team_labels, jersey_map, 
        output_json, output_csv,
        all_tracked_ids=all_tracked_ids
    )
    
    # 6. Visualization (Optional)
    if make_video:
        from vision.visualization import generate_debug_video
        if viz_dir is None:
            viz_dir = "/home/ubuntu/football/output_viz"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        viz_path = os.path.join(viz_dir, "debug_run.mp4")
        generate_debug_video(video_path, frames, jersey_map, viz_path)
    
    # Cleanup Memory
    del model
    del frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return payload

# ... (process_video remains mostly unchanged, but we could update it if needed, though this request is for debug mode)

def main():
    global NO_DB
    
    parser = argparse.ArgumentParser(description="Football Pipeline Orchestrator")
    parser.add_argument("--local_video", type=str, help="Path to local video file for debug mode")
    parser.add_argument("--no_db", action="store_true", help="Skip DB connections")
    parser.add_argument("--save_local", action="store_true", help="Save output to ./output folder (or --output_dir)")
    parser.add_argument("--make_video", action="store_true", help="Generate debug video output")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to process")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    
    args = parser.parse_args()
    
    if args.no_db:
        NO_DB = True
        print("[debug] DB connections disabled.")
        
    if args.local_video:
        # DEBUG PATH
        if not os.path.exists(args.local_video):
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
            
        base_name = os.path.splitext(os.path.basename(args.local_video))[0]
        out_json = os.path.join(out_dir, f"{base_name}_stats.json")
        out_csv = os.path.join(out_dir, f"{base_name}_stats.csv")
        
        try:
            run_pipeline(
                args.local_video, 
                out_json, 
                out_csv, 
                make_video=args.make_video, 
                max_frames=args.max_frames,
                viz_dir=out_dir # Save video to same dir
            )
            print(f"Finished. Output saved to {out_dir}")
        except Exception as e:
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        sys.exit(0)
    else:
        # PRODUCTION PATH
        start_polling_loop()

if __name__ == "__main__":
    main()
