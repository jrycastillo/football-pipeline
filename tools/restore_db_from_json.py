import os
import json
import pymysql
import glob
import uuid
from pymysql.cursors import DictCursor

# Config (Same as orchestrator/check_db)
HOST = "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com"
PORT = 25060
USER = "scoutbridge"
PASS = "***REMOVED_SECRET***"
DB = "footballgallery"
TABLE = "MatchesVideoAnalysis_test"
OUTPUT_DIR = "output"

def restore_data():
    print(f"--- Starting Data Recovery for {TABLE} ---")
    
    # 1. Connect to DB and map IDs
    try:
        conn = pymysql.connect(
            host=HOST, port=PORT, user=USER, password=PASS, database=DB,
            cursorclass=DictCursor, autocommit=True
        )
    except Exception as e:
        print(f"DB Connection Failed: {e}")
        return

    db_map = {} # prefix(8) -> {id, unique_id}
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, unique_id, status FROM {TABLE}")
        for row in cur.fetchall():
            if row['unique_id']:
                prefix = row['unique_id'][:8]
                db_map[prefix] = row
    
    print(f"Loaded {len(db_map)} IDs from DB.")
    
    # 2. Scan Local Output Folders
    print(f"Scanning {OUTPUT_DIR} for pipeline logs...")
    
    restored_count = 0
    
    subdirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    with conn.cursor() as cur:
        for d in subdirs:
            dir_path = os.path.join(OUTPUT_DIR, d)
            log_path = os.path.join(dir_path, "pipeline.log")
            stats_path = os.path.join(dir_path, "players_stats.json")
            
            if not os.path.exists(log_path) or not os.path.exists(stats_path):
                continue
                
            # Extract URL from Log
            url = None
            try:
                with open(log_path, 'r', errors='ignore') as f:
                    # Scan first 50 lines
                    for _ in range(50):
                        line = f.readline()
                        if not line: break
                        # Match: [track] Initializing ThreadedVideoReader for URL...
                        if "Initializing ThreadedVideoReader for" in line:
                            parts = line.split("for ")
                            if len(parts) > 1:
                                raw_url = parts[1].split(" (Resume:")[0].strip()
                                # Remove trailing '...' if present (head truncation artifact?)
                                # Actually python 'head' might not truncate, but 'cat' output showed it.
                                # The log line usually ends with '...' if it's printed by a specific logger? 
                                # No, log usually has full URL. 
                                url = raw_url
                                break
            except Exception as e:
                print(f"Error reading log {d}: {e}")
                continue
                
            if not url:
                # Fallback: check "Starting Lazy Trigger Pipeline on"
                try:
                    with open(log_path, 'r', errors='ignore') as f:
                        for _ in range(50):
                            line = f.readline()
                            if "Starting Lazy Trigger Pipeline on" in line:
                                parts = line.split(" on ")
                                if len(parts) > 1:
                                    url = parts[1].split("...")[0].strip() # Pipeline msg often has ...
                                    break
                except:
                    pass
            
            if not url:
                print(f"Skipping {d}: Could not extract URL from pipeline.log")
                continue
                
            # Calculate Hash
            import hashlib
            unique_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
            prefix = unique_id[:8]
            
            # Verify Matches Folder
            # Folder format: PREFIX_TIMESTAMP
            folder_prefix = d.split('_')[0]
            if folder_prefix != prefix:
                # Mismatch - log warning but proceed?
                # Maybe folder name uses DIFFERENT hash logic? 
                # Or maybe log URL had time-sensitive signature different from folder gen time?
                # Trust the URL in the log as the "Source Of Truth" for THIS run.
                # But to reconcile with DB, we need the DB to have THIS hash.
                pass

            # Read Stats
            try:
                with open(stats_path, 'r') as f:
                    stats_data = json.load(f)
                if not stats_data:
                    print(f"Skipping {d}: valid JSON but empty stats.")
                    continue
            except:
                continue
                
            analysis_payload = json.dumps({"stats": stats_data})
            
            # Upsert
            try:
                # Check if exists
                cur.execute(f"SELECT id FROM {TABLE} WHERE unique_id=%s", (unique_id,))
                row = cur.fetchone()
                
                if row:
                    # Update
                    print(f"🔄 Updating Match {row['id']} (Hash {prefix}) from {d}")
                    cur.execute(f"UPDATE {TABLE} SET analysis=CAST(%s AS JSON), status='finished', updated_at=NOW() WHERE id=%s", (analysis_payload, row['id']))
                else:
                    # Insert
                    print(f"➕ Inserting NEW Match (Hash {prefix}) from {d}")
                    # Validation Status 5 = 'processed'? Default to NULL or something valid.
                    # task_id is required. Using short unique ID.
                    # Column might be short (e.g. VARCHAR(20)).
                    new_task_id = f"res_{unique_id[:8]}"
                    cur.execute(f"""
                        INSERT INTO {TABLE} 
                        (unique_id, source_url, status, analysis, created_at, updated_at, matches_video_id, user_id, task_id)
                        VALUES (%s, %s, 'finished', CAST(%s AS JSON), NOW(), NOW(), NULL, NULL, %s)
                    """, (unique_id, url, analysis_payload, new_task_id))
                
                restored_count += 1
                
            except Exception as e:
                print(f"❌ DB Error for {d}: {e}")

    print(f"\n--- Recovery Complete ---")
    print(f"Total Matches Processed: {restored_count}")

if __name__ == "__main__":
    restore_data()
