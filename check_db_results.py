import pymysql
import os
import json
from pymysql.cursors import DictCursor

# Config matches orchestrator.py fallback/defaults
HOST = "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com"
PORT = 25060
USER = "scoutbridge"
PASS = "***REMOVED_SECRET***"
DB = "footballgallery"
TABLE = "MatchesVideoAnalysis_test"

def check_db():
    print(f"Connecting to {HOST}:{PORT}/{DB} table {TABLE}...")
    try:
        conn = pymysql.connect(
            host=HOST, port=PORT, user=USER, password=PASS, database=DB,
            cursorclass=DictCursor, connect_timeout=10
        )
        with conn.cursor() as cur:
            # 1. Total Count
            cur.execute(f"SELECT COUNT(*) as count FROM {TABLE}")
            count = cur.fetchone()['count']
            print(f"Total Rows in {TABLE}: {count}")
            
            # 2. Status Distribution
            cur.execute(f"SELECT status, COUNT(*) as c FROM {TABLE} GROUP BY status")
            print("Status Distribution:")
            for row in cur.fetchall():
                print(f"  {row['status']}: {row['c']}")
                
            # 3. List Verified Matches (Finished)
            print("\n✅ FINISHED MATCHES with DATA (Searching...):")
            # Query explicitly for matches with players in stats
            cur.execute(f"SELECT id, source_url, updated_at, analysis FROM {TABLE} WHERE status='finished' AND JSON_LENGTH(analysis->'$.stats') > 0 ORDER BY updated_at DESC LIMIT 1")
            
            rows = cur.fetchall()
            if not rows:
                print("  (No matches with Stats data found)")
            
            found_data = False
            for row in rows:
                if found_data: break 
                
                # Parse Stats
                try:
                    analysis = json.loads(row['analysis']) if isinstance(row['analysis'], str) else row['analysis']
                    stats = analysis.get('stats', {})
                    num_players = len(stats)
                    
                    print(f"  [ID: {row['id']}] {row['updated_at']} | URL: ...{row['source_url'][-30:] if row['source_url'] else 'None'} | Count: {num_players}")
                    
                    if num_players > 0:
                        found_data = True
                        print(f"\n  📊 STATS SUMMARY for Match {row['id']} (Top Distances):")
                        sorted_players = sorted(stats.items(), key=lambda x: x[1].get('distance_m', 0), reverse=True)
                        print(f"  {'PID':<10} | {'Jersey':<8} | {'Team':<10} | {'Role':<10} | {'Dist (m)':<10} | {'Speed (m/s)':<10}")
                        print("  " + "-"*80)
                        
                        for pid, data in sorted_players[:5]:
                            jersey = data.get('jersey_number', 'N/A')
                            team = data.get('team', 'Unknown')
                            role = data.get('role', 'Player')
                            dist = f"{data.get('distance_m', 0):.1f}"
                            speed = f"{data.get('top_speed', 0):.1f}"
                            print(f"  {str(pid):<10} | {str(jersey):<8} | {str(team):<10} | {str(role):<10} | {dist:<10} | {speed:<10}")
                        print("\n")
                        
                except Exception as e:
                    print(f"  Error parsing analysis JSON for ID {row['id']}: {e}")
            
            # 4. Recent Failures
            print("\n❌ RECENT FAILURES (Last 5):")
            cur.execute(f"SELECT id, source_url, updated_at, error FROM {TABLE} WHERE status='failed' ORDER BY updated_at DESC LIMIT 5")
            for row in cur.fetchall():
                 err_short = (row['error'] or "Unknown Error")[:80]
                 print(f"  [ID: {row['id']}] {row['updated_at']} | Error: {err_short}...")

    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    check_db()
