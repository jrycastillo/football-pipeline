#!/usr/bin/env python3
import os, pymysql
from pymysql.cursors import DictCursor

conn = pymysql.connect(
    host="db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com",
    port=25060, user="scoutbridge", password="***REMOVED_SECRET***",
    database="footballgallery", cursorclass=DictCursor
)

print("=" * 80)
print("DATABASE STATUS")
print("=" * 80)

with conn.cursor() as cur:
    # Running
    cur.execute("SELECT COUNT(*) as c FROM MatchesVideoAnalysis_test WHERE status='running'")
    running = cur.fetchone()['c']
    print(f"\n🔄 Running: {running}")
    
    # Finished
    cur.execute("SELECT COUNT(*) as c FROM MatchesVideoAnalysis_test WHERE status='finished'")
    finished = cur.fetchone()['c']
    print(f"✅ Finished: {finished}")
    
    # Failed
    cur.execute("SELECT COUNT(*) as c FROM MatchesVideoAnalysis_test WHERE status='failed'")
    failed = cur.fetchone()['c']
    print(f"❌ Failed: {failed}")
    
    # Recent activity
    cur.execute("""
        SELECT id, status, created_at, updated_at 
        FROM MatchesVideoAnalysis_test 
        WHERE updated_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)
        ORDER BY updated_at DESC LIMIT 10
    """)
    recent = cur.fetchall()
    print(f"\n⏰ Updated in last hour: {len(recent)}")
    for r in recent:
        print(f"   • ID {r['id']:5} | {r['status']:10} | Updated: {r['updated_at']}")

conn.close()
print("\n" + "=" * 80)
