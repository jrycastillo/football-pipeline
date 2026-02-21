
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DB'),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10
    )
except Exception as e:
    print(f"DB Connection Failed: {e}")
    exit(1)

ids = ['162b6abe208946b', '14c0f4e8c4af40d', '69a33466fc234db']
placeholders = ', '.join(['%s'] * len(ids))

with conn.cursor() as cursor:
    # 1. Check current status
    check_sql = f"SELECT video_id, status, source_url FROM MatchesVideoAnalysis_test WHERE video_id IN ({placeholders})"
    cursor.execute(check_sql, ids)
    rows = cursor.fetchall()
    print("Current Status:")
    for r in rows:
        print(f"  {r['video_id']}: {r['status']} | URL: {r.get('source_url', 'None')[:50]}...")

    # 2. Reset to PENDING
    update_sql = f"UPDATE MatchesVideoAnalysis_test SET status = 'PENDING', progress = 0, last_updated = NOW() WHERE video_id IN ({placeholders})"
    cursor.execute(update_sql, ids)
    conn.commit()
    print(f"\n✅ Reset {cursor.rowcount} videos to PENDING.")
    
conn.close()
