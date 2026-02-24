
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

try:
    conn = pymysql.connect(
        host=os.getenv('MYSQL_HOST'),
        port=int(os.getenv('MYSQL_PORT', 25060)),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DB'),
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10
    )
except Exception as e:
    print(f"DB Connection Failed: {e}")
    exit(1)

with conn.cursor() as cursor:
    # 1. Check current status
    check_sql = f"SELECT unique_id, status, source_url FROM MatchesVideoAnalysis_test WHERE source_url LIKE '%69a33466fc234db%'"
    cursor.execute(check_sql)
    rows = cursor.fetchall()
    print("Current Status:")
    for r in rows:
        print(f"  {r['unique_id']}: {r['status']} | URL: {r.get('source_url', 'None')}")

    # 2. Reset to queued
    update_sql = f"UPDATE MatchesVideoAnalysis_test SET status = 'queued', updated_at = NOW() WHERE source_url LIKE '%69a33466fc234db%'"
    cursor.execute(update_sql)
    conn.commit()
    print(f"\n✅ Reset {cursor.rowcount} videos to queued.")
    
conn.close()
