import pymysql
import json

MYSQL_HOST = "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com"
MYSQL_PORT = 25060
MYSQL_USER = "scoutbridge"
MYSQL_PASS = "***REMOVED_SECRET***"
MYSQL_DB = "footballgallery"
TABLE_NAME = "MatchesVideoAnalysis_test"

def check_stats():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cursor:
        sql = f"SELECT id, analysis, updated_at FROM {TABLE_NAME} WHERE status='finished' ORDER BY updated_at DESC LIMIT 5"
        cursor.execute(sql)
        results = cursor.fetchall()
        
        print("--- LAST 5 FINISHED VIDEOS ---")
        for row in results:
            analysis = row['analysis']
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            
            stats = analysis.get("stats", {})
            
            # Count players in stats
            num_players = len(stats) if isinstance(stats, dict) else len(stats) # Handle list or dict
            
            print(f"ID: {row['id']} | Time: {row['updated_at']} | Stats Entries: {num_players}")
            if num_players > 0:
                print(f"  > Sample: {str(stats)[:100]}...")

if __name__ == "__main__":
    check_stats()
