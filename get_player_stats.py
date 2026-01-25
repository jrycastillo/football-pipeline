import pymysql
import json

MYSQL_HOST = "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com"
MYSQL_PORT = 25060
MYSQL_USER = "scoutbridge"
MYSQL_PASS = "***REMOVED_SECRET***"
MYSQL_DB = "footballgallery"
TABLE_NAME = "MatchesVideoAnalysis_test"

def get_full_stats():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cursor:
        sql = f"SELECT id, analysis, updated_at FROM {TABLE_NAME} WHERE status='finished' ORDER BY updated_at DESC LIMIT 1"
        cursor.execute(sql)
        row = cursor.fetchone()
        
        if row:
            print(f"Video ID: {row['id']} | Finished: {row['updated_at']}")
            analysis = row['analysis']
            if isinstance(analysis, str):
                analysis = json.loads(analysis)
            
            stats = analysis.get("stats", {})
            print(f"\n--- PLAYER STATS ({len(stats)} players) ---")
            
            for jersey, data in stats.items():
                role = data.get("role", "Unknown")
                team = data.get("team", "Unknown")
                player_stats = data.get("stats", {})
                goals = player_stats.get("goals_total", 0)
                passes = player_stats.get("passes_total", 0)
                fouls = player_stats.get("fouls_total", 0)
                print(f"#{jersey} ({role}) - Team: {team} | Goals: {goals} | Passes: {passes} | Fouls: {fouls}")

if __name__ == "__main__":
    get_full_stats()
