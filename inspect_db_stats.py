import pymysql
import json
import os

conn = pymysql.connect(
    host='db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com',
    port=25060,
    user='scoutbridge',
    password='***REMOVED_SECRET***',
    database='footballgallery',
    cursorclass=pymysql.cursors.DictCursor
)

with conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT matches_video_id, user_id, updated_at, analysis FROM MatchesVideoAnalysis_test WHERE status='finished' AND updated_at > NOW() - INTERVAL 4 HOUR ORDER BY updated_at DESC LIMIT 1")
        row = cursor.fetchone()

if not row:
    print("No videos finished in the last 4 hours.")
    exit()

print(f"Video: {row.get('matches_video_id')} (User: {row.get('user_id')})")
print(f"Finished at: {row.get('updated_at')}")

if isinstance(row['analysis'], str):
    analysis_data = json.loads(row['analysis'])
else:
    analysis_data = row['analysis']

stats = analysis_data['stats'] if analysis_data and 'stats' in analysis_data else {}

aggregated = {}

# Sum up all stats across all players
for pid, p_data in stats.items():
    s = p_data.get('stats', {})
    for k, v in s.items():
        if isinstance(v, (int, float)):
             aggregated[k] = aggregated.get(k, 0) + v

print("\n--- Non-Zero Stats Summary ---")
for k, v in aggregated.items():
    if v > 0:
        print(f"{k}: {v}")


print("\n--- Zero Stats (Gaps) ---")
for k, v in aggregated.items():
    if v == 0:
        print(f"{k}: 0")

first_pid = next(iter(stats))
print(f"\n--- Schema Check (Player {first_pid}) ---")
print(list(stats[first_pid]['stats'].keys()))

