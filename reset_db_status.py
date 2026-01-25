import pymysql

MYSQL_HOST = "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com"
MYSQL_PORT = 25060
MYSQL_USER = "scoutbridge"
MYSQL_PASS = "***REMOVED_SECRET***"
MYSQL_DB = "footballgallery"
TABLE_NAME = "MatchesVideoAnalysis_test"

def reset_all():
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor, autocommit=True
    )
    with conn.cursor() as cursor:
        # Check current count
        cursor.execute(f"SELECT count(*) as count FROM {TABLE_NAME}")
        before = cursor.fetchone()['count']
        print(f"Found {before} total records in {TABLE_NAME}")
        
        # Delete ALL rows
        cursor.execute(f"DELETE FROM {TABLE_NAME}")
        print(f"Deleted all records!")
        
        # Verify
        cursor.execute(f"SELECT count(*) as count FROM {TABLE_NAME}")
        after = cursor.fetchone()['count']
        print(f"Remaining records: {after}")

if __name__ == "__main__":
    reset_all()
