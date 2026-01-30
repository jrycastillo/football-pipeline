#!/usr/bin/env python3
"""
Test database connection and verify table structure.
"""

import os
import sys
import pymysql
from pymysql.cursors import DictCursor

# Load credentials from environment or hardcode
MYSQL_HOST = os.getenv("MYSQL_HOST", "db-mysql-sgp1-18289-do-user-18922201-0.f.db.ondigitalocean.com")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "25060"))
MYSQL_USER = os.getenv("MYSQL_USER", "scoutbridge")
MYSQL_PASS = os.getenv("MYSQL_PASSWORD", "***REMOVED_SECRET***")
MYSQL_DB = os.getenv("MYSQL_DB", "footballgallery")
TABLE_NAME = os.getenv("TABLE_NAME", "MatchesVideoAnalysis_test")

print("=" * 60)
print("DATABASE CONNECTION TEST")
print("=" * 60)
print(f"\nHost: {MYSQL_HOST}")
print(f"Port: {MYSQL_PORT}")
print(f"User: {MYSQL_USER}")
print(f"Database: {MYSQL_DB}")
print(f"Table: {TABLE_NAME}")
print()

# Test connection
try:
    print("🔌 Testing connection...")
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        cursorclass=DictCursor,
        connect_timeout=10
    )
    print("✅ Connected successfully!")
    print()

    with conn.cursor() as cur:
        # Check if table exists
        print(f"📋 Checking table '{TABLE_NAME}'...")
        cur.execute("SHOW TABLES LIKE %s", (TABLE_NAME,))
        table_exists = cur.fetchone()

        if table_exists:
            print(f"✅ Table '{TABLE_NAME}' exists")
            print()

            # Get table structure
            print("📊 Table structure:")
            cur.execute(f"DESCRIBE {TABLE_NAME}")
            columns = cur.fetchall()
            for col in columns:
                nullable = "NULL" if col['Null'] == 'YES' else "NOT NULL"
                print(f"   • {col['Field']:30} {col['Type']:20} {nullable}")
            print()

            # Count records
            cur.execute(f"SELECT COUNT(*) as count FROM {TABLE_NAME}")
            count = cur.fetchone()['count']
            print(f"📈 Total records: {count}")
            print()

            # Show recent records
            if count > 0:
                print("🔍 Recent records (last 5):")
                cur.execute(f"""
                    SELECT id, matches_video_id, status, created_at, updated_at
                    FROM {TABLE_NAME}
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                records = cur.fetchall()
                for rec in records:
                    video_id = rec['matches_video_id'] or 'NULL'
                print(f"   • ID {rec['id']:5} | Video: {str(video_id):15} | Status: {rec['status']:10} | Created: {rec['created_at']}")
                print()

                # Count by status
                print("📊 Records by status:")
                cur.execute(f"""
                    SELECT status, COUNT(*) as count
                    FROM {TABLE_NAME}
                    GROUP BY status
                """)
                statuses = cur.fetchall()
                for stat in statuses:
                    print(f"   • {stat['status']:15} : {stat['count']:5} records")
                print()

                # Check for recent processing
                cur.execute(f"""
                    SELECT COUNT(*) as count
                    FROM {TABLE_NAME}
                    WHERE updated_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)
                """)
                recent = cur.fetchone()['count']
                print(f"⏰ Updated in last hour: {recent}")
                print()

        else:
            print(f"❌ Table '{TABLE_NAME}' does NOT exist!")
            print("\n⚠️  You may need to create the table first.")
            print("\nRun this to create it:")
            print(f"   CREATE TABLE {TABLE_NAME} (...)")
            print()

    conn.close()
    print("=" * 60)
    print("✅ DATABASE CONNECTION TEST PASSED")
    print("=" * 60)
    sys.exit(0)

except pymysql.err.OperationalError as e:
    print(f"❌ Connection failed: {e}")
    print("\n🔍 Possible causes:")
    print("   1. Incorrect credentials")
    print("   2. Firewall blocking connection")
    print("   3. Database server down")
    print("   4. VPN required")
    print()
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
