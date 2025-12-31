#!/usr/bin/env python3
"""
Status API Server for Video Processing Pipeline
Provides endpoints to check video processing status and logs.
"""

from flask import Flask, jsonify, request
import pymysql
from pymysql.cursors import DictCursor
import os
import yaml
import subprocess

app = Flask(__name__)

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

MYSQL_HOST = config["env"]["MYSQL_HOST"]
MYSQL_PORT = int(config["env"]["MYSQL_PORT"])
MYSQL_USER = config["env"]["MYSQL_USER"]
MYSQL_PASS = "***REMOVED_SECRET***"  # Hardcoded for now
MYSQL_DB = config["env"]["MYSQL_DB"]
ANALYSIS_TABLE = "MatchesVideoAnalysis_test"

LOG_FILE = os.path.join(os.path.dirname(__file__), "polling_service.log")


def get_db():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS,
        database=MYSQL_DB, cursorclass=DictCursor, autocommit=True
    )


@app.route("/")
def index():
    return jsonify({"service": "Video Processing Status API", "endpoints": [
        "/status - Overall processing status",
        "/status/<video_id> - Specific video status",
        "/logs - Recent log entries",
        "/logs/<video_id> - Logs for specific video"
    ]})


@app.route("/status")
def overall_status():
    """Get overall processing status summary."""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Get status counts
        cur.execute(f"SELECT status, COUNT(*) as count FROM {ANALYSIS_TABLE} GROUP BY status")
        status_counts = {row["status"]: row["count"] for row in cur.fetchall()}
        
        # Get recent completions
        cur.execute(f"""
            SELECT source_url, status, updated_at 
            FROM {ANALYSIS_TABLE} 
            WHERE status = 'finished'
            ORDER BY updated_at DESC 
            LIMIT 5
        """)
        recent = cur.fetchall()
        
        conn.close()
        
        return jsonify({
            "status_counts": status_counts,
            "recent_completions": [{
                "video": row["source_url"].split("/")[-1].split("?")[0] if row["source_url"] else "unknown",
                "status": row["status"],
                "updated": str(row["updated_at"])
            } for row in recent]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status/<video_id>")
def video_status(video_id):
    """Get status for a specific video by ID or filename pattern."""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute(f"""
            SELECT status, source_url, updated_at, LENGTH(analysis) as analysis_size, error
            FROM {ANALYSIS_TABLE} 
            WHERE source_url LIKE %s OR unique_id LIKE %s
            ORDER BY updated_at DESC
            LIMIT 1
        """, (f"%{video_id}%", f"%{video_id}%"))
        
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"error": "Video not found", "video_id": video_id}), 404
        
        return jsonify({
            "video_id": video_id,
            "status": row["status"],
            "has_stats": (row["analysis_size"] or 0) > 10,
            "analysis_size": row["analysis_size"],
            "error": row["error"],
            "updated": str(row["updated_at"]),
            "source": row["source_url"].split("/")[-1].split("?")[0] if row["source_url"] else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs")
def recent_logs():
    """Get recent log entries."""
    lines = request.args.get("lines", 50, type=int)
    try:
        if os.path.exists(LOG_FILE):
            result = subprocess.run(
                ["tail", "-n", str(min(lines, 200)), LOG_FILE],
                capture_output=True, text=True
            )
            log_lines = result.stdout.strip().split("\n") if result.stdout else []
        else:
            log_lines = ["Log file not found"]
        
        return jsonify({
            "log_file": LOG_FILE,
            "lines": len(log_lines),
            "content": log_lines[-50:]  # Return last 50 lines max
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logs/<video_id>")
def video_logs(video_id):
    """Get log entries for a specific video."""
    try:
        if os.path.exists(LOG_FILE):
            result = subprocess.run(
                ["grep", "-i", video_id, LOG_FILE],
                capture_output=True, text=True
            )
            log_lines = result.stdout.strip().split("\n") if result.stdout else []
            log_lines = [l for l in log_lines if l]  # Remove empty lines
        else:
            log_lines = ["Log file not found"]
        
        return jsonify({
            "video_id": video_id,
            "lines": len(log_lines),
            "content": log_lines[-30:]  # Return last 30 matches
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset-stale", methods=["POST"])
def reset_stale():
    """Reset stale 'running' entries to 'queued' - but ONLY if they have no stats."""
    try:
        hours = request.args.get("hours", 1, type=int)
        
        conn = get_db()
        cur = conn.cursor()
        
        # Only reset entries WITHOUT analysis data
        cur.execute(f"""
            UPDATE {ANALYSIS_TABLE} 
            SET status = 'queued'
            WHERE status = 'running' 
            AND updated_at < NOW() - INTERVAL %s HOUR
            AND (analysis IS NULL OR LENGTH(analysis) < 10)
        """, (hours,))
        
        affected = cur.rowcount
        conn.commit()
        conn.close()
        
        return jsonify({
            "reset_count": affected,
            "hours_threshold": hours,
            "note": "Only reset entries WITHOUT existing stats"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Video Processing Status API on port 8080...")
    app.run(host="0.0.0.0", port=8080, debug=False)
