import os
import time
import random
import json
import uuid
import requests
import pymysql
import yaml
import torch
import traceback
import argparse
import sys
import subprocess
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()
from utils.health_monitor import get_health_monitor

# Load Config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Environment Variables (Override Config if set)
MYSQL_HOST = os.getenv("MYSQL_HOST", CONFIG["env"]["MYSQL_HOST"])
MYSQL_PORT = int(os.getenv("MYSQL_PORT", CONFIG["env"]["MYSQL_PORT"]))
MYSQL_USER = os.getenv("MYSQL_USER", CONFIG["env"]["MYSQL_USER"])
MYSQL_PASS = os.getenv("MYSQL_PASSWORD")  # Required: Set via environment variable
if not MYSQL_PASS:
    print("[WARNING] MYSQL_PASSWORD not set. Database operations will fail.")
MYSQL_DB = os.getenv("MYSQL_DB", CONFIG["env"]["MYSQL_DB"])
ANALYSIS_TABLE = os.getenv("TABLE_NAME", CONFIG["env"]["TABLE_NAME"])
PIPELINE_FPS = CONFIG.get("heuristics", {}).get("FPS", 25)

SBG_BASE = os.getenv("SBG_BASE", CONFIG["env"]["SBG_BASE"]).rstrip("/")
SBG_LIST_URL = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN = os.getenv("SBG_TOKEN", CONFIG["env"]["SBG_TOKEN"])

# Global flag for DB connection
NO_DB = os.getenv("NO_DB", "0") == "1"

# Initialize health monitor
health = get_health_monitor()

# Database Helpers
def _conn():
    if NO_DB: return None
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, database=MYSQL_DB,
        cursorclass=DictCursor, autocommit=True
    )


DB_PERSISTENCE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS `matches` (
      `analysis_id` VARCHAR(128) NOT NULL,
      `matches_video_id` VARCHAR(128) NULL,
      `user_id` VARCHAR(128) NULL,
      `source_url` TEXT NULL,
      `output_dir` VARCHAR(1024) NULL,
      `roster_file` VARCHAR(1024) NULL,
      `status` VARCHAR(32) NOT NULL DEFAULT 'finished',
      `roster_json` JSON NULL,
      `player_stats_json` JSON NULL,
      `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (`analysis_id`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `roster_players` (
      `id` BIGINT NOT NULL AUTO_INCREMENT,
      `analysis_id` VARCHAR(128) NOT NULL,
      `team_name` VARCHAR(128) NOT NULL,
      `team_color` VARCHAR(128) NULL,
      `jersey_number` INT NOT NULL,
      `roster_player_json` JSON NULL,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uniq_roster_player` (`analysis_id`, `team_name`, `jersey_number`),
      CONSTRAINT `fk_roster_players_match`
        FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
        ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `roster_known_stats` (
      `id` BIGINT NOT NULL AUTO_INCREMENT,
      `analysis_id` VARCHAR(128) NOT NULL,
      `metric` VARCHAR(128) NOT NULL,
      `team_name` VARCHAR(128) NULL,
      `stat_value_json` JSON NULL,
      `known_stats_json` JSON NULL,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uniq_roster_known_stat` (`analysis_id`, `metric`, `team_name`),
      CONSTRAINT `fk_roster_known_stats_match`
        FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
        ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `ai_player_stats` (
      `id` BIGINT NOT NULL AUTO_INCREMENT,
      `analysis_id` VARCHAR(128) NOT NULL,
      `player_key` VARCHAR(128) NOT NULL,
      `jersey_number` INT NULL,
      `team_name` VARCHAR(128) NULL,
      `player_name` VARCHAR(255) NULL,
      `verification_status` VARCHAR(64) NULL,
      `stats_json` JSON NULL,
      `player_json` JSON NOT NULL,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uniq_ai_player_stats` (`analysis_id`, `player_key`),
      CONSTRAINT `fk_ai_player_stats_match`
        FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
        ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `events` (
      `id` BIGINT NOT NULL AUTO_INCREMENT,
      `analysis_id` VARCHAR(128) NOT NULL,
      `event_index` INT NOT NULL,
      `event_type` VARCHAR(64) NULL,
      `frame` INT NULL,
      `time_s` DOUBLE NULL,
      `primary_player` VARCHAR(128) NULL,
      `secondary_player` VARCHAR(128) NULL,
      `confidence` DOUBLE NULL,
      `identity_confidence` DOUBLE NULL,
      `identity_confidence_receiver` DOUBLE NULL,
      `status` VARCHAR(64) NULL,
      `event_json` JSON NOT NULL,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uniq_event` (`analysis_id`, `event_index`),
      CONSTRAINT `fk_events_match`
        FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
        ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS `clips` (
      `id` BIGINT NOT NULL AUTO_INCREMENT,
      `analysis_id` VARCHAR(128) NOT NULL,
      `clip_index` INT NOT NULL,
      `clip_path` VARCHAR(1024) NOT NULL,
      `event_type` VARCHAR(64) NULL,
      `player` VARCHAR(128) NULL,
      `player_boxed` BOOLEAN NOT NULL DEFAULT FALSE,
      `upload_status` VARCHAR(64) NOT NULL DEFAULT 'pending',
      `time_s` DOUBLE NULL,
      `event_frame` INT NULL,
      `size_bytes` BIGINT NULL,
      `clip_json` JSON NOT NULL,
      `upload_json` JSON NULL,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uniq_clip` (`analysis_id`, `clip_index`),
      CONSTRAINT `fk_clips_match`
        FOREIGN KEY (`analysis_id`) REFERENCES `matches` (`analysis_id`)
        ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


DB_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "schema.sql")


def _split_sql_statements(sql_text):
    statements = []
    current = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip()
            statements.append(statement[:-1].strip())
            current = []
    if current:
        statements.append("\n".join(current).strip())
    return statements


def _persistence_ddl():
    if os.path.exists(DB_SCHEMA_PATH):
        with open(DB_SCHEMA_PATH, "r") as f:
            return _split_sql_statements(f.read())
    return DB_PERSISTENCE_DDL


def _json_or_none(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_file(path, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)

def _sha1(s: str) -> str:
    import hashlib
    from urllib.parse import urlparse
    # Strip query parameters from signed URLs (they contain timestamps)
    if s and s.startswith("http"):
        parsed = urlparse(s)
        s = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _default_analysis_id(video_id, source_url, output_dir):
    if video_id and str(video_id) != "unknown":
        return str(video_id)
    basis = source_url or output_dir or str(uuid.uuid4())
    return f"analysis_{_sha1(basis)[:16]}"


def _primary_event_player(event):
    for key in ("player", "from", "by"):
        if event.get(key) is not None:
            return str(event.get(key))
    return None


def _secondary_event_player(event):
    for key in ("to", "on", "against"):
        if event.get(key) is not None:
            return str(event.get(key))
    return None


def _coerce_int_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_time_s(event, vid_stride):
    t = _coerce_float_or_none(event.get("time_s"))
    if t is not None:
        return t
    frame = _coerce_int_or_none(event.get("frame"))
    if frame is None:
        return None
    return round(((frame + 1) * max(1, int(vid_stride or 1))) / float(PIPELINE_FPS or 25), 3)


def _load_uploads_by_clip(output_dir):
    uploads = _load_json_file(os.path.join(output_dir, "uploads_manifest.json"), default=[])
    out = {}
    if not isinstance(uploads, list):
        return out
    for row in uploads:
        if isinstance(row, dict) and row.get("clip"):
            out[row["clip"]] = row
    return out


def _build_db_rows(analysis_id, output_dir, roster_file=None, vid_stride=None):
    roster = _load_json_file(roster_file, default=None)
    player_stats = _load_json_file(os.path.join(output_dir, "player_stats.json"), default={})
    raw_tracks = _load_json_file(os.path.join(output_dir, "raw_tracks.json"), default=[])
    clips_manifest = _load_json_file(os.path.join(output_dir, "clips_manifest.json"), default=[])
    uploads_by_clip = _load_uploads_by_clip(output_dir)

    roster_players = []
    known_stats = []
    if isinstance(roster, dict):
        for team_name, team_info in (roster.get("teams") or {}).items():
            if not isinstance(team_info, dict):
                continue
            team_color = team_info.get("color")
            for number in team_info.get("roster") or []:
                jersey_number = _coerce_int_or_none(number)
                if jersey_number is None:
                    continue
                roster_players.append({
                    "analysis_id": analysis_id,
                    "team_name": str(team_name),
                    "team_color": team_color,
                    "jersey_number": jersey_number,
                    "roster_player_json": {
                        "team": team_name,
                        "color": team_color,
                        "jersey_number": number,
                    },
                })

        raw_known_stats = roster.get("known_stats") or {}
        if isinstance(raw_known_stats, dict):
            for metric, value in raw_known_stats.items():
                if isinstance(value, dict):
                    for team_name, team_value in value.items():
                        known_stats.append({
                            "analysis_id": analysis_id,
                            "metric": str(metric),
                            "team_name": str(team_name),
                            "stat_value_json": team_value,
                            "known_stats_json": raw_known_stats,
                        })
                else:
                    known_stats.append({
                        "analysis_id": analysis_id,
                        "metric": str(metric),
                        "team_name": None,
                        "stat_value_json": value,
                        "known_stats_json": raw_known_stats,
                    })

    ai_players = []
    if isinstance(player_stats, dict):
        for player_key, player in player_stats.items():
            if not isinstance(player, dict):
                continue
            ai_players.append({
                "analysis_id": analysis_id,
                "player_key": str(player_key),
                "jersey_number": _coerce_int_or_none(player.get("jersey_number")),
                "team_name": player.get("team"),
                "player_name": player.get("player_name"),
                "verification_status": player.get("verification_status"),
                "stats_json": player.get("stats") or {},
                "player_json": player,
            })

    events = []
    if isinstance(raw_tracks, list):
        for idx, event in enumerate(raw_tracks):
            if not isinstance(event, dict):
                continue
            events.append({
                "analysis_id": analysis_id,
                "event_index": idx,
                "event_type": event.get("type"),
                "frame": _coerce_int_or_none(event.get("frame")),
                "time_s": _event_time_s(event, vid_stride),
                "primary_player": _primary_event_player(event),
                "secondary_player": _secondary_event_player(event),
                "confidence": _coerce_float_or_none(event.get("confidence")),
                "identity_confidence": _coerce_float_or_none(event.get("identity_confidence")),
                "identity_confidence_receiver": _coerce_float_or_none(
                    event.get("identity_confidence_receiver")),
                "status": event.get("status", "unverified"),
                "event_json": event,
            })

    clips = []
    if isinstance(clips_manifest, list):
        for idx, clip in enumerate(clips_manifest):
            if not isinstance(clip, dict):
                continue
            clip_rel = clip.get("clip")
            upload = uploads_by_clip.get(clip_rel, {})
            clips.append({
                "analysis_id": analysis_id,
                "clip_index": idx,
                "clip_path": clip_rel,
                "event_type": clip.get("type"),
                "player": str(clip.get("player")) if clip.get("player") is not None else None,
                "player_boxed": bool(clip.get("player_boxed", False)),
                "upload_status": clip.get("upload_status") or upload.get("status") or "pending",
                "time_s": _coerce_float_or_none(clip.get("time_s")),
                "event_frame": _coerce_int_or_none(clip.get("event_frame")),
                "size_bytes": _coerce_int_or_none(clip.get("size_bytes")),
                "clip_json": clip,
                "upload_json": upload or None,
            })

    return {
        "roster": roster,
        "player_stats": player_stats,
        "roster_players": roster_players,
        "known_stats": known_stats,
        "ai_players": ai_players,
        "events": events,
        "clips": clips,
    }


def _print_persistence_dry_run(analysis_id, rows):
    print("[db-persist] dry run: no database writes")
    print("[db-persist] CREATE TABLE DDL:")
    for ddl in _persistence_ddl():
        print(ddl.strip() + ";")
    print("[db-persist] planned rows:")
    print(f"  matches: 1 ({analysis_id})")
    print(f"  roster_players: {len(rows['roster_players'])}")
    print(f"  roster_known_stats: {len(rows['known_stats'])}")
    print(f"  ai_player_stats: {len(rows['ai_players'])}")
    print(f"  events: {len(rows['events'])}")
    print(f"  clips: {len(rows['clips'])}")


def persist_run_to_db(analysis_id, output_dir, roster_file=None, matches_video_id=None,
                      user_id=None, source_url=None, status="finished", vid_stride=None,
                      dry_run=False):
    rows = _build_db_rows(analysis_id, output_dir, roster_file=roster_file, vid_stride=vid_stride)

    if dry_run:
        _print_persistence_dry_run(analysis_id, rows)
        return rows
    if NO_DB:
        print("[db-persist] skipped: DB connections disabled")
        return rows

    with _conn() as conn, conn.cursor() as cur:
        for ddl in _persistence_ddl():
            cur.execute(ddl)

        cur.execute("""
            INSERT INTO `matches`
              (`analysis_id`, `matches_video_id`, `user_id`, `source_url`, `output_dir`,
               `roster_file`, `status`, `roster_json`, `player_stats_json`,
               `created_at`, `updated_at`)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
              `matches_video_id`=VALUES(`matches_video_id`),
              `user_id`=VALUES(`user_id`),
              `source_url`=VALUES(`source_url`),
              `output_dir`=VALUES(`output_dir`),
              `roster_file`=VALUES(`roster_file`),
              `status`=VALUES(`status`),
              `roster_json`=VALUES(`roster_json`),
              `player_stats_json`=VALUES(`player_stats_json`),
              `updated_at`=NOW()
        """, (
            analysis_id, str(matches_video_id) if matches_video_id is not None else None,
            user_id, source_url, output_dir, roster_file, status,
            _json_or_none(rows["roster"]), _json_or_none(rows["player_stats"])
        ))

        for table in ("roster_players", "roster_known_stats", "ai_player_stats",
                      "events", "clips"):
            cur.execute(f"DELETE FROM `{table}` WHERE `analysis_id`=%s", (analysis_id,))

        if rows["roster_players"]:
            cur.executemany("""
                INSERT INTO `roster_players`
                  (`analysis_id`, `team_name`, `team_color`, `jersey_number`, `roster_player_json`)
                VALUES (%s, %s, %s, %s, %s)
            """, [
                (r["analysis_id"], r["team_name"], r["team_color"], r["jersey_number"],
                 _json_or_none(r["roster_player_json"]))
                for r in rows["roster_players"]
            ])

        if rows["known_stats"]:
            cur.executemany("""
                INSERT INTO `roster_known_stats`
                  (`analysis_id`, `metric`, `team_name`, `stat_value_json`, `known_stats_json`)
                VALUES (%s, %s, %s, %s, %s)
            """, [
                (r["analysis_id"], r["metric"], r["team_name"],
                 _json_or_none(r["stat_value_json"]), _json_or_none(r["known_stats_json"]))
                for r in rows["known_stats"]
            ])

        if rows["ai_players"]:
            cur.executemany("""
                INSERT INTO `ai_player_stats`
                  (`analysis_id`, `player_key`, `jersey_number`, `team_name`, `player_name`,
                   `verification_status`, `stats_json`, `player_json`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                (r["analysis_id"], r["player_key"], r["jersey_number"], r["team_name"],
                 r["player_name"], r["verification_status"], _json_or_none(r["stats_json"]),
                 _json_or_none(r["player_json"]))
                for r in rows["ai_players"]
            ])

        if rows["events"]:
            cur.executemany("""
                INSERT INTO `events`
                  (`analysis_id`, `event_index`, `event_type`, `frame`, `time_s`,
                   `primary_player`, `secondary_player`, `confidence`, `identity_confidence`,
                   `identity_confidence_receiver`, `status`, `event_json`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                (r["analysis_id"], r["event_index"], r["event_type"], r["frame"], r["time_s"],
                 r["primary_player"], r["secondary_player"], r["confidence"],
                 r["identity_confidence"], r["identity_confidence_receiver"], r["status"],
                 _json_or_none(r["event_json"]))
                for r in rows["events"]
            ])

        if rows["clips"]:
            cur.executemany("""
                INSERT INTO `clips`
                  (`analysis_id`, `clip_index`, `clip_path`, `event_type`, `player`,
                   `player_boxed`, `upload_status`, `time_s`, `event_frame`, `size_bytes`,
                   `clip_json`, `upload_json`)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, [
                (r["analysis_id"], r["clip_index"], r["clip_path"], r["event_type"],
                 r["player"], r["player_boxed"], r["upload_status"], r["time_s"],
                 r["event_frame"], r["size_bytes"], _json_or_none(r["clip_json"]),
                 _json_or_none(r["upload_json"]))
                for r in rows["clips"]
            ])

    print(f"[db-persist] wrote analysis_id={analysis_id}: "
          f"{len(rows['roster_players'])} roster players, "
          f"{len(rows['known_stats'])} known stat rows, "
          f"{len(rows['ai_players'])} AI players, "
          f"{len(rows['events'])} events, {len(rows['clips'])} clips")
    health.record_db_query(success=True)
    return rows

def upsert_status_row(matches_video_id, user_id, source_url, status, task_id,
                      validation_status_id=None, analysis=None, error=None):
    if NO_DB:
        print(f"[db-mock] Upserting status: {status} for {matches_video_id} (Error: {error})")
        return

    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    mv_id_num = int(matches_video_id) if (matches_video_id is not None and str(matches_video_id).isdigit()) else None
    payload = analysis.copy() if isinstance(analysis, dict) else (analysis or {})
    if isinstance(payload, dict):
        payload.setdefault("matches_video_key", str(matches_video_id) if matches_video_id is not None else None)
        payload.setdefault("source_url", source_url)
        payload.setdefault("user_id", user_id)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",",":")) if payload is not None else None

    try:
        sel_sql = f"SELECT id FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1"
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(sel_sql, (unique_id,))
            row = cur.fetchone()
            if row:
                upd_sql = f"""
                UPDATE {ANALYSIS_TABLE}
                   SET matches_video_id=%s, user_id=%s, validation_status_id=%s,
                       source_url=%s, task_id=%s, status=%s, analysis=CAST(%s AS JSON),
                       error=%s, updated_at=NOW()
                 WHERE id=%s
                """
                cur.execute(upd_sql, (mv_id_num, user_id, validation_status_id, source_url,
                                      task_id, status, payload_json, error, row["id"]))
            else:
                ins_sql = f"""
                INSERT INTO {ANALYSIS_TABLE}
                  (matches_video_id, user_id, unique_id, validation_status_id,
                   source_url, task_id, status, analysis, error, created_at, updated_at)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, CAST(%s AS JSON), %s, NOW(), NOW())
                """
                cur.execute(ins_sql, (mv_id_num, user_id, unique_id, validation_status_id,
                                      source_url, task_id, status, payload_json, error))
            # Log success
            has_stats = analysis is not None and 'stats' in (analysis or {})
            if has_stats:
                print(f"[db] ✅ SUCCESS: Stats dumped for {matches_video_id} (status={status})")
            else:
                print(f"[db] ✅ Status updated for {matches_video_id}: {status}")
            health.record_db_query(success=True)
    except Exception as e:
        print(f"[db] ❌ FAILED to upsert: {e}")
        health.record_db_query(success=False)

# A 'running' row older than this is a crashed/killed run (the pipeline
# subprocess itself times out at 24h), so the video is eligible again.
# Without this, an orchestrator crash left the row 'running' forever and
# the video was silently skipped on every subsequent poll.
RUNNING_STALE_HOURS = 25


def is_video_processed(matches_video_id, source_url, retry_failed=False):
    if NO_DB: return False
    unique_id = _sha1(source_url or f"{matches_video_id or ''}")
    try:
        with _conn() as conn, conn.cursor() as cur:
            sql = (f"SELECT status, updated_at, "
                   f"TIMESTAMPDIFF(HOUR, updated_at, NOW()) AS age_h "
                   f"FROM {ANALYSIS_TABLE} WHERE unique_id=%s LIMIT 1")
            cur.execute(sql, (unique_id,))
            row = cur.fetchone()
            health.record_db_query(success=True)
            if not row:
                return False
            status = row["status"]
            if status == "finished":
                return True
            if status == "running":
                age_h = row.get("age_h")
                if age_h is not None and age_h >= RUNNING_STALE_HOURS:
                    print(f"[db] Reclaiming stale 'running' video {matches_video_id} "
                          f"(last update {age_h}h ago)")
                    return False
                return True
            if status == "failed":
                if retry_failed:
                    print(f"[db] Retrying previously failed video {matches_video_id} "
                          f"(--retry_failed)")
                    return False
                return True
    except Exception as e:
        print(f"[db] Error checking status: {e}")
        health.record_db_query(success=False)
    return False

def run_pipeline(video_path, output_dir, max_frames=None, no_db=False, video_id=None, user_id=None, spaces_url=None,
                 locking_mode=2, jnr_stride=None, vid_stride=None, tracking_mode="bytetrack", make_video=False,
                 task_id=0, roster_file=None, clip_events=None, clip_pad_s=3.0,
                 write_db=False, db_dry_run=False, analysis_id=None, pitch_homography=False):
    """
    Unified metadata-aware pipeline wrapper.
    Delegates to pipeline_consolidated.py and handles DB updates.
    """
    import cv2
    import tempfile
    
    local_video_path = video_path
    use_streaming = True
    temp_video_path = None
    filename = os.path.basename(video_path.split("?")[0]) or "video.mp4"

    # 1. Download/Streaming Hybrid Logic
    # USER REQUEST: Always download to temp for reliable processing
    if video_path.startswith("http"):
        print(f"[pipeline] Downloading to temp: {video_path}...")
        use_streaming = False  # Always download

        if not use_streaming:
            temp_dir = tempfile.mkdtemp(prefix="pf_")
            temp_video_path = os.path.join(temp_dir, filename)
            # Retry transient network failures — a single blip used to mark the
            # video 'failed' permanently (failed rows are not re-polled).
            download_attempts = 3
            last_err = None
            for attempt in range(1, download_attempts + 1):
                try:
                    resp = requests.get(video_path, stream=True, timeout=3600)
                    resp.raise_for_status()
                    with open(temp_video_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    local_video_path = temp_video_path
                    print(f"[pipeline] Downloaded to {temp_video_path}")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"[pipeline] Download attempt {attempt}/{download_attempts} failed: {e}")
                    if attempt < download_attempts:
                        time.sleep(10 * attempt)
            if last_err is not None:
                print(f"[pipeline] Download Failed after {download_attempts} attempts: {last_err}")
                if not no_db:
                    upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=f"Download failed: {last_err}")
                return False

    # 2. Initial status update (In Progress) - Use 'running' (7 chars) instead of 'in_progress' (11 chars)
    health.record_video_start(video_id, user_id)
    if not no_db:
        print(f"[pipeline] Setting status 'running' for {video_id} (Task {task_id})...")
        upsert_status_row(video_id, user_id, spaces_url or video_path, "running", task_id)

    try:
        # 3. Execute Subprocess
        cmd = [
            sys.executable, "pipeline_consolidated.py",
            "--video", local_video_path,
            "--output_dir", output_dir,
            "--locking_mode", str(locking_mode)
        ]
        if not make_video:
            cmd.append("--no_video_output")
        
        if max_frames:
            cmd.extend(["--max_frames", str(max_frames)])
        if jnr_stride:
            cmd.extend(["--jnr_stride", str(jnr_stride)])
        # Round 12 fix: Always pass --vid_stride from config if not explicitly provided.
        # Previously, omitting --vid_stride caused pipeline to default to stride=1 (every frame),
        # making processing 3x slower than intended and causing V4 (60fps) to timeout at 24h.
        if vid_stride:
            cmd.extend(["--vid_stride", str(vid_stride)])
        else:
            vid_stride_cfg = CONFIG.get("heuristics", {}).get("VID_STRIDE", 3)
            cmd.extend(["--vid_stride", str(vid_stride_cfg)])
        if tracking_mode:
            cmd.extend(["--tracking_mode", tracking_mode])
        if roster_file:
            cmd.extend(["--roster_file", roster_file])
        if clip_events:
            cmd.extend(["--clip_events", clip_events, "--clip_pad_s", str(clip_pad_s)])
        # Nabeel v3 pitch-keypoint homography (opt-in; default off until the
        # keypoint model is retrained on our footage — see docs/PITCH_HOMOGRAPHY.md).
        if pitch_homography:
            cmd.append("--pitch_homography")

        print(f"[pipeline] Executing Core: {' '.join(cmd)}")
        # GPU OOM guard: hold a permit while the subprocess owns GPU memory;
        # on a CUDA OOM death the permit is leaked so concurrency shrinks.
        _gate_acquired = False
        if _GPU_GATE is not None:
            _GPU_GATE.acquire()
            _gate_acquired = True
        _oom_leak = False
        try:
            # Increased timeout to 24 hours for H100 full matches
            result = subprocess.run(cmd, env=os.environ, timeout=86400)
            if result.returncode != 0:
                _oom_leak = _run_dir_is_cuda_oom(output_dir)
        finally:
            if _gate_acquired:
                global _GPU_GATE_LEAKS
                with _GPU_GATE_LOCK:
                    if _oom_leak and _GPU_GATE_LEAKS < _GPU_GATE_MAX_LEAKS:
                        _GPU_GATE_LEAKS += 1
                        print(f"[OOMGuard] CUDA OOM detected — permanently reducing "
                              f"pipeline concurrency (leaked permit "
                              f"{_GPU_GATE_LEAKS}/{_GPU_GATE_MAX_LEAKS})")
                    else:
                        _GPU_GATE.release()

        if result.returncode != 0:
            print(f"[pipeline] Core failed with code {result.returncode}")
            if not no_db:
                upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=f"Core exit {result.returncode}")
            return False

        # 3. Handle Results & DB - Use 'finished' to match DB ENUM
        stats_path = os.path.join(output_dir, "player_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)

            # Extract frame count if available
            frames_processed = stats_data.get('metadata', {}).get('total_frames', 0)
            health.record_video_complete(video_id, frames_processed)

            if write_db:
                run_analysis_id = analysis_id or _default_analysis_id(
                    video_id, spaces_url or video_path, output_dir)
                print(f"[db-persist] Persisting normalized analysis rows ({run_analysis_id})...")
                try:
                    persist_run_to_db(
                        run_analysis_id,
                        output_dir,
                        roster_file=roster_file,
                        matches_video_id=video_id,
                        user_id=user_id,
                        source_url=spaces_url or video_path,
                        status="finished",
                        vid_stride=vid_stride or CONFIG.get("heuristics", {}).get("VID_STRIDE", 3),
                        dry_run=db_dry_run,
                    )
                except Exception as e:
                    print(f"[db-persist] failed (non-fatal): {e}")
                    health.record_db_query(success=False)

            if not no_db:
                print(f"[pipeline] Updating DB with stats for {video_id}...")
                upsert_status_row(video_id, user_id, spaces_url or video_path, "finished", task_id, analysis={"stats": stats_data})
            return True
        else:
            print(f"[pipeline] Missing player_stats.json in {output_dir}")
            health.record_video_failure(video_id, "Missing output file")
            if not no_db:
                # Without this the row stayed 'running' forever
                upsert_status_row(video_id, user_id, spaces_url or video_path,
                                  "failed", task_id, error="Missing player_stats.json")
            return False

    except Exception as e:
        print(f"[pipeline] Error: {e}")
        health.record_video_failure(video_id, str(e))
        if not no_db:
            upsert_status_row(video_id, user_id, spaces_url or video_path, "failed", task_id, error=str(e))
        return False
    finally:
        # Cleanup temp
        if temp_video_path and os.path.exists(temp_video_path):
            print(f"[pipeline] Cleaning up temp file...")
            try:
                os.remove(temp_video_path)
                temp_dir = os.path.dirname(temp_video_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except OSError as e:
                print(f"[pipeline] Warning: Could not remove temp files: {e}")

# Polling Logic
def fetch_pending_videos():
    """Fetch list of pending videos from ScoutBridge API (all pages)."""
    print(f"[poll] Fetching from {SBG_LIST_URL}...")
    headers = {
        "Authorization": f"Bearer {SBG_TOKEN}",
        "Content-Type": "application/json"
    }
    
    all_items = []
    page = 1
    
    try:
        while True:
            resp = requests.get(f"{SBG_LIST_URL}?page={page}&size=100", headers=headers, timeout=10)
            health.record_api_call(success=(resp.status_code == 200))
            
            if resp.status_code != 200:
                print(f"[poll] Error fetching videos: {resp.status_code} - {resp.text}")
                break
                
            data = resp.json()
            items = data.get("items", [])
            all_items.extend(items)
            
            total_pages = data.get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        
        print(f"[poll] Found {len(all_items)} pending videos (across {page} page(s)).")
        return all_items
        
    except Exception as e:
        print(f"[poll] Exception fetching videos: {e}")
        health.record_api_call(success=False)
        return all_items  # Return whatever we got so far


def process_spaces_video(video_item, save_local=True, no_db=True, max_frames=None,
                         locking_mode=2, jnr_stride=None, vid_stride=None,
                         tracking_mode="bytetrack", make_video=False,
                         roster_file=None, clip_events=None, clip_pad_s=3.0,
                         write_db=False, db_dry_run=False, analysis_id=None,
                         pitch_homography=False):
    """
    Process a single video from SPACES using the unified run_pipeline wrapper.
    """
    video_id = video_item.get("id", "unknown")
    spaces_url = video_item.get("spacesURL")
    filename = video_item.get("filename", "video.mp4")
    
    # Extract UserID from fileLocation or filename
    # Usually: matches_upload/{userID}/{filename}
    file_loc = video_item.get("fileLocation", "")
    if "/" in file_loc:
        user_id = file_loc.split("/")[1]
    else:
        user_id = filename.split("_")[0] if "_" in filename else "unknown"
    
    print(f"[process] Starting: {filename}")
    print(f"[process] Starting: {filename}")
    print(f"[process] Video ID: {video_id}, User ID: {user_id}")
    
    # Generate unique TaskID (pseudo-unique 32-bit int)
    # Avoids 'Duplicate entry 0' error
    import random
    task_id = int(time.time() * 1000) % 1000000000 + random.randint(0, 100000)
    print(f"[process] Generated Task ID: {task_id}")
    
    if not spaces_url:
        print(f"[process] Error: No spacesURL for {video_id}")
        return {"status": "error", "error": "No spacesURL"}
    
    # Create output directory
    out_dir = f"./output/{video_id}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Process using the new unified wrapper
    # Process using the new unified wrapper
    success = run_pipeline(
        video_path=spaces_url,
        output_dir=out_dir,
        max_frames=max_frames,
        no_db=no_db,
        video_id=video_id,
        user_id=user_id,
        spaces_url=spaces_url,
        locking_mode=locking_mode,
        jnr_stride=jnr_stride,
        vid_stride=vid_stride,
        tracking_mode=tracking_mode,
        make_video=make_video,
        task_id=task_id,
        roster_file=roster_file,
        clip_events=clip_events,
        clip_pad_s=clip_pad_s,
        write_db=write_db,
        db_dry_run=db_dry_run,
        analysis_id=analysis_id,
        pitch_homography=pitch_homography
    )

    if success:
        return {"status": "success", "video_id": video_id, "stats_path": os.path.join(out_dir, "player_stats.json")}
    else:
        return {"status": "error", "video_id": video_id, "error": "Pipeline failed"}


import concurrent.futures
import threading

# GPU OOM guard (only armed when --parallel > 1): every pipeline subprocess
# must hold a permit. When a run dies of CUDA OOM the permit is deliberately
# NOT returned, so effective concurrency shrinks by one — parallel runs that
# overload the GPU degrade to fewer workers instead of failing repeatedly.
# Never shrinks below 1.
_GPU_GATE = None
_GPU_GATE_LEAKS = 0
_GPU_GATE_MAX_LEAKS = 0
_GPU_GATE_LOCK = threading.Lock()


def _run_dir_is_cuda_oom(output_dir):
    """True if the failed run's pipeline_error.txt points at CUDA OOM."""
    try:
        with open(os.path.join(output_dir, "pipeline_error.txt")) as f:
            return "out of memory" in f.read().lower()
    except OSError:
        return False

def start_polling_loop(poll_interval=60, max_videos=None, min_size_mb=0, max_size_mb=float('inf'), 
                       locking_mode=2, jnr_stride=None, vid_stride=None,
                       make_video=False, parallel_workers=1, max_frames=None,
                       video_ids_filter=None, roster_file=None, clip_events=None,
                       clip_pad_s=3.0, write_db=False, db_dry_run=False,
                       analysis_id=None, pitch_homography=False, retry_failed=False):
    """
    Continuously poll for pending videos and process them.
    
    Args:
        poll_interval: Seconds between polls (default 60)
        max_videos: Max videos to PROCESS (submit) before stopping
        min_size_mb: Minimum file size filter in MB
        max_size_mb: Maximum file size filter in MB
        locking_mode: Mode passed down to pipeline
        jnr_stride: Stride passed down to pipeline
        parallel_workers: Number of concurrent pipeline jobs
    """
    print(f"[poll] Starting polling loop (interval={poll_interval}s, workers={parallel_workers})...")
    print(f"[poll] Size filter: {min_size_mb}MB - {max_size_mb}MB")
    if video_ids_filter:
        print(f"[poll] Video ID filter: {video_ids_filter}")
    
    # Arm the GPU OOM guard only for genuinely parallel runs.
    global _GPU_GATE, _GPU_GATE_MAX_LEAKS
    if parallel_workers > 1:
        _GPU_GATE = threading.Semaphore(parallel_workers)
        _GPU_GATE_MAX_LEAKS = parallel_workers - 1

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers)

    processed_count = 0
    processed_ids = set()
    futures = []  # Track submitted futures for error handling
    
    while True:
        try:
            videos = fetch_pending_videos()
            
            if not videos:
                print(f"[poll] No pending videos. Waiting {poll_interval}s...")
                time.sleep(poll_interval)
                continue
            
            # Filter by size and already processed
            submitted_in_this_cycle = 0
            
            for video in videos:
                video_id = video.get("id")
                file_size_mb = video.get("fileSize", 0) / 1024 / 1024
                
                # Skip already processed (local cache)
                if video_id in processed_ids:
                    continue
                
                # Skip if not in whitelist (when --video_ids is used)
                if video_ids_filter and video_id not in video_ids_filter:
                    continue
                
                # Skip if outside size range
                if file_size_mb < min_size_mb or file_size_mb > max_size_mb:
                    continue
                
                # Skip if already in DB (Persistent check)
                if is_video_processed(video_id, video.get("spacesURL"),
                                      retry_failed=retry_failed):
                   print(f"[poll] Already processed (DB): {video_id}")
                   processed_ids.add(video_id)
                   continue

                # No filename filter - process all videos
                
                # Submit to worker pool
                print(f"[poll] Submitting {video_id} ({file_size_mb:.1f}MB) to worker pool...")

                # Frame Limit (Debug Override)
                limit_frames = max_frames

                future = executor.submit(
                    process_spaces_video,
                    video, save_local=True, no_db=NO_DB,
                    locking_mode=locking_mode,
                    jnr_stride=jnr_stride,
                    vid_stride=vid_stride,
                    make_video=make_video,
                    max_frames=limit_frames,
                    roster_file=roster_file,
                    clip_events=clip_events,
                    clip_pad_s=clip_pad_s,
                    write_db=write_db,
                    db_dry_run=db_dry_run,
                    analysis_id=analysis_id,
                    pitch_homography=pitch_homography
                )
                futures.append(future)

                processed_ids.add(video_id)
                processed_count += 1
                submitted_in_this_cycle += 1
                
                # Check max limit
                if max_videos and processed_count >= max_videos:
                    print(f"[poll] Reached max_videos limit ({max_videos}). Stopping submissions.")
                    print(f"[poll] Waiting for {len(futures)} running tasks to complete...")
                    executor.shutdown(wait=True)  # Wait for all tasks to complete
                    return
            
            if submitted_in_this_cycle == 0:
                 print(f"[poll] No new actionable videos found this cycle.")
            else:
                 print(f"[poll] Submitted {submitted_in_this_cycle} new jobs.")

            # Clean up completed futures and check for errors
            completed_futures = [f for f in futures if f.done()]
            for future in completed_futures:
                try:
                    future.result()  # Re-raise any exceptions from worker
                except Exception as e:
                    print(f"[poll] Worker task failed with error: {e}")
                    traceback.print_exc()
                    health.record_error("worker_task_failure", str(e))
            futures = [f for f in futures if not f.done()]  # Keep only running futures

            # Print health summary every 10 cycles
            if processed_count % 10 == 0 and processed_count > 0:
                health.print_summary()
                health.save_snapshot()

            # Wait before next poll
            print(f"[poll] Active workers: {len([f for f in futures if not f.done()])}")
            print(f"[poll] Cycle complete. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("[poll] Interrupted by user. Stopping...")
            print(f"[poll] Waiting for {len([f for f in futures if not f.done()])} running tasks to complete...")
            executor.shutdown(wait=True)  # Wait for graceful shutdown
            break
        except Exception as e:
            print(f"[poll] Error in polling loop: {e}")
            traceback.print_exc()
            time.sleep(poll_interval)


# ... (process_video remains mostly unchanged, but we could update it if needed, though this request is for debug mode)

def main():
    global NO_DB
    
    parser = argparse.ArgumentParser(description="Football Pipeline Orchestrator")
    parser.add_argument("--local_video", type=str, help="Path to local video file or SPACES URL for debug mode")
    parser.add_argument("--no_db", action="store_true", help="Skip DB connections")
    parser.add_argument("--save_local", action="store_true", help="Save output to ./output folder (or --output_dir)")
    parser.add_argument("--make_video", action="store_true", help="Generate debug video output")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to process")
    parser.add_argument("--resume_frame", type=int, default=0, help="Start processing from this frame index")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--persist_output_dir", type=str,
                        help="Persist an existing finished output directory without running the pipeline")
    parser.add_argument("--roster_file", type=str,
                        help="Optional roster JSON passed through to the pipeline and persisted with --write_db")
    parser.add_argument("--clip_events", type=str,
                        help="Comma-separated event types to clip; writes clips_manifest.json")
    parser.add_argument("--clip_pad_s", type=float, default=3.0,
                        help="Seconds of video kept on each side of a clipped event")
    parser.add_argument("--pitch_homography", action="store_true",
                        help="Enable Nabeel v3 pitch-keypoint homography (px->meters). "
                             "Opt-in; default off until the keypoint model is retrained on our footage.")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-process videos whose DB status is 'failed' (default: skip them)")
    parser.add_argument("--analysis_id", type=str,
                        help="Stable idempotency key for normalized DB persistence")
    parser.add_argument("--write_db", action="store_true",
                        help="Persist roster, AI stats, events and clips to normalized MySQL tables")
    parser.add_argument("--db_dry_run", action="store_true",
                        help="Print persistence DDL and planned row counts without writing")
    
    # Polling mode arguments
    parser.add_argument("--poll", action="store_true", help="Enable polling mode to fetch from SPACES")
    parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between polls (default 60)")
    parser.add_argument("--max_videos", type=int, help="Max videos to process before stopping")
    parser.add_argument("--min_size_mb", type=float, default=0, help="Minimum video size in MB")
    parser.add_argument("--max_size_mb", type=float, default=float('inf'), help="Maximum video size in MB")
    parser.add_argument("--locking_mode", type=int, choices=[1, 2, 3], default=2, help="Internal pipeline locking mode")
    parser.add_argument("--jnr_stride", type=int, help="Internal pipeline JNR stride (frames)")
    parser.add_argument("--vid_stride", type=int, help="Internal pipeline VIDEO stride (skip frames)")
    parser.add_argument("--tracking_mode", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Tracking backend")
    
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent pipelines (default 1)")
    parser.add_argument("--video_ids", type=str, help="Comma-separated list of video IDs to process (whitelist filter)")
    
    args = parser.parse_args()

    if args.db_dry_run:
        args.write_db = True
    
    if args.no_db:
        NO_DB = True
        print("[debug] DB connections disabled.")

    if args.persist_output_dir:
        if not args.write_db:
            print("Error: --persist_output_dir requires --write_db or --db_dry_run.")
            sys.exit(1)
        if not os.path.isdir(args.persist_output_dir):
            print(f"Error: Output directory {args.persist_output_dir} not found.")
            sys.exit(1)

        run_analysis_id = args.analysis_id or _default_analysis_id(None, None, args.persist_output_dir)
        try:
            persist_run_to_db(
                run_analysis_id,
                args.persist_output_dir,
                roster_file=args.roster_file,
                matches_video_id=None,
                user_id=None,
                source_url=None,
                status="finished",
                vid_stride=args.vid_stride or CONFIG.get("heuristics", {}).get("VID_STRIDE", 3),
                dry_run=args.db_dry_run,
            )
        except Exception as e:
            print(f"[db-persist] failed: {e}")
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
        
    if args.local_video:
        # DEBUG PATH - Single video
        if not args.local_video.startswith("http") and not os.path.exists(args.local_video):
            print(f"Error: Video file {args.local_video} not found.")
            sys.exit(1)
            
        print(f"Running Debug Mode on {args.local_video}")
        
        # Determine output directory
        if args.output_dir:
            out_dir = args.output_dir
        elif args.save_local:
            out_dir = "./output"
        else:
            out_dir = "."
            
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        try:
            # We use the new unified wrapper
            success = run_pipeline(
                video_path=args.local_video, 
                output_dir=out_dir,
                no_db=args.no_db, 
                max_frames=args.max_frames,
                video_id=os.path.splitext(os.path.basename(args.local_video))[0].split('?')[0],
                user_id="local_user",
                locking_mode=args.locking_mode,
                jnr_stride=args.jnr_stride,
                vid_stride=args.vid_stride,
                tracking_mode=args.tracking_mode,
                make_video=args.make_video,
                roster_file=args.roster_file,
                clip_events=args.clip_events,
                clip_pad_s=args.clip_pad_s,
                write_db=args.write_db,
                db_dry_run=args.db_dry_run,
                analysis_id=args.analysis_id,
                pitch_homography=args.pitch_homography
            )
            if success:
                print(f"Finished successfully. Output in {out_dir}")
            else:
                print("Pipeline execution failed.")
                sys.exit(1)
        except Exception as e:
            print(f"Pipeline failed: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        sys.exit(0)
        
    elif args.poll:
        # POLLING MODE - Fetch from SPACES and process
        print(f"[main] Starting SPACES polling mode (Parallel Workers: {args.parallel})...")
        video_ids_filter = set(args.video_ids.split(',')) if args.video_ids else None
        start_polling_loop(
            poll_interval=args.poll_interval,
            max_videos=args.max_videos,
            min_size_mb=args.min_size_mb,
            max_size_mb=args.max_size_mb,
            locking_mode=args.locking_mode,
            jnr_stride=args.jnr_stride,
            vid_stride=args.vid_stride,
            make_video=args.make_video,
            parallel_workers=args.parallel,
            max_frames=args.max_frames,
            video_ids_filter=video_ids_filter,
            roster_file=args.roster_file,
            clip_events=args.clip_events,
            clip_pad_s=args.clip_pad_s,
            write_db=args.write_db,
            db_dry_run=args.db_dry_run,
            analysis_id=args.analysis_id,
            pitch_homography=args.pitch_homography,
            retry_failed=args.retry_failed
        )
    else:
        # Show help if no mode specified
        parser.print_help()
        print("\n[main] Use --local_video for single video or --poll for SPACES polling mode.")

if __name__ == "__main__":
    main()
