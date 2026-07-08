"""
tools/upload_highlights.py — push a run's event clips to the ScoutBridge
highlights API (DigitalOcean Spaces behind it; admin verification queue reads
from there).

Consumes the run's clips_manifest.json (written by stats/event_clipper.py) and
POSTs each clip with its metadata. Writes uploads_manifest.json next to it,
mapping every clip to the returned storage id / file_key / file_url — the
references the admin UI needs to find the clip again.

API notes (from the backend's example):
- Auth is the X-Internal-Api-Key header. The key is NOT stored in this repo:
  pass --api_key or set SBG_HIGHLIGHTS_API_KEY in the environment.
- confidence_level is expected in PERCENT (70 -> stored as 0.7); our manifest
  confidences are 0-1, so they are converted here.
- Empty optional fields (matches_video_id, analysis_id, ...) are omitted
  rather than sent blank.

Usage (from the worker, where the clips live):
    python3 tools/upload_highlights.py --run_dir output/run_hb5_showcase \
        --user_id <app user id> [--matches_video_id X] [--analysis_id Y] \
        [--max_conf 0.75] [--types goal,shot] [--dry_run]
"""

import argparse
import json
import os
import sys
import time

DEFAULT_ENDPOINT = "https://api-staging.scoutbridge.net/football-gallery/api/v2/files/highlights"


def build_fields(entry, args):
    """Multipart form fields for one manifest entry (file handled separately)."""
    fields = {
        "user_id": args.user_id,
        "event_type": entry.get("type") or "unknown",
    }
    conf = entry.get("confidence")
    if isinstance(conf, (int, float)):
        # API expects percent: 70 -> stored 0.7
        fields["confidence_level"] = str(int(round(conf * 100)))
    ts = entry.get("time_s")
    if isinstance(ts, (int, float)):
        fields["timestamp_sec"] = str(ts)
    player = entry.get("player")
    if isinstance(player, int) or (isinstance(player, str) and str(player).isdigit()):
        fields["jersey_number"] = str(player)
    if args.matches_video_id:
        fields["matches_video_id"] = args.matches_video_id
    if args.analysis_id:
        fields["analysis_id"] = args.analysis_id
    return fields


def select_entries(manifest, args):
    out = []
    types = set(t.strip() for t in args.types.split(",") if t.strip()) if args.types else None
    for entry in manifest:
        if types and entry.get("type") not in types:
            continue
        conf = entry.get("confidence")
        if args.max_conf is not None and isinstance(conf, (int, float)) and conf > args.max_conf:
            continue
        if args.min_conf is not None and isinstance(conf, (int, float)) and conf < args.min_conf:
            continue
        out.append(entry)
    if args.limit:
        out = out[: args.limit]
    return out


def upload_one(session, endpoint, api_key, clip_path, fields, retries=1):
    """POST one clip. Returns (ok, response_dict_or_error_string)."""
    for attempt in range(retries + 1):
        try:
            with open(clip_path, "rb") as fh:
                resp = session.post(
                    endpoint,
                    headers={"accept": "application/json",
                             "X-Internal-Api-Key": api_key},
                    files={"file": (os.path.basename(clip_path), fh, "video/mp4")},
                    data=fields,
                    timeout=120,
                )
            if resp.status_code >= 500 and attempt < retries:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
            body = resp.json()
            if not body.get("response"):
                return False, f"API rejected: {json.dumps(body)[:300]}"
            return True, body.get("data", {})
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            return False, f"{type(e).__name__}: {e}"
    return False, "unreachable"


def upload_run(run_dir, user_id, api_key, endpoint=DEFAULT_ENDPOINT,
               matches_video_id=None, analysis_id=None, types=None,
               max_conf=None, min_conf=None, limit=None, dry_run=False):
    """Upload a finished run's clips. Callable from the pipeline
    (--upload_highlights) or the CLI below. Returns (ok_count, total_selected);
    never raises — a failed batch is reported in uploads_manifest.json."""
    class _Args:
        pass
    args = _Args()
    args.user_id = user_id
    args.matches_video_id = matches_video_id
    args.analysis_id = analysis_id
    args.types = types
    args.max_conf = max_conf
    args.min_conf = min_conf
    args.limit = limit

    manifest_path = os.path.join(run_dir, "clips_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[Upload] No clips_manifest.json in {run_dir} — nothing to upload")
        return 0, 0
    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = select_entries(manifest, args)
    print(f"[Upload] {len(entries)}/{len(manifest)} clips selected from {manifest_path}")

    session = None
    if not dry_run:
        import requests
        session = requests.Session()

    results = []
    ok_count = 0
    for entry in entries:
        clip_rel = entry.get("clip")
        clip_path = os.path.join(run_dir, clip_rel) if clip_rel else None
        fields = build_fields(entry, args)
        record = {"clip": clip_rel, "fields": fields}
        if not clip_path or not os.path.exists(clip_path):
            record["status"] = "missing_file"
            print(f"  MISSING {clip_rel}")
        elif dry_run:
            record["status"] = "dry_run"
            print(f"  DRY {clip_rel} -> {fields}")
        else:
            try:
                ok, data = upload_one(session, endpoint, api_key, clip_path, fields)
            except Exception as e:
                ok, data = False, f"{type(e).__name__}: {e}"
            if ok:
                ok_count += 1
                record["status"] = "uploaded"
                record["remote"] = {k: data.get(k) for k in
                                    ("id", "file_key", "file_url", "created_at")}
                print(f"  OK  {clip_rel} -> id={data.get('id')}")
            else:
                record["status"] = "failed"
                record["error"] = str(data)
                print(f"  FAIL {clip_rel}: {data}")
        results.append(record)

    out_path = os.path.join(run_dir, "uploads_manifest.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Upload] {ok_count}/{len(entries)} uploaded; record -> {out_path}")
    return ok_count, len(entries)


def main():
    parser = argparse.ArgumentParser(description="Upload a run's event clips to the highlights API")
    parser.add_argument("--run_dir", required=True, help="Run output dir containing clips_manifest.json + clips/")
    parser.add_argument("--user_id", required=True, help="App user id the match belongs to")
    parser.add_argument("--matches_video_id", default=None)
    parser.add_argument("--analysis_id", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api_key", default=None,
                        help="X-Internal-Api-Key value; defaults to env SBG_HIGHLIGHTS_API_KEY")
    parser.add_argument("--types", default=None, help="Comma-separated event types to upload (default: all)")
    parser.add_argument("--max_conf", type=float, default=None,
                        help="Only upload clips at or below this confidence (admin-queue mode)")
    parser.add_argument("--min_conf", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Upload at most N clips")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be uploaded, send nothing")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("SBG_HIGHLIGHTS_API_KEY")
    if not api_key and not args.dry_run:
        print("No API key: pass --api_key or set SBG_HIGHLIGHTS_API_KEY", file=sys.stderr)
        sys.exit(2)

    ok_count, total = upload_run(
        args.run_dir, args.user_id, api_key, endpoint=args.endpoint,
        matches_video_id=args.matches_video_id, analysis_id=args.analysis_id,
        types=args.types, max_conf=args.max_conf, min_conf=args.min_conf,
        limit=args.limit, dry_run=args.dry_run)
    sys.exit(0 if (args.dry_run or ok_count == total) else 1)


if __name__ == "__main__":
    main()
