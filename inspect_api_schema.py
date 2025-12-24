import os
import requests
import json

# ======================= CONFIG =======================
SBG_BASE      = os.getenv("SBG_BASE", "https://api-staging.scoutbridge.net/football-gallery/api").rstrip("/")
SBG_LIST_URL  = f"{SBG_BASE}/v2/files/list/video/for-match-analysis"
SBG_TOKEN     = os.getenv("SBG_TOKEN", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjY1NzgwNDQsInN1YiI6ImFudG9uaW9qaGFuY2Vkcmljays1QGdtYWlsLmNvbSIsInVzZXJfaWQiOiJlZTIyYWFlNiJ9.YZ4r1Ahp--tX0h2rO-FnV8ZusfOshPZO7sn4Kat_J1E")

def inspect_schema():
    headers = {"Authorization": f"Bearer {SBG_TOKEN}"}
    try:
        print(f"[diagnostic] Fetching from {SBG_LIST_URL}...")
        resp = requests.get(SBG_LIST_URL, headers=headers)
        
        if resp.status_code != 200:
            print(f"[diagnostic] ERROR: Status {resp.status_code}")
            print(resp.text)
            return

        data = resp.json()
        
        # 1. Inspect Wrapper
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            print(f"[diagnostic] Root Keys: {list(data.keys())}")
            if "items" in data: items = data["items"]
            elif "data" in data: items = data["data"]
            elif "files" in data: items = data["files"]
        
        print(f"[diagnostic] Found {len(items)} items.")
        
        if items:
            print("\n[diagnostic] --- RAW JSON OF FIRST ITEM ---")
            print(json.dumps(items[0], indent=2))
            print("\n[diagnostic] --- END RAW JSON ---")
            
            # 2. Check for Linkage Keys
            first = items[0]
            keys = first.keys()
            potential_links = [k for k in keys if "id" in k.lower() or "match" in k.lower() or "analysis" in k.lower()]
            print(f"[diagnostic] Potential Linkage Keys: {potential_links}")
            
            print("\n[diagnostic] --- VIDEOS ~100MB (50-200MB) ---")
            found_candidates = False
            for i, item in enumerate(items):
                size_mb = item.get("fileSize", 0) / (1024 * 1024)
                if 50 <= size_mb <= 200:
                    print(f"{i}: {size_mb:.2f} MB - {item.get('spacesURL')}")
                    found_candidates = True
            
            if not found_candidates:
                print("No videos found between 50MB and 200MB.")

    except Exception as e:
        print(f"[diagnostic] Exception: {e}")

if __name__ == "__main__":
    inspect_schema()
