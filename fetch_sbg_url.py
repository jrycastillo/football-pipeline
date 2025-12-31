import requests
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

url = f"{config['env']['SBG_BASE']}/v2/files/list/video/for-match-analysis"
token = config['env']['SBG_TOKEN']
print(f"Fetching from {url}...")
try:
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Content: {resp.text[:500]}") # Print first 500 chars
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    if items:
        print(f"Found {len(items)} videos:")
        for idx, item in enumerate(items):
            print(f"[{idx}] {item.get('filename')} (Size: {item.get('fileSize')})")
            print(f"    URL: {item.get('spacesURL', item.get('fileLocation'))}")
    else:
        print("No videos found.")
except Exception as e:
    print(f"Error: {e}")
