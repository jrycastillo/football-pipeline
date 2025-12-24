import requests
import os
import yaml

def test_download():
    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    token = os.getenv("SBG_TOKEN", config["env"]["SBG_TOKEN"])
    list_url = "https://api-staging.scoutbridge.net/football-gallery/api/v2/files/list/video/for-match-analysis"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"Fetching list from {list_url}...")
    resp = requests.get(list_url, headers=headers)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    
    if not items:
        print("No videos found.")
        return
        
    video = items[0]
    video_url = video["spacesURL"]
    filename = video["filename"]
    
    print(f"Attempting to download {filename} from {video_url[:50]}...")
    
    try:
        # Use requests to download
        r = requests.get(video_url, stream=True, timeout=10)
        r.raise_for_status()
        print("Success! Head markers received.")
        # Just download first 1KB
        with open("test_chunk.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                break
        print("Test chunk saved successfully.")
    except Exception as e:
        print(f"Download FAILED: {e}")

if __name__ == "__main__":
    test_download()
