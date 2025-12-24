
import json

try:
    with open("output/raw_tracks.json", "r") as f:
        data = json.load(f)
    
    print(f"Root type: {type(data)}")
    if isinstance(data, list):
        print(f"Root len: {len(data)}")
        if len(data) > 0:
            print(f"Index 0 type: {type(data[0])}")
            if isinstance(data[0], list):
                print("Detected nested list. Fixing...")
                tracks = data[0]
                # Double check inside
                if len(tracks) > 0 and isinstance(tracks[0], dict):
                    print("Found dicts inside index 0.")
                    with open("output/raw_tracks_fixed.json", "w") as f:
                        json.dump(tracks, f, indent=2)
                    print("Saved to output/raw_tracks_fixed.json")
                else:
                    print("Index 0 is list but empty or not dicts.")
            else:
                 print("Index 0 is not list.")
    else:
        print("Root is not list.")

except Exception as e:
    print(f"Error: {e}")
