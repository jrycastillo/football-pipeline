"""
Label jersey crops using Gemini API
"""
import os
import json
import time
import base64
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai

# Configuration
CROPS_DIR = Path("data/manual_label_crops")
OUTPUT_FILE = Path("data/gemini_labels.json")
BATCH_SIZE = 50  # Save progress every N crops

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def label_crop(model, image_path, max_retries=3):
    """Get jersey number from Gemini for a single crop."""
    for attempt in range(max_retries):
        try:
            img_data = encode_image(image_path)
            
            response = model.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": img_data
                },
                "What jersey number is visible on this football player's shirt? Reply with ONLY the number (1-99). If no number is visible or unreadable, reply with 'UNK'."
            ])
            
            text = response.text.strip()
            
            # Parse response
            if text.upper() == 'UNK' or 'unreadable' in text.lower() or 'unclear' in text.lower() or 'cannot' in text.lower():
                return -1
            
            # Extract number
            import re
            match = re.search(r'\b(\d{1,2})\b', text)
            if match:
                num = int(match.group(1))
                if 1 <= num <= 99:
                    return num
            
            return -1
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait_time = 60 * (attempt + 1)
                print(f"\n⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}")
                return -1
    return -1

def main():
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY='your-api-key'")
        return
    
    # Initialize Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    
    # Get all crop files
    crop_files = sorted([f for f in CROPS_DIR.iterdir() if f.suffix == '.jpg'])
    print(f"Found {len(crop_files)} crops to label")
    
    # Load existing progress
    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing labels")
    
    # Process crops
    labeled = 0
    skipped = 0
    
    for i, crop_path in enumerate(tqdm(crop_files, desc="Labeling")):
        filename = crop_path.name
        
        # Skip if already labeled
        if filename in results:
            skipped += 1
            continue
        
        # Get label from Gemini
        label = label_crop(model, crop_path)
        results[filename] = label
        labeled += 1
        
        # Rate limiting (1 request per second for safety)
        time.sleep(1.0)
        
        # Save progress periodically
        if labeled % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved progress: {labeled} new labels")
    
    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    # Delete unreadable crops
    deleted = 0
    for filename, label in results.items():
        if label <= 0:
            crop_path = CROPS_DIR / filename
            if crop_path.exists():
                crop_path.unlink()
                deleted += 1
    
    # Stats
    valid_labels = [v for v in results.values() if v > 0]
    print(f"\n✅ Labeling complete!")
    print(f"   Total processed: {len(results)}")
    print(f"   Valid labels: {len(valid_labels)}")
    print(f"   Deleted (unreadable): {deleted}")
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
