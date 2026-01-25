"""
Label jersey crops using local Qwen VLM (fast, no rate limits)
"""
import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import re

# Configuration
CROPS_DIR = Path("data/manual_label_crops")
OUTPUT_FILE = Path("data/qwen_labels.json")

def main():
    print("=" * 50)
    print("Labeling crops with local Qwen VLM")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading Qwen2.5-VL-3B...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("Model loaded!")
    
    # Get crop files
    crop_files = sorted([f for f in CROPS_DIR.iterdir() if f.suffix == '.jpg'])
    print(f"\n2. Found {len(crop_files)} crops to label")
    
    # Load existing progress
    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"   Loaded {len(results)} existing labels")
    
    prompt = "What jersey number is on this football player's shirt? Reply with ONLY the number (1-99). If no number visible, say UNK."
    
    # Process crops
    labeled = 0
    for crop_path in tqdm(crop_files, desc="Labeling"):
        filename = crop_path.name
        
        # Skip if already labeled
        if filename in results:
            continue
        
        try:
            img = Image.open(crop_path).convert("RGB")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=10)
                output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Parse response
            if "unk" in output.lower() or "cannot" in output.lower():
                results[filename] = -1
            else:
                match = re.search(r'\b(\d{1,2})\b', output)
                if match:
                    num = int(match.group(1))
                    results[filename] = num if 1 <= num <= 99 else -1
                else:
                    results[filename] = -1
            
            labeled += 1
            
            # Save progress every 100
            if labeled % 100 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error on {filename}: {e}")
            results[filename] = -1
    
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
    valid = [v for v in results.values() if v > 0]
    print(f"\n✅ Complete!")
    print(f"   Total processed: {len(results)}")
    print(f"   Valid labels: {len(valid)}")
    print(f"   Deleted (unreadable): {deleted}")
    print(f"\nLabels saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
