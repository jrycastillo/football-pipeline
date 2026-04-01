#!/usr/bin/env python3
"""
Download SoccerNet JNR 2023 Dataset
Run this script to download the dataset, then upload to Google Drive
"""
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import os
import sys
from pathlib import Path

# Configuration
LOCAL_DIR = "/Users/ronan/Babak/data/soccernet_jnr"
SPLITS = ["train", "test", "challenge"]

def main():
    print("=" * 70)
    print("SoccerNet JNR 2023 Dataset Downloader")
    print("=" * 70)
    print(f"\nDownload location: {LOCAL_DIR}")
    print(f"Splits to download: {SPLITS}")
    print(f"Expected size: ~5-10 GB")
    print(f"Expected time: 10-60 minutes (depending on internet speed)")
    print("\n" + "=" * 70)

    # Confirm
    response = input("\nProceed with download? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Download cancelled.")
        sys.exit(0)

    # Create directory
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"\n✅ Created directory: {LOCAL_DIR}")

    # Initialize downloader
    print("\n🔄 Initializing SoccerNet downloader...")
    try:
        downloader = SNdl(LocalDirectory=LOCAL_DIR)
    except Exception as e:
        print(f"\n❌ Error initializing downloader: {e}")
        print("\nMake sure you have installed SoccerNet:")
        print("  pip install SoccerNet")
        sys.exit(1)

    # Download
    print("\n🔄 Starting download...")
    print("=" * 70)
    try:
        downloader.downloadDataTask(task="jersey-2023", split=SPLITS)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try downloading one split at a time:")
        print("     SPLITS = ['train']  # in the script")
        print("  3. Check if you have enough disk space (~10-15 GB)")
        sys.exit(1)

    # Verify
    print("\n" + "=" * 70)
    print("✅ Download complete!")
    print("=" * 70)

    # Check what was downloaded
    dataset_path = Path(LOCAL_DIR) / "jersey-2023"
    if dataset_path.exists():
        print(f"\n📁 Dataset location: {dataset_path}")

        # Count files
        for split in SPLITS:
            split_path = dataset_path / split / "images"
            if split_path.exists():
                jpg_files = list(split_path.rglob("*.jpg"))
                print(f"   - {split}: {len(jpg_files):,} images")

        # Check disk usage
        print(f"\n💾 Checking disk usage...")
        os.system(f"du -sh {dataset_path}")

        print("\n" + "=" * 70)
        print("Next steps:")
        print("  1. Verify the data looks correct")
        print("  2. Upload to Google Drive:")
        print(f"     Upload folder: {dataset_path}")
        print("  3. You can now delete the local copy if needed")
        print("=" * 70)
    else:
        print(f"\n⚠️  Warning: Expected path not found: {dataset_path}")
        print("Check the download output above for errors.")

if __name__ == "__main__":
    main()
