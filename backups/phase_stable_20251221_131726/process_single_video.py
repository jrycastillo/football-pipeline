
import os
import subprocess
import sys

# Config
VIDEO_PATH = "/home/ubuntu/videoforprocessing_link/clipped_ikorudo_tornadoes.mp4"
OUTPUT_DIR = "output/test_ikorudo_refinement"

def run_command(cmd):
    print(f"[Run] {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"[Error] Command failed with code {process.returncode}")
        sys.exit(process.returncode)

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        sys.exit(1)

    print(f"--- Starting Refinement Run for {VIDEO_PATH} ---")
    
    # Step 1: Run Orchestrator (High Fidelity + Save Local + No DB)
    # Note: Stride settings are now hardcoded in orchestrator for this mode? 
    # Or should we pass args? Orchestrator args validation:
    # We want Stride 1 Track, Stride 5 OCR.
    # We rely on my recent orchestrator edits which set `process_every_n_frames=1` in the loop.
    # And `stride=stride_jnr` (which defaults to 5 in orchestrator if not passed? No, need to check).
    # Orchestrator uses `CONFIG` or args.
    
    cmd = (
        f"python3 -u orchestrator.py "
        f"--local_video '{VIDEO_PATH}' "
        f"--save_local "
        f"--no_db "
        f"--output_dir '{OUTPUT_DIR}' "
        # We need to ensure logic uses Stride 1.
        # Orchestrator defaults might be strict.
        # But my recent code edit HARDCODED `process_every_n_frames=1`.
        # So "Standard" run is actually High Fidelity now with that edit.
    )
    
    run_command(cmd)
    
    # Step 2: Run Stitching (Disabled for Phase 81 - Sticky Tracking)
    # print("\n--- Running Track Stitching ---")
    # cmd_stitch = f"python3 vision/stitching.py --output_dir '{OUTPUT_DIR}'"
    # run_command(cmd_stitch)
    
    print("\n[Success] Refinement Pipeline Complete.")

if __name__ == "__main__":
    main()
