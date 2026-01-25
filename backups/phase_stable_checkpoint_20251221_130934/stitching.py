import json
import numpy as np
import os
import math

class TrackStitcher:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tracking_data = self._load_json(os.path.join(output_dir, "tracking_data.json"))
        # Use RAW stats (with Unknowns)
        self.stats_data = self._load_json(os.path.join(output_dir, "raw_players_stats.json"))
        self.log_file = os.path.join(output_dir, "debug_stitching_log.txt")
        
        # Maps Frame -> Tracks
        self.frame_map = {f["frame_idx"]: f["boxes"] for f in self.tracking_data}
        self.sorted_frames = sorted(self.frame_map.keys())
        
        # Track Lifetime
        self.track_lifetimes = {} # id -> {"start": f, "end": f, "path": [(f, x, y)]}
        self._build_lifetimes()

    def _load_json(self, path):
        if not os.path.exists(path): return {}
        with open(path, "r") as f:
            return json.load(f)

    def _log(self, msg):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

    def _build_lifetimes(self):
        """Constructs lifetime / trajectory for every track ID"""
        for frame_idx in self.sorted_frames:
            boxes = self.frame_map[frame_idx]
            for b in boxes:
                tid = b["id"]
                if tid is None: continue
                
                x1, y1, x2, y2 = b["xyxy"]
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                
                if tid not in self.track_lifetimes:
                    self.track_lifetimes[tid] = {"start": frame_idx, "end": frame_idx, "path": []}
                
                self.track_lifetimes[tid]["end"] = frame_idx
                self.track_lifetimes[tid]["path"].append((frame_idx, cx, cy))

    def stitch(self):
        self._log("--- Starting Track Stitching ---")
        merged_count = 0
        
        # Identify "Anchor" Tracks (Known Jersey Numbers)
        anchors = {} # JerseyNum -> [TrackIDs]
        unknowns = []
        
        for tid, data in self.stats_data.items():
            jnr = str(data.get("jersey_number", "Unknown"))
            try:
                if "Unknown_" in tid:
                    tid_int = int(tid.replace("Unknown_", ""))
                else:
                    tid_int = int(tid)
            except:
                continue # Skip non-integer keys if any
                
            if jnr.lower() not in ["unknown", "null", "none"]:
                if jnr not in anchors: anchors[jnr] = []
                anchors[jnr].append(tid_int)
            else:
                unknowns.append(tid_int)
                
        self._log(f"Found {len(anchors)} identified players and {len(unknowns)} unknown tracks.")
        
        # Attempt to merge Unknowns into Anchors
        # Logic: If Player 10 ends at F100, and Unknown X starts at F110 near F100 position -> Merge.
        
        # Process each Jersey Group
        matches = [] # (UnknownID, TargetJersey, TargetTrackID, Gap, Dist)
        
        for jnr, track_ids in anchors.items():
            # Get all segments for this player
            segments = []
            for tid in track_ids:
                if tid in self.track_lifetimes:
                     segments.append({"id": tid, "life": self.track_lifetimes[tid]})
            
            # Sort segments by start time
            segments.sort(key=lambda x: x["life"]["start"])
            
            # Iterate through Unknowns
            for uid in unknowns:
                if uid not in self.track_lifetimes: continue
                ulife = self.track_lifetimes[uid]
                
                # Check alignment with ANY segment of this player
                for seg in segments:
                    slife = seg["life"]
                    sid = seg["id"]
                    
                    # Case 1: Unknown is AFTER Known
                    # Gap > 0 and Gap < 60 frames (2 seconds)
                    gap = ulife["start"] - slife["end"]
                    if 0 < gap < 60:
                        # Spatial Check
                        end_pos = slife["path"][-1] # (f, x, y)
                        start_pos = ulife["path"][0] # (f, x, y)
                        
                        dist = math.sqrt((end_pos[1]-start_pos[1])**2 + (end_pos[2]-start_pos[2])**2)
                        
                        # Max Speed constraint: Player moves < 10px per frame?
                        # 60 frames gap -> max 600px move.
                        # Stricter: 300px
                        if dist < 300:
                            matches.append((uid, jnr, sid, gap, dist))
                            
                    # Case 2: Unknown is BEFORE Known (Reverse Stitch)
                    gap_rev = slife["start"] - ulife["end"]
                    if 0 < gap_rev < 60:
                        end_pos = ulife["path"][-1]
                        start_pos = slife["path"][0]
                        dist = math.sqrt((end_pos[1]-start_pos[1])**2 + (end_pos[2]-start_pos[2])**2)
                        
                        if dist < 300:
                             matches.append((uid, jnr, sid, gap_rev, dist))

        # Filter Best Matches
        # An unknown can only match to ONE player.
        # Pick smallest Gap/Dist combo.
        
        # Sort matches by Gap then Distance
        matches.sort(key=lambda x: (x[3], x[4]))
        
        params_merged_unknowns = set()
        final_merges = {} # UnknownID -> TargetJersey
        
        for uid, jnr, sid, gap, dist in matches:
            if uid in params_merged_unknowns: continue
            
            self._log(f"MERGE: Unknown Track {uid} -> Player {jnr} (Main Track {sid}) [Gap: {gap}, Dist: {dist:.1f}]")
            final_merges[uid] = jnr
            params_merged_unknowns.add(uid)
            merged_count += 1
            
        # Execute Merge in Stats Data
        for uid, target_jnr in final_merges.items():
            # Find the main stats entry for this JNR
            # (Ideally we merge into the specific track ID, but stats output keys are Track IDs)
            # We want to essentially COPY Unknown stats into one of the Player stats buckets.
            # OR we rename the Unknown Track key to have the Jersey Number?
            # Current JSON format: "TrackID": { ... "jersey_number": "10" }
            
            # Strategy: Be additive.
            # Find the "Primary" Track ID for this Jersey (longest duration?)
            target_ids = anchors[target_jnr]
            # Simplest: Just set the "jersey_number" of the Unknown track to the Target Jersey.
            # And let the post-processor re-aggregate later? 
            # OR aggregate NOW.
            
            # The User requested: "MERGE the 'Unknown' stats (Distance, Passes) into 'Player 10'."
            # Since `players_stats.json` is keyed by Track ID, we must modify the entry for `uid`
            # to have `jersey_number`: jnr.
            # AND we should virtually combine them.
            
            uid_str = str(uid)
            if uid_str in self.stats_data:
                self.stats_data[uid_str]["jersey_number"] = target_jnr
                # Optionally add a flag
                self.stats_data[uid_str]["stitched_to"] = target_jnr

        # Save Updated Stats
        out_file = os.path.join(self.output_dir, "players_stats.json")
        with open(out_file, "w") as f:
            json.dump(self.stats_data, f, indent=2)
            
        self._log(f"--- Stitching Complete. {merged_count} tracks merged. ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    stitcher = TrackStitcher(args.output_dir)
    stitcher.stitch()
