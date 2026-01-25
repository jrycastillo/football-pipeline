
import json
import numpy as np
from collections import defaultdict, Counter

class TrackerDiagnostics:
    def __init__(self, log_path="output/tracking_metrics.jsonl"):
        self.log_path = log_path
        # Append mode by default to avoid accidental wipes on read
        # Caller is responsible for cleaning up if needed
            
        self.track_histories = defaultdict(int)
        self.jersey_histories = defaultdict(lambda: defaultdict(int))
        self.lock_histories = defaultdict(lambda: {"total_locked": 0, "changes": 0, "last_state": None, "first_lock_frame": None, "start_frame": None})
        self.reactivations_total = 0
        self.risky_match_total = 0 # Task 4: Merge Risk Audit
        self.cumulative_unmatched_reasons = Counter()
        
    def log_frame(self, frame_idx, tracks, detections, matches_info):
        """
        matches_info: dict with keys 'matched_count', 'refind_count', 'mean_iou', 'mean_reid', 'unmatched_reasons'
        """
        # 1. Active Tracks Stats
        active_ids = [t.track_id for t in tracks if t.is_activated]
        new_ids = [t.track_id for t in tracks if t.start_frame == frame_idx]
        
        # Update histograms
        current_locked_count = 0
        for t in tracks:
            # Duration
            if t.is_activated:
                self.track_histories[t.track_id] += 1
            
            # Anti-Merge Guard (Task D)
            if hasattr(t, 'jersey_det') and t.jersey_det:
                num, conf = t.jersey_det
                if conf > 0.85: # High confidence only
                    self._update_jersey_history(t.track_id, num)

            # Lock Metrics (Task E)
            # Check for locked_number
            if hasattr(t, 'locked_number') and t.locked_number is not None:
                current_locked_count += 1
                self.lock_histories[t.track_id]["total_locked"] += 1
                
                if self.lock_histories[t.track_id]["first_lock_frame"] is None:
                     self.lock_histories[t.track_id]["first_lock_frame"] = frame_idx
                     if hasattr(t, 'start_frame'):
                         self.lock_histories[t.track_id]["start_frame"] = t.start_frame

                # Check Change
                if self.lock_histories[t.track_id]["last_state"] is not None:
                    if str(self.lock_histories[t.track_id]["last_state"]) != str(t.locked_number):
                        self.lock_histories[t.track_id]["changes"] += 1
                self.lock_histories[t.track_id]["last_state"] = t.locked_number
            else:
                # Was locked, now unlocked -> Change?
                if self.lock_histories[t.track_id]["last_state"] is not None:
                    self.lock_histories[t.track_id]["changes"] += 1
                    self.lock_histories[t.track_id]["last_state"] = None

        # 3. Reactivations
        refind_count = matches_info.get("refind_count", 0)
        self.reactivations_total += refind_count

        entry = {
            "frame_idx": int(frame_idx),
            "det_count": len(detections),
            "active_track_count": len(active_ids),
            "locked_track_count": current_locked_count,
            "new_tracks_count": len(new_ids),
            "matched_pairs": matches_info.get("matched_count", 0),
            "refind_count": refind_count,
            "mean_iou": float(matches_info.get("mean_iou", 0.0)),
            "mean_siglip_sim": float(matches_info.get("mean_reid", 0.0)),
            "unmatched_reasons": matches_info.get("unmatched_reasons", {}),
            "active_ids": active_ids # Enable post-hoc unique count
        }
        
        # Write to file (append)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def _update_jersey_history(self, track_id, jersey_num):
        self.jersey_histories[track_id][str(jersey_num)] += 1
        
    def read_log(self):
        data = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            pass
        return data

    def generate_summary(self, identity_manager=None):
        """
        Computes final acceptance metrics.
        """
        log_data = self.read_log()
        total_frames = max([e["frame_idx"] for e in log_data] + [0]) if log_data else 0
        
        # Rebuild histograms from log data if empty (post-hoc analysis)
        if not self.track_histories and log_data:
            # Try to recover from 'active_ids' field if available
            self.track_histories = defaultdict(int)
            self.reactivations_total = 0
            for entry in log_data:
                # Reactivations
                self.reactivations_total += entry.get("refind_count", 0)
                # Durations
                active_ids = entry.get("active_ids", [])
                for tid in active_ids:
                    self.track_histories[tid] += 1
            
        total_unique = len(self.track_histories)
        
        # Rates
        rate = (total_unique / total_frames) * 1000 if total_frames > 0 else 0
        reactivation_rate = (self.reactivations_total / total_frames) * 1000 if total_frames > 0 else 0
        
        # Lengths
        lengths = list(self.track_histories.values())
        p10 = 0; p50 = 0; p90 = 0
        median_len = 0
        short_ratio = 0
        
        if lengths:
            lengths_sorted = sorted(lengths)
            p10 = np.percentile(lengths_sorted, 10)
            p50 = np.percentile(lengths_sorted, 50)
            p90 = np.percentile(lengths_sorted, 90)
            median_len = p50
            
            short_tracks = len([l for l in lengths if l < 5])
            short_ratio = short_tracks / len(lengths)
            
            # Histogram
            hist_bins = [0, 5, 10, 50, 100, 500, 1000, 99999]
            hist_counts, _ = np.histogram(lengths, bins=hist_bins)
            track_length_histogram = {
                f"{hist_bins[i]}-{hist_bins[i+1]}": int(hist_counts[i]) for i in range(len(hist_bins)-1)
            }
        else:
            track_length_histogram = {}

        # --- Task A: Detection Stability & Track Health ---
        # Dropout Rate: % frames with < 16 active tracks (Proxy for failure)
        dropout_frames = 0
        det_counts_list = []
        low_conf_ratios = [] # Not strictly logged per frame, need detector raw info.
                             # Proxy: We can check 'det_count' vs 'active_track_count'
        
        if log_data:
             for entry in log_data:
                 det_c = entry.get("det_count", 0)
                 act_c = entry.get("active_track_count", 0)
                 det_counts_list.append(det_c)
                 if act_c < 16:
                     dropout_frames += 1
                     
        dropout_rate_pct = (dropout_frames / total_frames * 100) if total_frames > 0 else 0.0
        dets_per_frame_mean = np.mean(det_counts_list) if det_counts_list else 0.0
        
        # New/Lost Tracks per 1000
        # New Tracks = Total Unique? No, new_tracks_count per frame sum?
        # new_ids_per_1000_frames uses total_unique.
        # Let's stick with that.
        
        # Lost Tracks?
        # We don't log "lost_stracks" count explicitly in log_frame, but we can imply?
        # No, 'lost_stracks' is internal state.
        # We'll rely on short_tracks_pct as proxy for rapid loss.

            
        # Anti-Merge / Jersey Lock Changes Report
        merge_conflicts = []
        lock_changes_total = 0
        tracks_locked_count = 0
        total_active_duration = sum(self.track_histories.values())
        total_locked_duration = 0
        
        # Aggregate Lock Stats
        for tid, info in self.lock_histories.items():
            if info["total_locked"] > 0:
                tracks_locked_count += 1
                total_locked_duration += info["total_locked"]
            lock_changes_total += info["changes"]
            
        lock_rate_tracks = (tracks_locked_count / total_unique * 100) if total_unique > 0 else 0.0
        lock_rate_frames = (total_locked_duration / total_active_duration * 100) if total_active_duration > 0 else 0.0
        
        # Median Frames to Lock
        frames_to_lock_list = []
        for tid, info in self.lock_histories.items():
            if info["first_lock_frame"] is not None and info["start_frame"] is not None:
                frames_to_lock_list.append(info["first_lock_frame"] - info["start_frame"])
        
        median_frames_to_lock = 0
        if frames_to_lock_list:
             median_frames_to_lock = np.median(frames_to_lock_list)
        
        # Use IdentityManager if provided (Preferred/Accurate)
        if identity_manager:
            merge_conflicts = identity_manager.detect_merge_conflicts()
        else:
            # Fallback to local proxy (less accurate)
            for tid, counts in self.jersey_histories.items():
                major_nums = {n: c for n, c in counts.items() if c >= 3}
                if len(major_nums) > 1:
                    merge_conflicts.append({
                        "track_id": tid,
                        "conflicting_jerseys": major_nums,
                        "source": "proxy_diagnostics"
                    })

        # Aggregate Unmatched Reasons
        unmatched_reasons_agg = defaultdict(int)
        if log_data:
            for entry in log_data:
                 reasons = entry.get("unmatched_reasons", {})
                 for reason, count in reasons.items():
                     unmatched_reasons_agg[reason] += count

        summary = {
            "total_frames": total_frames,
            "raw_unique_track_ids": total_unique,
            "new_ids_per_1000_frames": round(rate, 2),
            "new_track_ratio": round(total_unique / (sum([e.get("det_count", 0) for e in log_data]) if log_data else 1), 4),
            "reactivations_per_1000_frames": round(reactivation_rate, 2),
            "reactivations_total": self.reactivations_total,
            "median_track_length_frames": round(median_len, 1),
            "track_length_p10": round(p10, 1),
            "track_length_p50": round(p50, 1),
            "track_length_p90": round(p90, 1),
            "tracks_shorter_than_5_frames_pct": round(short_ratio * 100, 2),
            "dropout_rate_pct": round(dropout_rate_pct, 2),
            "dets_per_frame_mean": round(dets_per_frame_mean, 2),
            
            # Tasks D & E Metrics
            "consolidated_player_count": len(set([m['track_id'] for m in merge_conflicts])) if merge_conflicts else total_unique, # Proxy
            "merge_error_count": len(merge_conflicts),
            "risky_match_total": self.risky_match_total if self.risky_match_total > 0 else sum([e.get("unmatched_reasons", {}).get("risky_match", 0) for e in log_data]) if log_data else 0,
            "lock_rate_tracks_pct": round(lock_rate_tracks, 1),
            "lock_rate_frames_pct": round(lock_rate_frames, 1),
            "lock_changes_total": lock_changes_total,
            "median_frames_to_lock": round(median_frames_to_lock, 1),
            
            "unmatched_reasons_total": unmatched_reasons_agg,
            "merge_conflicts_details": merge_conflicts,
            "track_length_histogram": track_length_histogram
        }
        
        # Write Summary
        summary_path = self.log_path.replace(".jsonl", ".json").replace("metrics", "summary")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"📊 [TrackerDiagnostics] Summary written to {summary_path}")
        return summary
