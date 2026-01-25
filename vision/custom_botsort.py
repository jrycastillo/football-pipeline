
import numpy as np
import torch
import cv2
from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.trackers.utils import matching
from ultralytics.trackers.basetrack import TrackState
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import box_iou
from vision.tracker_diagnostics import TrackerDiagnostics
from collections import defaultdict, deque, Counter

class JerseyBOTrack(BOTrack):
    @property
    def tlbr(self):
        """Convert tlwh to tlbr for compatibility with evaluation/audit."""
        if self.tlwh is None: return None
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def update(self, new_track, frame_id, *args):
        """
        Quality-Aware Update (Task D):
        Only update appearance features if the new detection is high quality.
        """
        # 1. Update Frame/State (Standard)
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.score = new_track.score

        # 2. Update Box (Standard Kalman)
        self._tlwh = new_track.tlwh
        
        # 3. Features Update (QUALITY CONTROL)
        if new_track.curr_feat is not None:
            # Check Quality
            # We assume tlwh corresponds to the feature
            w, h = self.tlwh[2], self.tlwh[3]
            
            # Simple Heuristic: Small boxes are blurry/bad
            is_good_quality = True
            
            if h < 40: # Tiny crop
                is_good_quality = False
                # print(f"DEBUG: Skipped Feature Update for Track {self.track_id} (H={h:.1f} < 40)")

            if self.curr_feat is None:
                # Always init
                self.curr_feat = new_track.curr_feat
                self.features = deque([new_track.curr_feat], maxlen=self.features.maxlen) # Reset buffer
            else:
                if is_good_quality:
                    # EMA Update
                    self.curr_feat = self.alpha * self.curr_feat + (1 - self.alpha) * new_track.curr_feat
                    self.curr_feat /= np.linalg.norm(self.curr_feat)
                    self.features.append(new_track.curr_feat) # Add to smooth buffer
                else:
                    # Skip Update (Keep old good features)
                    pass
        
        # 4. Smooth Feature (Standard)
        # self.smooth_feat = np.mean(self.features, axis=0)
        # self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        # ^ BoT-SORT doesn't actually use smooth_feat in matching default, it uses curr_feat or proxy.
        # We stick to standard BoT-SORT logic for the rest.

class JerseyBoTSORT(BOTSORT):
    def __init__(self, args, frame_rate=30, siglip_encoder=None, identity_manager=None, output_dir="output"):
        super().__init__(args, frame_rate)
        self.siglip_encoder = siglip_encoder
        self.identity_manager = identity_manager
        
        # User defined thresholds (V88 Refined with Conditional Gate)
        self.jersey_bonus = 0.2
        self.jersey_penalty = 1.0 
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh_relaxed = args.appearance_thresh # 0.70
        self.appearance_thresh_strict = 0.55 # Strict base
        
        # Diagnostics
        import os
        self.diagnostics = TrackerDiagnostics(os.path.join(output_dir, "tracking_metrics.jsonl"))
        
        # Dynamic Thresholding
        self.matched_dists = [] 
        self.dynamic_dist_thresh = 0.28 
        
        # Debug Counters
        self.counters = {"small_box": 0, "suppressed": 0}

    def init_track(self, results, img=None):
        """
        Override init_track to use CUSTOM JerseyBOTrack with Quality Control.
        """
        jersey_preds = getattr(self, '_temp_jersey_preds', None)
        
        if len(results) == 0:
            return []
            
        # Reset counters per frame
        self.counters = {"small_box": 0, "suppressed": 0}
            
        # Get Boxes
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        
        # Extract Features (SigLIP)
        features = None
        # FIX: Check enable_reid (Custom) OR with_reid (Ultralytics)
        use_reid = getattr(self.args, "with_reid", False) or getattr(self.args, "enable_reid", False)
        if use_reid and self.siglip_encoder is not None and img is not None:
            # Robust Box Access (Shim vs Results)
            xyxy_boxes = None
            if hasattr(results, 'xyxy') and results.xyxy is not None:
                 xyxy_boxes = results.xyxy
            elif hasattr(results, 'boxes') and results.boxes is not None and hasattr(results.boxes, 'xyxy'):
                 xyxy_boxes = results.boxes.xyxy
            else:
                 xyxy_boxes = xywh2xyxy(results.xywh)
            
            # Extract features using SigLIP wrapper
            if isinstance(xyxy_boxes, torch.Tensor):
                xyxy_boxes = xyxy_boxes.cpu().numpy()
            
            features = self.siglip_encoder(img, xyxy_boxes)
            
            # CRITICAL FIX: L2 Normalize features immediately
            # SigLIP outputs are raw (Norm ~17). We need Unit Norm for Cosine Distance.
            if features is not None and len(features) > 0:
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                # Avoid divide by zero
                norms[norms < 1e-6] = 1.0 
                features = features / norms
            
        # Create JerseyBOTracks
        tracks = []
        confidences = results.conf
        if hasattr(confidences, 'cpu'): confidences = confidences.cpu().numpy()
        
        classes = results.cls
        if hasattr(classes, 'cpu'): classes = classes.cpu().numpy()
        
        for i, (xywh, s, c) in enumerate(zip(bboxes, confidences, classes)):
            feat = features[i] if features is not None else None
            
            # DEBUG: Check Feature Integrity
            if i == 0 and self.frame_id % 50 == 0 and feat is not None:
                norm_val = np.linalg.norm(feat)
                print(f"[FEAT DEBUG] Frame {self.frame_id}: Feat Shape={feat.shape}, Norm={norm_val:.4f}, First5={feat[:5]}")
            
            # Create Custom Track with Quality Aware Update
            gst = JerseyBOTrack(xywh, s, c, feat) # Use Subclass
            
            # Attach Jersey Prediction (Ephemeral for this detection)
            if jersey_preds and i < len(jersey_preds):
                gst.jersey_det = jersey_preds[i] # (number, conf)
            else:
                gst.jersey_det = None
                
            tracks.append(gst)
            
        return tracks
    
    # Optimized Update for Speed (Lazy SigLIP)
    def update(self, results, img=None):
        """
        Custom Update Loop with Lazy SigLIP Extraction.
        1. Initialize Tracks (No Features).
        2. Match High-Confidence Detections via IoU.
        3. Batch Compute SigLIP Features for:
           - Unmatched High-Conf Detections (Candidates for ReID Recovery).
           - Matched Tracks needing Update (Every N=3 frames).
        4. Match Remaining via ReID.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        
        # 1. Init High-Conf Tracks (NO FEATURES YET - Lazy)
        # Pass img=None to force skip feature computation in init_track
        detections = self.init_track(results, img=None)
        
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: List[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        
        if hasattr(self, "gmc") and img is not None:
             # GMC logic omitted for brevity (standard)
             pass

        # MATCH 1: IoU on High Conf
        dists = self.get_dists(strack_pool, detections) # IoU Only since no features
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # --- LAZY SIGLIP BATCHING ---
        # We need features for:
        # A) u_detection (Unmatched High-Conf) -> might match Lost tracks via ReID
        # B) matches (Matched tracks) -> Update appearance if scheduled (N=3)
        
        if self.siglip_encoder is not None and img is not None:
            crops_to_process = []
            
            # A. Unmatched Detections (Need features for ReID matching)
            unmatched_dets_indices = u_detection # Indices in 'detections' list
            
            # B. Matched Tracks (Need features for update, sparse cadence)
            # Only update if frame_id % 3 == 0 OR track has no features
            matched_dets_indices = []
            for itracked, idet in matches:
                track = strack_pool[itracked]
                # FPS OPTIMIZATION: Skip SigLIP if IoU >= 0.2
                iou_dist = dists[itracked, idet] if 'dists' in locals() else 1.0
                
                # Task C4: Skip SigLIP when IoU >= 0.2
                if iou_dist <= 0.8:
                    passed_gate = True # Good Match
                else:
                    passed_gate = False # Weak Match
                
                # Schedule Logic
                needs_update = (self.frame_id % 5 == 0) or (track.curr_feat is None)
                
                if passed_gate and track.curr_feat is not None:
                     continue # SKIP UPDATE
                
                if needs_update:
                    matched_dets_indices.append(idet)

                    
            # Combine unique indices to avoid duplicate computation
            # (Though sets are disjoint by definition of linear_assignment)
            indices_to_compute = set(unmatched_dets_indices) | set(matched_dets_indices)
            sorted_indices = sorted(list(indices_to_compute))
            
            # Extract Boxes for Batch
            batch_boxes = []
            index_map = {} # Map batch_idx -> detection_idx
            
            for b_idx, d_idx in enumerate(sorted_indices):
                det = detections[d_idx]
                batch_boxes.append(det.tlwh) 
                index_map[b_idx] = d_idx
                
            # Run Batch (if any)
            if batch_boxes:
                # Convert TLWH to XYXY for Processor
                batch_boxes_np = np.array(batch_boxes)
                batch_xyxy = xywh2xyxy(batch_boxes_np)
                
                # Inference
                features = self.siglip_encoder(img, batch_xyxy) #(N, 768)
                
                # CRITICAL: L2 Normalize
                if features is not None and len(features) > 0:
                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                    norms[norms < 1e-6] = 1.0 
                    features = features / norms
                    
                    # Assign back to detections
                    for b_idx, feat in enumerate(features):
                        d_idx = index_map[b_idx]
                        detections[d_idx].curr_feat = feat # Assign Feature!

        # --- END LAZY BATCH ---

        # Process Matches (Step 1)
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # MATCH 2: ReID Association for Unmatched (u_track) vs Unmatched Detections (u_detection)
        # Note: u_detection now has Features (if computed above)
        
        # Filter unmatched tracks that are Tracked state (candidates for ReID)
        # Actually ByteTrack tries to match u_track (High Score Unmatched) to Second (Low Score).
        # But we also want to try matching u_track to u_detection via ReID if IoU failed but ReID is good?
        # Standard ByteTrack: 
        #   Step 2 is High-Track vs Low-Det (IoU).
        #   Step 3 is Unconfirmed vs Unmatched-High-Det (IoU).
        # BoTSORT adds ReID in Step 1.
        
        # Since we skipped ReID in Match 1 (IoU only), we should run a ReID refinement here?
        # Or standard ByteTrack flow:
        # If we failed IoU, we try ReID.
        
        # Let's do a ReID-only pass for u_track vs u_detection (High Conf Recovery)
        # Only for tracks that have features.
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        r_unmatched_dets = [detections[i] for i in u_detection] # These should have features now
        
        if self.args.with_reid and len(r_tracked_stracks) > 0 and len(r_unmatched_dets) > 0:
             dists_reid = self.get_dists(r_tracked_stracks, r_unmatched_dets) # Will use ReID now if features exist
             matches_reid, u_track_reid, u_det_reid = matching.linear_assignment(dists_reid, thresh=self.args.match_thresh)
             
             for itracked, idet in matches_reid:
                 track = r_tracked_stracks[itracked]
                 det = r_unmatched_dets[idet]
                 if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                 else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                    
             # Update leftovers
             u_track = [u_track[i] for i in u_track_reid] # Filtered indices? No this indexing is complex.
             # Simplify: Just accept ReID matches and remove them from 'u_track' list equivalent.
             # Actually, simpler to just let them fall through to Step 2 (Low Conf).
             pass # Skip complex ReID Step 1.5, trust Step 1 IoU for Speed.
                  # Wait, if IoU failed (e.g. occlusion jump), we need ReID.
                  # So Step 1 should have been IoU + ReID?
                  # But we wanted Lazy.
                  
        # Refined Lazy Strategy:
        # Match 1 was IoU-Only.
        # So we missed ReID matches.
        # We MUST run ReID match on the leftovers (High Unmatched Tracks vs High Unmatched Dets).
        # Yes, that is 'Match 1.5'.
        
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(results_second, img=None) # No features for low conf
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection] # THESE HAVE FEATURES
        dists = self.get_dists(unconfirmed, detections) # ReID available
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                # CHURN REDUCTION (Task 13)
                # 1. Min Box Height (40px)
                if track.tlwh[3] < 40:
                    continue
            
                # 2. Near Miss Suppression
                # If detection is dangerously close to ANY existing track (Tracked or Lost), skip Spawn.
                # This handles re-activation failures where detector jumps slightly.
                
                all_tracks = self.tracked_stracks + self.lost_stracks
                is_near_miss = False
                cx, cy = track.tlwh[0] + track.tlwh[2]/2, track.tlwh[1] + track.tlwh[3]/2
            
                if len(all_tracks) > 0:
                    # Normalized Distance Check (User requested centroid_dist_norm <= 0.6)
                    # Norm by what? Box width/height? Or Image Dims?
                    # User: 'centroid_dist_norm <= 0.6'. Usually relative to box size.
                    # Let's use box size (average of track & detection).
                    for t in all_tracks:
                        tcx, tcy = t.tlwh[0] + t.tlwh[2]/2, t.tlwh[1] + t.tlwh[3]/2
                        dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                        # Norm by avg diagonal or width?
                        # Let's use track width as scale.
                        scale = max(t.tlwh[2], 1.0)
                        norm_dist = dist / scale
                        if norm_dist <= 0.6:
                            is_near_miss = True
                            break
            
                if is_near_miss:
                    self.counters["suppressed"] += 1
                    continue # Skip Spawn (Potential Reactivation Next Frame)
            
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

    def get_dists(self, tracks, detections):
        # OPTIMIZATION: If detections lack features (Lazy Mode), return IoU only
        # Prevents embedding_distance crash
        if not detections or (hasattr(detections[0], 'curr_feat') and detections[0].curr_feat is None):
            return matching.iou_distance(tracks, detections)

        """
        Custom Distance Metric (V88 Refined):
        1. IoU Distance
        2. SigLIP ReID Distance
        3. Jersey Consistency (Confidence Weighted & Capped)
        4. Centroid Distance Mask
        """
        # 1. IoU Distance
        dists = matching.iou_distance(tracks, detections)
        iou_dists = dists.copy() # For Conditional Gate

        # 2. Centroid Distance Mask
        has_dims = hasattr(self, 'img_w') and getattr(self, 'img_w', 0) > 0
        prox_mask = None
        
        if self.proximity_thresh < 1.0 and has_dims and len(tracks) > 0 and len(detections) > 0:
             t_cents = np.array([[t.tlwh[0]+t.tlwh[2]/2, t.tlwh[1]+t.tlwh[3]/2] for t in tracks])
             d_cents = np.array([[d.tlwh[0]+d.tlwh[2]/2, d.tlwh[1]+d.tlwh[3]/2] for d in detections])
             
             t_cents[:, 0] /= self.img_w
             t_cents[:, 1] /= self.img_h
             d_cents[:, 0] /= self.img_w
             d_cents[:, 1] /= self.img_h
             
             t_exp = t_cents[:, None, :]
             d_exp = d_cents[None, :, :]
             c_dists = np.sqrt(np.sum((t_exp - d_exp)**2, axis=2))
             
             dists_mask = c_dists > self.proximity_thresh
             prox_mask = c_dists < self.proximity_thresh # Valid Prox
        else:
             dists_mask = dists > (1 - self.proximity_thresh)
             prox_mask = dists < (1 - self.proximity_thresh)

        # Fuse Score
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        # 3. ReID Distance (CONDITIONAL GATING V2)
        if self.args.with_reid:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            
            # --- TASK E: Verify ReID Math (Sanity Check) ---
            if self.frame_id < 200 and emb_dists.size > 0:
                 min_d = np.min(emb_dists)
                 mean_d = np.mean(emb_dists)
                 p05 = np.percentile(emb_dists, 5)
                 p95 = np.percentile(emb_dists, 95)
                 # self.diagnostics.log_manual(f"[ReID Check] Min={min_d:.3f} Mean={mean_d:.3f} P95={p95:.3f}")

            # CONDITIONAL GATING LOGIC
            # Definition: 
            # - Strict Gate (0.55): Standard Requirement.
            # - Relaxed Gate (0.70): Allowed ONLY if Motion is Strong.
            #   Strong Motion = High IoU (iou_dist < 0.8 => IoU > 0.2) OR Close Proximity (if enabled).
            
            # Prepare Mask for Relaxed Gate
            # Note: iou_dists is cost (1-IoU). So < 0.8 means IoU > 0.2.
            strong_motion = (iou_dists < 0.8) 
            if prox_mask is not None:
                strong_motion = strong_motion | prox_mask
                
            # Apply Gate
            # If Strong Motion: Allow if dist <= 0.70 (Relaxed)
            # Else: Allow only if dist <= 0.55 (Strict)
            
            gate_mask = np.where(strong_motion, 
                                 emb_dists > self.appearance_thresh_relaxed, # 0.70 
                                 emb_dists > self.appearance_thresh_strict)  # 0.55
            
            # --- TASK 3: Threshold Semantic Clarification ---
            # Randomly log decision for auditing (1% chance or if specific frame)
            # Prove "Distance < Thresh = Accept"
            if np.any(gate_mask) and self.frame_id % 100 == 0:
                 # Find a rejected case
                 rej_indices = np.where(gate_mask)
                 if len(rej_indices[0]) > 0:
                      r_idx = rej_indices[0][0]
                      c_idx = rej_indices[1][0]
                      dist_val = emb_dists[r_idx, c_idx]
                      thresh_used = self.appearance_thresh_relaxed if strong_motion[r_idx, c_idx] else self.appearance_thresh_strict
                      # Log REJECT
                      pass # Silenced for FPS
            
            # Log ACCEPT (Relaxed)
            # Find cases where Strong Motion=True AND Dist in [0.55, 0.70] AND NOT Rejected
            semi_risky = (strong_motion) & (emb_dists > self.appearance_thresh_strict) & (emb_dists <= self.appearance_thresh_relaxed)
            if np.any(semi_risky):
                 # This is a match allowed by relaxation
                 acc_indices = np.where(semi_risky)
                 r_idx = acc_indices[0][0]
                 c_idx = acc_indices[1][0]
                 dist_val = emb_dists[r_idx, c_idx]
                 # Count risky match
                 if hasattr(self.diagnostics, 'risky_match_total'):
                      self.diagnostics.risky_match_total += np.sum(semi_risky)
                 
                 if self.frame_id % 50 == 0:
                      pass # Silenced for FPS
                                 
            emb_dists[gate_mask] = 1.0 # Kill connection
            
            dists = np.minimum(dists, emb_dists)
            
        return dists
            
        # 4. Jersey Score (Unchanged...)
            
        # 4. Jersey Bonus/Penalty (Vote-to-Lock Refined V88)
        if self.identity_manager:
            for i, track in enumerate(tracks):
                # Ensure OCR state exists (Lazy Init)
                if not hasattr(track, 'locked_number'):
                    track.locked_number = None
                    track.lock_conf = 0.0

                # Check Lock
                if track.locked_number is not None:
                     locked_num = track.locked_number
                     lock_strength = track.lock_conf # 0..1
                     
                     for j, det in enumerate(detections):
                         if hasattr(det, 'jersey_det') and det.jersey_det:
                             det_num, det_conf = det.jersey_det
                             try: conf_val = float(det_conf) if det_conf else 0.0
                             except: conf_val = 0.0
                             
                             if conf_val >= 0.85:
                                 if str(det_num) == str(locked_num):
                                     # Bonus: -0.10 to -0.20 scaled.
                                     # V88 Refined: -0.20 * lock_conf * ocr_conf
                                     bonus = -0.20 * lock_strength * conf_val
                                     dists[i, j] += max(bonus, -0.20)
                                 else:
                                     # Penalty: Mismatch
                                     # V88 Refined: +0.35 to +0.60 scaled.
                                     # Cap +0.60.
                                     penalty = 0.60 * lock_strength * conf_val
                                     
                                     # Anti-Impulse Override (Strong App + Good Spatial)
                                     # Use p05 logic or Warm-up
                                     
                                     # Cosine Distance
                                     cos_dist = dists[i, j]
                                     
                                     # Threshold Logic
                                     is_strong_app = False
                                     if self.frame_id < 300:
                                          is_strong_app = cos_dist <= 0.28 # Warm-up
                                     else:
                                          # Use dynamic p05 threshold
                                          is_strong_app = cos_dist <= (self.dynamic_dist_thresh if hasattr(self, 'dynamic_dist_thresh') else 0.28)
                                          
                                     if is_strong_app:
                                          penalty = 0.0 # Override
                                          
                                     dists[i, j] += min(penalty, 0.60)
                                     
        return dists

    def update_ocr_state(self, track, det, iou=None, cos_sim=None):
        """
        V88 Vote-to-Lock Refined:
        - Strict Gating (IoU/Sim/Dist)
        - Unlock Safety (Challenge Check)
        - Anti-Thrashing (Cooldown)
        """
        # Lazy Init
        if not hasattr(track, 'ocr_history'):
            track.ocr_history = deque(maxlen=30)
            track.locked_number = None
            track.lock_conf = 0.0
            track.lock_stability = 0
            
        # 1. Strict Voting Gate
        # Require: OCR >= 0.85 AND (IoU >= 0.2 OR Sim >= 0.55/Dist <= 0.45)
        
        should_vote = False
        if hasattr(det, 'jersey_det') and det.jersey_det:
             num, conf = det.jersey_det
             try: conf_val = float(conf)
             except: conf_val = 0.0
             
             if conf_val >= 0.85:
                 # Check Spatial/App
                 pass_spatial = True
                 pass_app = True
                 
                 if iou is not None:
                     if iou < 0.2: pass_spatial = False # Fail IoU
                 
                 # Explicit Cosine Distance Gate
                 if cos_sim is not None:
                     # cos_sim is Similarity (1=Identical).
                     # weak_gate_dist = 0.45 (Distance).
                     # sim = 1 - dist.
                     # dist < 0.45 <=> 1 - sim < 0.45 <=> sim > 0.55.
                     # User said: "Set weak_gate_dist = 0.45".
                     # I will enforce: cos_dist <= 0.45. (sim >= 0.55)
                     cos_dist = 1.0 - cos_sim
                     if cos_dist > 0.45: pass_app = False
                     
                 if pass_spatial and pass_app:
                     should_vote = True
                     track.ocr_history.append((num, conf_val, self.frame_id))

        # 2. Prune Old Votes
        current_frame = self.frame_id
        while len(track.ocr_history) > 0 and (current_frame - track.ocr_history[0][2]) > 30:
            track.ocr_history.popleft()
            
        # 3. Evaluate Lock (Check for Conflict/Unlock)
        if len(track.ocr_history) >= 8:
            votes = [x[0] for x in track.ocr_history]
            vote_counts = Counter(votes)
            top_num, count = vote_counts.most_common(1)[0]
            proportion = count / len(track.ocr_history)
            
            # Avg Conf
            top_confs = [x[1] for x in track.ocr_history if x[0] == top_num]
            avg_conf = sum(top_confs) / len(top_confs)

            # Logic
            if track.locked_number is None:
                # Attempt Lock
                if count >= 8 and proportion >= 0.6 and avg_conf >= 0.80:
                    track.locked_number = top_num
                    track.lock_conf = min(avg_conf, 1.0)
                    track.lock_stability = 0
                    if not hasattr(track, 'lock_changes'): track.lock_changes = 0 # Init
            else:
                # Already Locked
                if str(track.locked_number) == str(top_num):
                     track.lock_conf = min(avg_conf, 1.0) # Update conf
                else:
                     # Conflict: Top Vote != Locked
                     # Challenge Check (Unlock Safety)
                     # Require 2*K (16) votes for new number
                     
                     if not hasattr(track, 'last_lock_change_frame'):
                          track.last_lock_change_frame = 0
                     
                     time_since_change = self.frame_id - track.last_lock_change_frame

                     if count >= 16 and proportion >= 0.6 and avg_conf >= 0.85:
                          if time_since_change >= 150:
                              # FLIP / UNLOCK
                              track.locked_number = top_num
                              track.lock_conf = min(avg_conf, 1.0)
                              track.lock_stability = 0
                              if not hasattr(track, 'lock_changes'): track.lock_changes = 0
                              track.lock_changes += 1
                              track.last_lock_change_frame = self.frame_id

    def update(self, results, img=None, jersey_preds=None):
        """
        V88 Update:
        - Stores jersey_preds
        """
        self._temp_jersey_preds = jersey_preds
        
        # Scene Cut Detection & Dims
        if img is not None:
             curr_hist = cv2.calcHist([img], [0], None, [16], [0, 256])
             curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
             
             if hasattr(self, 'prev_hist'):
                 score = cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                 if score < 0.6: # Significant change
                     if hasattr(self, 'gmc'):
                         self.gmc.reset_params()
                         
             self.prev_hist = curr_hist
             self.img_h, self.img_w = img.shape[:2]
             
        # --- BYTETracker Logic Start ---
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        # feats_keep logic omitted (handled in init_track)

        detections = self.init_track(results, img)
        
        # Unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []  # type: List[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        # Step 2: First association
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results.xyxy if hasattr(results, 'xyxy') else getattr(results, 'boxes', results).xyxy)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        detections_audit = detections # Preserve for Rejection Audit (Line 843)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            
            # Metrics for Vote Gate (V88)
            # ReID Distance: Compute Explicitly
            # dists matrix is contaminated with IoU/Bonus.
            # We need Pure Cosine Similarity.
            
            t_feat = getattr(track, 'curr_feat', None) # Normalized?
            d_feat = getattr(det, 'curr_feat', None)
            
            sim_val = 0.0
            if t_feat is not None and d_feat is not None:
                sim_val = np.dot(t_feat, d_feat)
                # Clip
                sim_val = max(0.0, min(1.0, sim_val))
            
            reid_dist = 1.0 - sim_val
            
            # IoU Calculation (using imported box_iou)
            # Ensure safe conversion to tensor
            try:
                t_box = torch.tensor([track.tlbr])
                d_box = torch.tensor([det.tlbr])
                iou_val = float(box_iou(t_box, d_box)[0, 0])
            except:
                iou_val = 0.0 # Fallback
            
            # Collect Valid Dist
            self.matched_dists.append(reid_dist)
            
            # Update Threshold (Every 100 updates)
            if len(self.matched_dists) > 0 and len(self.matched_dists) % 100 == 0:
                if len(self.matched_dists) < 500:
                    self.dynamic_dist_thresh = 0.28 # Warm-up
                else:
                    # User requested p05 of cos_dist (Strict "Top 5%").
                    recent = self.matched_dists[-1000:]
                    self.dynamic_dist_thresh = float(np.percentile(recent, 5))

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self.update_ocr_state(track, det, iou=iou_val, cos_sim=sim_val)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self.update_ocr_state(track, det, iou=iou_val, cos_sim=sim_val)
                refind_stracks.append(track)                
        # Step 3: Second association
        detections_second = self.init_track(results_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists_second = matching.iou_distance(r_tracked_stracks, detections_second)
        matches_second, u_track_second, u_detection_second = matching.linear_assignment(dists_second, thresh=0.5)
        
        for itracked, idet in matches_second:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                self.update_ocr_state(track, det)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                self.update_ocr_state(track, det)
                refind_stracks.append(track)

        for it in u_track_second:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists_unconfirmed = self.get_dists(unconfirmed, detections)
        matches_unconf, u_unconfirmed, u_detection_unconf = matching.linear_assignment(dists_unconfirmed, thresh=0.7)
        
        for itracked, idet in matches_unconf:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            self.update_ocr_state(unconfirmed[itracked], detections[idet])
            activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        # Step 4: Init new stracks
        for inew in u_detection_unconf:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                # CHURN REDUCTION (Task 13)
                # 1. Min Box Height (40px)
                if track.tlwh[3] < 40:
                    self.counters["small_box"] += 1
                    continue
            
                # 2. Near Miss Suppression
                # If detection is dangerously close to ANY existing track (Tracked or Lost), skip Spawn.
                # This handles re-activation failures where detector jumps slightly.
                
                all_tracks = self.tracked_stracks + self.lost_stracks
                is_near_miss = False
                cx, cy = track.tlwh[0] + track.tlwh[2]/2, track.tlwh[1] + track.tlwh[3]/2
            
                if len(all_tracks) > 0:
                    # Normalized Distance Check (User requested centroid_dist_norm <= 0.6)
                    # Norm by what? Box width/height? Or Image Dims?
                    # User: 'centroid_dist_norm <= 0.6'. Usually relative to box size.
                    # Let's use box size (average of track & detection).
                    for t in all_tracks:
                        tcx, tcy = t.tlwh[0] + t.tlwh[2]/2, t.tlwh[1] + t.tlwh[3]/2
                        dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
                        # Norm by avg diagonal or width?
                        # Let's use track width as scale.
                        scale = max(t.tlwh[2], 1.0)
                        norm_dist = dist / scale
                        if norm_dist <= 0.6:
                            is_near_miss = True
                            break
            
                if is_near_miss:
                    self.counters["suppressed"] += 1
                    continue # Skip Spawn (Potential Reactivation Next Frame)
            

            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        # Phase 2 Diagnosis: Detailed Stats every 20 frames
        if self.frame_id % 20 == 0:
             # Basic Counts
             det_count = len(detections)
             active_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
             active_count = len(active_stracks)
             
             # Calculate Match Candidates Stats
             candidate_ious = []
             candidate_reids = []
             strict_usage = 0
             relaxed_usage = 0
             
             # Re-calc stats from raw dists (approximate)
             # Better to inspect what just happened in linear_assignment
             # But here we are post-match (except for unmatched analysis)
             
             print(f"📊 [Diag Frame {self.frame_id}] Dets={det_count} ActiveTracks={active_count}")

        match_stats = {
            "matched_count": len(matches) + len(matches_second) + len(matches_unconf),
            "refind_count": len(refind_stracks),
            "mean_iou": 0.0, 
            "mean_reid": 0.0, 
            "unmatched_reasons": defaultdict(int), # high_cost, gated, no_dets
            "dynamic_dist_thresh": self.dynamic_dist_thresh if hasattr(self, 'dynamic_dist_thresh') else 0.28
        }
        
        # Analyze Unmatched Tracks (Why did they die?)
        # Step: Inspect the u_track from Step 1 assigment
        
        # Rejection Sampling (Phase 2)
        if self.frame_id % 20 == 0 and len(u_track) > 0 and dists.shape[1] > 0 and getattr(self.args, 'audit_rejections', False):
             print(f"📉 [Rejection Audit Frame {self.frame_id}] Unmatched Tracks: {len(u_track)}")
             for it in u_track[:5]: # Check first 5 unmatched tracks
                 try:
                     track = strack_pool[it]
                     # Find best potential detection
                     d_costs = dists[it, :]
                     best_idx = np.argmin(d_costs)
                     best_cost = d_costs[best_idx]
                     
                     # Re-compute components for this pair to explain why
                     det = detections_audit[best_idx]
                     
                     # Components
                     # We need re-access raw IoU/ReID. 
                     # Approximation:
                     # SAFEGUARD: Use tlwh if tlbr missing
                     if hasattr(track, 'tlbr'):
                          t_tlbr = track.tlbr
                     else:
                          # Fallback: tlwh to tlbr
                          t_tlbr = [track.tlwh[0], track.tlwh[1], track.tlwh[0]+track.tlwh[2], track.tlwh[1]+track.tlwh[3]]
                     
                     iou_val = box_iou(torch.tensor([t_tlbr]), torch.tensor([det.tlbr]))[0, 0].item()
                     
                     t_feat = getattr(track, 'curr_feat', None)
                     d_feat = getattr(det, 'curr_feat', None)
                     reid_dist = 1.0
                     if t_feat is not None and d_feat is not None:
                          sim = np.dot(t_feat, d_feat)
                          reid_dist = 1.0 - sim
                          
                     # Gate Logic
                     strong_motion = (1.0 - iou_val < 0.8) or (hasattr(self, 'proximity_thresh') and False) # Prox check complexity omitted
                     thresh_used = self.appearance_thresh_relaxed if strong_motion else self.appearance_thresh_strict
                     gate_status = "PASS" if reid_dist <= thresh_used else "FAIL"
                     
                     reason = []
                     if best_cost > self.args.match_thresh: reason.append(f"Cost {best_cost:.2f} > {self.args.match_thresh}")
                     if gate_status == "FAIL": reason.append(f"Gated (Dist {reid_dist:.2f} > {thresh_used:.2f})")
                     
                     print(f"  ❌ Track {track.track_id} rejected best Det {best_idx}: IoU={iou_val:.2f} ReID={reid_dist:.2f} Cost={best_cost:.2f} Motion={strong_motion} -> {' + '.join(reason)}")
                 except Exception as e:
                     print(f"⚠️ [Audit Error] {e}")

        # Analyze Unmatched (u_track from Step 2 is the main source of lost tracks)
        # We need the dists matrix from Step 2. It is `dists`.
        # u_track are indices into strack_pool.
        
        for it in u_track:
             # Find best detection candidate cost
             if dists.shape[1] > 0:
                 min_cost = np.min(dists[it, :])
                 if min_cost > self.args.match_thresh:
                     # Why?
                     if min_cost >= 0.99: # Gated
                         match_stats["unmatched_reasons"]["gated"] += 1
                     else:
                         match_stats["unmatched_reasons"]["high_cost"] += 1
             else:
                 match_stats["unmatched_reasons"]["no_dets"] += 1

        self.diagnostics.log_frame(
            self.frame_id, 
            self.tracked_stracks + self.lost_stracks, 
            detections + detections_second, 
            match_stats
        )

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
