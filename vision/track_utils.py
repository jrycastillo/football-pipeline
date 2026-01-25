import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) of two boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def greedy_match(detections, tracks, iou_threshold=0.3):
    """
    Greedy assignment of detections to tracks based on IoU.
    detections: list of [x1, y1, x2, y2]
    tracks: dict of {track_id: [x1, y1, x2, y2]}
    
    Returns:
        matches: list of (det_idx, track_id)
        unmatched_detections: list of det_idx
        unmatched_tracks: list of track_id
    """
    if not detections:
        return [], [], list(tracks.keys())
    if not tracks:
        return [], list(range(len(detections))), []

    track_ids = list(tracks.keys())
    track_boxes = [tracks[tid] for tid in track_ids]
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(detections), len(track_ids)))
    for i, det in enumerate(detections):
        for j, trk in enumerate(track_boxes):
            iou_matrix[i, j] = calculate_iou(det, trk)

    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(track_ids)

    # Greedy matching
    while np.any(iou_matrix > iou_threshold):
        # Find highest IoU
        idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        det_idx, trk_idx_local = idx
        val = iou_matrix[idx]
        
        if val < iou_threshold:
            break
            
        tid = track_ids[trk_idx_local]
        matches.append((det_idx, tid))
        
        # Remove matched det and track from future consideration
        iou_matrix[det_idx, :] = -1
        iou_matrix[:, trk_idx_local] = -1
        
        if det_idx in unmatched_detections:
            unmatched_detections.remove(det_idx)
        if tid in unmatched_tracks:
            unmatched_tracks.remove(tid)
            
    return matches, unmatched_detections, unmatched_tracks
