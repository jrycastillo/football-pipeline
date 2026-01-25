import numpy as np
from scipy.cluster.vq import kmeans2
from vision.color_utils import get_closest_color_name

def cluster_teams_by_color(color_samples):
    """
    Input: {track_id: [[r,g,b], [r,g,b]...]}
    Output: {track_id: team_id (0 or 1)}
    """
    if not color_samples:
        return {}
        
    # 1. Aggregate mean color per track
    track_means = {}
    all_means = []
    
    for tid, samples in color_samples.items():
        if not samples: continue
        # Calculate mean of samples for this track
        arr = np.array(samples)
        mean_rgb = np.mean(arr, axis=0) # [R, G, B]
        track_means[tid] = mean_rgb
        all_means.append(mean_rgb)
        
    if len(all_means) < 2:
        return {tid: 0 for tid in track_means}
        
    # 2. Run K-Means (k=2) on ALL track means
    # data needs to be float32
    data = np.array(all_means, dtype=np.float32)
    
    # Check variance. If very low, maybe only 1 team?
    # But usually there's referee or GK. We'll force 2 clusters.
    
    try:
        centroids, labels = kmeans2(data, 2, minit='points')
    except Exception as e:
        print(f"[clustering] K-Means failed: {e}")
        return {tid: 0 for tid in track_means}
        
    # 3. Assign
    # labels corresponds to 'all_means' order
    
    # We need to map back to track IDs
    # But dictionaries are unordered (pre-3.7) or ordered (3.7+).
    # To be safe, we should have used a list of (tid, mean).
    
    ordered_tids = list(track_means.keys())
    ordered_means = np.array([track_means[tid] for tid in ordered_tids], dtype=np.float32)
    
    # Predict labels for each track
    # Simple distance to centroids
    final_map = {}
    
    c0 = centroids[0]
    # Sort centroids by brightness or something to be deterministic?
    # Or just return them.
    
    # If same name, add simple suffix or just keep (user asked for visual color, if both red, then "Red" is truth).
    # But usually teams are distinct.
    
    c0_name = get_closest_color_name(centroids[0])
    c1_name = get_closest_color_name(centroids[1])

    print(f"[clustering] Team 0 Color: {c0_name} ({centroids[0]})")
    print(f"[clustering] Team 1 Color: {c1_name} ({centroids[1]})")
    
    final_map = {}
    
    # Map track_id directly to COLOR NAME
    for i, tid in enumerate(ordered_tids):
        rgb = ordered_means[i]
        d0 = np.sum((rgb - centroids[0])**2)
        d1 = np.sum((rgb - centroids[1])**2)
        
        final_map[tid] = c0_name if d0 < d1 else c1_name
        
    print(f"[clustering] Clustered {len(ordered_tids)} tracks into {c0_name} / {c1_name}")
    return final_map
