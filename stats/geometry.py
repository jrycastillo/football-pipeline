import math
import numpy as np

def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def is_in_zone(pt, zone_rect):
    x, y = pt
    x1, y1, x2, y2 = zone_rect
    return x1 <= x <= x2 and y1 <= y <= y2

def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang)
