import numpy as np
import cv2

# Standard Pitch Dimensions (Meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Goal Dimensions
GOAL_WIDTH = 7.32

class Camera:
    def __init__(self, homography_matrix=None, frame_width=None, frame_height=None):
        if homography_matrix is not None:
             self.H = np.array(homography_matrix)
        else:
             # Round 16: Resolution-aware scaling
             # Map actual video dimensions to standard pitch (105m x 68m)
             self.H = np.eye(3)
             fw = frame_width or 1920
             fh = frame_height or 1080
             self.H[0, 0] = PITCH_LENGTH / fw   # 105.0 / 1920 ≈ 0.0547
             self.H[1, 1] = PITCH_WIDTH / fh     # 68.0 / 1080  ≈ 0.0630
             
    def project_point(self, x, y):
        """
        Convert Image (px) to Pitch (meters).
        """
        p = np.array([x, y, 1.0])
        mapped = np.dot(self.H, p)
        if mapped[2] != 0:
            xm = mapped[0] / mapped[2]
            ym = mapped[1] / mapped[2]
            return (xm, ym)
        return (0.0, 0.0)

    def calculate_distance(self, p1, p2):
        """
        Calculate Euclidean distance in METERS between two Image Points (px).
        """
        x1_m, y1_m = self.project_point(p1[0], p1[1])
        x2_m, y2_m = self.project_point(p2[0], p2[1])
        return np.sqrt((x2_m - x1_m)**2 + (y2_m - y1_m)**2)

    def is_in_penalty_box(self, point_px):
        """
        Check if point is in Penalty Box.
        BoxDims: 16.5m from goal line, 40.3m wide (centered).
        """
        xm, ym = self.project_point(point_px[0], point_px[1])
        
        # Assuming origin (0,0) is top-left corner, Goal at X=0 and X=105
        # Penalty Box 1 (Left Goal): X in [0, 16.5], Y in [13.84, 54.16] (Centered on 68/2 = 34)
        
        box_depth = 16.5
        box_width_half = 20.15 # 40.3 / 2
        cy = PITCH_WIDTH / 2.0
        
        # Check Box 1 (Left)
        if 0 <= xm <= box_depth:
            if cy - box_width_half <= ym <= cy + box_width_half:
                return True
                
        # Check Box 2 (Right)
        if PITCH_LENGTH - box_depth <= xm <= PITCH_LENGTH:
            if cy - box_width_half <= ym <= cy + box_width_half:
                return True
                
        return False

    def is_side_channel(self, point_px):
        """
        Check if point is in Side Channels (Outer 15% of width).
        """
        xm, ym = self.project_point(point_px[0], point_px[1])
        
        margin = PITCH_WIDTH * 0.15 # 10.2m
        
        # Top Channel
        if 0 <= ym <= margin: return True
        # Bottom Channel
        if PITCH_WIDTH - margin <= ym <= PITCH_WIDTH: return True
        
        return False
        
    def get_shot_cone_angle(self, ball_px):
        """
        Calculate angle subtended by goal posts from ball position.
        """
        try:
            bx, by = self.project_point(ball_px[0], ball_px[1])
            
            # Goal Posts (Left Goal for simplicity, or find nearest)
            # Goal Center (0, 34)
            # Posts: (0, 30.34), (0, 37.66)
            
            # Determine nearest goal
            if bx < PITCH_LENGTH / 2:
                # Left Goal
                gx = 0
            else:
                # Right Goal
                gx = PITCH_LENGTH
            
            mid_y = PITCH_WIDTH / 2.0
            post1 = (gx, mid_y - GOAL_WIDTH/2)
            post2 = (gx, mid_y + GOAL_WIDTH/2)
            
            # Vector Ball->P1
            v1 = (post1[0]-bx, post1[1]-by)
            v2 = (post2[0]-bx, post2[1]-by)
            
            # Angle
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 * mag2 == 0: return 0.0
            
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            angle = np.arccos(np.clip(dot / (mag1 * mag2), -1.0, 1.0))
            return np.degrees(angle)
        except:
             return 0.0

    def is_opponent_in_cone(self, ball_px, opponents_px):
        """
        Check if any opponent is inside the triangle (Ball, Post1, Post2).
        """
        # (Simplified: Check if opponent is 'between' ball and goal)
        # For now, just check distance threshold in direction of goal
        return False # Placeholder
