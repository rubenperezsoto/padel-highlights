from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol


# Thresholds and Constants
BALL_CLOSE_THRESHOLD = 50.0  # Pixels
BALL_SPEED_THRESHOLD = 5.0   # Pixels per frame
PLAYER_SPEED_THRESHOLD = 2.0 # Pixels per frame
DIRECTION_CHANGE_ANGLE_THRESHOLD = 45.0 # Degrees


class FrameDetection(Protocol):
    frame_index: int
    timestamp: float
    ball_visible: bool
    ball_x: Optional[float]
    ball_y: Optional[float]
    players: List[Dict[str, float]] # List of {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}


class FeatureExtractor:
    """
    Stateless class to compute temporal features from a sliding window of detections.
    """

    @staticmethod
    def compute_features(window_frames: List[FrameDetection | Dict[str, Any]], height: float = 720.0) -> Dict[str, float]:
        """
        Computes features for a given window of frames.
        Normalizes pixel-based features by the image height to be resolution-independent.
        """
        if not window_frames:
            return {}

        # Normalize input to dicts for easier access if they are objects
        frames = []
        for f in window_frames:
            if hasattr(f, "__dict__") or not isinstance(f, dict):
                frames.append({
                    "frame_index": getattr(f, "frame_index"),
                    "timestamp": getattr(f, "timestamp"),
                    "ball_visible": getattr(f, "ball_visible"),
                    "ball_x": getattr(f, "ball_x"),
                    "ball_y": getattr(f, "ball_y"),
                    "players": getattr(f, "players")
                })
            else:
                frames.append(f)

        num_frames = len(frames)
        
        # 1. ball_visible_ratio
        visible_count = sum(1 for f in frames if f["ball_visible"])
        ball_visible_ratio = visible_count / num_frames if num_frames > 0 else 0.0

        # 2. Ball speed features
        ball_speeds = []
        ball_velocities = []
        for i in range(1, num_frames):
            f1, f2 = frames[i-1], frames[i]
            if f1["ball_visible"] and f2["ball_visible"] and \
               f1["ball_x"] is not None and f2["ball_x"] is not None:
                dx = f2["ball_x"] - f1["ball_x"]
                dy = f2["ball_y"] - f1["ball_y"]
                dist = math.sqrt(dx**2 + dy**2)
                ball_speeds.append(dist / height) # Normalize
                ball_velocities.append((dx / height, dy / height)) # Normalize

        mean_ball_speed = sum(ball_speeds) / len(ball_speeds) if ball_speeds else 0.0
        
        std_ball_speed = 0.0
        if len(ball_speeds) > 1:
            variance = sum((s - mean_ball_speed)**2 for s in ball_speeds) / len(ball_speeds)
            std_ball_speed = math.sqrt(variance)

        # 3. num_ball_direction_changes
        num_ball_direction_changes = 0
        for i in range(1, len(ball_velocities)):
            v1 = ball_velocities[i-1]
            v2 = ball_velocities[i]
            
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                angle = math.degrees(math.acos(cos_theta))
                if angle > DIRECTION_CHANGE_ANGLE_THRESHOLD:
                    num_ball_direction_changes += 1

        # 4. Ball-Player distance features
        min_dist_ball_player = float('inf')
        num_frames_ball_close = 0
        all_min_dists = []
        
        # Normalize threshold
        normalized_ball_close_threshold = BALL_CLOSE_THRESHOLD / height

        for f in frames:
            if f["ball_visible"] and f["ball_x"] is not None:
                frame_min_dist = float('inf')
                for p in f["players"]:
                    px = (p["x1"] + p["x2"]) / 2
                    py = (p["y1"] + p["y2"]) / 2
                    dist = math.sqrt((f["ball_x"] - px)**2 + (f["ball_y"] - py)**2)
                    if dist < frame_min_dist:
                        frame_min_dist = dist
                
                if frame_min_dist != float('inf'):
                    norm_dist = frame_min_dist / height
                    all_min_dists.append(norm_dist)
                    if norm_dist < min_dist_ball_player:
                        min_dist_ball_player = norm_dist
                    if norm_dist < normalized_ball_close_threshold:
                        num_frames_ball_close += 1
        
        mean_dist_ball_player = sum(all_min_dists) / len(all_min_dists) if all_min_dists else -1.0
        std_dist_ball_player = 0.0
        if len(all_min_dists) > 1:
            variance = sum((d - mean_dist_ball_player)**2 for d in all_min_dists) / len(all_min_dists)
            std_dist_ball_player = math.sqrt(variance)

        if min_dist_ball_player == float('inf'):
            min_dist_ball_player = -1.0

        # 5. Player speed features
        player_speeds = []
        num_players_moving = 0
        
        # Normalize threshold
        normalized_player_speed_threshold = PLAYER_SPEED_THRESHOLD / height

        max_players = max((len(f["players"]) for f in frames), default=0)
        for p_idx in range(max_players):
            p_speeds = []
            for i in range(1, num_frames):
                if p_idx < len(frames[i-1]["players"]) and p_idx < len(frames[i]["players"]):
                    p1 = frames[i-1]["players"][p_idx]
                    p2 = frames[i]["players"][p_idx]
                    
                    c1x, c1y = (p1["x1"] + p1["x2"]) / 2, (p1["y1"] + p1["y2"]) / 2
                    c2x, c2y = (p2["x1"] + p2["x2"]) / 2, (p2["y1"] + p2["y2"]) / 2
                    
                    dist = math.sqrt((c2x - c1x)**2 + (c2y - c1y)**2)
                    p_speeds.append(dist / height) # Normalize
            
            if p_speeds:
                avg_p_speed = sum(p_speeds) / len(p_speeds)
                player_speeds.append(avg_p_speed)
                if avg_p_speed > normalized_player_speed_threshold:
                    num_players_moving += 1

        mean_player_speed = sum(player_speeds) / len(player_speeds) if player_speeds else 0.0

        return {
            "ball_visible_ratio": ball_visible_ratio,
            "mean_ball_speed": mean_ball_speed,
            "std_ball_speed": std_ball_speed,
            "num_ball_direction_changes": float(num_ball_direction_changes),
            "min_dist_ball_player": min_dist_ball_player,
            "mean_dist_ball_player": mean_dist_ball_player,
            "std_dist_ball_player": std_dist_ball_player,
            "num_frames_ball_close": float(num_frames_ball_close),
            "mean_player_speed": mean_player_speed,
            "num_players_moving": float(num_players_moving)
        }
