from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np

class ServeFormationDetector:
    """
    Detects if players are in a "Serve Formation" based on their positions.
    """
    def __init__(self, net_y: float = 0.5, movement_threshold: float = 0.05):
        self.net_y = net_y
        self.movement_threshold = movement_threshold # Max movement (normalized) to be considered static

    def is_serve_formation(self, frames: List[Dict[str, Any]], height: float = 720.0) -> bool:
        """
        Analyzes a window of frames (e.g., 1 second) to check for static serve positions.
        
        Args:
            frames: List of frame data dictionaries, each containing 'players' list.
            height: Image height for normalization.
        """
        if not frames:
            return False

        # 1. Check for 4 players (ideal case) or at least 3
        # We take the median number of players detected across the window
        num_players_counts = [len(f["players"]) for f in frames]
        median_players = np.median(num_players_counts)
        
        if median_players < 3:
            return False # Not enough players to confirm formation

        # 2. Check if players are static
        # We track the centroid of each player across the window
        # Since we don't have IDs in the simple dict, we'll just check if the *set* of positions is stable.
        # A simple heuristic: calculate the variance of player centroids.
        
        # Collect all centroids for all frames
        all_centroids = []
        for f in frames:
            frame_centroids = []
            for p in f["players"]:
                cx = (p["x1"] + p["x2"]) / 2
                cy = (p["y1"] + p["y2"]) / 2
                frame_centroids.append((cx, cy))
            all_centroids.append(sorted(frame_centroids)) # Sort to match roughly by position

        # Check stability
        # We need to make sure we are comparing the "same" player across frames.
        # Sorting by Y then X usually works for separating players on court.
        
        # Let's simplify: Check if the bounding boxes in the first and last frame are similar
        first_frame_players = sorted(frames[0]["players"], key=lambda p: p["y1"])
        last_frame_players = sorted(frames[-1]["players"], key=lambda p: p["y1"])
        
        if len(first_frame_players) != len(last_frame_players):
            return False # Player count changed, likely movement or occlusion

        max_movement = 0.0
        for p1, p2 in zip(first_frame_players, last_frame_players):
            c1x, c1y = (p1["x1"] + p1["x2"]) / 2, (p1["y1"] + p1["y2"]) / 2
            c2x, c2y = (p2["x1"] + p2["x2"]) / 2, (p2["y1"] + p2["y2"]) / 2
            dist = np.sqrt((c2x - c1x)**2 + (c2y - c1y)**2)
            max_movement = max(max_movement, dist / height) # Normalize

        if max_movement > self.movement_threshold:
            return False # Players are moving too much

        # 3. Check for Serve Configuration
        # One side should have 2 players, other side 2 players (split by net_y)
        # Or at least 1 and 1 if occlusion.
        
        # Use the middle frame for position check
        middle_frame = frames[len(frames)//2]
        top_players = 0
        bottom_players = 0
        
        for p in middle_frame["players"]:
            cy = (p["y1"] + p["y2"]) / 2
            if cy < self.net_y:
                top_players += 1
            else:
                bottom_players += 1
                
        # Valid configurations: 2 vs 2, 2 vs 1, 1 vs 2 (due to occlusion)
        if top_players >= 1 and bottom_players >= 1:
             return True
             
        return False
