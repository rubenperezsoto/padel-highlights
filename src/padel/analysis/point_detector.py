from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

class PointDetector:
    """
    Detects rally intervals from model probabilities using temporal smoothing.
    """
    def __init__(
        self, 
        threshold: float = 0.5, 
        min_duration_seconds: float = 2.0,
        min_gap_seconds: float = 3.0,
        smoothing_window_seconds: float = 1.5,
        buffer_seconds: float = 1.5,
        net_y: float | None = None
    ):
        self.threshold = threshold
        self.min_duration_seconds = min_duration_seconds
        self.min_gap_seconds = min_gap_seconds
        self.smoothing_window_seconds = smoothing_window_seconds
        self.buffer_seconds = buffer_seconds
        self.net_y = net_y

    def detect(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        """
        Detects rally intervals (start, end) from a DataFrame with 'timestamp' and 'prob'.
        """
        if df.empty:
            return []

        # 1. Calculate smoothing window size in ticks
        dt = df["timestamp"].diff().mean()
        window_size = max(1, int(self.smoothing_window_seconds / dt))
        
        # 2. Apply smoothing (Moving Average)
        smoothed_prob = df["prob"].rolling(window=window_size, center=True).mean().fillna(df["prob"])
        
        # 3. Thresholding
        in_play = (smoothed_prob >= self.threshold).astype(int)
        
        # 4. Find raw continuous intervals
        raw_intervals = []
        is_active = False
        start_ts = 0.0
        
        for i in range(len(in_play)):
            if in_play.iloc[i] == 1 and not is_active:
                is_active = True
                start_ts = df["timestamp"].iloc[i]
            elif in_play.iloc[i] == 0 and is_active:
                is_active = False
                end_ts = df["timestamp"].iloc[i]
                raw_intervals.append({"start": start_ts, "end": end_ts})
        
        if is_active:
            raw_intervals.append({"start": start_ts, "end": df["timestamp"].iloc[-1]})

        if not raw_intervals:
            return []

        # 5. Merge intervals that are close to each other
        merged_intervals = []
        if raw_intervals:
            current = raw_intervals[0]
            for next_int in raw_intervals[1:]:
                gap = next_int["start"] - current["end"]
                if gap <= self.min_gap_seconds:
                    # Merge
                    current["end"] = next_int["end"]
                else:
                    merged_intervals.append(current)
                    current = next_int
            merged_intervals.append(current)

        # 6. Filter by duration and add buffers
        final_intervals = []
        for interval in merged_intervals:
            # 7. Net Crossing Heuristic & Trimming
            if self.net_y is not None:
                crossings_info = self._get_crossings_info(df, interval)
                if crossings_info["count"] < 2:
                    continue # Skip if ball doesn't cross the net at least twice
                
                # Trim end to last crossing + a small extra buffer
                # This prevents long "walking" segments at the end of a point
                interval["end"] = min(interval["end"], crossings_info["last_ts"] + self.buffer_seconds)
            
            duration = interval["end"] - interval["start"]
            if duration >= self.min_duration_seconds:
                final_intervals.append({
                    "start": max(0, interval["start"] - self.buffer_seconds),
                    "end": interval["end"]
                })
                
        return final_intervals

    def _get_crossings_info(self, df: pd.DataFrame, interval: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes net crossings during the interval.
        Returns count and timestamp of the last crossing.
        """
        mask = (df["timestamp"] >= interval["start"]) & (df["timestamp"] <= interval["end"])
        subset = df[mask]
        
        if subset.empty or "ball_y" not in subset.columns:
            return {"count": 2, "last_ts": interval["end"]} # Assume valid if no data
            
        # Filter only frames where ball is visible
        visible_ball = subset[subset["ball_visible"] == True]
        if len(visible_ball) < 5:
            return {"count": 2, "last_ts": interval["end"]}
            
        # Count crossings
        sides = (visible_ball["ball_y"] > self.net_y).astype(int)
        cross_mask = (sides.diff().abs() == 1)
        crossings_count = cross_mask.sum()
        
        last_ts = interval["end"]
        if crossings_count > 0:
            last_ts = visible_ball[cross_mask]["timestamp"].max()
            
        return {"count": crossings_count, "last_ts": last_ts}
