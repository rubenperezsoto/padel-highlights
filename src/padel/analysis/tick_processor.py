from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from src.padel.analysis.feature_extractor import FeatureExtractor


class TickProcessor:
    """
    Processor to resample frame-level detections into fixed-frequency ticks
     and compute temporal features using a sliding window.
    """

    def __init__(
        self, 
        tick_frequency: float = 5.0, 
        window_size_seconds: float = 1.5
    ) -> None:
        self.tick_frequency = tick_frequency
        self.window_size_seconds = window_size_seconds
        self.detections_df: Optional[pd.DataFrame] = None
        self.labels: List[Dict[str, float]] = []

    def load_detections(self, source: Union[pd.DataFrame, str, Path]) -> None:
        """
        Loads detections from a pandas DataFrame or a Parquet file.
        Expected columns: frame_index, timestamp, ball_visible, ball_x, ball_y, players
        """
        if isinstance(source, (str, Path)):
            self.detections_df = pd.read_parquet(source)
        else:
            self.detections_df = source

        # Ensure timestamp is sorted
        self.detections_df = self.detections_df.sort_values("timestamp").reset_index(drop=True)

    def load_labels(self, json_path: Union[str, Path]) -> None:
        """
        Loads rally labels from a JSON file.
        Expected format: [{"start": 10.5, "end": 15.2}, ...]
        """
        with open(json_path, "r") as f:
            self.labels = json.load(f)

    def _get_label_for_timestamp(self, timestamp: float) -> int:
        """Returns 1 if timestamp is within any labeled rally interval, else 0."""
        for interval in self.labels:
            if interval["start"] <= timestamp <= interval["end"]:
                return 1
        return 0

    def process(self, height: float = 720.0) -> pd.DataFrame:
        """
        Processes the detections into ticks and extracts features.
        Returns a DataFrame where each row is a tick.
        """
        if self.detections_df is None:
            raise ValueError("Detections not loaded. Call load_detections first.")

        start_time = self.detections_df["timestamp"].min()
        end_time = self.detections_df["timestamp"].max()
        
        # Generate tick timestamps
        tick_timestamps = np.arange(start_time, end_time, 1.0 / self.tick_frequency)
        
        tick_results = []
        
        for ts in tick_timestamps:
            # Define window: [ts - window_size, ts]
            window_start = ts - self.window_size_seconds
            window_end = ts
            
            # Filter frames in window
            mask = (self.detections_df["timestamp"] >= window_start) & \
                   (self.detections_df["timestamp"] <= window_end)
            window_df = self.detections_df[mask]
            
            if window_df.empty:
                continue
                
            # Convert window to list of dicts for FeatureExtractor
            window_frames = window_df.to_dict("records")
            
            # Extract features
            features = FeatureExtractor.compute_features(window_frames, height=height)
            
            # Build tick row
            row = {
                "timestamp": ts,
                "ball_y": window_df.iloc[len(window_df)//2]["ball_y"] if "ball_y" in window_df.columns else None,
                "ball_visible": window_df.iloc[len(window_df)//2]["ball_visible"] if "ball_visible" in window_df.columns else False,
                **features
            }
            
            # Add label if labels are available
            if self.labels:
                row["in_play"] = self._get_label_for_timestamp(ts)
                
            tick_results.append(row)
            
        return pd.DataFrame(tick_results)
