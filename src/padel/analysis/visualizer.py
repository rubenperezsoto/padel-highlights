import cv2
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict

class PadelVisualizer:
    """
    Handles all visualization tasks for the Padel Highlights project.
    """
    
    @staticmethod
    def plot_predictions(
        ticks_path: Path, 
        model_path: Path, 
        output_path: Path, 
        labels_path: Optional[Path] = None
    ):
        """
        Plots model predictions over time against ground truth.
        """
        # Load data
        df = pd.read_parquet(ticks_path)
        
        # Load model
        saved_data = joblib.load(model_path)
        model = saved_data["model"]
        feature_names = saved_data["feature_names"]
        
        # Predict probabilities
        X = df[feature_names].fillna(0.0)
        probs = model.predict_proba(X)[:, 1]
        df["prob"] = probs
        
        # Load labels if available
        if labels_path and labels_path.exists():
            with open(labels_path, "r") as f:
                labels = json.load(f)
            
            def is_in_play(ts):
                for interval in labels:
                    if interval["start"] <= ts <= interval["end"]:
                        return 1
                return 0
            
            df["ground_truth"] = df["timestamp"].apply(is_in_play)
        else:
            df["ground_truth"] = 0

        # Plotting
        plt.figure(figsize=(15, 6))
        plt.fill_between(df["timestamp"], 0, df["ground_truth"], color='gray', alpha=0.2, label='Ground Truth (Rally)')
        plt.plot(df["timestamp"], df["prob"], label='Predicted Probability', color='blue', linewidth=1)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Probability / Label')
        plt.title(f'Model Predictions: {ticks_path.name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

    @staticmethod
    def draw_net_line(video_path: Path, net_y_norm: float, output_path: Path):
        """
        Draws the detected net line on a sample frame from the video.
        """
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            cap.release()
            return
        
        height, width = frame.shape[:2]
        net_y = int(net_y_norm * height)
        
        # Draw the net line
        cv2.line(frame, (0, net_y), (width, net_y), (0, 255, 0), 3)
        cv2.putText(frame, f"Detected Net (y={net_y_norm:.4f})", (50, net_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), frame)
        cap.release()
        print(f"Net visualization saved to {output_path}")
