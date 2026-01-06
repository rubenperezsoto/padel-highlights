import argparse
import json
from pathlib import Path
import cv2
import pandas as pd
import joblib
from src.padel.analysis.point_detector import PointDetector
from src.padel.analysis.net_detector import NetDetector

def main():
    parser = argparse.ArgumentParser(description="Detect rally points from a ticks.parquet file.")
    parser.add_argument("--ticks", type=Path, required=True, help="Path to ticks.parquet")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--video", type=Path, help="Path to original video (required for net heuristic)")
    parser.add_argument("--output", type=Path, default=Path("data/detected_points.json"), help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum rally duration in seconds")
    parser.add_argument("--min-gap", type=float, default=3.0, help="Minimum gap between rallies to keep them separate")
    parser.add_argument("--smoothing", type=float, default=1.5, help="Smoothing window in seconds")
    parser.add_argument("--buffer", type=float, default=1.5, help="Buffer seconds to add before/after each point")
    parser.add_argument("--use-net-heuristic", action="store_true", help="Filter rallies that don't cross the net")
    
    args = parser.parse_args()
    
    if not args.ticks.exists():
        print(f"Error: Ticks file {args.ticks} not found.")
        return
    if not args.model.exists():
        print(f"Error: Model file {args.model} not found.")
        return

    # 1. Detect Net Line if heuristic is enabled
    net_y = None
    if args.use_net_heuristic:
        if not args.video or not args.video.exists():
            print("Error: --video is required when using --use-net-heuristic.")
            return
        print(f"Detecting net line in {args.video}...")
        net_detector = NetDetector()
        net_y = net_detector.detect(args.video)
        print(f"Net line detected at y={net_y:.4f}")

    # 2. Load data and model
    print(f"Loading data from {args.ticks}...")
    df = pd.read_parquet(args.ticks)
    
    print(f"Loading model from {args.model}...")
    saved_data = joblib.load(args.model)
    model = saved_data["model"]
    feature_names = saved_data["feature_names"]
    
    # 2. Predict probabilities
    print("Predicting probabilities...")
    X = df[feature_names].fillna(0.0)
    df["prob"] = model.predict_proba(X)[:, 1]
    
    # 3. Detect points
    print("Detecting points...")
    
    # Convert normalized net_y to absolute pixels if it exists
    absolute_net_y = None
    serve_detector = None
    
    if net_y is not None:
        cap = cv2.VideoCapture(str(args.video))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        absolute_net_y = net_y * height
        print(f"Absolute net line at y={absolute_net_y:.1f} pixels")
        
        # Initialize ServeFormationDetector with absolute net_y
        from src.padel.analysis.serve_detector import ServeFormationDetector
        serve_detector = ServeFormationDetector(net_y=absolute_net_y)

    detector = PointDetector(
        threshold=args.threshold,
        min_duration_seconds=args.min_duration,
        min_gap_seconds=args.min_gap,
        smoothing_window_seconds=args.smoothing,
        buffer_seconds=args.buffer,
        net_y=absolute_net_y,
        serve_detector=serve_detector
    )
    intervals = detector.detect(df)
    
    # 4. Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(intervals, f, indent=4)
        
    print(f"Success! Detected {len(intervals)} points.")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
