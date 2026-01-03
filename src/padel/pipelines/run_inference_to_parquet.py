from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import pandas as pd
from src.padel.tracking.yolo_handler import YOLOHandler
from src.padel.analysis.tick_processor import TickProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a video and save features to Parquet.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument("--output", type=Path, default=Path("data/ticks.parquet"), help="Output Parquet path.")
    parser.add_argument("--max-frames", type=int, help="Limit number of frames to process.")
    parser.add_argument("--tick-freq", type=float, default=5.0, help="Tick frequency in Hz.")
    
    args = parser.parse_args()
    
    if not args.video.exists():
        print(f"Error: Video {args.video} not found.")
        return

    # 1. Run Inference (YOLO + TrackNet)
    print(f"Running inference (YOLO + TrackNet) on {args.video}...")
    yolo = YOLOHandler()
    
    # Try to initialize TrackNet, but don't fail if it's not configured
    tracknet = None
    try:
        from src.padel.tracking.trackvnet_handler import TrackVNETHandler
        tracknet = TrackVNETHandler()
        print("TrackNet initialized successfully.")
    except Exception as e:
        print(f"Warning: TrackNet could not be initialized: {e}")
        print("Proceeding with YOLO only.")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_data = {} # frame_index -> data dict
    
    # Process YOLO detections
    print("Processing YOLO detections...")
    for det in yolo.run_on_video(args.video):
        if args.max_frames and det.frame_index >= args.max_frames:
            break
            
        if det.frame_index not in frame_data:
            frame_data[det.frame_index] = {
                "frame_index": det.frame_index,
                "timestamp": det.frame_index / fps,
                "ball_visible": False,
                "ball_x": None,
                "ball_y": None,
                "players": []
            }
        
        frame_data[det.frame_index]["players"].append({
            "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2
        })

    # Process TrackNet detections if available
    if tracknet:
        print("Processing TrackNet detections...")
        for det in tracknet.run_on_video(args.video):
            if args.max_frames and det.frame_index >= args.max_frames:
                break
            
            if det.frame_index in frame_data:
                frame_data[det.frame_index]["ball_visible"] = det.visible
                frame_data[det.frame_index]["ball_x"] = det.x
                frame_data[det.frame_index]["ball_y"] = det.y
            else:
                # This might happen if YOLO missed a frame entirely
                frame_data[det.frame_index] = {
                    "frame_index": det.frame_index,
                    "timestamp": det.frame_index / fps,
                    "ball_visible": det.visible,
                    "ball_x": det.x,
                    "ball_y": det.y,
                    "players": []
                }

    if not frame_data:
        print("No detections found.")
        return

    # Convert dict to sorted list
    sorted_frames = [frame_data[i] for i in sorted(frame_data.keys())]
    detections_df = pd.DataFrame(sorted_frames)
    print(f"Inference complete. Processed {len(detections_df)} frames.")

    # 2. Run TickProcessor
    print(f"Resampling to {args.tick_freq} Hz and extracting features...")
    processor = TickProcessor(tick_frequency=args.tick_freq, window_size_seconds=1.5)
    processor.load_detections(detections_df)
    
    ticks_df = processor.process()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    ticks_df.to_parquet(args.output)
    print(f"Success! Saved {len(ticks_df)} ticks to {args.output}")


if __name__ == "__main__":
    main()
