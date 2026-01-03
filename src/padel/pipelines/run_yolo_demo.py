from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from src.padel.tracking.yolo_handler import YOLOHandler


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="Path to input video.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path to weights."
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    handler = YOLOHandler(model_name=args.model)
    count = 0
    frames_processed = 0

    print(f"Starting YOLO inference on {args.video}...")
    
    # We need to track frame indices manually if we want to stop at max_frames
    # but the generator yields one detection per player per frame.
    current_frame = -1
    
    for det in handler.run_on_video(args.video):
        if det.frame_index != current_frame:
            current_frame = det.frame_index
            frames_processed += 1
            
        print(f"frame={det.frame_index} id={det.player_id} box=[{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}] conf={det.confidence:.2f}")
        count += 1
        
        if args.max_frames and frames_processed >= args.max_frames:
            break

    print(f"Processed {frames_processed} frames. Found {count} player detections.")


if __name__ == "__main__":
    main()
