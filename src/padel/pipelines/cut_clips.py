import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm

def cut_clips(video_path, intervals_path, output_dir):
    # 1. Load intervals
    with open(intervals_path, "r") as f:
        intervals = json.load(f)
    
    if not intervals:
        print("No intervals to cut.")
        return

    # 2. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard MP4 codec

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cutting {len(intervals)} clips from {video_path.name}...")

    for i, interval in enumerate(intervals):
        start_frame = int(interval["start"] * fps)
        end_frame = int(interval["end"] * fps)
        num_frames = end_frame - start_frame
        
        if num_frames <= 0:
            continue

        output_path = output_dir / f"clip_{i+1:03d}.mp4"
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"Generating {output_path.name} ({interval['start']:.1f}s - {interval['end']:.1f}s)...")
        
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()

    cap.release()
    print(f"Success! Clips saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut video clips based on detected intervals.")
    parser.add_argument("--video", type=Path, required=True, help="Path to original video")
    parser.add_argument("--intervals", type=Path, required=True, help="Path to intervals JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("data/clips"), help="Output directory for clips")
    
    args = parser.parse_args()
    cut_clips(args.video, args.intervals, args.output_dir)
