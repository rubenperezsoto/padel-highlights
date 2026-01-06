import argparse
from pathlib import Path
from src.padel.analysis.net_detector import NetDetector
from src.padel.analysis.visualizer import PadelVisualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    
    detector = NetDetector()
    net_y = detector.detect(args.video)
    
    PadelVisualizer.draw_net_line(
        video_path=args.video,
        net_y_norm=net_y,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
