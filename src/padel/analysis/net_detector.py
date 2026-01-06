import cv2
import numpy as np
from pathlib import Path

class NetDetector:
    """
    Detects the net line (white tape) in a padel video.
    """
    def __init__(self, sample_frames: int = 5):
        self.sample_frames = sample_frames

    def detect(self, video_path: Path) -> float:
        """
        Returns the y-coordinate (normalized 0-1) of the net line.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.5 # Default to middle
        
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detected_y = []
        
        # Sample a few frames from the middle of the video
        for i in range(self.sample_frames):
            frame_idx = int(total_frames * (0.2 + 0.6 * i / self.sample_frames))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 1. Focus on the middle vertical third of the frame
            roi_y_start = int(height * 0.3)
            roi_y_end = int(height * 0.7)
            roi = frame[roi_y_start:roi_y_end, :]
            
            # 2. Convert to grayscale and detect edges
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 3. Hough Line Transform to find horizontal lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width//4, maxLineGap=20)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is horizontal-ish
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 5: # Very horizontal
                        detected_y.append((y1 + y2) / 2 + roi_y_start)
        
        cap.release()
        
        if not detected_y:
            print("Warning: Could not detect net line. Defaulting to 0.5")
            return 0.5
            
        # Return median to avoid outliers
        median_y = np.median(detected_y)
        return median_y / height

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    
    detector = NetDetector()
    y = detector.detect(args.video)
    print(f"Detected Net Y: {y:.4f}")
