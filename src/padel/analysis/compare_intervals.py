import json
from pathlib import Path
from typing import List, Dict

class IntervalComparator:
    """
    Compares detected rally intervals with ground truth labels.
    """
    
    @staticmethod
    def compare(detected_path: Path, ground_truth_path: Path):
        """
        Calculates precision and recall for detected intervals.
        """
        if not detected_path.exists():
            print(f"Error: Detected intervals file {detected_path} not found.")
            return
        if not ground_truth_path.exists():
            print(f"Error: Ground truth file {ground_truth_path} not found.")
            return

        with open(detected_path, "r") as f:
            detected = json.load(f)
        with open(ground_truth_path, "r") as f:
            gt = json.load(f)

        found_gt = 0
        for g in gt:
            for d in detected:
                # Check overlap
                if max(g["start"], d["start"]) < min(g["end"], d["end"]):
                    found_gt += 1
                    break
                    
        correct_detected = 0
        for d in detected:
            for g in gt:
                if max(g["start"], d["start"]) < min(g["end"], d["end"]):
                    correct_detected += 1
                    break
                    
        print(f"--- Comparison: {detected_path.name} vs {ground_truth_path.name} ---")
        print(f"GT Intervals Found: {found_gt}/{len(gt)} ({found_gt/len(gt)*100:.1f}%)")
        print(f"Correct Detections: {correct_detected}/{len(detected)} ({correct_detected/len(detected)*100:.1f}%)")
        print("-" * 50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detected", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args()
    
    IntervalComparator.compare(args.detected, args.gt)
