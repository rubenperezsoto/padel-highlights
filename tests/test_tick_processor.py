import pandas as pd
import json
from pathlib import Path
from src.padel.analysis.tick_processor import TickProcessor

def test_tick_processor():
    # 1. Create mock detections (30 FPS, 2 seconds)
    data = []
    for i in range(60):
        ts = i / 30.0
        # Ball moving in a line
        data.append({
            "frame_index": i,
            "timestamp": ts,
            "ball_visible": True,
            "ball_x": 100.0 + i,
            "ball_y": 100.0,
            "players": [{"x1": 50, "y1": 50, "x2": 70, "y2": 70}]
        })
    
    df = pd.DataFrame(data)
    
    # 2. Create mock labels JSON
    labels = [{"start": 0.5, "end": 1.5}]
    labels_path = Path("test_labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
        
    try:
        # 3. Initialize and run TickProcessor
        processor = TickProcessor(tick_frequency=5.0, window_size_seconds=1.0)
        processor.load_detections(df)
        processor.load_labels(labels_path)
        
        result_df = processor.process()
        
        print("Tick Processor Results (First 5 rows):")
        print(result_df.head())
        
        # 4. Verifications
        assert not result_df.empty
        assert "in_play" in result_df.columns
        
        # Check labeling logic
        # Ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6...
        # Labels: 0.5 to 1.5
        # 0.6, 0.8, 1.0, 1.2, 1.4 should be in_play=1
        in_play_ticks = result_df[result_df["in_play"] == 1]["timestamp"].tolist()
        print(f"\nIn-play ticks: {in_play_ticks}")
        
        import math
        def is_in_list(val, lst):
            return any(math.isclose(val, x, rel_tol=1e-9) for x in lst)

        assert is_in_list(0.6, in_play_ticks)
        assert is_in_list(1.4, in_play_ticks)
        assert not is_in_list(0.4, in_play_ticks)
        assert not is_in_list(1.6, in_play_ticks)
        
        print("\nTest passed!")
        
    finally:
        if labels_path.exists():
            labels_path.unlink()

if __name__ == "__main__":
    test_tick_processor()
