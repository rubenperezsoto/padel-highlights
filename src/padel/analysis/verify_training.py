import pandas as pd
import json
from pathlib import Path
import subprocess
from src.padel.analysis.tick_processor import TickProcessor

def verify_training_pipeline():
    # 1. Create mock detections (30 FPS, 10 seconds)
    data = []
    for i in range(300):
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
    parquet_path = Path("test_ticks.parquet")
    
    # 2. Generate ticks using TickProcessor
    processor = TickProcessor(tick_frequency=5.0, window_size_seconds=1.0)
    processor.load_detections(df)
    ticks_df = processor.process()
    ticks_df.to_parquet(parquet_path)
    
    # 3. Create mock labels JSON
    labels = [{"start": 2.0, "end": 5.0}, {"start": 7.0, "end": 9.0}]
    labels_path = Path("test_labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)
        
    try:
        # 4. Run training script
        print("Running training script...")
        cmd = [
            "python3", "-m", "src.padel.analysis.train_rally_classifier",
            "--ticks", str(parquet_path),
            "--labels", str(labels_path),
            "--output", "test_model.joblib"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        assert result.returncode == 0
        assert Path("test_model.joblib").exists()
        
        print("\nVerification successful!")
        
    finally:
        # Cleanup
        for p in [parquet_path, labels_path, Path("test_model.joblib")]:
            if p.exists():
                p.unlink()

if __name__ == "__main__":
    verify_training_pipeline()
