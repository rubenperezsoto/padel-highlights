from src.padel.analysis.feature_extractor import FeatureExtractor

def test_feature_extractor():
    # Mock window of 5 frames
    window = [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "ball_visible": True,
            "ball_x": 100.0,
            "ball_y": 100.0,
            "players": [{"x1": 90, "y1": 90, "x2": 110, "y2": 110}] # Player close to ball
        },
        {
            "frame_index": 1,
            "timestamp": 0.033,
            "ball_visible": True,
            "ball_x": 110.0,
            "ball_y": 110.0,
            "players": [{"x1": 100, "y1": 100, "x2": 120, "y2": 120}]
        },
        {
            "frame_index": 2,
            "timestamp": 0.066,
            "ball_visible": False, # Ball missing
            "ball_x": None,
            "ball_y": None,
            "players": [{"x1": 110, "y1": 110, "x2": 130, "y2": 130}]
        },
        {
            "frame_index": 3,
            "timestamp": 0.1,
            "ball_visible": True,
            "ball_x": 130.0,
            "ball_y": 100.0, # Direction change
            "players": [{"x1": 120, "y1": 120, "x2": 140, "y2": 140}]
        },
        {
            "frame_index": 4,
            "timestamp": 0.133,
            "ball_visible": True,
            "ball_x": 140.0,
            "ball_y": 90.0,
            "players": [{"x1": 130, "y1": 130, "x2": 150, "y2": 150}]
        }
    ]

    features = FeatureExtractor.compute_features(window)
    
    print("Computed Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    assert features["ball_visible_ratio"] == 0.8
    assert features["num_frames_ball_close"] > 0
    print("\nTest passed!")

if __name__ == "__main__":
    test_feature_extractor()
