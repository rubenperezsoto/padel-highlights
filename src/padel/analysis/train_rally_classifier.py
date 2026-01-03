from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_and_label_data(
    parquet_path: Union[str, Path], 
    labels_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Loads per-tick features from a parquet file and optionally applies labels from a JSON file.
    """
    df = pd.read_parquet(parquet_path)
    
    if labels_path:
        with open(labels_path, "r") as f:
            labels = json.load(f)
            
        def is_in_play(ts: float) -> int:
            for interval in labels:
                if interval["start"] <= ts <= interval["end"]:
                    return 1
            return 0
            
        df["in_play"] = df["timestamp"].apply(is_in_play)
        
    # Handle missing values (e.g., from missing ball detections)
    # For simplicity, we fill with -1.0 or 0.0 depending on the feature
    df = df.fillna(0.0)
    
    return df


def train_model(
    df: pd.DataFrame, 
    target_col: str = "in_play",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a RandomForestClassifier on the provided DataFrame.
    Excludes 'timestamp' and the target column from features.
    """
    # Features: all columns except timestamp and target
    feature_cols = [c for c in df.columns if c not in ["timestamp", target_col]]
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols
    }


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the model and prints performance metrics.
    """
    y_pred = model.predict(X_test)
    
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model: RandomForestClassifier, feature_names: List[str], output_path: Union[str, Path]) -> None:
    """
    Saves the trained model and feature names to disk.
    """
    data = {
        "model": model,
        "feature_names": feature_names
    }
    joblib.dump(data, output_path)
    print(f"\nModel saved to {output_path}")


def predict_ticks(df: pd.DataFrame, model_path: Union[str, Path]) -> pd.Series:
    """
    Loads a saved model and predicts in_play probabilities for a new tick DataFrame.
    """
    saved_data = joblib.load(model_path)
    model = saved_data["model"]
    feature_names = saved_data["feature_names"]
    
    # Ensure the input DataFrame has the required features
    X = df[feature_names].fillna(0.0)
    
    # Return probabilities for the 'in_play' class (class 1)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a padel rally classifier.")
    parser.add_argument("--ticks", type=Path, nargs="+", required=True, help="Paths to one or more ticks.parquet files")
    parser.add_argument("--labels", type=Path, nargs="+", help="Paths to corresponding labels.json files (must match order of --ticks)")
    parser.add_argument("--output", type=Path, default=Path("rally_model.joblib"), help="Model output path")
    
    args = parser.parse_args()
    
    all_dfs = []
    
    if args.labels and len(args.ticks) != len(args.labels):
        print("Error: Number of --ticks files must match number of --labels files.")
    else:
        for i, ticks_path in enumerate(args.ticks):
            if ticks_path.exists():
                labels_path = args.labels[i] if args.labels else None
                print(f"Loading and labeling data from {ticks_path}...")
                df = load_and_label_data(ticks_path, labels_path)
                all_dfs.append(df)
            else:
                print(f"Error: File {ticks_path} not found.")
        
        if not all_dfs:
            print("No data loaded. Exiting.")
        else:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            if "in_play" not in combined_df.columns:
                print("Error: Target column 'in_play' not found. Please provide labels.json files.")
            else:
                print(f"Training model on {len(combined_df)} total ticks...")
                results = train_model(combined_df)
                
                evaluate_model(results["model"], results["X_test"], results["y_test"])
                
                save_model(results["model"], results["feature_names"], args.output)
