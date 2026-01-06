import argparse
from pathlib import Path
from src.padel.analysis.visualizer import PadelVisualizer

def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions over time.")
    parser.add_argument("--ticks", type=Path, required=True, help="Path to ticks.parquet")
    parser.add_argument("--labels", type=Path, help="Path to labels.json")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the plot")
    
    args = parser.parse_args()
    
    PadelVisualizer.plot_predictions(
        ticks_path=args.ticks,
        labels_path=args.labels,
        model_path=args.model,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
