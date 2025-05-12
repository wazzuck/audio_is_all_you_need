import os
import argparse
import numpy as np
# import sys # No longer needed

# Add src directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # No longer needed

# Updated imports
# from src.data_loader import load_and_process_data
# from src.utils import save_pickle
# from src import config
from data_loader import load_and_process_data
from utils import save_pickle
import config

# Default paths (relative to project root) - Now fetched from config
# DEFAULT_DATA_DIR = 'data'
# DEFAULT_OUTPUT_DIR = 'data/processed'

def main(args):
    """Runs the data loading and preprocessing."""
    print("Starting data preprocessing...")

    # Use args.data_dir and args.output_dir which default to config values
    urbansound_dir = os.path.join(args.data_dir, 'UrbanSound8K')
    metadata_path = os.path.join(urbansound_dir, 'metadata', 'UrbanSound8K.csv')
    audio_dir = os.path.join(urbansound_dir, 'audio')

    if not os.path.exists(metadata_path) or not os.path.exists(audio_dir):
        print(f"Error: UrbanSound8K data not found in {urbansound_dir}.")
        print("Please run 'python download_data.py' first.")
        return

    print(f"Loading metadata from: {metadata_path}")
    print(f"Loading audio from: {audio_dir}")

    features, labels, folds, class_mapping = load_and_process_data(metadata_path, audio_dir)

    if features is None or len(features) == 0:
        print("Error: No features were extracted. Please check the logs.")
        return

    # Save the processed data
    os.makedirs(args.output_dir, exist_ok=True)
    save_pickle(features, os.path.join(args.output_dir, 'features.pkl'))
    save_pickle(labels, os.path.join(args.output_dir, 'labels.pkl'))
    save_pickle(folds, os.path.join(args.output_dir, 'folds.pkl'))
    save_pickle(class_mapping, os.path.join(args.output_dir, 'class_mapping.pkl'))

    print(f"Preprocessing complete. Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess UrbanSound8K audio data into Mel spectrogram features.')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, # Use config default
                        help=f'Directory where the UrbanSound8K dataset is located (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.PROCESSED_DIR, # Use config default
                        help=f'Directory to save the processed features (default: {config.PROCESSED_DIR})')

    args = parser.parse_args()
    main(args) 