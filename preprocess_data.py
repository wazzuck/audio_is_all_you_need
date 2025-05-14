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
import huggingface_hub # Added for HF upload
import shutil # Added for potential cleanup

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

    # Upload to Hugging Face Hub
    print(f"Attempting to upload processed data to Hugging Face Hub repository: {config.HF_REPO_ID}")
    print("Please ensure you have logged in via 'huggingface-cli login' and have write access to the repository.")
    try:
        # Attempt to create the repo first.
        # exist_ok=True will not raise an error if the repo already exists and is accessible.
        try:
            huggingface_hub.create_repo(
                repo_id=config.HF_REPO_ID,
                repo_type="dataset",
                exist_ok=True 
            )
            print(f"Ensured repository {config.HF_REPO_ID} exists or was created.")
        except Exception as create_e:
            # This might catch other errors if exist_ok=True isn't enough for some edge cases (e.g. permissions)
            print(f"Note: Problem during create_repo (repo may already exist or other permission issue): {create_e}")
            # We'll proceed to upload_folder anyway, as it will provide a more specific error if the issue persists.

        repo_url = huggingface_hub.upload_folder(
            folder_path=args.output_dir,
            repo_id=config.HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Add processed UrbanSound8K data from {args.output_dir}",
            path_in_repo="." # Upload to the root of the dataset repo
        )
        print(f"Successfully uploaded processed data to: {repo_url}")

        # Optional: Clean up local processed files after upload if desired
        # print(f"Cleaning up local directory: {args.output_dir}")
        # shutil.rmtree(args.output_dir)
        # print("Local cleanup complete.")

    except Exception as e:
        print(f"Error during Hugging Face Hub operation: {e}")
        print("Please check your authentication, internet connection, repository permissions, and ensure the repo_id is correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess UrbanSound8K audio data into Mel spectrogram features.')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR, # Use config default
                        help=f'Directory where the UrbanSound8K dataset is located (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.PROCESSED_DIR, # Use config default
                        help=f'Directory to save the processed features (default: {config.PROCESSED_DIR})')

    args = parser.parse_args()
    main(args) 