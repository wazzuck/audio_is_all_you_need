import os
import numpy as np
import tensorflow as tf
import argparse
import sys

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import (load_audio_robust, extract_mel_spectrogram, 
                             pad_or_truncate_spectrogram, MAX_DURATION_S, 
                             SAMPLE_RATE, HOP_LENGTH, N_MELS, NUM_CLASSES)
from src.utils import load_pickle
from src.model import build_cnn_model # Import the model builder
from src import config # Import config

# Default paths - Now fetched from config
# DEFAULT_MODEL_DIR = 'models'
# DEFAULT_PROCESSED_DIR = 'data/processed' # To load class mapping
DEFAULT_FOLD = 1 # Default to using the model trained excluding fold 1

def predict_sound_class(audio_path, model_weights_path, class_mapping):
    """Loads an audio file, preprocesses it, builds the model structure,
       loads weights from .safetensors, and predicts its class."""
    print(f"Loading audio file: {audio_path}")
    audio = load_audio_robust(audio_path, sr=SAMPLE_RATE)
    if audio is None:
        print("Failed to load audio file.")
        return

    # --- Preprocess the audio (same steps as in data_loader/train) ---
    target_audio_len = int(MAX_DURATION_S * SAMPLE_RATE)
    if len(audio) < target_audio_len:
        audio = np.pad(audio, (0, target_audio_len - len(audio)), mode='constant')
    elif len(audio) > target_audio_len:
        audio = audio[:target_audio_len]

    mel_spec = extract_mel_spectrogram(audio)

    # Calculate expected spectrogram length
    target_len = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / HOP_LENGTH))
    mel_spec_processed = pad_or_truncate_spectrogram(mel_spec, target_len)

    # Determine input shape for model building
    input_shape = (N_MELS, target_len, 1)
    print(f"Expected model input shape: {input_shape}")

    # Add batch dimension for prediction
    mel_spec_batch = mel_spec_processed[np.newaxis, ..., np.newaxis]

    # --- Build Model and Load Weights ---
    print("Building model architecture...")
    model = build_cnn_model(input_shape=input_shape, num_classes=NUM_CLASSES)

    print(f"Loading model weights from: {model_weights_path}")
    try:
        # TensorFlow expects the path without the suffix in some versions?
        # Let's try loading directly first, then maybe strip suffix if it fails.
        model.load_weights(model_weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure the weights file exists and TensorFlow/safetensors library is correctly installed.")
        # Attempting without suffix (older TF behavior?)
        # try:
        #    weights_path_no_suffix, _ = os.path.splitext(model_weights_path)
        #    print(f"Retrying load with path: {weights_path_no_suffix}")
        #    model.load_weights(weights_path_no_suffix)
        # except Exception as e2:
        #    print(f"Error loading weights (retry without suffix): {e2}")
        #    return
        return

    print("Making prediction...")
    predictions = model.predict(mel_spec_batch)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_mapping.get(predicted_index, "Unknown")
    confidence = predictions[0][predicted_index]

    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class} (ID: {predicted_index})")
    print(f"Confidence: {confidence:.4f}")
    print("\nFull Probabilities:")
    for i, prob in enumerate(predictions[0]):
        class_name = class_mapping.get(i, f"Unknown Class {i}")
        print(f"  {class_name:<20}: {prob:.4f}")

def main(args):
    # Load class mapping
    class_mapping_path = os.path.join(args.processed_dir, 'class_mapping.pkl')
    class_mapping = load_pickle(class_mapping_path)
    if class_mapping is None:
        print(f"Error: Class mapping not found at {class_mapping_path}. Cannot proceed.")
        print("Run preprocessing script first.")
        return

    # Construct model weights path
    model_filename = f'model_fold_{args.fold_num}.safetensors' # Changed extension
    model_weights_path = os.path.join(args.model_dir, model_filename)

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found: {model_weights_path}")
        print(f"Ensure you have trained the model for fold {args.fold_num} using scripts/train.py")
        return

    if not os.path.exists(args.audio_file):
        print(f"Error: Input audio file not found: {args.audio_file}")
        return

    predict_sound_class(args.audio_file, model_weights_path, class_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the class of an urban sound audio file using a trained model (.safetensors weights).')
    parser.add_argument('audio_file', type=str, help='Path to the input audio file (.wav recommended).')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                        help=f'Directory containing the trained model weights (.safetensors) (default: {config.MODEL_DIR})')
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR,
                        help=f'Directory containing the class mapping pickle file (default: {config.PROCESSED_DIR})')
    parser.add_argument('--fold_num', type=int, default=DEFAULT_FOLD,
                        help=f'Which fold\'s trained model weights to use for prediction (default: {DEFAULT_FOLD})')

    args = parser.parse_args()
    main(args) 