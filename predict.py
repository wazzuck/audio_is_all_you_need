import os
import numpy as np
import torch
import argparse
from safetensors.torch import load_file as load_safetensors

# Assuming data_loader, utils, model, config are in the same directory or PYTHONPATH is set
from data_loader import (
    load_audio_robust, extract_mel_spectrogram,
    pad_or_truncate_spectrogram, MAX_DURATION_S,
    SAMPLE_RATE, HOP_LENGTH, N_MELS, NUM_CLASSES
)
from utils import load_pickle
from model import AudioCNN # Import the PyTorch model
import config

DEFAULT_FOLD_DISPLAY = 1 # For user-facing default, actual filename might be 0-indexed

def predict_sound_class(audio_path, model_weights_path, class_mapping, device):
    """Loads an audio file, preprocesses it, builds the PyTorch model,
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

    # Calculate expected spectrogram length based on MAX_DURATION_S
    # This should match the target_len used during training for consistency
    target_len_processed_spec = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / HOP_LENGTH))
    mel_spec_processed = pad_or_truncate_spectrogram(mel_spec, target_len_processed_spec)

    # Ensure N_MELS matches the processed spectrogram
    if mel_spec_processed.shape[0] != N_MELS:
        print(f"Warning: Processed spectrogram has {mel_spec_processed.shape[0]} mel bands, but model expects {N_MELS}.")
        # Potentially resize or error out here if critical

    print(f"Processed Mel spectrogram shape: {mel_spec_processed.shape} (Expected: {N_MELS}, {target_len_processed_spec})")

    # Convert to PyTorch tensor and add batch dimension
    # Model expects (batch, n_mels, time_frames)
    mel_spec_tensor = torch.tensor(mel_spec_processed, dtype=torch.float32).unsqueeze(0) # Add batch dim

    # --- Build Model and Load Weights ---
    print("Building PyTorch model architecture...")
    # target_len_estimate_for_fc_input_calc should be the actual time dimension of the input spectrogram
    model = AudioCNN(n_mels=N_MELS, num_classes=NUM_CLASSES, target_len_estimate_for_fc_input_calc=mel_spec_tensor.shape[2])
    model.to(device)

    print(f"Loading model weights from: {model_weights_path}")
    try:
        state_dict = load_safetensors(model_weights_path, device=device) # Load to target device directly
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights using safetensors: {e}")
        print("Ensure the weights file exists, is a valid .safetensors PyTorch model, and matches the model architecture.")
        return

    model.eval() # Set model to evaluation mode

    print("Making prediction...")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model(mel_spec_tensor.to(device))
    
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_index_tensor = torch.max(probabilities, dim=1)

    predicted_index = predicted_index_tensor.item()
    confidence_score = confidence.item()
    
    predicted_class = class_mapping.get(predicted_index, "Unknown")

    print("\n--- Prediction Results ---")
    print(f"Predicted Class: {predicted_class} (ID: {predicted_index})")
    print(f"Confidence: {confidence_score:.4f}")
    print("\nFull Probabilities:")
    for i, prob in enumerate(probabilities.squeeze().tolist()): # Squeeze to remove batch dim for iteration
        class_name = class_mapping.get(i, f"Unknown Class {i}")
        print(f"  {class_name:<20}: {prob:.4f}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class mapping
    class_mapping_path = os.path.join(args.processed_dir, 'class_mapping.pkl')
    class_mapping = load_pickle(class_mapping_path)
    if class_mapping is None:
        print(f"Error: Class mapping not found at {class_mapping_path}. Cannot proceed.")
        print("Run preprocessing script first.")
        return

    # Construct model weights path. Folds in filenames are 0-indexed by train.py
    # If user provides 1-indexed fold, adjust it. Args.fold_num is 0-indexed based on train.py convention.
    model_filename = f'model_fold_{args.fold_num}_best.safetensors' 
    model_weights_path = os.path.join(args.model_dir, model_filename)

    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found: {model_weights_path}")
        print(f"Ensure you have trained the model for fold {args.fold_num} and it saved a '_best.safetensors' file.")
        return

    if not os.path.exists(args.audio_file):
        print(f"Error: Input audio file not found: {args.audio_file}")
        return

    predict_sound_class(args.audio_file, model_weights_path, class_mapping, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict the class of an urban sound audio file using a trained PyTorch model (.safetensors weights).')
    parser.add_argument('audio_file', type=str, help='Path to the input audio file (.wav recommended).')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                        help=f'Directory containing the trained model weights (.safetensors) (default: {config.MODEL_DIR})')
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR,
                        help=f'Directory containing the class mapping pickle file (default: {config.PROCESSED_DIR})')
    # The train.py script saves folds as fold_0, fold_1 etc.
    # So, if default is Fold 1 (user-facing), it means index 0 for filename.
    # Let's assume fold_num is 0-indexed for consistency with filenames now.
    parser.add_argument('--fold_num', type=int, default=0, 
                        help='Which fold\'s trained model weights to use for prediction (0-indexed, default: 0).')

    args = parser.parse_args()
    main(args) 