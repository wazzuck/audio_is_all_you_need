import os
import random
import argparse
import numpy as np
import torch
import librosa # For resample if needed by load_audio_robust
# from playsound import playsound # For playing sound - REPLACED
import subprocess # For calling aplay
import safetensors.torch

# Project specific imports
import config
from model import AudioCNN
from utils import load_pickle
# Import necessary functions and constants from data_loader
from data_loader import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, MAX_DURATION_S,
    load_audio_robust, extract_mel_spectrogram, pad_or_truncate_spectrogram,
    CLASS_MAPPING
)

# --- Print imported CLASS_MAPPING for debugging ---
print(f"DEBUG: Imported CLASS_MAPPING: {CLASS_MAPPING}")
# --- End debug print ---

def get_random_sound_file(base_audio_dir):
    """Gets a random .wav file from the UrbanSound8K audio subdirectories."""
    all_wav_files = []
    for root, _, files in os.walk(base_audio_dir):
        for file in files:
            if file.endswith(".wav"):
                all_wav_files.append(os.path.join(root, file))
    if not all_wav_files:
        raise FileNotFoundError(f"No .wav files found in {base_audio_dir}. Please ensure UrbanSound8K is downloaded and in the correct location.")
    return random.choice(all_wav_files)

def preprocess_audio_for_inference(audio_path):
    """Loads and preprocesses a single audio file for model inference."""
    # Calculate target length for spectrogram
    target_spec_len = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / HOP_LENGTH))
    
    audio = load_audio_robust(audio_path, sr=SAMPLE_RATE)
    if audio is None:
        raise ValueError(f"Could not load audio from {audio_path}")

    # Pad or truncate audio to MAX_DURATION_S * SAMPLE_RATE length
    target_audio_len = int(MAX_DURATION_S * SAMPLE_RATE)
    if len(audio) < target_audio_len:
        audio = np.pad(audio, (0, target_audio_len - len(audio)), mode='constant')
    elif len(audio) > target_audio_len:
        audio = audio[:target_audio_len]

    mel_spec = extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    
    # Pad/truncate spectrogram
    mel_spec_processed = pad_or_truncate_spectrogram(mel_spec, target_spec_len)
    
    return mel_spec_processed

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Get a random sound file ---
    urban_sound_audio_dir = os.path.join(config.DATA_DIR, "UrbanSound8K", "audio")
    try:
        random_audio_file = get_random_sound_file(urban_sound_audio_dir)
        print(f"\nSelected random sound: {random_audio_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check that UrbanSound8K dataset is downloaded and located in: {urban_sound_audio_dir}")
        return

    # --- 2. Play the sound ---
    print("Playing sound using aplay...")
    try:
        # playsound(random_audio_file) # Old method
        # Use aplay -D hw:0,0 for direct ALSA playback to the working device
        playback_command = ['aplay', '-D', 'hw:0,0', random_audio_file]
        process = subprocess.run(playback_command, check=True, capture_output=True, text=True)
        print(f"aplay finished. stdout: {process.stdout} stderr: {process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error playing sound with aplay: {e}")
        print(f"aplay stdout: {e.stdout}")
        print(f"aplay stderr: {e.stderr}")
        print("Ensure 'aplay' is installed (from alsa-utils) and hw:0,0 is the correct audio device.")
        # Continue to prediction even if playback fails
    except FileNotFoundError:
        print("Error: 'aplay' command not found. Please install 'alsa-utils'.")
        # Continue to prediction even if playback fails
    except Exception as e:
        print(f"An unexpected error occurred during sound playback with aplay: {e}")
        # Continue to prediction even if playback fails

    # --- 3. Load Model ---
    model_path = os.path.join(config.MODEL_DIR, f"model_fold_{args.fold_num}_best.safetensors")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Ensure you have trained the model for fold {args.fold_num} and it's saved in {config.MODEL_DIR}")
        return

    # CLASS_MAPPING is already idx -> class name, which is what we need for display.
    # The previous inversion to idx_to_class was creating class_name -> idx,
    # which then caused .get(integer_index) to fail.
    num_classes = len(CLASS_MAPPING)

    # --- Print num_classes for debugging ---
    # The debug print for idx_to_class is removed as it's no longer defined this way.
    print(f"DEBUG: num_classes from CLASS_MAPPING: {num_classes}")
    # --- End debug print ---

    # Determine n_mels and target_len from constants for model initialization
    n_mels_for_model = N_MELS
    target_len_for_model_calc = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / HOP_LENGTH))


    model = AudioCNN(n_mels=n_mels_for_model, num_classes=num_classes, target_len_estimate_for_fc_input_calc=target_len_for_model_calc)
    
    try:
        state_dict = safetensors.torch.load_file(model_path, device=str(device))
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        return
        
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # --- 4. Preprocess the sound for the model ---
    print("Preprocessing audio for model...")
    try:
        mel_spectrogram = preprocess_audio_for_inference(random_audio_file)
    except ValueError as e:
        print(f"Error preprocessing audio: {e}")
        return

    # Convert to tensor, add batch and channel dimensions
    # Model expects (batch, n_mels, time_frames) and adds channel dim internally
    input_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0).to(device)

    # --- 5. Make Prediction ---
    print("Making prediction...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class_name = CLASS_MAPPING.get(predicted_idx.item(), "Unknown")
    
    print(f"\n--- Prediction Results ---")
    print(f"File: {os.path.basename(random_audio_file)}")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Confidence: {confidence.item():.4f}")

    print(f"\nFull Probabilities:")
    # Ensure we iterate based on the actual number of classes from the mapping
    # for i in range(num_classes): # Original line
    for i in range(num_classes): # Iterate based on num_classes which is len(CLASS_MAPPING)
        class_name = CLASS_MAPPING.get(i, f"Unmapped_Index_{i}") # Use CLASS_MAPPING directly
        # Ensure probability index is within bounds of the output tensor
        if i < probabilities.shape[1]:
            print(f"  {class_name}: {probabilities[0, i].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a random city sound and predict its class using a trained model.")
    parser.add_argument("--fold_num", type=int, default=0,
                        help="The fold number of the trained model to use (0-9). Default: 0.")
    
    # Potentially add argument for custom audio file path later
    # parser.add_argument("--audio_file", type=str, default=None,
    # help="Optional path to a specific audio file to predict. If None, a random sound is chosen.")

    args = parser.parse_args()
    main(args) 