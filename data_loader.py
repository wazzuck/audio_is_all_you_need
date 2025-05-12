import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf # Added for robust audio loading

# Constants from the paper/common practice
# Using 4s slices as per paper's finding
MAX_DURATION_S = 4.0 
SAMPLE_RATE = 22050 # Standard for many audio tasks, adjust if dataset varies significantly
N_MELS = 128 # Number of Mel bands
N_FFT = 2048 # Window size for FFT
HOP_LENGTH = 512 # Hop length for FFT

# Define class mapping based on dataset description
CLASS_MAPPING = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing", 3: "dog_bark",
    4: "drilling", 5: "engine_idling", 6: "gun_shot", 7: "jackhammer",
    8: "siren", 9: "street_music"
}
NUM_CLASSES = len(CLASS_MAPPING)

def load_audio_robust(filepath, sr=SAMPLE_RATE):
    """Loads audio file using soundfile, falling back to librosa if needed."""
    try:
        # Try soundfile first (often handles more formats/corruptions)
        audio, file_sr = sf.read(filepath, dtype='float32')
        # Resample if necessary
        if file_sr != sr:
            audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr)
            # If stereo, convert to mono by averaging
            if audio.ndim > 1:
                 audio = np.mean(audio, axis=0)
        else:
             # If stereo, convert to mono by averaging
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        return audio
    except Exception as e_sf:
        print(f"Soundfile failed for {filepath}: {e_sf}. Trying librosa.")
        try:
            # Fallback to librosa
            audio, file_sr = librosa.load(filepath, sr=sr, mono=True)
            return audio
        except Exception as e_lr:
            print(f"Librosa also failed for {filepath}: {e_lr}. Skipping file.")
            return None

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Extracts a Mel spectrogram from an audio signal."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def pad_or_truncate_spectrogram(spec, target_len):
    """Pads or truncates the spectrogram along the time axis."""
    if spec.shape[1] < target_len:
        pad_width = target_len - spec.shape[1]
        # Pad with minimum value (silence in log-scale)
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.min(spec))
    elif spec.shape[1] > target_len:
        spec = spec[:, :target_len]
    return spec

def load_and_process_data(metadata_path, audio_dir):
    """Loads metadata, processes audio files into Mel spectrograms, and returns features and labels."""
    metadata = pd.read_csv(metadata_path)
    features = []
    labels = []
    folds = []

    # Calculate target length based on max duration
    target_len = int(np.ceil(MAX_DURATION_S * SAMPLE_RATE / HOP_LENGTH))
    print(f"Target spectrogram length: {target_len} frames")

    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing audio files"):
        filename = os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        class_id = row['classID']
        fold = row['fold']

        if not os.path.exists(filename):
            print(f"Warning: File not found {filename}. Skipping.")
            continue

        audio = load_audio_robust(filename, sr=SAMPLE_RATE)
        if audio is None:
            continue # Skip if loading failed

        # Pad or truncate audio to MAX_DURATION_S * SAMPLE_RATE length before spectrogram
        # This ensures all spectrograms *start* with the same time dimension assumption
        # Note: This is slightly different from padding the spectrogram itself, might be better
        target_audio_len = int(MAX_DURATION_S * SAMPLE_RATE)
        if len(audio) < target_audio_len:
            audio = np.pad(audio, (0, target_audio_len - len(audio)), mode='constant')
        elif len(audio) > target_audio_len:
            audio = audio[:target_audio_len]

        mel_spec = extract_mel_spectrogram(audio)

        # Pad/truncate spectrogram just in case length varies slightly due to edge effects
        mel_spec_processed = pad_or_truncate_spectrogram(mel_spec, target_len)

        features.append(mel_spec_processed)
        labels.append(class_id)
        folds.append(fold)

    features = np.array(features)
    labels = np.array(labels)
    folds = np.array(folds)

    # Add channel dimension for CNN input (channels_last format)
    features = features[..., np.newaxis]

    print(f"Processed {len(features)} files.")
    print(f"Features shape: {features.shape}") # Should be (num_samples, n_mels, target_len, 1)
    print(f"Labels shape: {labels.shape}")
    print(f"Folds shape: {folds.shape}")

    return features, labels, folds, CLASS_MAPPING 