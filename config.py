import os

# --- New Path Calculation for Assets/Checkpoints above project root ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Project root
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR) # Directory above project root

# Define the base assets path relative to the parent directory
ASSETS_SUBDIR = os.path.join('assets', 'audio_is_all_you_need')
ASSETS_BASE_DIR = os.path.join(_PARENT_DIR, ASSETS_SUBDIR)

# --- Directories ---

# Base directory for all data
# DATA_DIR = '../assets/audio_is_all_you_need' # This can be replaced by ASSETS_BASE_DIR if it's the same, or defined using it.
                                              # For now, let's keep it as it might be used independently elsewhere.
                                              # If its only purpose was to define PROCESSED_DIR, we can remove/refactor it.
DATA_DIR = ASSETS_BASE_DIR # Assuming DATA_DIR should also point to the assets base. If it's truly a *different* '../assets/...' it needs to be defined carefully.
                           # Based on the desired output, it seems DATA_DIR and ASSETS_BASE_DIR should resolve to the same path.

# Directory for preprocessed features, labels, etc.
PROCESSED_DIR = os.path.join(ASSETS_BASE_DIR, 'data')

# Directory to save final trained model weights/files
MODEL_DIR = os.path.join(ASSETS_BASE_DIR, 'models') # Changed to use ASSETS_BASE_DIR for an absolute path

# Define checkpoint directory within the assets base dir
CHECKPOINT_SUBDIR = 'checkpoints'
CHECKPOINT_BASE_DIR = os.path.join(ASSETS_BASE_DIR, CHECKPOINT_SUBDIR)

# File to store the last training state for resuming (inside the checkpoint dir)
TRAINING_STATE_FILE = os.path.join(CHECKPOINT_BASE_DIR, 'training_state.json')


# --- Training Parameters ---

# Default number of training epochs
DEFAULT_EPOCHS = 50

# Default batch size for training
DEFAULT_BATCH_SIZE = 32


# --- External Services ---

# Hugging Face Hub Repository ID
HF_REPO_ID = "wazzuck/audio_is_all_you_need"

# Hugging Face Hub Repository ID for Models - PLEASE UPDATE THIS TO YOUR ACTUAL MODEL REPO NAME
HF_MODEL_REPO_ID = "wazzuck/audio_is_all_you_need" # Changed to match the desired model repo ID

# Wandb Project Name
WANDB_PROJECT = "audio_is_all_you_need"

# Wandb Entity (Username or Team Name) - **PLEASE VERIFY/UPDATE THIS**
WANDB_ENTITY = "nevillebryce-ai-learning" # Replace with your actual Wandb entity if different


# --- Potentially Add More ---
# e.g., parameters from src/data_loader.py if desired:
# SAMPLE_RATE = 22050
# MAX_DURATION_S = 4.0
# N_MELS = 128
# HOP_LENGTH = 512
# NUM_CLASSES = 10

# TEMP_DOWNLOAD_DIR = "../assets/audio_is_all_you_need" # This could also potentially use ASSETS_BASE_DIR
TEMP_DOWNLOAD_DIR = ASSETS_BASE_DIR 