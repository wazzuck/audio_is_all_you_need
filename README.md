# Audio is All You Need - UrbanSound8K Classification

This project implements a CNN-based model, inspired by common audio classification architectures, to classify sounds from the UrbanSound8K dataset. It uses Mel spectrograms as input features.

## Features

*   Data downloading and preprocessing scripts.
*   CNN model defined using PyTorch.
*   10-fold cross-validation training script.
*   Checkpointing for resumable training (saving PyTorch model states).
*   Saving best model weights per fold in `.safetensors` format.
*   Prediction script for classifying new audio files.
*   Centralized configuration in `config.py`.

## Directory Structure

The project's main directory contains the scripts and configuration. Generated assets like processed data, model checkpoints, and final models are stored outside the project root in a sibling `assets/` directory by default (configurable in `config.py`).

```
../assets/
└── audio_is_all_you_need/      # Base directory for generated assets (see ASSETS_BASE_DIR in config.py)
    ├── checkpoints/            # Saved model checkpoints for resuming (see CHECKPOINT_BASE_DIR)
    │   ├── fold_*/              # Checkpoints for each fold
    │   │   ├── epoch_*.pt       # PyTorch state dict (model, optimizer)
    │   │   ├── epoch_*.safetensors # Model state dict (weights only)
    │   │   └── best_model_this_fold.pt # Best model for the fold (PyTorch state)
    │   └── training_state.json # Tracks last completed fold/epoch
    ├── data/                   # Processed features (see PROCESSED_DIR in config.py)
    │   ├── features.pkl
    │   ├── labels.pkl
    │   ├── folds.pkl
    │   └── class_mapping.pkl
    └── models/                 # Saved final best model weights (see MODEL_DIR in config.py)
        └── model_fold_*_best.safetensors
. (Project Root)
├── 00_setup.sh                     # Initial Anaconda setup script
├── 01_setup.sh                     # Conda environment and dependency setup script
├── config.py                       # Central configuration file (paths, training params, HF/W&B IDs)
├── create_test_tone.py             # Script to create a test WAV file
├── data/
│   └── UrbanSound8K/               # Downloaded dataset (created by download_data.py in this structure)
│       ├── audio/
│       └── metadata/
├── data_loader.py                  # Data loading, feature extraction, constants
├── download_data.py                # Script to download UrbanSound8K
├── model.py                        # CNN model definition (PyTorch)
├── notes                           # Personal notes file
├── notebooks/                      # Jupyter notebooks (if any)
├── papers/                         # Related papers (if any)
├── play_and_predict_random.py      # Example script to play and predict a random slice
├── predict.py                      # Script to predict class for an audio file
├── preprocess_data.py              # Script to preprocess audio and extract features
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── slides/                         # Related slides (if any)
├── test_playsound_app.py           # Test script for playsound
├── test_tone.wav                   # A test audio file
├── train.py                        # Model training script (PyTorch)
├── train_explainer.md              # Detailed explanation of train.py
├── train_explainer.txt             # Plain text version of train_explainer.md
├── utils.py                        # Utility functions (pickle/json loading/saving)
├── wandb/                          # Directory for Weights & Biases local artifacts
├── .gitignore
└── AI @ FAC - week 5 - project - audio is all you need.pdf # Project description PDF
```
*(Note: Directories like `../assets/audio_is_all_you_need/...`, `data/UrbanSound8K/`, and `wandb/` are created by running the scripts or by external tools like W&B).*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-directory>
    ```

2.  **Python Version Prerequisite:**
    This project is developed with **Python 3.11**. The setup scripts will configure a Conda environment with this Python version.

3.  **Run the initial Anaconda setup script (`00_setup.sh`):**
    This script will help you download and run the Anaconda installer if you don't have Conda already. Follow the prompts during the installation.
    ```bash
    bash 00_setup.sh
    ```
    *If you already have Conda installed and initialized, you might be able to skip this step, but ensure your Conda base environment is accessible.*

4.  **IMPORTANT: Close and Reopen Terminal (after `00_setup.sh` if run):**
    If you ran `00_setup.sh`, you **must** close your current terminal window and open a new one. This ensures that the changes made by the Anaconda installer (like updating your system's PATH) are recognized by the shell.

5.  **Run the environment setup script (`01_setup.sh`):**
    In the **new** terminal window (if applicable), navigate back to the project directory and run the second script. This script will:
    *   Create a Conda environment named `audio_env` (or your chosen name, e.g., `tf_env` is still in the script, update if necessary) with Python 3.11.
    *   Activate the `audio_env` environment *within the script* to ensure correct package installation.
    *   Install the required Python dependencies from `requirements.txt` (now including PyTorch) into `audio_env`.
    ```bash
    cd <repository-directory> # If needed
    bash 01_setup.sh
    ```
    *Note: The `01_setup.sh` script might still refer to `tf_env`. You may need to update the `ENV_NAME` variable within `01_setup.sh` to `audio_env` or your preferred name, and ensure `requirements.txt` is up-to-date for PyTorch before running it.*

6.  **(Optional) Automatic Environment Activation:**
    For convenience, you might want the `audio_env` environment to activate automatically. (Instructions for direnv or manual shell configuration remain the same).

## Usage

**IMPORTANT: Ensure the `audio_env` (or your chosen Conda environment name) is activated (e.g., `conda activate audio_env`) before running any Python scripts.**

Follow these steps in order:

1.  **Download the UrbanSound8K Dataset:**
    Run the download script. This will download the dataset into the `data/` directory (specifically `data/UrbanSound8K/`) using the `soundata` library.
    ```bash
    python download_data.py
    ```

2.  **Preprocess the Data:**
    Run the preprocessing script. This loads the audio files, extracts Mel spectrograms, and saves the processed features, labels, fold information, and class mapping.
    By default, processed files are saved to `../assets/audio_is_all_you_need/data/` (configurable via `PROCESSED_DIR` in `config.py`).
    The script will also attempt to upload these processed files to a Hugging Face Hub dataset repository specified by `HF_REPO_ID` in `config.py`.
    ```bash
    python preprocess_data.py
    ```
    *   You can optionally specify `--data_dir` (where UrbanSound8K is, defaults to `DATA_DIR` in `config.py`) and `--output_dir` (where to save processed files, defaults to `PROCESSED_DIR` in `config.py`).

3.  **Train the Model:**
    Run the training script. This performs 10-fold cross-validation using PyTorch.
    See `train_explainer.md` for a detailed breakdown of this script.
    ```bash
    python train.py
    ```
    *   **Arguments:**
        *   `--processed_dir`: Directory containing the processed features (default: `../assets/audio_is_all_you_need/data/`, from `config.PROCESSED_DIR`). If files are not found locally, the script will attempt to download them from the Hugging Face Hub dataset repository specified by `HF_REPO_ID` in `config.py`.
        *   `--model_dir`: Directory to save the final best `.safetensors` weights for each fold (default: `../assets/audio_is_all_you_need/models/`, from `config.MODEL_DIR`).
        *   `--epochs`: Number of epochs to train per fold (default: `config.DEFAULT_EPOCHS`).
        *   `--batch_size`: Training batch size (default: `config.DEFAULT_BATCH_SIZE`).
        *   `--learning_rate`: Initial learning rate for the Adam optimizer (default: 0.001).
    *   **Checkpointing & Resuming:**
        *   The script saves PyTorch model checkpoints (`.pt` files containing model state, optimizer state, epoch, and validation loss) and Safetensors model weights (`.safetensors`) after each epoch. These are stored in `../assets/audio_is_all_you_need/checkpoints/fold_<fold_num>/` (path from `config.CHECKPOINT_BASE_DIR`).
        *   Epoch-wise `.safetensors` checkpoints are also uploaded to a Hugging Face Hub model repository specified by `config.HF_MODEL_REPO_ID`.
        *   The script saves the last completed fold and epoch in `../assets/audio_is_all_you_need/checkpoints/training_state.json`. If interrupted, running it again automatically attempts to resume from the last saved state.
    *   **Output & Logging:**
        *   The best model weights for each fold (`model_fold_X_best.safetensors`) are saved in the directory specified by `--model_dir` and also uploaded to the Hugging Face Hub model repository (`config.HF_MODEL_REPO_ID`).
        *   Training logs and cross-validation results are printed to the console.
        *   If configured (see `WANDB_PROJECT`, `WANDB_ENTITY` in `config.py`), metrics are logged to Weights & Biases.

4.  **Predict on a New Audio File:**
    Use the prediction script to classify a single audio file using a trained PyTorch model.
    ```bash
    python predict.py path/to/your/audio.wav --fold_num <fold_to_use>
    ```
    *   Replace `path/to/your/audio.wav` with the actual path to your audio file.
    *   Replace `<fold_to_use>` with the 0-indexed fold number whose trained model you want to use (e.g., `0` for the first fold). This corresponds to the model saved as `model_fold_0_best.safetensors` in the models directory.
    *   **Arguments:**
        *   `audio_file` (Required): Path to the input audio file.
        *   `--model_dir`: Directory containing the saved `.safetensors` model weights (default: `../assets/audio_is_all_you_need/models/`, from `config.MODEL_DIR`).
        *   `--processed_dir`: Directory containing `class_mapping.pkl` (default: `../assets/audio_is_all_you_need/data/`, from `config.PROCESSED_DIR`).
        *   `--fold_num`: Which fold's trained model weights to use (0-indexed, default: 0).
    *   **Output:** The script will print the predicted class, confidence score, and the full probability distribution over all classes.

## Model Architecture

The model uses a series of Convolutional blocks followed by temporal pooling and a final classification head. (The conceptual architecture remains the same, PyTorch implementation details are in `model.py`).

```
Input: (Batch, Channels, Mel Bands, Time Steps)  [e.g., (None, 1, 128, 173)] # PyTorch NCHW format
  │
  ▼
Conv2D (32 filters, 3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2,2) -> Dropout (0.2)
  │
  ▼
Conv2D (64 filters, 3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2,2) -> Dropout (0.2)
  │
  ▼
Conv2D (128 filters, 3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2,4) -> Dropout (0.2)
  │                                               Shape: (Batch, 128, Freq', Time')
  ▼
Mean over Time Axis (dim=3 in PyTorch)            Shape: (Batch, 128, Freq')
  │ (Permuted for Conv1D if channels are not first, but PyTorch Conv1D takes (Batch, Channels, Length))
  │ Here, input to Conv1D is (Batch, 128 channels, Freq' length)
  ▼
Conv1D (64 filters, kernel=3, padding=1, ReLU)    Shape: (Batch, 64, Freq')
  │
  ▼
Flatten                                           Shape: (Batch, Freq' * 64)
  │
  ▼
Linear (128 units, ReLU) -> Dropout (0.5)
  │
  ▼
Linear (10 units)                                 Shape: (Batch, 10) # Output logits
  │ (Softmax is typically applied in the loss function, e.g., CrossEntropyLoss)
  ▼
Output: Logits per Class (or Probabilities if Softmax is applied post-hoc)
```

## Configuration

Key parameters are centralized in `config.py`. This includes:
*   **Directory Paths:** Paths for storing downloaded raw data (`DATA_DIR`), processed data (`PROCESSED_DIR`), model checkpoints (`CHECKPOINT_BASE_DIR`), and final trained models (`MODEL_DIR`). These typically point to locations within `../assets/audio_is_all_you_need/` by default.
*   **Default Training Parameters:** `DEFAULT_EPOCHS`, `DEFAULT_BATCH_SIZE`.
*   **External Service Integration:**
    *   Hugging Face Hub: Repository IDs for datasets (`HF_REPO_ID`) and models (`HF_MODEL_REPO_ID`).
    *   Weights & Biases: Project (`WANDB_PROJECT`) and entity (`WANDB_ENTITY`) for experiment tracking.

Modify `config.py` to change these settings for your environment or preferences. 