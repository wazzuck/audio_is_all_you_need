# Audio is All You Need - UrbanSound8K Classification

This project implements a CNN-based model, inspired by common audio classification architectures, to classify sounds from the UrbanSound8K dataset. It uses Mel spectrograms as input features.

## Features

*   Data downloading and preprocessing scripts.
*   CNN model defined using TensorFlow/Keras.
*   10-fold cross-validation training script.
*   Checkpointing for resumable training.
*   Saving best model weights per fold in `.safetensors` format.
*   Prediction script for classifying new audio files.
*   Centralized configuration in `config.py`.

## Directory Structure

```
../assets/
│   └── audio_is_all_you_need/
│       └── checkpoints/            # Saved model checkpoints for resuming
│           ├── fold_*/              # Checkpoints for each fold
│           │   └── epoch_**/        # Checkpoint for each epoch
│           │       └── ... (tf saved model format)
│           └── training_state.json # Tracks last completed fold/epoch
.
├── config.py                       # Central configuration file
├── data/
│   ├── UrbanSound8K/               # Downloaded dataset (created by download_data.py)
│   │   ├── audio/
│   │   └── metadata/
│   └── processed/                  # Processed features (created by preprocess_data.py)
│       ├── features.pkl
│       ├── labels.pkl
│       ├── folds.pkl
│       └── class_mapping.pkl
├── data_loader.py                  # Data loading, feature extraction, constants
├── download_data.py                # Script to download UrbanSound8K
├── model.py                        # CNN model definition
├── models/                         # Saved final best model weights (created by train.py)
│   └── model_fold_*_best.safetensors
├── notes                           # Personal notes file
├── notebooks/                      # Jupyter notebooks (if any)
├── papers/                         # Related papers (if any)
├── predict.py                      # Script to predict class for an audio file
├── preprocess_data.py              # Script to preprocess audio and extract features
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── slides/                         # Related slides (if any)
├── train.py                        # Model training script
├── utils.py                        # Utility functions (pickle/json loading/saving)
.gitignore
└── AI @ FAC - week 5 - project - audio is all you need.pdf # Project description PDF
```
*(Note: Some directories like `../assets`, `data/UrbanSound8K`, `data/processed`, and `models` are created by running the scripts).*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-directory>
    ```
2.  **Run the initial Anaconda setup script:**
    This script will download and run the Anaconda installer. Follow the prompts during the installation.
    ```bash
    bash 00_setup.sh
    ```
3.  **IMPORTANT: Close and Reopen Terminal:**
    After `00_setup.sh` finishes, you **must** close your current terminal window and open a new one. This ensures that the changes made by the Anaconda installer (like updating your system's PATH) are recognized by the shell.

4.  **Run the environment setup script:**
    In the **new** terminal window, navigate back to the project directory and run the second script. This will typically create the specific conda environment for this project and install the required dependencies from `requirements.txt`.
    ```bash
    cd <repository-directory> # If needed
    bash 01_setup.sh
    ```
5.  **Activate the Conda Environment (if needed):**
    The `01_setup.sh` script *might* activate the environment automatically. If not, or if you need to activate it in a later session, use the command specified in `01_setup.sh` (usually something like `conda activate <your_env_name>`).

## Usage

Follow these steps in order:

1.  **Download the UrbanSound8K Dataset:**
    Run the download script. This will download the dataset into the `data/` directory using the `soundata` library.
    ```bash
    python download_data.py
    ```

2.  **Preprocess the Data:**
    Run the preprocessing script. This loads the audio files, extracts Mel spectrograms, and saves the processed features, labels, fold information, and class mapping into the `data/processed/` directory as pickle files.
    ```bash
    python preprocess_data.py
    ```
    *   You can optionally specify `--data_dir` (where UrbanSound8K is) and `--output_dir` (where to save processed files), but they default to the paths in `config.py`.

3.  **Train the Model:**
    Run the training script. This performs 10-fold cross-validation.
    ```bash
    python train.py
    ```
    *   **Arguments:**
        *   `--processed_dir`: Directory containing the output from `preprocess_data.py` (default: `data/processed`).
        *   `--model_dir`: Directory to save the final best `.safetensors` weights for each fold (default: `models`).
        *   `--epochs`: Number of epochs to train per fold (default: 50 from `config.py`).
        *   `--batch_size`: Training batch size (default: 32 from `config.py`).
    *   **Checkpointing & Resuming:** The script saves checkpoints (full model state) after each epoch to `assets/audio_is_all_you_need/checkpoints/`. It also saves the last completed fold and epoch in `training_state.json`. If the script is interrupted, simply run it again, and it will automatically resume from the last saved state.
    *   **Output:** The best model weights for each fold (`model_fold_X_best.safetensors`) will be saved in the specified `--model_dir`. Training logs and cross-validation results will be printed to the console.

4.  **Predict on a New Audio File:**
    Use the prediction script to classify a single audio file.
    ```bash
    python predict.py path/to/your/audio.wav --fold_num <fold_to_use>
    ```
    *   Replace `path/to/your/audio.wav` with the actual path to your audio file.
    *   Replace `<fold_to_use>` with the fold number whose trained model you want to use for prediction (e.g., `1`). This corresponds to the model saved as `models/model_fold_1_best.safetensors`.
    *   **Arguments:**
        *   `audio_file` (Required): Path to the input audio file.
        *   `--model_dir`: Directory containing the saved `.safetensors` model weights (default: `models`).
        *   `--processed_dir`: Directory containing `class_mapping.pkl` (default: `data/processed`).
        *   `--fold_num`: Which fold's trained model weights to use (default: 1).
    *   **Output:** The script will print the predicted class, confidence score, and the full probability distribution over all classes.

## Model Architecture

The model uses a series of Convolutional blocks followed by temporal pooling and a final classification head.

```
Input: (Batch, Mel Bands, Time Steps, 1)  [e.g., (None, 128, 173, 1)]
  │
  ▼
Conv2D (32 filters, 3x3, same padding) -> BatchNorm -> ReLU -> MaxPool (2,2) -> Dropout (0.2)
  │
  ▼
Conv2D (64 filters, 3x3, same padding) -> BatchNorm -> ReLU -> MaxPool (2,2) -> Dropout (0.2)
  │
  ▼
Conv2D (128 filters, 3x3, same padding) -> BatchNorm -> ReLU -> MaxPool (2,4) -> Dropout (0.2)
  │                                               Shape: (Batch, Freq', Time', 128)
  ▼
Lambda: Reduce Mean over Time Axis (axis=2)       Shape: (Batch, Freq', 128)
  │
  ▼
Conv1D (64 filters, kernel=3, same padding, ReLU) Shape: (Batch, Freq', 64)
  │
  ▼
Flatten                                           Shape: (Batch, Freq' * 64)
  │
  ▼
Dense (128 units, ReLU) -> Dropout (0.5)
  │
  ▼
Dense (10 units, Softmax)                         Shape: (Batch, 10)
  │
  ▼
Output: Probabilities per Class
```

## Configuration

Key parameters like directory paths, default training settings (epochs, batch size), and external service details (like WandB/Hugging Face integration added previously) can be modified in `config.py`. 