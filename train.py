# This script is designed to train a machine learning model to classify audio sounds.
# It uses a technique called a Convolutional Neural Network (CNN), specifically tailored for audio data.
# The training process involves feeding the model with examples of sounds and their correct categories,
# allowing it to learn patterns and make predictions on new, unseen sounds.

# --- Core Libraries ---
# os: Stands for "operating system". This library allows the script to interact with the computer's operating system,
# for example, to create directories (folders) or check if files exist.
import os
# numpy: Short for "Numerical Python". It's a fundamental package for scientific computing in Python.
# It provides powerful tools for working with arrays (lists of numbers, grids of numbers, etc.),
# which are essential for handling data in machine learning.
import numpy as np
# torch: This is the main library for PyTorch, an open-source machine learning framework developed by Facebook's AI Research lab.
# PyTorch provides the building blocks for creating and training neural networks.
import torch
# torch.nn: A sub-module of PyTorch specifically for building neural networks. 'nn' stands for "neural network".
# It contains pre-defined layers (like convolutional layers, linear layers), loss functions, and other utilities.
import torch.nn as nn
# torch.optim: A sub-module of PyTorch that provides optimization algorithms.
# Optimizers are used during training to adjust the model's internal parameters to minimize errors.
import torch.optim as optim
# Dataset and DataLoader: These are PyTorch utilities for managing and loading data efficiently.
# - Dataset: A class that represents your dataset, providing a way to access individual data samples.
# - DataLoader: Wraps a Dataset and provides an iterator to easily feed data to the model in batches (small groups).
from torch.utils.data import Dataset, DataLoader
# ReduceLROnPlateau: A learning rate scheduler from PyTorch.
# The learning rate is a parameter that controls how much the model adjusts its parameters during training.
# This scheduler automatically reduces the learning rate if the model's performance on a validation set stops improving,
# which can help the model converge to a better solution.
from torch.optim.lr_scheduler import ReduceLROnPlateau
# StratifiedKFold: A technique from scikit-learn (a popular machine learning library) for cross-validation.
# Cross-validation is a method to evaluate how well a model will generalize to new data.
# "Stratified" means it tries to ensure that each "fold" (subset of data) has a similar proportion of samples from each class.
# "K-Fold" means the data is split into 'k' folds; the model is trained on 'k-1' folds and tested on the remaining one, repeating 'k' times.
from sklearn.model_selection import StratifiedKFold
# accuracy_score: A function from scikit-learn to calculate the accuracy of predictions.
# Accuracy is the proportion of correct predictions out of the total predictions.
from sklearn.metrics import accuracy_score
# argparse: A standard Python library for parsing command-line arguments.
# This allows users to run the script with different settings (e.g., number of training cycles) without modifying the code itself.
import argparse
# json: A standard Python library for working with JSON (JavaScript Object Notation) data.
# JSON is a lightweight, human-readable format for data exchange, often used for configuration files or saving simple states.
import json
# huggingface_hub: A library from Hugging Face (a company and community focused on AI).
# This library allows interaction with the Hugging Face Hub, a platform for sharing AI models, datasets, and demos.
# This script uses it to potentially download data and upload trained models.
import huggingface_hub
# wandb: Stands for "Weights & Biases". It's a tool for tracking and visualizing machine learning experiments.
# It can log metrics (like accuracy and loss), system information, and model configurations,
# making it easier to compare different training runs and understand model behavior.
import wandb
# save_file as save_safetensors: A function from the `safetensors` library.
# Safetensors is a new, secure, and fast format for saving and loading model weights (the learned parameters).
# It's an alternative to PyTorch's default `.pt` or `.pth` format.
from safetensors.torch import save_file as save_safetensors
# time: A standard Python library for time-related functions.
# Here, it's used to measure how long each training epoch (cycle through the data) takes.
import time # For timing epochs
# tqdm: A library that provides a fast, extensible progress bar for loops in Python.
# It makes long-running processes more user-friendly by showing visual feedback.
from tqdm import tqdm # Added tqdm import

# --- Project-Specific Modules ---
# These are other Python files within this project that contain helper functions or definitions.
# utils: Likely contains general utility functions used across the project (e.g., for loading/saving files).
from utils import load_pickle, save_pickle
# model: Contains the definition of the `AudioCNN` neural network architecture.
from model import AudioCNN # Import the PyTorch model
# data_loader: Might contain constants or functions related to data loading or properties, like the number of classes.
from data_loader import NUM_CLASSES # Assuming this is still relevant and correct. NUM_CLASSES would be the total number of sound categories.
# config: A Python file that centralizes configuration settings for the project,
# such as file paths, default training parameters, and API keys or identifiers for external services.
import config

# --- Global Configuration Variables ---
# These variables are loaded from the `config.py` file and define where certain files are stored.
# CHECKPOINT_BASE_DIR: The base directory where model checkpoints (snapshots of the model during training) will be saved.
# This allows training to be resumed if it's interrupted.
CHECKPOINT_BASE_DIR = config.CHECKPOINT_BASE_DIR
# TRAINING_STATE_FILE: The name of a JSON file that will store the current state of the training process
# (e.g., which fold and epoch was last completed). This is used for resuming training.
TRAINING_STATE_FILE = config.TRAINING_STATE_FILE

# --- PyTorch Dataset Definition ---
# This class defines how our audio data should be handled by PyTorch.
# It inherits from `torch.utils.data.Dataset`, which is the base class for all PyTorch datasets.
class SoundDataset(Dataset):
    # The `__init__` method is the constructor for the class. It's called when you create a `SoundDataset` object.
    # - features: A NumPy array containing the Mel spectrograms (the input data for the model).
    # - labels: A NumPy array containing the corresponding correct categories for each spectrogram.
    # - device: Specifies where the data should be processed (CPU or GPU).
    def __init__(self, features, labels, device):
        # Convert the input `features` (NumPy array) into PyTorch Tensors.
        # Tensors are PyTorch's primary data structure, similar to NumPy arrays but with added capabilities for GPU acceleration and automatic differentiation.
        # `dtype=torch.float32` means the numbers in the tensor will be 32-bit floating-point numbers (numbers with decimals).
        self.features = torch.tensor(features, dtype=torch.float32)
        # Convert the input `labels` (NumPy array) into PyTorch Tensors.
        # `dtype=torch.long` means the numbers will be 64-bit integers (whole numbers).
        # The `CrossEntropyLoss` function (used later for calculating errors) expects labels to be of type long.
        self.labels = torch.tensor(labels, dtype=torch.long)
        # Store the `device` (e.g., 'cuda' for GPU, 'cpu' for CPU).
        # This can be used later to ensure data is on the correct device, although in this script,
        # data is moved to the device before creating the DataLoader.
        self.device = device

    # The `__len__` method should return the total number of samples in the dataset.
    # PyTorch's DataLoader uses this to know how many items to iterate over.
    def __len__(self):
        return len(self.features) # Returns the number of spectrograms (and corresponding labels).

    # The `__getitem__` method should return a single sample from the dataset at a given index `idx`.
    # PyTorch's DataLoader calls this method to get individual data points or batches.
    # - idx: The index of the data sample to retrieve.
    def __getitem__(self, idx):
        # This comment indicates that data is assumed to be pre-loaded onto the device.
        # If features were very large and couldn't all fit in GPU memory, one might load/move them here on-demand.
        # For this project, data is explicitly moved to the device before being wrapped by DataLoader.
        # Returns the feature tensor (one spectrogram) and its corresponding label tensor for the given index.
        return self.features[idx], self.labels[idx]

# --- Helper Functions for Training State and Checkpoints ---
# These functions manage saving and loading the progress of the training, including model weights.

# Function to save the current training state (fold and epoch number) to a JSON file.
# This allows the training to be resumed from where it left off if interrupted.
# - state_file: The full path to the JSON file where the state will be saved.
# - fold_num: The index of the current cross-validation fold being processed or just completed.
# - epoch_num: The number of the current training epoch being processed or just completed.
def save_training_state_json(state_file, fold_num, epoch_num):
    # `os.makedirs` creates the directory (folder) for the `state_file` if it doesn't already exist.
    # `os.path.dirname(state_file)` gets the directory part of the file path.
    # `exist_ok=True` means it won't raise an error if the directory already exists.
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    # Create a Python dictionary to store the state.
    state = {'last_fold': fold_num, 'last_epoch': epoch_num}
    # Open the `state_file` in write mode ('w').
    # The `with` statement ensures the file is properly closed even if errors occur.
    with open(state_file, 'w') as f:
        # `json.dump` writes the Python `state` dictionary to the file `f` in JSON format.
        json.dump(state, f)
    # Print a confirmation message to the console.
    print(f"Saved training state: Fold {fold_num}, Epoch {epoch_num}")

# Function to load a previously saved training state from a JSON file.
# - state_file: The full path to the JSON file from which to load the state.
def load_training_state_json(state_file):
    # Check if the `state_file` actually exists.
    if os.path.exists(state_file):
        # Open the `state_file` in read mode ('r').
        with open(state_file, 'r') as f:
            try:
                # `json.load` reads the JSON data from the file `f` and converts it into a Python dictionary.
                state = json.load(f)
                # Check if the loaded state dictionary contains the expected keys ('last_fold' and 'last_epoch').
                if 'last_fold' in state and 'last_epoch' in state:
                    # If valid, print the found state and return it.
                    print(f"Found previous training state: {state}")
                    return state
            # `json.JSONDecodeError` occurs if the file is not valid JSON (e.g., corrupted).
            except json.JSONDecodeError:
                # If the file is corrupted, print an error message and proceed as if no state file was found.
                print(f"Error reading training state file: {state_file}. Starting fresh.")
    # If the file doesn't exist or was invalid, return a default state dictionary,
    # indicating that training should start from the beginning (fold 0, epoch 0).
    return {'last_fold': 0, 'last_epoch': 0}

# Function to save a PyTorch model checkpoint during training.
# A checkpoint includes the model's learned parameters (weights), the optimizer's state,
# the current epoch number, and the validation loss at that epoch.
# - epoch: The epoch number that was just completed.
# - model: The PyTorch model object itself (e.g., an instance of `AudioCNN`).
# - optimizer: The PyTorch optimizer object (e.g., an instance of `torch.optim.Adam`).
# - val_loss: The validation loss achieved at this epoch (used to monitor performance).
# - checkpoint_path: The full file path where the checkpoint will be saved (typically a `.pt` or `.pth` file).
def save_pytorch_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path):
    # Create the directory for the checkpoint if it doesn't exist.
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # `torch.save` serializes and saves a Python dictionary containing the checkpoint information.
    torch.save({
        'epoch': epoch,  # The epoch number.
        'model_state_dict': model.state_dict(),  # The model's state dictionary (contains all learned parameters).
        'optimizer_state_dict': optimizer.state_dict(),  # The optimizer's state dictionary (needed to resume optimization correctly).
        'val_loss': val_loss,  # The validation loss (can be used to choose the best model or for learning rate scheduling).
    }, checkpoint_path)
    # Print a confirmation message.
    print(f"Saved checkpoint to {checkpoint_path} (Epoch {epoch})")

# Function to load a PyTorch model checkpoint.
# This is used to resume training or to load a trained model for evaluation/inference.
# - model: An instance of the model architecture (e.g., `AudioCNN`) into which the weights will be loaded.
# - optimizer: An instance of the optimizer; its state will also be loaded if available in the checkpoint. Can be `None` if only loading model weights for inference.
# - checkpoint_path: The path to the `.pt` checkpoint file.
# - device: The device (CPU/GPU) where the model and its weights should be loaded.
def load_pytorch_checkpoint(model, optimizer, checkpoint_path, device):
    # Check if the checkpoint file exists.
    if not os.path.exists(checkpoint_path):
        # If not found, print a message and return default values:
        # start epoch 0, and 'infinity' as loss (so any real loss will be better).
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, float('inf')

    # Print a message indicating that loading is in progress.
    print(f"Loading checkpoint from {checkpoint_path}")
    # `torch.load` loads the checkpoint file.
    # `map_location=device` ensures that the tensors are loaded onto the specified device (CPU or GPU),
    # regardless of where they were saved from. This is important for portability.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # `model.load_state_dict()` loads the saved model parameters (weights and biases) into the `model` object.
    model.load_state_dict(checkpoint['model_state_dict'])
    # If an `optimizer` object is provided and its state is in the checkpoint, load it.
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get the epoch number from the checkpoint. `.get('epoch', 0)` provides a default of 0 if 'epoch' isn't found.
    # Add 1 because training should resume from the *next* epoch after the saved one.
    start_epoch = checkpoint.get('epoch', 0) + 1
    # Get the validation loss from the checkpoint, defaulting to infinity if not found.
    val_loss = checkpoint.get('val_loss', float('inf'))
    # Print information about the resumed state.
    print(f"Resuming from Epoch {start_epoch}, Last Val Loss: {val_loss:.4f}")
    # Return the epoch to start from and the last validation loss.
    return start_epoch, val_loss

# --- Main Training Function ---
# This function orchestrates the entire model training process, including cross-validation.
# - X_all: A NumPy array containing all feature data (Mel spectrograms) for the entire dataset.
# - y_all: A NumPy array containing all corresponding labels for `X_all`.
# - folds_all: A NumPy array indicating the cross-validation fold assignment for each sample in `X_all`.
# - num_classes_global: The total number of unique sound categories the model needs to classify (e.g., 10 for UrbanSound8K).
# - model_dir: The directory path where the final trained models (best model for each fold) will be saved.
# - epochs: The number of times the model will iterate over the entire training dataset for each fold. Default is 50.
# - batch_size: The number of training samples processed before the model's parameters are updated. Default is 32.
# - learning_rate: The initial learning rate for the optimizer. Default is 0.001.
def train_model(X_all, y_all, folds_all, num_classes_global, model_dir, epochs=50, batch_size=32, learning_rate=0.001):
    # --- Device Setup ---
    # Determine whether to use a CUDA-enabled GPU (if available) or the CPU.
    # `torch.cuda.is_available()` returns True if a compatible GPU is found and PyTorch can use it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # Log the device being used.

    # --- Directory Creation ---
    # Ensure that the base directory for saving checkpoints exists. If not, create it.
    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
    # Ensure that the directory for saving final models exists. If not, create it.
    os.makedirs(model_dir, exist_ok=True)

    # --- Initialization for Storing Results ---
    # `fold_accuracies`: A list to store the best validation accuracy achieved for each fold.
    fold_accuracies = []
    # `all_fold_histories`: A list to store the detailed history (loss/accuracy per epoch) for each fold.
    all_fold_histories = []

    # --- Load Initial Training State (for Resuming) ---
    # Call `load_training_state_json` to get the last saved fold and epoch.
    # This allows the script to resume from where it left off if it was interrupted.
    initial_state = load_training_state_json(TRAINING_STATE_FILE)
    # `start_fold_idx`: The index of the fold from which to start or resume training.
    # Folds are typically 0-indexed internally (e.g., for 10 folds, indices 0 through 9).
    start_fold_idx = initial_state['last_fold']

    # --- Logic to Advance Start Fold if Previous Fold Fully Completed ---
    # This logic checks if the script is resuming (`start_fold_idx > 0`) and if the last recorded epoch
    # for that fold (`initial_state['last_epoch']`) is equal to or greater than the total epochs planned.
    if start_fold_idx > 0 and initial_state['last_epoch'] >= epochs:
        # If the previous fold was fully completed, print a message and advance to the next fold.
        print(f"Fold {start_fold_idx} completed all {epochs} epochs. Starting next fold.")
        start_fold_idx += 1 # Increment to the next fold index.
        # When starting a new fold, the epoch count begins from 0.
        initial_epoch_for_next_fold = 0
    # If resuming a fold that was not fully completed:
    elif start_fold_idx > 0 :
        # `initial_epoch_for_next_fold` will be the epoch number recorded in the state file.
        # The training loop for that fold will then attempt to start from this epoch (or epoch + 1, handled later).
        initial_epoch_for_next_fold = initial_state['last_epoch']
        print(f"Resuming Fold {start_fold_idx} from epoch {initial_epoch_for_next_fold}")
    # If `start_fold_idx` is 0, it means we are starting training from the very first fold (or no state file was found).
    else: # start_fold_idx is 0
        initial_epoch_for_next_fold = 0 # Start the first fold from epoch 0.

    # Get the unique fold numbers present in the `folds_all` array (e.g., [0, 1, 2, ..., 9] for 10-fold CV).
    # `np.unique` finds the unique values, and `np.sort` ensures they are in ascending order.
    unique_fold_numbers = np.sort(np.unique(folds_all))
    
    # --- Determine Input Data Shape for Model ---
    # The model needs to know the dimensions of the input Mel spectrograms.
    # `X_all` is expected to be a 3D NumPy array: (number_of_samples, number_of_mel_bands, number_of_time_frames).
    if X_all.ndim != 3:
        # If `X_all` doesn't have 3 dimensions, raise an error with a descriptive message.
        raise ValueError(f"Expected X_all to have 3 dimensions (num_samples, n_mels, target_len), but got {X_all.ndim}")
    # `n_mels`: The number of Mel frequency bands (typically the height of the spectrogram).
    n_mels = X_all.shape[1] # Second dimension.
    # `target_len`: The number of time frames in the spectrogram (typically the width).
    target_len = X_all.shape[2] # Third dimension.
    print(f"Input data: n_mels={n_mels}, target_len={target_len}")


    # --- Outer Loop: Cross-Validation Folds ---
    # This loop iterates through each unique fold number identified earlier.
    # `enumerate` provides both the index (`fold_idx`, 0 to k-1) and the actual fold number (`fold_num_actual`).
    for fold_idx, fold_num_actual in enumerate(unique_fold_numbers):
        # `current_fold_history`: A dictionary to store training/validation loss and accuracy for each epoch of the current fold.
        current_fold_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        # --- Fold Skipping Logic (for Resuming) ---
        # If the current `fold_idx` is less than `start_fold_idx` (determined from the loaded state),
        # it means this fold was already completed in a previous run.
        if fold_idx < start_fold_idx:
            print(f"Skipping completed Fold {fold_num_actual} (index {fold_idx})")
            # Placeholder: If a full history from previous runs is needed for `all_fold_histories.pkl`,
            # logic to load and append that history would be required here.
            # For simplicity, this version just appends `None` for skipped folds.
            all_fold_histories.append(None)
            continue # Skip to the next iteration of the fold loop.

        # Print a header for the current fold being trained.
        print(f"\n--- Training Fold {fold_num_actual} (Index {fold_idx + 1}/{len(unique_fold_numbers)}) ---")
        
        # Determine the starting epoch for *this specific fold*.
        # If this is the fold we are resuming (`fold_idx == start_fold_idx`), use `initial_epoch_for_next_fold` (from loaded state).
        # Otherwise (it's a new fold being started after a resumed one), start from epoch 0.
        initial_epoch_for_this_fold = initial_epoch_for_next_fold if fold_idx == start_fold_idx else 0

        # --- Data Splitting for Current Fold ---
        # Divide the data into training and testing (validation) sets for the current fold.
        # `train_indices`: Indices of samples NOT in the current `fold_num_actual` (these are for training).
        train_indices = np.where(folds_all != fold_num_actual)[0]
        # `test_indices`: Indices of samples that ARE in the current `fold_num_actual` (these are for testing/validation).
        test_indices = np.where(folds_all == fold_num_actual)[0]

        # Use the indices to select the actual data for training and testing.
        X_train, X_test = X_all[train_indices], X_all[test_indices]
        y_train, y_test = y_all[train_indices], y_all[test_indices]

        # Print the shapes of the training and testing data to verify.
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # --- Create PyTorch Datasets and DataLoaders ---
        # Create `SoundDataset` instances for the training and testing sets.
        train_dataset = SoundDataset(X_train, y_train, device)
        test_dataset = SoundDataset(X_test, y_test, device)
        
        # Explicitly move the feature and label tensors within the datasets to the target `device` (CPU/GPU).
        # This is done before creating DataLoaders, ensuring data is on the correct device when accessed.
        # This is a common practice if the entire dataset (for a fold) can fit in GPU memory.
        train_dataset.features = train_dataset.features.to(device)
        train_dataset.labels = train_dataset.labels.to(device)
        test_dataset.features = test_dataset.features.to(device)
        test_dataset.labels = test_dataset.labels.to(device)

        # Create `DataLoader` instances.
        # - `train_loader`: For the training data. `shuffle=True` shuffles the data at the beginning of each epoch,
        #   which helps the model learn more robustly and prevents it from learning any order-specific patterns.
        # - `test_loader`: For the testing/validation data. `shuffle=False` is typically used for validation/testing
        #   as the order doesn't matter for evaluation, and it ensures consistent evaluation.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # --- Model, Optimizer, Criterion, Scheduler Setup for Current Fold ---
        # Create an instance of the `AudioCNN` model.
        # Pass necessary parameters: `n_mels`, `num_classes_global` (total categories), and `target_len` (time frames).
        # `.to(device)` moves the model's parameters and buffers to the specified `device` (CPU/GPU).
        model = AudioCNN(n_mels=n_mels, num_classes=num_classes_global, target_len_estimate_for_fc_input_calc=target_len).to(device)
        # If Weights & Biases (`wandb`) is active (a run has been initialized):
        if wandb.run:
            # `wandb.watch(model)` tells W&B to monitor the model's gradients and parameters during training.
            # `log="all"` logs both gradients and parameters. `log_freq=100` specifies logging every 100 batches.
            wandb.watch(model, log="all", log_freq=100)

        # Create an optimizer. `optim.Adam` is a popular and generally effective optimization algorithm.
        # `model.parameters()` provides the optimizer with the parameters of the model that it needs to adjust.
        # `lr=learning_rate` sets the initial learning rate.
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Create a loss function (criterion). `nn.CrossEntropyLoss` is commonly used for multi-class classification problems.
        # It combines LogSoftmax and Negative Log Likelihood Loss, and expects raw scores (logits) from the model.
        criterion = nn.CrossEntropyLoss()
        # Create a learning rate scheduler. `ReduceLROnPlateau` monitors a metric (here, validation loss, `mode='min'`).
        # If the metric doesn't improve for a certain number of epochs (`patience=5`), it reduces the learning rate
        # by a factor (`factor=0.2`). `min_lr` sets a lower bound on the learning rate.
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)

        # Define the directory path for saving checkpoints specific to *this* fold.
        fold_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, f'fold_{fold_num_actual}')
        
        # --- Variables for Early Stopping and Best Model Tracking in Current Fold ---
        # `best_val_loss_for_early_stop`: Stores the lowest validation loss seen so far in this fold. Initialized to infinity.
        best_val_loss_for_early_stop = float('inf')
        # `patience_counter`: Counts how many epochs the validation loss hasn't improved.
        patience_counter = 0
        # `early_stopping_patience`: The number of epochs to wait for improvement before stopping training early for this fold.
        early_stopping_patience = 10 # Based on original Keras setup mentioned in previous context.
        
        # `best_val_accuracy_this_fold`: Stores the highest validation accuracy seen so far in this fold.
        best_val_accuracy_this_fold = 0.0
        # `best_epoch_this_fold`: Stores the epoch number at which the `best_val_accuracy_this_fold` was achieved.
        best_epoch_this_fold = 0
        
        # --- Load Checkpoint if Resuming This Specific Fold ---
        # `start_epoch_from_checkpoint`: This will be the epoch to actually start from if a checkpoint is loaded.
        start_epoch_from_checkpoint = 0
        # If `initial_epoch_for_this_fold` is greater than 0, it means we are trying to resume this fold.
        if initial_epoch_for_this_fold > 0:
            # Construct the path to the checkpoint file of the *previous successfully completed epoch*.
            # For example, if `initial_epoch_for_this_fold` is 5 (meaning epoch 4 was the last completed),
            # we try to load `epoch_4.pt`.
            resume_checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{initial_epoch_for_this_fold -1}.pt')
            # Check if this checkpoint file exists.
            if os.path.exists(resume_checkpoint_path):
                # If it exists, load it. `load_pytorch_checkpoint` returns the epoch to *start from* (saved_epoch + 1)
                # and the validation loss from that checkpoint.
                start_epoch_from_checkpoint, _ = load_pytorch_checkpoint(model, optimizer, resume_checkpoint_path, device)
            else:
                # If the specific checkpoint is not found, print a warning and default to starting the fold from scratch (epoch 0).
                print(f"Warning: Checkpoint {resume_checkpoint_path} not found for resuming. Starting fold from scratch.")
                start_epoch_from_checkpoint = 0
        
        # `actual_start_epoch`: The true starting epoch for the inner loop.
        # It's the maximum of `initial_epoch_for_this_fold` (from global state) and `start_epoch_from_checkpoint` (from loaded fold checkpoint).
        # This ensures that if a specific checkpoint was loaded, we use its epoch, otherwise we use the global resume state.
        actual_start_epoch = max(initial_epoch_for_this_fold, start_epoch_from_checkpoint)
        if actual_start_epoch > 0:
             # Log if the start epoch was adjusted due to checkpoint loading.
             print(f"Adjusted start epoch for this fold to: {actual_start_epoch}")


        # Print a message indicating the start of training for the current fold and epoch.
        print(f"Starting training for Fold {fold_num_actual} from epoch {actual_start_epoch}...")
        # --- Inner Loop: Epochs ---
        # This loop iterates for the specified number of `epochs`.
        # An epoch is one complete pass through the entire training dataset for the current fold.
        for epoch in range(actual_start_epoch, epochs):
            # Record the start time of the epoch for duration calculation.
            epoch_start_time = time.time()
            # --- a. Training Phase ---
            # Set the model to "training mode" using `model.train()`.
            # This is important because some layers (like Dropout, BatchNorm) behave differently during training versus evaluation.
            model.train()
            # Initialize variables to accumulate loss and count correct predictions for this epoch's training phase.
            running_loss = 0.0  # Accumulates batch losses.
            correct_train = 0   # Counts correctly classified training samples.
            total_train = 0     # Counts total training samples processed.

            # Wrap `train_loader` with `tqdm` to display a progress bar for the training batches.
            # `desc` sets the description for the progress bar. `leave=False` means the bar is removed upon completion.
            train_pbar = tqdm(train_loader, desc=f"Fold {fold_num_actual} Epoch {epoch + 1}/{epochs} Training", leave=False)
            # Loop through each batch of data provided by `train_loader`.
            # `enumerate` provides `batch_idx` (batch index) and the batch itself (`inputs`, `targets`).
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                # Data (`inputs`, `targets`) is already on the `device` because it was moved in the `SoundDataset` or before DataLoader creation.
                
                # `optimizer.zero_grad()`: Clears the gradients of all optimized model parameters.
                # Gradients are accumulated by default in PyTorch, so they need to be zeroed out before each new backward pass.
                optimizer.zero_grad()
                
                # **Forward pass**: Feed the `inputs` (batch of spectrograms) through the `model`.
                # The model processes the inputs and produces `outputs` (raw scores or logits for each class).
                outputs = model(inputs)
                
                # **Calculate loss**: The `criterion` (CrossEntropyLoss) compares the model's `outputs` with the true `targets` (correct labels)
                # and calculates the `loss` – a single number representing how "wrong" the model was for this batch.
                loss = criterion(outputs, targets)
                
                # **Backward pass (Backpropagation)**: `loss.backward()` computes the gradients of the loss with respect to all model parameters
                # that have `requires_grad=True`. These gradients indicate how each parameter should be adjusted to reduce the loss.
                loss.backward()
                
                # **Optimizer step**: `optimizer.step()` updates the model's parameters based on the computed gradients
                # and the optimization algorithm (Adam, in this case).
                optimizer.step()

                # Accumulate the loss for this batch. `loss.item()` gets the scalar loss value.
                # Multiply by `inputs.size(0)` (batch size) because the loss is typically averaged over the batch.
                running_loss += loss.item() * inputs.size(0)
                # Get predictions: `torch.max(outputs.data, 1)` finds the class with the highest score (logit) for each sample in the batch.
                # It returns (max_values, max_indices). `predicted` will be the indices (predicted class labels).
                _, predicted = torch.max(outputs.data, 1)
                # Count total samples in this batch.
                total_train += targets.size(0)
                # Count correct predictions by comparing `predicted` with `targets`.
                correct_train += (predicted == targets).sum().item() # `.sum().item()` converts the count to a Python number.
            
            # Calculate average loss and accuracy for the entire training epoch.
            epoch_loss = running_loss / len(train_loader.dataset) # Divide total loss by total number of samples.
            epoch_acc = correct_train / total_train               # Divide correct predictions by total samples.
            # Store the epoch's training loss and accuracy in the history dictionary.
            current_fold_history['loss'].append(epoch_loss)
            current_fold_history['accuracy'].append(epoch_acc)

            # --- b. Validation Phase ---
            # Set the model to "evaluation mode" using `model.eval()`.
            # This disables layers like Dropout and makes BatchNorm use its learned running statistics.
            model.eval()
            # Initialize variables for validation loss and accuracy.
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            # `with torch.no_grad()`: Context manager that disables gradient calculations.
            # This is important for evaluation/inference as it reduces memory usage and speeds up computation,
            # since we are not updating the model here.
            with torch.no_grad():
                # Optionally, wrap `test_loader` with `tqdm` for a validation progress bar.
                val_pbar = tqdm(test_loader, desc=f"Fold {fold_num_actual} Epoch {epoch + 1}/{epochs} Validation", leave=False)
                # Loop through each batch of validation data.
                for inputs, targets in val_pbar:
                    # Data is already on the `device`.
                    # Forward pass: Get model predictions on the validation batch.
                    outputs = model(inputs)
                    # Calculate loss on the validation batch.
                    loss = criterion(outputs, targets)
                    # Accumulate validation loss.
                    val_running_loss += loss.item() * inputs.size(0)
                    # Get predicted class labels.
                    _, predicted = torch.max(outputs.data, 1)
                    # Count total validation samples.
                    total_val += targets.size(0)
                    # Count correct validation predictions.
                    correct_val += (predicted == targets).sum().item()

            # Calculate average validation loss and accuracy for the epoch.
            val_loss = val_running_loss / len(test_loader.dataset)
            val_acc = correct_val / total_val
            # Store the epoch's validation loss and accuracy.
            current_fold_history['val_loss'].append(val_loss)
            current_fold_history['val_accuracy'].append(val_acc)
            
            # Calculate how long the epoch took.
            epoch_duration = time.time() - epoch_start_time
            # Print a summary of the epoch's performance.
            print(f"Fold {fold_num_actual} | Epoch {epoch + 1}/{epochs} | Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # --- c. Logging and Saving at the end of each epoch ---
            # If a Weights & Biases run is active:
            if wandb.run:
                # Log metrics to W&B. This helps track experiment progress and compare runs.
                # Metrics are prefixed with `fold_{fold_num_actual}/` to organize them by fold in W&B.
                # `step` is set to a global step counter to align plots across folds if desired.
                wandb.log({
                    f"fold_{fold_num_actual}/train_loss": epoch_loss,
                    f"fold_{fold_num_actual}/train_accuracy": epoch_acc,
                    f"fold_{fold_num_actual}/val_loss": val_loss,
                    f"fold_{fold_num_actual}/val_accuracy": val_acc,
                    "epoch": epoch +1 # Log current epoch number (1-indexed for clarity in logs).
                }, step=epoch + (fold_idx * epochs)) # Global step across all folds.

            # Update the learning rate scheduler. `scheduler.step(val_loss)` passes the current validation loss.
            # If `val_loss` hasn't improved for `patience` epochs, the scheduler will reduce the learning rate.
            scheduler.step(val_loss)

            # Save a PyTorch checkpoint (`.pt` file) for the current epoch.
            # This includes model state, optimizer state, epoch number, and validation loss.
            checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{epoch}.pt')
            save_pytorch_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path)
            
            # Save a .safetensors checkpoint (model weights only) for this epoch and attempt to upload to Hugging Face Hub.
            safetensors_epoch_filename = f'epoch_{epoch}.safetensors'
            safetensors_epoch_local_path = os.path.join(fold_checkpoint_dir, safetensors_epoch_filename)
            try:
                # `save_safetensors` saves only the model's `state_dict()` (parameters).
                save_safetensors(model.state_dict(), safetensors_epoch_local_path)
                print(f"Saved epoch {epoch} safetensors to {safetensors_epoch_local_path}")
                
                # Define the path where the file will be stored within the Hugging Face Hub repository.
                hf_path_in_repo = f"fold_{fold_num_actual}/{safetensors_epoch_filename}"
                # Attempt to upload the .safetensors file.
                hf_upload_url = huggingface_hub.upload_file(
                    path_or_fileobj=safetensors_epoch_local_path, # Local file to upload.
                    path_in_repo=hf_path_in_repo,                 # Path within the HF repo.
                    repo_id=config.HF_MODEL_REPO_ID,              # Target model repository ID from config.
                    repo_type="model",                            # Type of repository.
                    commit_message=f"Upload model checkpoint: Fold {fold_num_actual}, Epoch {epoch}" # Commit message for the upload.
                )
                print(f"Uploaded epoch {epoch} safetensors for fold {fold_num_actual} to: {hf_upload_url}")
            except Exception as e:
                # Catch any errors during saving or uploading and print a message.
                print(f"Error saving or uploading epoch {epoch} safetensors for fold {fold_num_actual}: {e}")

            # Save the overall training state (current fold index and completed epoch number) to `training_state.json`.
            # `epoch + 1` is saved because `epoch` is 0-indexed, and we want to record the *completed* epoch count.
            save_training_state_json(TRAINING_STATE_FILE, fold_idx, epoch + 1)


            # --- d. Best Model Tracking and Early Stopping for the current fold ---
            # Check if the current epoch's validation accuracy is the best seen so far *for this fold*.
            if val_acc > best_val_accuracy_this_fold:
                best_val_accuracy_this_fold = val_acc # Update the best validation accuracy.
                best_epoch_this_fold = epoch          # Record the epoch number.
                # Define the path for saving the best model's PyTorch checkpoint for this fold.
                best_model_fold_path = os.path.join(fold_checkpoint_dir, 'best_model_this_fold.pt')
                # Save this checkpoint immediately.
                save_pytorch_checkpoint(epoch, model, optimizer, val_loss, best_model_fold_path)
                print(f"New best validation accuracy for fold {fold_num_actual}: {best_val_accuracy_this_fold:.4f} at epoch {epoch + 1}")

            # Early stopping logic:
            # If the current validation loss is *lower* (better) than `best_val_loss_for_early_stop`:
            if val_loss < best_val_loss_for_early_stop:
                best_val_loss_for_early_stop = val_loss # Update the best validation loss.
                patience_counter = 0                    # Reset the patience counter.
            # If validation loss did *not* improve:
            else:
                patience_counter += 1                   # Increment the patience counter.
            
            # If the `patience_counter` reaches the `early_stopping_patience` limit:
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold_num_actual}.")
                break # Exit the inner epoch loop for this fold.
        
        # --- After Epoch Loop (for the current fold is complete, either by finishing all epochs or by early stopping) ---
        print(f"Fold {fold_num_actual} finished. Best validation accuracy: {best_val_accuracy_this_fold:.4f} at epoch {best_epoch_this_fold + 1}")
        
        # Load the best performing model (based on validation accuracy) for this fold to save its weights as .safetensors.
        best_model_final_path_pt = os.path.join(fold_checkpoint_dir, 'best_model_this_fold.pt')
        # Check if the `best_model_this_fold.pt` file actually exists (it should if training ran for at least one epoch that improved val_acc).
        if os.path.exists(best_model_final_path_pt):
            print(f"Loading best model for fold {fold_num_actual} from {best_model_final_path_pt}")
            # Create a new instance of the model architecture.
            # This ensures we are loading weights into a "clean" model, not one that might have continued training past its best point.
            best_model_for_saving = AudioCNN(n_mels=n_mels, num_classes=num_classes_global, target_len_estimate_for_fc_input_calc=target_len).to(device)
            # Load the weights from the best PyTorch checkpoint into this new model instance.
            # Optimizer is `None` as we only need the model weights for saving the final .safetensors file.
            load_pytorch_checkpoint(best_model_for_saving, None, best_model_final_path_pt, device)
            
            # Define the path for saving the final .safetensors file for this fold's best model.
            # This is typically saved in the main `model_dir` provided as an argument to `train_model`.
            safetensors_path = os.path.join(model_dir, f'model_fold_{fold_num_actual}_best.safetensors')
            print(f"Saving best weights for fold {fold_num_actual} to: {safetensors_path}")
            try:
                # Save the `state_dict` (parameters) of the `best_model_for_saving` in .safetensors format.
                save_safetensors(best_model_for_saving.state_dict(), safetensors_path)
            except Exception as e:
                print(f"Error saving .safetensors: {e}")
        else:
            # If the best model checkpoint for the fold wasn't found (e.g., if no improvement was ever made or training was interrupted very early).
            print(f"Could not find best model checkpoint at {best_model_final_path_pt} for fold {fold_num_actual}. Cannot save .safetensors.")

        # Store the best validation accuracy achieved for this fold.
        fold_accuracies.append(best_val_accuracy_this_fold)
        # Store the detailed history (loss/accuracy per epoch) for this fold.
        all_fold_histories.append(current_fold_history)
        
        # Reset `initial_epoch_for_next_fold` to 0.
        # This ensures that if the script proceeds to the *next* fold (after this one is completed),
        # that new fold will start its epoch count from 0, unless it too is being resumed from a specific checkpoint.
        initial_epoch_for_next_fold = 0


    # --- Overall Results (After All Folds) ---
    # After the outer loop (all folds) has completed:
    if fold_accuracies: # Check if `fold_accuracies` list is not empty (i.e., at least one fold was run).
        # Calculate the mean (average) and standard deviation of the best validation accuracies across all folds.
        # This gives an overall performance measure of the model.
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print("\n--- Cross-Validation Summary ---")
        # Print the best validation accuracy for each individual fold.
        print(f"Individual Fold Best Validation Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        # Print the average and standard deviation.
        print(f"Average Best Validation Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        # If Weights & Biases is active, log these overall cross-validation metrics.
        if wandb.run:
            wandb.log({
                "overall_mean_val_accuracy": mean_accuracy,
                "overall_std_val_accuracy": std_accuracy,
                "individual_fold_val_accuracies": fold_accuracies # Log the list of accuracies too.
            })
    else:
        # If no folds were trained or completed (e.g., script was interrupted before any fold finished).
        print("No folds were trained or completed.")

    # Save the `all_fold_histories` list (containing detailed loss/accuracy curves for every epoch of every fold)
    # as a pickle file. Pickle is a Python-specific format for serializing and de-serializing Python objects.
    save_pickle(all_fold_histories, os.path.join(model_dir, 'all_fold_histories_pytorch.pkl'))
    print("\n--- Training Process Finished ---")
    # Load and print the final training state recorded in `training_state.json`.
    final_state_json = load_training_state_json(TRAINING_STATE_FILE)
    print(f"Final training state recorded: {final_state_json}")

    # This section for ensuring Hugging Face model repo exists seems redundant here,
    # as a similar block is present at the start of `main()` and epoch-wise uploads are attempted during training.
    # However, keeping it as it was in the original code for completeness of this commenting pass.
    # It might serve as a final check or attempt if earlier ones failed.
    print(f"Attempting to ensure Hugging Face model repository '{config.HF_MODEL_REPO_ID}' exists (repo_type='model')...")
    try:
        # `huggingface_hub.create_repo` will create the repository if it doesn't exist,
        # or do nothing if `exist_ok=True` and it already exists.
        repo_url_or_info = huggingface_hub.create_repo(
            repo_id=config.HF_MODEL_REPO_ID,
            repo_type="model",
            exist_ok=True # Don't raise an error if the repository already exists.
        )
        # `create_repo` usually returns a `RepoInfo` object on success, which has a `repo_id` attribute.
        if hasattr(repo_url_or_info, 'repo_id'):
            print(f"Successfully ensured/verified Hugging Face model repository: {repo_url_or_info.repo_id}")
        else: # Fallback for older versions or different return types.
            print(f"Successfully ensured/verified Hugging Face model repository: {config.HF_MODEL_REPO_ID}")
            
    except huggingface_hub.utils.HfHubHTTPError as e_http: # Catch specific Hugging Face HTTP errors.
        print(f"HTTP Error creating/verifying Hugging Face model repository '{config.HF_MODEL_REPO_ID}': {e_http}")
        print(f"This usually means the repository name is invalid, you lack permissions, or there was a Hub-side issue.")
        print("Model uploads will likely fail. Please check your Hugging Face Hub settings and permissions.")
    except Exception as e: # Catch any other exceptions during repo creation/verification.
        print(f"Error creating/verifying Hugging Face model repository '{config.HF_MODEL_REPO_ID}': {e}")
        print("Model uploads might fail.")

# --- Main Execution Block ---
# The `main(args)` function is the primary entry point that orchestrates the script's operations
# when it's run with command-line arguments.
# - args: An object (usually from `argparse`) containing the parsed command-line arguments.
def main(args):
    print("Checking for processed data...") # Initial message.

    # --- Initialize Weights & Biases (W&B) ---
    # Check if a W&B run is already active (e.g., if this script is part of a W&B sweep).
    if not wandb.run:
        try:
            # `wandb.init()` initializes a new W&B run.
            wandb.init(
                project=config.WANDB_PROJECT, # Project name from `config.py`.
                entity=config.WANDB_ENTITY,   # User or team name from `config.py`.
                config={ # A dictionary of hyperparameters and settings to log with the run.
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "processed_dir": args.processed_dir,
                    "model_dir": args.model_dir,
                    "num_classes": NUM_CLASSES, # Number of classes for the classification task.
                }
            )
            print(f"Weights & Biases initialized for project: {config.WANDB_PROJECT}, entity: {config.WANDB_ENTITY}")
        except Exception as e:
            # If W&B initialization fails, print an error and continue without W&B logging.
            print(f"Could not initialize Weights & Biases: {e}. Training will continue without W&B logging.")
    else:
        # If a W&B run is already active (e.g., from a W&B sweep agent).
        print(f"Using active W&B run: {wandb.run.name}")
        # Update the W&B config with current arguments, allowing changes if it's a sweep.
        wandb.config.update({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }, allow_val_change=True) # `allow_val_change=True` is important for sweeps.

    # --- Ensure Hugging Face Model Repository Exists (Initial Check) ---
    # This attempts to create or verify the existence of the Hugging Face model repository
    # specified in `config.HF_MODEL_REPO_ID` at the beginning of the main execution.
    # This is useful so that epoch-wise model uploads later in `train_model` have a place to go.
    print(f"Attempting to ensure Hugging Face model repository '{config.HF_MODEL_REPO_ID}' exists (repo_type='model')...")
    try:
        repo_url_or_info = huggingface_hub.create_repo(
            repo_id=config.HF_MODEL_REPO_ID,
            repo_type="model",
            exist_ok=True
        )
        if hasattr(repo_url_or_info, 'repo_id'):
            print(f"Successfully ensured/verified Hugging Face model repository: {repo_url_or_info.repo_id}")
        else:
            print(f"Successfully ensured/verified Hugging Face model repository: {config.HF_MODEL_REPO_ID}")
    except huggingface_hub.utils.HfHubHTTPError as e_http:
        print(f"HTTP Error creating/verifying Hugging Face model repository '{config.HF_MODEL_REPO_ID}': {e_http}")
        print("Model uploads may fail if the repository isn't correctly set up on Hugging Face Hub.")
    except Exception as e:
        print(f"Error creating/verifying Hugging Face model repository '{config.HF_MODEL_REPO_ID}': {e}")
        print("Model uploads might fail.")

    # --- Define Paths for Essential Processed Data Files ---
    # These files are expected to be generated by a separate preprocessing script (e.g., `preprocess_data.py`).
    # `args.processed_dir` is the directory where these files should be located (from command-line or config).
    features_path = os.path.join(args.processed_dir, 'features.pkl') # Path to Mel spectrogram features.
    labels_path = os.path.join(args.processed_dir, 'labels.pkl')     # Path to corresponding labels.
    folds_path = os.path.join(args.processed_dir, 'folds.pkl')       # Path to cross-validation fold assignments.
    # `class_mapping.pkl` might also be downloaded if present in the HF dataset.

    # --- Check if Processed Data Files Exist Locally ---
    all_files_exist = (
        os.path.exists(features_path) and
        os.path.exists(labels_path) and
        os.path.exists(folds_path)
    )

    # --- Download Data from Hugging Face Hub if Not Found Locally ---
    if not all_files_exist:
        print(f"One or more processed data files (features.pkl, labels.pkl, folds.pkl) not found in '{args.processed_dir}'.")
        print(f"Attempting to download from Hugging Face Hub: {config.HF_REPO_ID} to {args.processed_dir}")
        # Ensure the local target directory for download exists.
        os.makedirs(args.processed_dir, exist_ok=True)
        try:
            # `huggingface_hub.snapshot_download` downloads files from a repository on Hugging Face Hub.
            huggingface_hub.snapshot_download(
                repo_id=config.HF_REPO_ID,           # The ID of the *dataset* repository on Hugging Face Hub.
                repo_type="dataset",                 # Specifies that it's a dataset repository.
                local_dir=args.processed_dir,        # The local directory where files will be downloaded.
                local_dir_use_symlinks=False,        # Set to False to download actual files, not symlinks.
                allow_patterns=["features.pkl", "labels.pkl", "folds.pkl", "class_mapping.pkl"] # Specify which files to download.
            )
            print(f"Download attempt finished for {args.processed_dir}.")
            # Re-check if the files exist after the download attempt.
            all_files_exist = (
                os.path.exists(features_path) and
                os.path.exists(labels_path) and
                os.path.exists(folds_path)
            )
            if not all_files_exist:
                # If files are still missing, print an error and exit.
                print(f"Error: Required data files still missing after download attempt from {config.HF_REPO_ID}.")
                print(f"Please ensure features.pkl, labels.pkl, and folds.pkl are present in the Hugging Face dataset at the root level.")
                print("Alternatively, run 'python preprocess_data.py' to generate the data locally.")
                if wandb.run: wandb.finish() # Ensure W&B run is closed if exiting early.
                return # Exit the main function.
            else:
                print(f"Successfully found/downloaded required files in {args.processed_dir}.")
        except Exception as e:
            # Catch any errors during download (e.g., network issues, repo not found).
            print(f"Error during data download from Hugging Face Hub: {e}")
            print(f"Please ensure the repository '{config.HF_REPO_ID}' exists, is accessible, and contains the required files at its root.")
            print("Alternatively, run 'python preprocess_data.py' to generate the data locally.")
            if wandb.run: wandb.finish()
            return
    else:
        # If all files were found locally from the start.
        print(f"Found all required processed data files locally in {args.processed_dir}")

    # --- Load Preprocessed Data ---
    print("\nLoading preprocessed data...")
    try:
        # `load_pickle` (from `utils.py`) loads data from pickle files.
        X = load_pickle(features_path)     # Load Mel spectrogram features.
        y = load_pickle(labels_path)       # Load labels.
        folds = load_pickle(folds_path)    # Load fold assignments.
    except FileNotFoundError:
        # This is a safeguard; the earlier checks should ideally prevent this.
        print(f"Critical Error: Processed data files were expected but not found in {args.processed_dir}.")
        print("This should have been caught by earlier checks. Please investigate.")
        if wandb.run: wandb.finish()
        return
    except Exception as e: # Catch other potential errors during file loading.
        print(f"Error loading processed data files: {e}")
        if wandb.run: wandb.finish()
        return

    # Check if any of the loaded data components are `None` (which `load_pickle` might return on failure).
    if X is None or y is None or folds is None:
         print("Error: One or more data components (X, y, folds) are None after loading. Exiting.")
         if wandb.run: wandb.finish()
         return

    # --- Reshape Feature Data (X) if Necessary ---
    # The `AudioCNN` model likely expects input features (X) to be 3D: (num_samples, n_mels, target_len).
    # Sometimes, data might be loaded with an extra "channel" dimension (e.g., (num_samples, 1, n_mels, target_len) for mono audio).
    print(f"Shape of X after loading: {X.shape}")
    if X.ndim == 4 and X.shape[1] == 1: # If X is 4D and the second dimension (channel) is 1.
        print(f"Reshaping X from {X.shape} to 3D by squeezing dimension 1.")
        X = X.squeeze(1) # `squeeze(1)` removes the dimension of size 1 at index 1.
    elif X.ndim == 4: # If X is 4D but the channel dimension is not at index 1 or not size 1.
        # This handles a case where the channel dimension might be last (less common for PyTorch Conv2D, but possible).
        if X.shape[3] == 1: # If the last dimension is 1.
            print(f"Reshaping X from {X.shape} to 3D by squeezing dimension 3 (the last dimension).")
            X = X.squeeze(3)
        else:
            # If it's 4D but the channel dimension isn't clearly 1 (e.g., might be 3 for RGB, but that's unusual for this audio task),
            # print a warning as automatic squeezing might be incorrect.
            print(f"X is 4D with shape {X.shape}, but the channel dimension (assumed to be 1 or 3) is not of size 1. Cannot safely squeeze.")


    # --- Start Model Training ---
    print("Starting model training...")
    # Call the `train_model` function with all the prepared data and configuration parameters.
    train_model(X, y, folds,
                num_classes_global=NUM_CLASSES,      # Total number of sound categories.
                model_dir=args.model_dir,            # Directory to save final models.
                epochs=args.epochs,                  # Number of epochs per fold.
                batch_size=args.batch_size,          # Batch size for training.
                learning_rate=args.learning_rate)    # Initial learning rate.
    print("Training complete. Models saved as .safetensors files.")

    # --- Finish Weights & Biases Run ---
    if wandb.run: # If a W&B run was active.
        wandb.finish() # `wandb.finish()` properly closes the W&B run, ensuring all data is synced.
        print("Weights & Biases run finished.")

# --- Script Entry Point ---
# This `if __name__ == "__main__":` block ensures that the code inside it only runs
# when the script is executed directly (not when it's imported as a module into another script).
if __name__ == "__main__":
    # Create an `ArgumentParser` object to handle command-line arguments.
    parser = argparse.ArgumentParser(description='Train the UrbanSound8K CNN model using PyTorch with 10-fold cross-validation.')
    
    # Define command-line arguments the script can accept:
    # `--processed_dir`: Directory containing the preprocessed features.
    #    `type=str` specifies the argument type.
    #    `default=config.PROCESSED_DIR` sets a default value from `config.py`.
    #    `help` provides a description shown when users run the script with `-h` or `--help`.
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR,
                        help=f'Directory containing the processed features (default: {config.PROCESSED_DIR})')
    # `--model_dir`: Directory where trained model weights will be saved.
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                        help=f'Directory to save the trained model weights (default: {config.MODEL_DIR})')
    # `--epochs`: Number of training epochs per fold.
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS,
                        help=f'Number of training epochs per fold (default: {config.DEFAULT_EPOCHS})')
    # `--batch_size`: Training batch size.
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE,
                        help=f'Training batch size (default: {config.DEFAULT_BATCH_SIZE})')
    # `--learning_rate`: Initial learning rate for the optimizer.
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')

    # Parse the command-line arguments provided when the script is run.
    # The parsed arguments will be stored in the `args` object (e.g., `args.epochs`).
    args = parser.parse_args()
    
    # Call the `main` function, passing the parsed command-line arguments.
    main(args) 