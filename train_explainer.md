# `train.py` Explainer

## Overall Goal of `train.py`

The primary goal of `train.py` is to train an audio classification model
(`AudioCNN`) using preprocessed Mel spectrogram features. It employs a k-fold
cross-validation strategy to evaluate the model's performance robustly.
The script handles:
1.  Loading configuration and data.
2.  Optionally downloading data from Hugging Face Hub.
3.  Setting up the model, optimizer, and loss function.
4.  Iterating through training folds and epochs.
5.  Saving model checkpoints locally.
6.  Attempting to upload model checkpoints to Hugging Face Hub.
7.  Logging metrics (to console and potentially Weights & Biases).
8.  Resuming training from a previously saved state.

## 1. Setup and Configuration Phase

This phase involves importing necessary libraries and loading configurations.

*   **Inputs:**
    *   Python scripts: `utils.py`, `model.py`, `data_loader.py`, `config.py`.
    *   Command-line arguments (parsed by `argparse`).
*   **Actions:**
    *   Imports libraries like `torch`, `numpy`, `huggingface_hub`, `wandb`,
        `tqdm`, etc.
    *   Loads global constants and paths from `config.py` (e.g.,
        `PROCESSED_DIR`, `MODEL_DIR`, `HF_MODEL_REPO_ID`, `HF_REPO_ID`,
        `TRAINING_STATE_FILE`).
    *   Defines command-line arguments (`--processed_dir`, `--model_dir`,
        `--epochs`, etc.) allowing runtime customization. These arguments
        default to values from `config.py`.
*   **Outputs:**
    *   An `args` object containing parsed command-line arguments.
    *   Configuration variables available for the script.

```
   [config.py] ----> HF_MODEL_REPO_ID, PROCESSED_DIR, etc.
                     |
 [Command Line] ----> args (epochs, batch_size, etc.)
                     |
                     V
             [train.py Script]
                     |
                     V
           (Script Execution)
```

## 2. `main()` Function - Orchestration

The `main(args)` function is the primary entry point and controls the overall
workflow of the training process.

*   **Key Steps & Flow:**

    1.  **W&B Initialization (Attempt):**
        *   Tries to initialize Weights & Biases (`wandb.init`) for experiment
            tracking, using project and entity details from `config.py`.
        *   If it fails (e.g., user not logged in), it prints a warning and
            continues without W&B.

    2.  **Hugging Face Model Repository Creation/Verification (Attempt):**
        *   Prints: "Attempting to ensure Hugging Face model repository..."
        *   Calls `huggingface_hub.create_repo` with `repo_id=config.HF_MODEL_REPO_ID`,
            `repo_type="model"`, and `exist_ok=True`.
        *   This attempts to create the specified repository on Hugging Face Hub
            if it doesn't exist, or verifies it if it does.
        *   Logs success or failure (with error details). This step is crucial
            for later model uploads.

        ```
          [config.py] ---> HF_MODEL_REPO_ID
                          |
                          V
        main() --> create_repo(HF_MODEL_REPO_ID, type="model") --> [Hugging Face Hub]
                          |                                         (Creates or Verifies Repo)
                          V
                   (Logs Success/Error)
        ```

    3.  **Data Loading and Preprocessing Check:**
        *   **Inputs:**
            *   `args.processed_dir` (path to processed data, from `config.PROCESSED_DIR`).
            *   `config.HF_REPO_ID` (for downloading dataset if local files are missing).
        *   **Actions:**
            a.  Constructs paths for `features.pkl`, `labels.pkl`, `folds.pkl`.
            b.  Checks if these files exist locally in `args.processed_dir`.
            c.  **If files are missing:**
                *   Prints: "One or more processed data files ... not found..."
                *   Attempts to download these files (plus `class_mapping.pkl`)
                    from the Hugging Face Hub dataset repository specified by
                    `config.HF_REPO_ID` using `huggingface_hub.snapshot_download`.
                    The target is `args.processed_dir`.
                *   Re-checks if files exist after download. If still missing,
                    prints an error and exits.
            d.  **If files exist (locally or after download):**
                *   Loads `X` (features), `y` (labels), and `folds` from their
                    respective `.pkl` files using `utils.load_pickle`.
                *   Checks if any loaded data is `None`; if so, exits.
                *   Prints the shape of `X`.
                *   **Reshapes `X`:** If `X` is 4D (e.g., `(N, 1, H, W)` or
                    `(N, H, W, 1)`), it squeezes out the channel dimension
                    (assumed to be of size 1) to make `X` 3D (`(N, H, W)`), as
                    the model's `forward` method expects this and adds the
                    channel dimension internally.
        *   **Outputs:**
            *   `X` (numpy array, features, e.g., `(num_samples, n_mels, target_len)`)
            *   `y` (numpy array, labels, e.g., `(num_samples,)`)
            *   `folds` (numpy array, fold assignments, e.g., `(num_samples,)`)

        ```
        Is data in args.processed_dir?
              |
             NO --> Attempt HF Download (from config.HF_REPO_ID, type="dataset")
              |       |
             YES      V (Success?)
              |      NO --> Error, Exit
              |       |
              |      YES
              |       |
              V       V
        Load .pkl files (features, labels, folds) --> X, y, folds
                                                        |
                                                        V
                                            Reshape X (if 4D with channel=1)
                                                        |
                                                        V
                                              Call train_model()
        ```

    4.  **Call `train_model()`:**
        *   Invokes the core training function with `X`, `y`, `folds`, and other
            parameters like number of classes, model directory, epochs, batch
            size, and learning rate.

    5.  **W&B Finish:**
        *   If a W&B run was initialized, `wandb.finish()` is called to sync
            final data and close the run.

## 3. `SoundDataset` Class (PyTorch `Dataset`)

This class is a standard PyTorch `Dataset` wrapper for the audio features and
labels.

*   **Purpose:** To provide an interface for the PyTorch `DataLoader` to
    efficiently load data in batches.
*   **`__init__(self, features, labels, device)`:**
    *   **Inputs:** `features` (numpy array), `labels` (numpy array), `device`
        (torch device).
    *   **Action:** Converts features and labels to PyTorch tensors (`torch.float32`
        for features, `torch.long` for labels).
*   **`__len__(self)`:**
    *   **Action:** Returns the total number of samples in the dataset.
*   **`__getitem__(self, idx)`:**
    *   **Input:** `idx` (integer index).
    *   **Action:** Returns the feature tensor and label tensor for the sample
        at the given index. Data might be moved to the specified `device` here
        or can be pre-moved before creating the `DataLoader`.

## 4. Helper Functions for State and Checkpoints

These utilities manage saving and loading training progress and model states.

*   **`save_training_state_json(state_file, fold_num, epoch_num)`:**
    *   **Inputs:** `state_file` (path to JSON), `fold_num` (current fold index),
        `epoch_num` (current completed epoch).
    *   **Action:** Saves a dictionary `{'last_fold': fold_num, 'last_epoch': epoch_num}`
        to the specified JSON file. This allows training to be resumed.
*   **`load_training_state_json(state_file)`:**
    *   **Input:** `state_file` (path to JSON).
    *   **Action:** Reads the JSON file. If it exists and is valid, returns the
        state dictionary. Otherwise, returns a default state `{'last_fold': 0, 'last_epoch': 0}`
        to start training from scratch.
*   **`save_pytorch_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path)`:**
    *   **Inputs:** Current `epoch`, `model` object, `optimizer` object,
        `val_loss`, and the `checkpoint_path` (e.g., `epoch_X.pt`).
    *   **Action:** Saves the `model.state_dict()`, `optimizer.state_dict()`,
        epoch number, and validation loss to a `.pt` file.
*   **`load_pytorch_checkpoint(model, optimizer, checkpoint_path, device)`:**
    *   **Inputs:** `model` object, `optimizer` object, `checkpoint_path`, `device`.
    *   **Action:** Loads the state dicts for the model and optimizer from the
        checkpoint file. Returns the epoch to start from (saved epoch + 1) and
        the last validation loss.

## 5. `train_model()` Function - The Core Training Loop

This is the heart of the script where the actual model training and validation occur.

*   **Inputs:** `X_all`, `y_all`, `folds_all`, `num_classes_global`, `model_dir`,
    `epochs`, `batch_size`, `learning_rate`.
*   **Key Steps & Flow:**

    1.  **Initialization:**
        *   Sets the `device` (cuda if available, else cpu).
        *   Creates directories for `CHECKPOINT_BASE_DIR` and `model_dir` if
            they don't exist.
        *   Loads the initial training state (`initial_state`) using
            `load_training_state_json`. This determines `start_fold_idx` and
            `initial_epoch_for_next_fold`.
            *   Handles logic to advance `start_fold_idx` if the previous fold
                completed all its epochs.
        *   Gets `unique_fold_numbers` from `folds_all`.
        *   Determines `n_mels` and `target_len` from `X_all.shape`.

    2.  **Outer Loop: Cross-Validation Folds** (`for fold_idx, fold_num_actual in enumerate(unique_fold_numbers)`)
        *   **Fold Skipping Logic:**
            *   `if fold_idx < start_fold_idx:`: Prints "Skipping completed Fold..."
                and `continue`s to the next fold. This uses the state loaded
                earlier.
        *   **Data Preparation for Current Fold:**
            *   Splits `X_all`, `y_all` into `X_train`, `X_test`, `y_train`, `y_test`
                based on `fold_num_actual` and `folds_all`.
            *   Creates `SoundDataset` instances for training and testing sets.
            *   Moves dataset features and labels to the target `device`.
            *   Creates `DataLoader` instances (`train_loader`, `test_loader`) for
                batching and shuffling.
        *   **Model, Optimizer, Criterion, Scheduler Setup:**
            *   Initializes the `AudioCNN` model with `n_mels`, `num_classes_global`,
                and `target_len` (for FC layer calculation), and moves it to `device`.
            *   If W&B is active, calls `wandb.watch(model)`.
            *   Initializes `optimizer` (e.g., `torch.optim.Adam`).
            *   Initializes `criterion` (loss function, e.g., `nn.CrossEntropyLoss`).
            *   Initializes `scheduler` (learning rate scheduler, e.g.,
                `ReduceLROnPlateau`).
        *   **Checkpoint and State Variables for Fold:**
            *   Defines `fold_checkpoint_dir`.
            *   Initializes variables for early stopping (`best_val_loss_for_early_stop`,
                `patience_counter`) and tracking the best model within the fold
                (`best_val_accuracy_this_fold`, `best_epoch_this_fold`).
            *   `actual_start_epoch`: Determined by `initial_epoch_for_this_fold`
                (from overall state) and potentially loading a specific epoch
                checkpoint (`.pt` file) if resuming a partially completed fold.

        *   **Inner Loop: Epochs** (`for epoch in range(actual_start_epoch, epochs)`)

            *   **a. Training Phase:**
                *   `model.train()`: Sets model to training mode.
                *   Iterates through `train_loader` (wrapped with `tqdm` for a
                    progress bar: `Fold X Epoch Y/Z Training`).
                    *   `optimizer.zero_grad()`
                    *   `outputs = model(inputs)`: Forward pass.
                    *   `loss = criterion(outputs, targets)`: Calculate loss.
                    *   `loss.backward()`: Backpropagation.
                    *   `optimizer.step()`: Update model weights.
                    *   Accumulates `running_loss` and `correct_train` predictions.
                *   Calculates `epoch_loss` and `epoch_acc` for training.

            *   **b. Validation Phase:**
                *   `model.eval()`: Sets model to evaluation mode.
                *   `with torch.no_grad()`: Disables gradient calculations.
                *   Iterates through `test_loader` (wrapped with `tqdm`:
                    `Fold X Epoch Y/Z Validation`).
                    *   `outputs = model(inputs)`: Forward pass.
                    *   `loss = criterion(outputs, targets)`: Calculate loss.
                    *   Accumulates `val_running_loss` and `correct_val` predictions.
                *   Calculates `val_loss` and `val_acc` for validation.

            *   **c. Logging and Saving:**
                *   Prints epoch summary (train/val loss & accuracy, time).
                *   If W&B is active, logs metrics using `wandb.log()`.
                *   `scheduler.step(val_loss)`: Updates learning rate if needed.
                *   Saves PyTorch checkpoint (`.pt`): `save_pytorch_checkpoint()`.
                *   Saves Safetensors checkpoint (`.safetensors`):
                    *   Locally: `save_safetensors(model.state_dict(), local_path)`.
                    *   HF Upload Attempt: `huggingface_hub.upload_file()` to
                        `config.HF_MODEL_REPO_ID` (path in repo: `fold_X/epoch_Y.safetensors`).
                        Prints success or error (e.g., "401 Unauthorized", "404 Not Found").
                *   Saves overall training state: `save_training_state_json(TRAINING_STATE_FILE, fold_idx, epoch + 1)`.

            *   **d. Best Model and Early Stopping:**
                *   If current `val_acc` > `best_val_accuracy_this_fold`, updates
                    best metrics and saves the current model as `best_model_this_fold.pt`.
                *   Early stopping logic: If `val_loss` doesn't improve for
                    `early_stopping_patience` epochs, breaks the epoch loop.

        *   **After Epoch Loop (for the current fold):**
            *   Prints fold summary (best validation accuracy and epoch).
            *   Loads the `best_model_this_fold.pt`.
            *   Saves its state dict as `model_fold_{fold_num_actual}_best.safetensors`
                in the main `model_dir`.
            *   Appends `best_val_accuracy_this_fold` to `fold_accuracies`.
            *   Appends `current_fold_history` to `all_fold_histories`.
            *   Resets `initial_epoch_for_next_fold = 0` for the *next* fold.

    3.  **After All Folds (Overall Results):**
        *   Calculates and prints mean and standard deviation of
            `fold_accuracies`.
        *   If W&B is active, logs these overall metrics.
        *   Saves `all_fold_histories` to `all_fold_histories_pytorch.pkl` in
            `model_dir`.
        *   Prints final training state.

```
train_model() Flow:
-------------------
Load Training State (fold_N, epoch_M) --> start_fold_idx, initial_epoch_for_next_fold

FOR each fold_idx IN folds:
  IF fold_idx < start_fold_idx:
    SKIP FOLD
    CONTINUE

  Prepare Data (X_train/test, y_train/test, DataLoaders)
  Initialize Model, Optimizer, Loss, Scheduler
  actual_start_epoch = initial_epoch_for_next_fold OR from .pt checkpoint

  FOR each epoch FROM actual_start_epoch TO total_epochs:
    TRAIN PHASE (on train_loader with tqdm):
      Forward -> Loss -> Backward -> Optimize
      Metrics (train_loss, train_acc)

    VALIDATION PHASE (on test_loader with tqdm):
      Forward -> Loss
      Metrics (val_loss, val_acc)

    LOG epoch metrics (console, W&B)
    Scheduler.step()
    SAVE .pt checkpoint
    SAVE .safetensors checkpoint --> Attempt HF Upload (to HF_MODEL_REPO_ID)
    SAVE training_state.json (current fold_idx, epoch+1)
    Early Stopping Check / Update Best Model for Fold

  SAVE best .safetensors for the completed fold (to model_dir)
  Store fold metrics

Summarize & Log Cross-Validation Results
Save all_fold_histories.pkl
```

## 6. Model Architecture (`AudioCNN` - from `model.py`)

While `AudioCNN` is defined in `model.py`, `train.py` instantiates and uses it.
A brief overview:

*   **Input:** Expects a 3D tensor `(batch_size, n_mels, time_frames)`.
    *   The `forward` method in `AudioCNN` first `unsqueeze(1)` to add a channel
        dimension, making it `(batch_size, 1, n_mels, time_frames)` for 2D convolutions.
*   **Architecture (High-Level):**
    1.  **Encoder Blocks:** A series of `EncoderBlock` modules. Each typically
        contains:
        *   `nn.Conv2d` (2D Convolution)
        *   `nn.BatchNorm2d`
        *   `nn.ReLU`
        *   `nn.MaxPool2d` (reduces spatial dimensions, e.g., frequency and time)
        *   `nn.Dropout`
    2.  **Time Reduction:** After encoder blocks, the time dimension is reduced,
        often by taking the mean across the time axis (`x.mean(dim=3)` if input
        is `B, C, F, T`). This results in `(batch, channels, freq_reduced)`.
    3.  **1D Convolution:** `nn.Conv1d` applied along the (reduced) frequency
        axis. This treats the frequency bins as a sequence.
    4.  **Classification Head:**
        *   `nn.Flatten()`
        *   `nn.Linear()` (Fully Connected layers)
        *   `nn.ReLU()`
        *   `nn.Dropout()`
        *   Final `nn.Linear()` to output `num_classes` logits.
*   **Output:** `(batch_size, num_classes)` - raw logits for each class. The
    `nn.CrossEntropyLoss` in `train.py` handles applying softmax internally.

```
AudioCNN (Simplified):

Input (B, N_Mels, Time)
   |
   V
Unsqueeze (add channel dim) -> (B, 1, N_Mels, Time)
   |
   V
+---------------------+
| EncoderBlock 1      | (Conv2D, BatchNorm, ReLU, MaxPool, Dropout)
| (e.g., 1 -> 32 ch)  | Pool (2,2)
+---------------------+
   |
   V (B, 32, N_Mels/2, Time/2)
+---------------------+
| EncoderBlock 2      |
| (e.g., 32 -> 64 ch) | Pool (2,2)
+---------------------+
   |
   V (B, 64, N_Mels/4, Time/4)
+---------------------+
| EncoderBlock 3      |
| (e.g., 64 -> 128 ch)| Pool (2,4) ; more pooling on time
+---------------------+
   |
   V (B, 128, N_Mels/8, Time/16)
   |
   V
Mean across Time Dim    -> (B, 128, N_Mels/8)  [Shape: Batch, Channels, Freq_reduced]
   |
   V
+---------------------+
| Conv1D (128 -> 64 ch)| Applied along Freq_reduced axis
| ReLU                |
+---------------------+
   |
   V (B, 64, N_Mels/8)
   |
   V
Flatten
   |
   V
+---------------------+
| Fully Connected     | (Linear, ReLU, Dropout)
| Layers              |
+---------------------+
   |
   V
Output (B, Num_Classes) [Logits]
``` 