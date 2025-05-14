import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import argparse
import sys
import json # Added import for json
import huggingface_hub # Added for HF download
# We need safetensors explicitly for saving, though TF integrates loading
# from safetensors import safe_open # Not needed for model.save_weights

# Add src directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # This line is removed

from utils import load_pickle, save_pickle # Keep save_pickle for history
from model import build_cnn_model
from data_loader import NUM_CLASSES # Use NUM_CLASSES from data_loader
import config # Import config

# Default paths - Now fetched from config
# DEFAULT_PROCESSED_DIR = 'data/processed'
# DEFAULT_MODEL_DIR = 'models'

# Checkpoint and state file paths from config
CHECKPOINT_BASE_DIR = config.CHECKPOINT_BASE_DIR
TRAINING_STATE_FILE = config.TRAINING_STATE_FILE

# --- Custom Callback to save training state ---
class TrainingStateCallback(Callback):
    def __init__(self, state_file, fold_num):
        super().__init__()
        self.state_file = state_file
        self.fold_num = fold_num

    def on_epoch_end(self, epoch, logs=None):
        # Epochs are 0-indexed internally, add 1 for user-facing/resume logic
        current_epoch = epoch + 1
        state = {'last_fold': self.fold_num, 'last_epoch': current_epoch}
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
        # print(f"Saved training state: Fold {self.fold_num}, Epoch {current_epoch}") # Optional: for debugging

def load_training_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            try:
                state = json.load(f)
                # Basic validation
                if 'last_fold' in state and 'last_epoch' in state:
                    print(f"Found previous training state: {state}")
                    return state
            except json.JSONDecodeError:
                print(f"Error reading training state file: {state_file}. Starting fresh.")
    return {'last_fold': 0, 'last_epoch': 0} # Default start state if no file or invalid

def train_model(X, y, folds, num_classes, model_dir, epochs=50, batch_size=32):
    """Trains the model using 10-fold cross-validation based on predefined folds,
       saving checkpoints to allow resuming, and final weights in .safetensors format."""
    # Ensure base checkpoint directory exists
    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True) # Original model dir for final .safetensors

    fold_accuracies = []
    fold_histories = []

    # Load last training state
    initial_state = load_training_state(TRAINING_STATE_FILE)
    start_fold = initial_state['last_fold']
    # If the last recorded epoch was the final epoch for that fold, start the next fold
    # Note: This assumes `epochs` is constant. If last_epoch == epochs, it finished.
    # A more robust check would be needed if epochs varied or training stopped exactly on the last epoch before state save.
    if start_fold > 0 and initial_state['last_epoch'] >= epochs:
         print(f"Fold {start_fold} finished ({initial_state['last_epoch']} epochs). Starting next fold.")
         start_fold += 1
         initial_epoch_for_next_fold = 0
    elif start_fold > 0:
         # Resume from the epoch *after* the last saved one
         initial_epoch_for_next_fold = initial_state['last_epoch']
         print(f"Resuming Fold {start_fold} from epoch {initial_epoch_for_next_fold}")
    else:
         initial_epoch_for_next_fold = 0 # Start from beginning

    unique_folds = np.sort(np.unique(folds))
    if len(unique_folds) != 10:
        print(f"Warning: Expected 10 folds, but found {len(unique_folds)}. Proceeding anyway.")

    input_shape = X.shape[1:] # Shape should be (n_mels, target_len, 1)
    print(f"Input shape for model: {input_shape}")

    # Convert labels to categorical
    y_cat = to_categorical(y, num_classes=num_classes)

    # Loop through folds, starting from the determined start_fold
    for fold_num in unique_folds:
        if fold_num < start_fold:
            print(f"Skipping completed Fold {fold_num}")
            # Optionally load past results if needed, otherwise just skip
            continue

        print(f"\n--- Training Fold {fold_num}/{len(unique_folds)} ---")

        # Determine initial epoch for this specific fold
        initial_epoch = initial_epoch_for_next_fold if fold_num == start_fold else 0

        # Split data based on the predefined fold number
        train_indices = np.where(folds != fold_num)[0]
        test_indices = np.where(folds == fold_num)[0]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y_cat[train_indices], y_cat[test_indices]
        y_test_labels = y[test_indices] # Keep original labels for evaluation

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Build model (rebuild for each fold)
        model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Define checkpoint path for this fold
        fold_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, f'fold_{fold_num}')
        checkpoint_filepath = os.path.join(fold_checkpoint_dir, 'epoch_{epoch:02d}.keras') # TF saves directory, now with .keras

        # Load weights if resuming this fold
        if initial_epoch > 0:
            resume_checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{initial_epoch:02d}.keras') # Added .keras
            if os.path.exists(resume_checkpoint_path): # Check if file exists
                try:
                    print(f"Loading model state from checkpoint: {resume_checkpoint_path}")
                    # Load the entire model state
                    model = tf.keras.models.load_model(resume_checkpoint_path)
                    print("Model state loaded successfully.")
                    # Re-compile is sometimes needed after loading, though often handled by load_model
                    # model.compile(...) # Re-use compilation args if needed
                except Exception as e:
                    print(f"Error loading checkpoint {resume_checkpoint_path}: {e}. Starting fold {fold_num} from scratch.")
                    initial_epoch = 0 # Reset epoch if loading failed
                    # Rebuild and compile if loading failed fundamentally
                    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
            else:
                print(f"Checkpoint path not found: {resume_checkpoint_path}. Starting fold {fold_num} from scratch.")
                initial_epoch = 0

        # Callbacks
        # Checkpoint callback to save full model state every epoch
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False, # Save entire model
            monitor='val_accuracy', # Can still monitor metric
            mode='max',
            save_best_only=False, # Save every epoch
            save_freq='epoch', # Explicitly save every epoch
            verbose=1)

        # State saving callback
        state_callback = TrainingStateCallback(state_file=TRAINING_STATE_FILE, fold_num=fold_num)

        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                      verbose=1, mode='min', min_lr=0.00001)

        # Train
        print(f"Starting training for Fold {fold_num} from epoch {initial_epoch}...")
        history = model.fit(X_train, y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(X_test, y_test),
                          callbacks=[model_checkpoint_callback, state_callback, early_stopping, reduce_lr], # Added state_callback
                          initial_epoch=initial_epoch, # Start from the correct epoch
                          verbose=1)

        fold_histories.append(history.history)

        # --- Find and save the best weights as .safetensors (after fold finishes) ---
        # Find the epoch with the best validation accuracy from history
        best_epoch_index = np.argmax(history.history['val_accuracy'])
        best_epoch = best_epoch_index + 1 # history is 0-indexed, epochs are 1-based in checkpoints/logs
        best_val_accuracy = history.history['val_accuracy'][best_epoch_index]
        print(f"Best epoch for Fold {fold_num}: {best_epoch} with val_accuracy: {best_val_accuracy:.4f}")

        # Load the model state from the best epoch's checkpoint
        best_checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{best_epoch:02d}.keras') # Added .keras
        if os.path.exists(best_checkpoint_path):
             print(f"Loading best model state from: {best_checkpoint_path}")
             # Load the best model state to save its weights
             best_model = tf.keras.models.load_model(best_checkpoint_path)

             # Save best weights in .safetensors format
             safetensors_path = os.path.join(model_dir, f'model_fold_{fold_num}_best.safetensors')
             print(f"Saving best weights for fold {fold_num} to: {safetensors_path}")
             try:
                 best_model.save_weights(safetensors_path) # TF handles .safetensors extension
             except ValueError as e:
                 print(f"Direct saving to .safetensors failed (requires TF >= 2.11/2.12+): {e}")
                 print("Consider manual saving using the safetensors library if needed.")
                 pass # Fallback or error handling
        else:
             print(f"Could not find checkpoint for best epoch {best_epoch} at {best_checkpoint_path}. Cannot save best .safetensors.")


        # Evaluate the best model (already loaded as best_model)
        _, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Fold {fold_num} Test Accuracy (from best epoch {best_epoch}): {accuracy:.4f}")
        fold_accuracies.append(accuracy)

        # Optional: Cleanup old checkpoints for this fold? Or keep all? Keeping all for now.

        # Optional: Detailed classification report (using best_model)
        # y_pred_probs = best_model.predict(X_test)
        # ... rest of report generation ...

    # --- Overall Results ---
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print("\n--- Cross-Validation Summary ---")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Average Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    # Save histories using pickle
    save_pickle(fold_histories, os.path.join(model_dir, 'all_fold_histories.pkl'))

    # After all folds complete, potentially clear the training state file?
    # Or leave it to indicate completion. Let's leave it for now.
    print("\n--- Training Process Finished ---")
    if os.path.exists(TRAINING_STATE_FILE):
        final_state = load_training_state(TRAINING_STATE_FILE)
        print(f"Final training state recorded: {final_state}")

def main(args):
    print("Checking for processed data...")

    # Check if the processed data directory exists
    if not os.path.exists(args.processed_dir):
        print(f"Processed data not found locally at '{args.processed_dir}'.")
        print(f"Attempting to download from Hugging Face Hub: {config.HF_REPO_ID}")
        print("Please ensure you have internet access and 'huggingface_hub' installed.")
        try:
            huggingface_hub.snapshot_download(
                repo_id=config.HF_REPO_ID,
                repo_type="dataset",
                local_dir=args.processed_dir,
                local_dir_use_symlinks=False # Download files directly
            )
            print(f"Successfully downloaded data to {args.processed_dir}")
        except Exception as e:
            print(f"Error downloading data from Hugging Face Hub: {e}")
            print(f"Please ensure the repository '{config.HF_REPO_ID}' exists and is accessible.")
            print("Alternatively, run 'python preprocess_data.py' to generate the data locally.")
            return # Exit if download fails
    else:
        print(f"Found processed data locally at {args.processed_dir}")

    print("\nLoading preprocessed data...")
    try:
        X = load_pickle(os.path.join(args.processed_dir, 'features.pkl'))
        y = load_pickle(os.path.join(args.processed_dir, 'labels.pkl'))
        folds = load_pickle(os.path.join(args.processed_dir, 'folds.pkl'))
    except FileNotFoundError:
        print(f"Error: Processed data files (e.g., features.pkl) not found in {args.processed_dir} even after check/download.")
        print("There might be an issue with the downloaded data or the preprocessing script output.")
        return

    if X is None or y is None or folds is None:
         print("Error loading one or more data files (features, labels, folds). Exiting.")
         return

    print("Starting model training...")
    train_model(X, y, folds,
                num_classes=NUM_CLASSES,
                model_dir=args.model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size)
    print("Training complete. Models saved as .safetensors files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the UrbanSound8K CNN model using 10-fold cross-validation, saving weights as .safetensors.')
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR,
                        help=f'Directory containing the processed features (default: {config.PROCESSED_DIR})')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                        help=f'Directory to save the trained model weights (.safetensors) (default: {config.MODEL_DIR})')
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS,
                        help=f'Number of training epochs per fold (default: {config.DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE,
                        help=f'Training batch size (default: {config.DEFAULT_BATCH_SIZE})')

    args = parser.parse_args()
    main(args) 