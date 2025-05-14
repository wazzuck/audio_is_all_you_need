import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import argparse
import json
import huggingface_hub
import wandb
from safetensors.torch import save_file as save_safetensors
import time # For timing epochs
from tqdm import tqdm # Added tqdm import

from utils import load_pickle, save_pickle
from model import AudioCNN # Import the PyTorch model
from data_loader import NUM_CLASSES # Assuming this is still relevant and correct
import config

# Checkpoint and state file paths from config
CHECKPOINT_BASE_DIR = config.CHECKPOINT_BASE_DIR
TRAINING_STATE_FILE = config.TRAINING_STATE_FILE

# --- PyTorch Dataset ---
class SoundDataset(Dataset):
    def __init__(self, features, labels, device):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss expects long
        self.device = device # Store device to potentially move data in __getitem__ if not pre-loaded

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Data is already on device if pre-loaded, or can be moved here
        # For simplicity, assuming data is pre-loaded or small enough
        # If features are large, consider loading/moving them here
        return self.features[idx], self.labels[idx]

# --- Helper functions for training state and checkpoints ---
def save_training_state_json(state_file, fold_num, epoch_num):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    state = {'last_fold': fold_num, 'last_epoch': epoch_num}
    with open(state_file, 'w') as f:
        json.dump(state, f)
    print(f"Saved training state: Fold {fold_num}, Epoch {epoch_num}")

def load_training_state_json(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            try:
                state = json.load(f)
                if 'last_fold' in state and 'last_epoch' in state:
                    print(f"Found previous training state: {state}")
                    return state
            except json.JSONDecodeError:
                print(f"Error reading training state file: {state_file}. Starting fresh.")
    return {'last_fold': 0, 'last_epoch': 0}

def save_pytorch_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss, # Or val_accuracy, depending on what's monitored
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path} (Epoch {epoch})")

def load_pytorch_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, float('inf') # Start from epoch 0, with infinite loss

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1 # Resume from next epoch
    val_loss = checkpoint.get('val_loss', float('inf')) # or val_accuracy
    print(f"Resuming from Epoch {start_epoch}, Last Val Loss: {val_loss:.4f}")
    return start_epoch, val_loss

def train_model(X_all, y_all, folds_all, num_classes_global, model_dir, epochs=50, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    fold_accuracies = []
    all_fold_histories = [] # List to store history dicts for each fold

    initial_state = load_training_state_json(TRAINING_STATE_FILE)
    start_fold_idx = initial_state['last_fold'] # Folds are 0-indexed here

    # If the last recorded epoch for the start_fold_idx was the final one, advance to the next fold
    if start_fold_idx > 0 and initial_state['last_epoch'] >= epochs:
        print(f"Fold {start_fold_idx} completed all {epochs} epochs. Starting next fold.")
        start_fold_idx += 1
        initial_epoch_for_next_fold = 0
    elif start_fold_idx > 0 :
        initial_epoch_for_next_fold = initial_state['last_epoch'] # Resume from this epoch +1 internally
        print(f"Resuming Fold {start_fold_idx} from epoch {initial_epoch_for_next_fold}")
    else: # start_fold_idx is 0
        initial_epoch_for_next_fold = 0

    unique_fold_numbers = np.sort(np.unique(folds_all)) # e.g., [0, 1, ..., 9] if that's how folds are numbered
    
    # Determine n_mels and target_len from the input data
    # X_all shape expected: (num_samples, n_mels, target_len)
    if X_all.ndim != 3:
        raise ValueError(f"Expected X_all to have 3 dimensions (num_samples, n_mels, target_len), but got {X_all.ndim}")
    n_mels = X_all.shape[1]
    target_len = X_all.shape[2]
    print(f"Input data: n_mels={n_mels}, target_len={target_len}")


    for fold_idx, fold_num_actual in enumerate(unique_fold_numbers): # fold_idx is 0 to N-1
        current_fold_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        if fold_idx < start_fold_idx:
            print(f"Skipping completed Fold {fold_num_actual} (index {fold_idx})")
            # To keep all_fold_histories consistent, we might need to load and append past history
            # For now, let's assume we only care about new/resumed folds.
            # If all_fold_histories.pkl contains full history, this needs adjustment or just fill with None.
            all_fold_histories.append(None) # Placeholder
            continue

        print(f"\n--- Training Fold {fold_num_actual} (Index {fold_idx + 1}/{len(unique_fold_numbers)}) ---")
        
        initial_epoch_for_this_fold = initial_epoch_for_next_fold if fold_idx == start_fold_idx else 0

        train_indices = np.where(folds_all != fold_num_actual)[0]
        test_indices = np.where(folds_all == fold_num_actual)[0]

        X_train, X_test = X_all[train_indices], X_all[test_indices]
        y_train, y_test = y_all[train_indices], y_all[test_indices]

        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        train_dataset = SoundDataset(X_train, y_train, device)
        test_dataset = SoundDataset(X_test, y_test, device)
        
        # Move data to device before creating DataLoader if memory allows
        # Or do it in Dataset's __getitem__
        train_dataset.features = train_dataset.features.to(device)
        train_dataset.labels = train_dataset.labels.to(device)
        test_dataset.features = test_dataset.features.to(device)
        test_dataset.labels = test_dataset.labels.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model = AudioCNN(n_mels=n_mels, num_classes=num_classes_global, target_len_estimate_for_fc_input_calc=target_len).to(device)
        if wandb.run:
            wandb.watch(model, log="all", log_freq=100) # Log gradients and parameters

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)

        fold_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, f'fold_{fold_num_actual}')
        
        # Variables for early stopping and best model in fold
        best_val_loss_for_early_stop = float('inf')
        patience_counter = 0
        early_stopping_patience = 10 # From original Keras setup
        
        best_val_accuracy_this_fold = 0.0
        best_epoch_this_fold = 0
        
        # Load checkpoint if resuming this fold
        start_epoch_from_checkpoint = 0
        if initial_epoch_for_this_fold > 0:
            # Checkpoint path for the *previous* successfully completed epoch
            resume_checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{initial_epoch_for_this_fold -1}.pt') 
            if os.path.exists(resume_checkpoint_path):
                # load_pytorch_checkpoint returns the epoch to *start* from (saved_epoch + 1)
                start_epoch_from_checkpoint, _ = load_pytorch_checkpoint(model, optimizer, resume_checkpoint_path, device)
            else:
                print(f"Warning: Checkpoint {resume_checkpoint_path} not found for resuming. Starting fold from scratch.")
                start_epoch_from_checkpoint = 0 # Effectively initial_epoch_for_this_fold becomes 0
        
        actual_start_epoch = max(initial_epoch_for_this_fold, start_epoch_from_checkpoint)
        if actual_start_epoch > 0:
             print(f"Adjusted start epoch for this fold to: {actual_start_epoch}")


        print(f"Starting training for Fold {fold_num_actual} from epoch {actual_start_epoch}...")
        for epoch in range(actual_start_epoch, epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Wrap train_loader with tqdm for a progress bar
            train_pbar = tqdm(train_loader, desc=f"Fold {fold_num_actual} Epoch {epoch + 1}/{epochs} Training", leave=False)
            for batch_idx, (inputs, targets) in enumerate(train_pbar):
                # inputs, targets = inputs.to(device), targets.to(device) # Already on device
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct_train / total_train
            current_fold_history['loss'].append(epoch_loss)
            current_fold_history['accuracy'].append(epoch_acc)

            # Validation
            model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                # Optionally, wrap test_loader with tqdm as well for validation progress
                val_pbar = tqdm(test_loader, desc=f"Fold {fold_num_actual} Epoch {epoch + 1}/{epochs} Validation", leave=False)
                for inputs, targets in val_pbar:
                    # inputs, targets = inputs.to(device), targets.to(device) # Already on device
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            val_loss = val_running_loss / len(test_loader.dataset)
            val_acc = correct_val / total_val
            current_fold_history['val_loss'].append(val_loss)
            current_fold_history['val_accuracy'].append(val_acc)
            
            epoch_duration = time.time() - epoch_start_time
            print(f"Fold {fold_num_actual} | Epoch {epoch + 1}/{epochs} | Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if wandb.run:
                wandb.log({
                    f"fold_{fold_num_actual}/train_loss": epoch_loss,
                    f"fold_{fold_num_actual}/train_accuracy": epoch_acc,
                    f"fold_{fold_num_actual}/val_loss": val_loss,
                    f"fold_{fold_num_actual}/val_accuracy": val_acc,
                    "epoch": epoch +1 # Log current epoch number (1-indexed)
                }, step=epoch + (fold_idx * epochs)) # Global step

            scheduler.step(val_loss) # ReduceLROnPlateau

            # Save PyTorch checkpoint (.pt file)
            checkpoint_path = os.path.join(fold_checkpoint_dir, f'epoch_{epoch}.pt')
            save_pytorch_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path)
            
            # Save .safetensors checkpoint for this epoch and upload to Hugging Face Hub
            safetensors_epoch_filename = f'epoch_{epoch}.safetensors'
            safetensors_epoch_local_path = os.path.join(fold_checkpoint_dir, safetensors_epoch_filename)
            try:
                save_safetensors(model.state_dict(), safetensors_epoch_local_path)
                print(f"Saved epoch {epoch} safetensors to {safetensors_epoch_local_path}")
                
                hf_path_in_repo = f"fold_{fold_num_actual}/{safetensors_epoch_filename}"
                hf_upload_url = huggingface_hub.upload_file(
                    path_or_fileobj=safetensors_epoch_local_path,
                    path_in_repo=hf_path_in_repo,
                    repo_id=config.HF_REPO_ID, # Using the existing repo ID
                    repo_type="model",
                    commit_message=f"Upload model checkpoint: Fold {fold_num_actual}, Epoch {epoch}"
                )
                print(f"Uploaded epoch {epoch} safetensors for fold {fold_num_actual} to: {hf_upload_url}")
            except Exception as e:
                print(f"Error saving or uploading epoch {epoch} safetensors for fold {fold_num_actual}: {e}")

            # Save training state (fold and current completed epoch)
            save_training_state_json(TRAINING_STATE_FILE, fold_idx, epoch + 1)


            # Update best model for this fold
            if val_acc > best_val_accuracy_this_fold:
                best_val_accuracy_this_fold = val_acc
                best_epoch_this_fold = epoch
                # Save best model state for this fold immediately (or path to best checkpoint)
                best_model_fold_path = os.path.join(fold_checkpoint_dir, 'best_model_this_fold.pt')
                save_pytorch_checkpoint(epoch, model, optimizer, val_loss, best_model_fold_path)
                print(f"New best validation accuracy for fold {fold_num_actual}: {best_val_accuracy_this_fold:.4f} at epoch {epoch + 1}")

            # Early stopping
            if val_loss < best_val_loss_for_early_stop:
                best_val_loss_for_early_stop = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold_num_actual}.")
                break # Break from epoch loop
        
        # After fold training (or early stopping)
        print(f"Fold {fold_num_actual} finished. Best validation accuracy: {best_val_accuracy_this_fold:.4f} at epoch {best_epoch_this_fold + 1}")
        
        # Load the best model for this fold to save as .safetensors
        best_model_final_path_pt = os.path.join(fold_checkpoint_dir, 'best_model_this_fold.pt')
        if os.path.exists(best_model_final_path_pt):
            print(f"Loading best model for fold {fold_num_actual} from {best_model_final_path_pt}")
            # Create a new model instance to load the state into
            best_model_for_saving = AudioCNN(n_mels=n_mels, num_classes=num_classes_global, target_len_estimate_for_fc_input_calc=target_len).to(device)
            load_pytorch_checkpoint(best_model_for_saving, None, best_model_final_path_pt, device)
            
            safetensors_path = os.path.join(model_dir, f'model_fold_{fold_num_actual}_best.safetensors')
            print(f"Saving best weights for fold {fold_num_actual} to: {safetensors_path}")
            try:
                save_safetensors(best_model_for_saving.state_dict(), safetensors_path)
            except Exception as e:
                print(f"Error saving .safetensors: {e}")
        else:
            print(f"Could not find best model checkpoint at {best_model_final_path_pt} for fold {fold_num_actual}. Cannot save .safetensors.")

        fold_accuracies.append(best_val_accuracy_this_fold) # Store the best val_acc for this fold
        all_fold_histories.append(current_fold_history)
        
        # Reset for next fold if resuming a multi-fold run
        initial_epoch_for_next_fold = 0


    # --- Overall Results ---
    if fold_accuracies: # Check if any folds were run
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print("\n--- Cross-Validation Summary ---")
        print(f"Individual Fold Best Validation Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        print(f"Average Best Validation Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

        if wandb.run:
            wandb.log({
                "overall_mean_val_accuracy": mean_accuracy,
                "overall_std_val_accuracy": std_accuracy,
                "individual_fold_val_accuracies": fold_accuracies
            })
    else:
        print("No folds were trained or completed.")

    save_pickle(all_fold_histories, os.path.join(model_dir, 'all_fold_histories_pytorch.pkl'))
    print("\n--- Training Process Finished ---")
    final_state_json = load_training_state_json(TRAINING_STATE_FILE)
    print(f"Final training state recorded: {final_state_json}")

    # Ensure Hugging Face model repository exists
    try:
        huggingface_hub.create_repo(
            repo_id=config.HF_REPO_ID,
            repo_type="model",
            exist_ok=True
        )
        print(f"Ensured Hugging Face model repository {config.HF_REPO_ID} exists or was created.")
    except Exception as e:
        print(f"Warning: Could not create/verify Hugging Face model repository {config.HF_REPO_ID}: {e}. Model uploads might fail.")


def main(args):
    print("Checking for processed data...")

    # Initialize Weights & Biases
    if not wandb.run: # Check if a run is already active (e.g. from a sweep)
        try:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate, # Added LR to config
                    "processed_dir": args.processed_dir,
                    "model_dir": args.model_dir,
                    "num_classes": NUM_CLASSES,
                }
            )
            print(f"Weights & Biases initialized for project: {config.WANDB_PROJECT}, entity: {config.WANDB_ENTITY}")
        except Exception as e:
            print(f"Could not initialize Weights & Biases: {e}. Training will continue without W&B logging.")
    else:
        print(f"Using active W&B run: {wandb.run.name}")
        # Update W&B config if necessary from args, though usually set by sweep agent
        wandb.config.update({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }, allow_val_change=True)


    # Define paths for essential processed files
    features_path = os.path.join(args.processed_dir, 'features.pkl')
    labels_path = os.path.join(args.processed_dir, 'labels.pkl')
    folds_path = os.path.join(args.processed_dir, 'folds.pkl')
    # class_mapping_path = os.path.join(args.processed_dir, 'class_mapping.pkl') # Also downloaded if present

    # Check if all essential files exist
    all_files_exist = (
        os.path.exists(features_path) and
        os.path.exists(labels_path) and
        os.path.exists(folds_path)
    )

    if not all_files_exist:
        print(f"One or more processed data files (features.pkl, labels.pkl, folds.pkl) not found in '{args.processed_dir}'.")
        print(f"Attempting to download from Hugging Face Hub: {config.HF_REPO_ID} to {args.processed_dir}")
        os.makedirs(args.processed_dir, exist_ok=True) # Ensure target directory exists
        try:
            huggingface_hub.snapshot_download(
                repo_id=config.HF_REPO_ID,
                repo_type="dataset",
                local_dir=args.processed_dir, # Files will be placed directly in here
                local_dir_use_symlinks=False,
                allow_patterns=["features.pkl", "labels.pkl", "folds.pkl", "class_mapping.pkl"] # Specify exact files
            )
            print(f"Download attempt finished for {args.processed_dir}.")
            # Re-check if files exist after download
            all_files_exist = (
                os.path.exists(features_path) and
                os.path.exists(labels_path) and
                os.path.exists(folds_path)
            )
            if not all_files_exist:
                print(f"Error: Required data files still missing after download attempt from {config.HF_REPO_ID}.")
                print(f"Please ensure features.pkl, labels.pkl, and folds.pkl are present in the Hugging Face dataset at the root level.")
                print("Alternatively, run 'python preprocess_data.py' to generate the data locally.")
                if wandb.run: wandb.finish()
                return
            else:
                print(f"Successfully found/downloaded required files in {args.processed_dir}.")
        except Exception as e:
            print(f"Error during data download from Hugging Face Hub: {e}")
            print(f"Please ensure the repository '{config.HF_REPO_ID}' exists, is accessible, and contains the required files at its root.")
            print("Alternatively, run 'python preprocess_data.py' to generate the data locally.")
            if wandb.run: wandb.finish()
            return
    else:
        print(f"Found all required processed data files locally in {args.processed_dir}")

    print("\nLoading preprocessed data...")
    try:
        X = load_pickle(features_path)
        y = load_pickle(labels_path)
        folds = load_pickle(folds_path)
    except FileNotFoundError: # This is a safeguard, should ideally be caught by earlier checks
        print(f"Critical Error: Processed data files were expected but not found in {args.processed_dir}.")
        print("This should have been caught by earlier checks. Please investigate.")
        if wandb.run: wandb.finish()
        return
    except Exception as e: # Catch other potential loading errors
        print(f"Error loading processed data files: {e}")
        if wandb.run: wandb.finish()
        return

    if X is None or y is None or folds is None:
         print("Error: One or more data components (X, y, folds) are None after loading. Exiting.")
         if wandb.run: wandb.finish()
         return

    print(f"Shape of X after loading: {X.shape}")
    if X.ndim == 4 and X.shape[1] == 1:
        print(f"Reshaping X from {X.shape} to 3D by squeezing dimension 1.")
        X = X.squeeze(1)
    elif X.ndim == 4:
        if X.shape[3] == 1:
            print(f"Reshaping X from {X.shape} to 3D by squeezing dimension 3 (the last dimension).")
            X = X.squeeze(3)
        else:
            print(f"X is 4D with shape {X.shape}, but the channel dimension (assumed to be 1 or 3) is not of size 1. Cannot safely squeeze.")


    print("Starting model training...")
    train_model(X, y, folds,
                num_classes_global=NUM_CLASSES,
                model_dir=args.model_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate) # Pass LR
    print("Training complete. Models saved as .safetensors files.")

    if wandb.run:
        wandb.finish()
        print("Weights & Biases run finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the UrbanSound8K CNN model using PyTorch with 10-fold cross-validation.')
    parser.add_argument('--processed_dir', type=str, default=config.PROCESSED_DIR,
                        help=f'Directory containing the processed features (default: {config.PROCESSED_DIR})')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_DIR,
                        help=f'Directory to save the trained model weights (default: {config.MODEL_DIR})')
    parser.add_argument('--epochs', type=int, default=config.DEFAULT_EPOCHS,
                        help=f'Number of training epochs per fold (default: {config.DEFAULT_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE,
                        help=f'Training batch size (default: {config.DEFAULT_BATCH_SIZE})')
    parser.add_argument('--learning_rate', type=float, default=0.001, # Added LR argument
                        help='Initial learning rate (default: 0.001)')

    args = parser.parse_args()
    main(args) 