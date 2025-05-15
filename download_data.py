import soundata
import os
# import sys # No longer needed

# Add src directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # No longer needed

# Import config directly
# from src import config # Old import
import config # Use direct import

# Use config value
DATA_DIR = config.DATA_DIR
# TEMP_DOWNLOAD_DIR = config.TEMP_DOWNLOAD_DIR # No longer needed

def download_urbansound8k(base_data_dir=DATA_DIR): # Renamed for clarity
    """Downloads the UrbanSound8K dataset using soundata into a specific UrbanSound8K subdirectory."""
    
    # Define the specific target directory for UrbanSound8K
    urbansound8k_target_dir = os.path.join(base_data_dir, "UrbanSound8K")
    
    print(f"Attempting to download UrbanSound8K to {os.path.abspath(urbansound8k_target_dir)}...")
    
    # Ensure the target UrbanSound8K directory exists
    if not os.path.exists(urbansound8k_target_dir):
        os.makedirs(urbansound8k_target_dir)
        print(f"Created data directory: {urbansound8k_target_dir}")
    
    # Use urbansound8k_target_dir as data_home for soundata
    dataset = soundata.initialize('urbansound8k', data_home=urbansound8k_target_dir)
    # The download happens automatically if the data is not found
    # We just need to trigger the validation or accessing a clip
    try:
        dataset.validate()
        print("Dataset found and validated.")
    except Exception as e:
        print(f"Validation failed, attempting download: {e}")
        # Attempting to access data might trigger download if validation check didn't
        try:
            # Force overwrite to ensure a fresh download, ignoring cached corrupted files
            paths = dataset.download(partial_download=False, cleanup=True, force_overwrite=False)
            if paths:
                 print("Download successful.")
                 dataset.validate() # Validate after download
                 print("Dataset validated after download.")
                 # --- Add this to check actual file paths ---
                 try:
                     clip_ids = dataset.get_track_ids() # Or get_clip_ids() depending on soundata version
                     if clip_ids:
                         first_clip_id = clip_ids[0]
                         clip_data = dataset.track(first_clip_id) # Or clip()
                         if clip_data and clip_data.audio_path:
                             print(f"DEBUG: Path of first audio file according to soundata: {clip_data.audio_path}")
                         else:
                             print("DEBUG: Could not retrieve audio path for the first clip.")
                     else:
                         print("DEBUG: No clip IDs found in the dataset.")
                 except Exception as debug_e:
                     print(f"DEBUG: Error trying to get clip path: {debug_e}")
                 # --- End of added debug section ---
            else:
                print("Download might have failed or was interrupted. Check logs.")
        except Exception as download_e:
            print(f"An error occurred during download: {download_e}")
            print("Please ensure you have internet connectivity and sufficient disk space.")
            print("You might need to manually download from https://urbansounddataset.weebly.com/urbansound8k.html")

if __name__ == "__main__":
    # Check/create the DATA_DIR (this is data_home now)
    # if not os.path.exists(DATA_DIR):
    #     os.makedirs(DATA_DIR) # This is now handled within download_urbansound8k
    download_urbansound8k()
    # Refer to the final data directory in the message
    # Construct the path to the UrbanSound8K directory for the final message
    final_urbansound8k_path = os.path.join(DATA_DIR, "UrbanSound8K")
    print(f"Script finished. Check the '{os.path.abspath(final_urbansound8k_path)}' directory for the UrbanSound8K dataset.")