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

def download_urbansound8k(data_home=DATA_DIR): # Use DATA_DIR as default data_home
    """Downloads the UrbanSound8K dataset using soundata."""
    print(f"Attempting to download UrbanSound8K to {os.path.abspath(data_home)}...")
    # Ensure the data_home directory exists (soundata might need this)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
        print(f"Created data directory: {data_home}")
    
    dataset = soundata.initialize('urbansound8k', data_home=data_home)
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
            paths = dataset.download(partial_download=False, cleanup=True, force_overwrite=True)
            if paths:
                 print("Download successful.")
                 dataset.validate() # Validate after download
                 print("Dataset validated after download.")
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
    print(f"Script finished. Check the '{DATA_DIR}' directory for the UrbanSound8K dataset.")