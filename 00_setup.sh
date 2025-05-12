#!/bin/bash

echo "This script will download and start the Anaconda installer for Linux x86_64."
echo "Please follow the prompts from the installer."
echo "It is recommended to install Anaconda for the current user only."

# Define Anaconda version and installer filename
ANACONDA_VERSION="2024.02-1" # Check for the latest version if needed
INSTALLER_FILENAME="Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"
DOWNLOAD_URL="https://repo.anaconda.com/archive/${INSTALLER_FILENAME}"

# Define download location
DOWNLOAD_DIR="/tmp"
INSTALLER_PATH="${DOWNLOAD_DIR}/${INSTALLER_FILENAME}"

echo "Downloading Anaconda installer..."
wget -O "${INSTALLER_PATH}" "${DOWNLOAD_URL}"

if [ $? -ne 0 ]; then
  echo "Error downloading Anaconda installer. Please check the URL or your internet connection."
  exit 1
fi

echo "Download complete. Starting the installer..."

# Make the installer executable
chmod +x "${INSTALLER_PATH}"

# Run the installer
"${INSTALLER_PATH}"

# Check if installation was successful (basic check by installer exit code)
if [ $? -ne 0 ]; then
  echo "Anaconda installation may have failed or was cancelled."
  # Clean up installer
  # rm "${INSTALLER_PATH}"
  exit 1
fi

echo "Anaconda installation script finished."
echo "-------------------------------------------------------------"
echo "IMPORTANT: Please CLOSE this terminal and open a NEW one."
echo "This is necessary for the shell configuration changes (like adding conda to PATH) to take effect."
echo "After opening a new terminal, run the 01_setup.sh script."
echo "-------------------------------------------------------------"

# Optional: Clean up the installer script after successful execution?
# Usually the installer asks if you want to remove it.
# Uncomment the line below if you want to force removal:
# rm "${INSTALLER_PATH}"

exit 0
