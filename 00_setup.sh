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

# ANSI Color Codes
BOLD_RED='\033[1;31m'
NC='\033[0m' # No Color

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
INSTALL_EXIT_CODE=$?

# Optional: Clean up the installer script regardless of exit code (unless user wants to retry)
# Uncomment the line below if you want to force removal:
# rm "${INSTALLER_PATH}"

if [ ${INSTALL_EXIT_CODE} -ne 0 ]; then
  echo "Anaconda installation may have failed or was cancelled (Exit Code: ${INSTALL_EXIT_CODE})."
  exit 1
fi

echo "Anaconda installation script finished."
echo "---------------------------------------------------------------------"
# Print the prominent warning message in bold red
echo -e "${BOLD_RED}"
echo "*********************************************************************"
echo "*                                                                   *"
echo "*      ANACONDA INSTALLATION COMPLETE - ACTION REQUIRED!            *"
echo "*                                                                   *"
echo "* You MUST close this terminal/SSH session and open a NEW one.      *"
echo "* This is required for shell configuration changes to take effect.    *"
echo "* The script cannot safely force this closure for you.              *"
echo "*                                                                   *"
echo "* After opening a new terminal, run the 01_setup.sh script.         *"
echo "*                                                                   *"
echo "*********************************************************************"
# Reset color
echo -e "${NC}"
echo "---------------------------------------------------------------------"

exit 0
