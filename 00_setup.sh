#!/bin/bash

# Set Git config (optional, keep if desired)
git config --global user.email "user@example.com"
git config --global user.name "User Name"

echo "--- Setting up Miniconda ---"

# Determine architecture based on hostname
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == "rpi" || "$HOSTNAME" == "penguin" ]]; then
  echo "Hostname is 'rpi' or 'penguin'. Selecting Miniconda for aarch64."
  MINICONDA_SCRIPT="Miniconda3-latest-Linux-aarch64.sh"
else
  echo "Hostname is '$HOSTNAME'. Selecting Miniconda for x86_64."
  MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
fi

MINICONDA_URL="https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}"
INSTALLER_PATH="$HOME/miniconda3/miniconda.sh" # Consistent installer path

# Create directory and download installer
mkdir -p "$HOME/miniconda3"
echo "Downloading $MINICONDA_SCRIPT..."
wget "$MINICONDA_URL" -O "$INSTALLER_PATH"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error downloading Miniconda from $MINICONDA_URL. Exiting."
    exit 1
fi

# Install Miniconda
echo "Installing Miniconda..."
bash "$INSTALLER_PATH" -b -u -p "$HOME/miniconda3"
if [ $? -ne 0 ]; then
    echo "Error installing Miniconda. Exiting."
    # Clean up installer if installation failed but download succeeded
    rm "$INSTALLER_PATH"
    exit 1
fi

# Clean up installer
rm "$INSTALLER_PATH"

# Initialize Conda
echo "Initializing Conda..."
source "$HOME/miniconda3/bin/activate"
# The following line might require shell restart to take full effect
conda init --all

echo "--- Conda Setup complete! ---"
echo "Please restart your shell or run 'source ~/.bashrc' (or ~/.zshrc etc.) for Conda initialization to take effect."

echo "Miniconda installation script finished."
echo "---------------------------------------------------------------------"
# Print the prominent warning message in bold red
echo -e "${BOLD_RED}"
echo "*********************************************************************"
echo "*                                                                   *"
echo "*      MINICONDA INSTALLATION COMPLETE - ACTION REQUIRED!            *"
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
