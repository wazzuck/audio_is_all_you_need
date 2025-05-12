#!/bin/bash
# Source the user's bash configuration to make conda available
# This assumes conda init has been run for the bash shell.
if [ -f ~/.bashrc ]; then
    echo "Sourcing ~/.bashrc to find conda..."
    source ~/.bashrc
else
    echo "Warning: ~/.bashrc not found. Conda might not be available in the script's PATH."
fi

echo "--- Setting up Conda Environment and Installing Dependencies ---"

ENV_NAME="tf_env"
PYTHON_VERSION="3.11"

# Now, try to find conda root using conda info (should be available if .bashrc worked)
_CONDA_ROOT=$(conda info --base 2>/dev/null) # Try to get conda root silently

# If conda info worked, source the specific conda.sh - more robust
if [ -n "$_CONDA_ROOT" ] && [ -f "$_CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "Found Conda root: $_CONDA_ROOT. Sourcing profile.d/conda.sh..."
    source "$_CONDA_ROOT/etc/profile.d/conda.sh"
else 
    echo "Warning: Could not reliably find Conda root or profile.d/conda.sh. Proceeding, but activation might rely solely on sourced ~/.bashrc."
fi

echo "Checking if conda environment '$ENV_NAME' exists..."
# Check if environment exists
# Add error handling for the initial conda check
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command still not found after sourcing ~/.bashrc. Cannot proceed."
    echo "Please ensure Conda is installed and initialized correctly (e.g., run 'conda init bash')."
    exit 1
fi

if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo "Error creating conda environment '$ENV_NAME'. Exiting."
        exit 1
    fi
    echo "Environment '$ENV_NAME' created successfully."
fi

# Activate the environment
# This should work now because conda command is available and conda.sh was sourced.
echo "Activating conda environment '$ENV_NAME'..."
conda activate $ENV_NAME
if [ $? -ne 0 ]; then
    echo "Error activating conda environment '$ENV_NAME'. Exiting."
    exit 1
fi
echo "Environment '$ENV_NAME' activated."


# Install requirements inside the activated environment
echo "Installing packages from requirements.txt into '$ENV_NAME'..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing packages from requirements.txt. Exiting."
    exit 1
fi

if [ "$HOSTNAME" != "penguin" ]; then
    echo "Installing tmux via conda..."
    conda install -y -c conda-forge tmux # Add -y for non-interactive install
    if [ $? -ne 0 ]; then
        echo "Error installing tmux via conda. Exiting."
        exit 1
    fi
fi

apt-get update
apt-get install vim
