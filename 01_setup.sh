#!/bin/bash
# Source the user's bash configuration to make conda available
# This assumes conda init has been run for the bash shell.
if [ -f ~/.bashrc ]; then
    echo "Sourcing ~/.bashrc to find conda..."
    source ~/.bashrc
else
    echo "Warning: ~/.bashrc not found. Conda might not be available in the script's PATH."
fi

echo "--- Setting up Conda Environment and Installing Dependencies in base environment ---"

# Now, try to find conda root using conda info (should be available if .bashrc worked)
_CONDA_ROOT=$(conda info --base 2>/dev/null) # Try to get conda root silently

# If conda info worked, source the specific conda.sh - more robust
if [ -n "$_CONDA_ROOT" ] && [ -f "$_CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "Found Conda root: $_CONDA_ROOT. Sourcing profile.d/conda.sh..."
    source "$_CONDA_ROOT/etc/profile.d/conda.sh"
else 
    echo "Warning: Could not reliably find Conda root or profile.d/conda.sh. Proceeding, but conda commands might rely solely on sourced ~/.bashrc."
fi

# Add error handling for the initial conda check
if ! command -v conda &> /dev/null; then
    echo "Error: 'conda' command still not found after sourcing ~/.bashrc. Cannot proceed."
    echo "Please ensure Conda is installed and initialized correctly (e.g., run 'conda init bash')."
    exit 1
fi

# No longer creating or activating a specific environment
# Python version is determined by the existing active (base) environment.
# echo "Checking if conda environment '$ENV_NAME' exists..."

# Activate the environment
# echo "Activating conda environment '$ENV_NAME'..."
# conda activate $ENV_NAME
# if [ $? -ne 0 ]; then
#     echo "Error activating conda environment '$ENV_NAME'. Exiting."
#     exit 1
# fi
# echo "Environment '$ENV_NAME' activated."
echo "Ensuring current environment (expected to be base) is used for installations."


# Ensure pip is installed in the current conda environment
echo "Ensuring pip is installed in the current conda environment..."
conda install -y pip
if [ $? -ne 0 ]; then
    echo "Error installing pip via conda. Exiting."
    exit 1
fi

# Install pygobject via conda-forge as pip build can be problematic
echo "Installing pygobject via conda-forge..."
conda install -y -c conda-forge pygobject
if [ $? -ne 0 ]; then
    echo "Error installing pygobject via conda-forge. Exiting."
    exit 1
fi

# Install playsound via conda-forge
echo "Installing playsound via conda-forge..."
conda install -y -c conda-forge playsound
if [ $? -ne 0 ]; then
    echo "Error installing playsound via conda-forge. Exiting."
    exit 1
fi

# Install requirements inside the current (base) environment
echo "Installing packages from requirements.txt into the current conda environment..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing packages from requirements.txt. Exiting."
    exit 1
fi

if [ "$HOSTNAME" != "penguin" ]; then
    echo "Installing tmux via conda into the current conda environment..."
    conda install -y -c conda-forge tmux # Add -y for non-interactive install
    if [ $? -ne 0 ]; then
        echo "Error installing tmux via conda. Exiting."
        exit 1
    fi
fi

sudo apt-get update
sudo apt-get install -y vim htop 

# --- Configure .bashrc for automatic \'base\' conda environment activation ---
echo ""
echo "--- Configuring .bashrc for automatic \'base\' conda environment activation ---"

BASHRC_FILE="$HOME/.bashrc"
ENV_TO_ACTIVATE="base" # Explicitly set to base
ACTIVATION_COMMAND="conda activate $ENV_TO_ACTIVATE"
MARKER_COMMENT="# Added by audio_is_all_you_need setup to activate $ENV_TO_ACTIVATE conda environment"

if [ -f "$BASHRC_FILE" ]; then
    # Check if Conda itself is initialized in .bashrc
    if ! grep -q -E "(conda initialize|conda.sh)" "$BASHRC_FILE"; then
        echo ""
        echo "WARNING: Conda does not appear to be fully initialized in your $BASHRC_FILE." >&2
        echo "         The line '$ACTIVATION_COMMAND\' will be added, but it might not work on new logins." >&2
        echo "         Please ensure \'conda init bash\' has been run and its output is correctly sourced in $BASHRC_FILE." >&2
        echo "         Look for a block of code starting with \'# >>> conda initialize >>>\' in $BASHRC_FILE." >&2
        echo ""
    fi

    # Check if our specific marker comment already exists to prevent duplicate entries
    if grep -Fxq "$MARKER_COMMENT" "$BASHRC_FILE"; then
        echo "Activation command for '$ENV_TO_ACTIVATE\' (marked by '$MARKER_COMMENT\') already found in $BASHRC_FILE."
        echo "No changes made to $BASHRC_FILE."
    else
        echo "Adding '$ACTIVATION_COMMAND\' to $BASHRC_FILE for automatic activation in new bash shells."
        # Append the marker and the command
        echo "" >> "$BASHRC_FILE" # Add a newline for separation
        echo "$MARKER_COMMENT" >> "$BASHRC_FILE"
        echo "$ACTIVATION_COMMAND" >> "$BASHRC_FILE"
        echo ""
        echo "Successfully added activation command to $BASHRC_FILE."
        echo "To make this effective for future sessions, no further action is needed; new bash terminals will attempt to activate '$ENV_TO_ACTIVATE\' automatically."
        echo "To apply to your CURRENT terminal session, you can run: source $BASHRC_FILE"
        echo "(Note: If your current shell is already \'base\', sourcing might not show an immediate change beyond what \'conda init\' does.)"
        echo ""
        echo "To UNDO this automatic activation in the future, manually edit $BASHRC_FILE" 
        echo "and remove the following lines:"
        echo "$MARKER_COMMENT"
        echo "$ACTIVATION_COMMAND"
    fi
else
    echo "WARNING: $BASHRC_FILE not found. Could not configure automatic environment activation for new bash shells." >&2
fi
# echo "Automatic environment activation for a specific environment in .bashrc has been removed from this script." # Old message, removing

echo "--- Setup script finished ---"
