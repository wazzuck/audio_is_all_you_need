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

# --- Attempt to add conda activation to .bashrc for future login shells ---
echo ""
echo "--- Configuring .bashrc for automatic 'tf_env' activation ---"

BASHRC_FILE="$HOME/.bashrc"
ENV_NAME="tf_env" # Ensure this matches the environment name used in this script
ACTIVATION_COMMAND="conda activate $ENV_NAME"
MARKER_COMMENT="# Added by audio_is_all_you_need setup to activate $ENV_NAME"

if [ -f "$BASHRC_FILE" ]; then
    # Check if Conda itself is initialized in .bashrc
    # This check is basic; a more robust check would look for the conda hook function.
    if ! grep -q -E "(conda initialize|conda.sh)" "$BASHRC_FILE"; then
        echo ""
        echo "WARNING: Conda does not appear to be fully initialized in your $BASHRC_FILE." >&2
        echo "         The line '$ACTIVATION_COMMAND' will be added, but it might not work on new logins." >&2
        echo "         Please ensure 'conda init bash' has been run and its output is correctly sourced in $BASHRC_FILE." >&2
        echo "         Look for a block of code starting with '# >>> conda initialize >>>' in $BASHRC_FILE." >&2
        echo ""
    fi

    # Check if our specific marker comment already exists to prevent duplicate entries
    if grep -Fxq "$MARKER_COMMENT" "$BASHRC_FILE"; then
        echo "'$ENV_NAME' activation command (marked by '$MARKER_COMMENT') already found in $BASHRC_FILE."
        echo "No changes made to $BASHRC_FILE."
    else
        echo "Adding '$ACTIVATION_COMMAND' to $BASHRC_FILE for automatic activation in new bash shells."
        # Append the marker and the command
        echo "" >> "$BASHRC_FILE" # Add a newline for separation
        echo "$MARKER_COMMENT" >> "$BASHRC_FILE"
        echo "$ACTIVATION_COMMAND" >> "$BASHRC_FILE"
        echo ""
        echo "Successfully added activation command to $BASHRC_FILE."
        echo "To make this effective for future sessions, no further action is needed; new bash terminals will attempt to activate '$ENV_NAME' automatically."
        echo "To apply to your CURRENT terminal session, you can run: source $BASHRC_FILE"
        echo "(Note: If you run 'source $BASHRC_FILE' and it activates '$ENV_NAME', this script will still finish in its own subshell.)"
        echo ""
        echo "To UNDO this automatic activation in the future, manually edit $BASHRC_FILE" 
        echo "and remove the following lines (or lines between the markers if more were added):"
        echo "$MARKER_COMMENT"
        echo "$ACTIVATION_COMMAND"
    fi
else
    echo "WARNING: $BASHRC_FILE not found. Could not configure automatic environment activation for new bash shells." >&2
fi

echo "--- Setup script finished ---"
