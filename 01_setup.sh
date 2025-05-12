echo "--- Setting up Conda Environment and Installing Dependencies ---"

ENV_NAME="tf_env"
PYTHON_VERSION="3.11"

echo "Checking if conda environment '$ENV_NAME' exists..."
# Check if environment exists
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
# Sourcing activate script is more reliable in scripts than `conda activate`
echo "Activating conda environment '$ENV_NAME'..."
# Find CONDA_PREFIX (might not be set if base isn't active)
_CONDA_ROOT=$(conda info --base)
echo "CONDA ROOT: $_CONDA_ROOT"
if [ -z "$_CONDA_ROOT" ]; then
    echo "Could not determine Conda base directory. Please ensure Conda is initialized." 
    exit 1
fi
source "$_CONDA_ROOT/etc/profile.d/conda.sh"
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
