#!/bin/bash

# --- Color Logger ---
RESET="\033[0m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"

log() {
    local level="$1"
    local color="$2"
    local message="$3"
    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${color}[${timestamp}] ${level} - ${message}${RESET}"
}

info() { log "INFO" "$GREEN" "$1"; }
warn() { log "WARNING" "$YELLOW" "$1"; }
error() { log "ERROR" "$RED" "$1"; exit 1; }

# --- Pipeline ---

# Step 1: Clone the Tracko repository
info "Cloning Tracko repository..."
# Clone only if the repository does not already exist
if [ -d "traco_2024" ]; then
    warn "Tracko repository already exists. Skipping clone."
else
    info "Cloning Tracko repository..."
    git clone https://github.com/ankilab/traco_2024.git || error "Failed to clone Tracko repository."
    info "Tracko repository cloned successfully."
fi

# Step 2: Setup Conda environment
env_name="traco_env"
info "Setting up Conda environment: $env_name"
if conda env list | grep -q "$env_name"; then
    warn "Conda environment '$env_name' already exists. Skipping creation."
    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate "$env_name" || error "Failed to activate Conda environment."
else
    conda create -n "$env_name" python=3.10 -y || error "Failed to create Conda environment."
    info "Conda environment '$env_name' created successfully."
fi

# Step 3: Install required packages
info "Installing required packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt || error "Failed to install packages from requirements.txt."
    info "Packages installed successfully."
else
    error "requirements.txt not found."
fi

# Step 4: Clone SAM2 repository
info "Checking SAM2 repository..."
if [ -d "sam2" ]; then
    warn "SAM2 repository already exists. Skipping clone."
else
    info "Cloning SAM2 repository..."
    git clone https://github.com/facebookresearch/sam2.git || error "Failed to clone SAM2 repository."
    info "SAM2 repository cloned successfully."
fi

# Step 5: Download SAM2 model weights
info "Checking SAM2 model weights..."
if [ -f "sam2/checkpoints/sam2.1_hiera_large.pt" ]; then
    warn "SAM2 model weights already exist. Skipping download."
else
    info "Downloading SAM2 model weights..."
    (cd sam2/checkpoints && bash download_ckpts.sh) || error "Failed to download SAM2 model weights."
    info "SAM2 model weights downloaded successfully."
fi

# Step 6: Build SAM2 model
info "Installing SAM2 model..."
(cd sam2 && pip install -e .) || error "Failed to install SAM2 model."
info "SAM2 model installed successfully."

# Step 7: Run the pipeline
info "Running pipeline script..."
python pipeline/pipeline.py \
--video_dir "./traco_2024/training" \
--csv_dir "./traco_2024/training" \
--data_dir "./data" \
--yolo_db_dir "./data/yolo_dataset" || error "Pipeline script failed."
info "Pipeline executed successfully."