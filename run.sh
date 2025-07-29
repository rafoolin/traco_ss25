#!/bin/bash

# --- Pipeline ---

# Step 1: Clone the Tracko repository

# Clone Code from Tracko Github repo
echo -e "\033[1;33mCloning Tracko repository...\033[0m"
# Clone only if the repository does not already exist
if [ -d "traco_2024" ]; then
    echo -e "\033[1;33mRepository already exists. Skipping clone.\033[0m"
else
    echo -e "\033[1;33mCloning repository...\033[0m"
    git clone https://github.com/ankilab/traco_2024.git
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to clone repository. Please check your internet connection or the repository URL.\033[0m"
        exit 1
    fi
fi

echo -e "\033[1;33mRepository cloned successfully.\033[0m"

# Step 2: Install required packages and run virtual environment setup

echo -e "\033[1;33mSetting up conda environment...\033[0m"
conda_env="traco_env"

# Check if conda is installed
if [ ! -d "env" ]; then
    conda create -n $conda_env python=3.10 -y
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to create conda environment. Please check your conda installation.\033[0m"
        exit 1
    fi
    echo -e "\033[1;33mConda environment created successfully.\033[0m"
else
    echo -e "\033[1;33mConda environment already exists. Skipping creation.\033[0m"
fi

source activate $conda_env
echo -e "\033[1;33mInstalling required packages...\033[0m"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to install required packages. Please check the requirements file.\033[0m"
        exit 1
    fi
    echo -e "\033[1;33mPackages installed successfully.\033[0m"
else
    echo -e "\033[1;33mrequirements.txt file not found. Please ensure it exists in the current directory.\033[0m"
    exit 1
fi


# Step 2: Extract frames from videos
echo -e "\033[1;33mExtracting frames from videos...\033[0m"
# If frames folder do exists, skip extraction
if [ -d "data/frames" ]; then
    echo -e "\033[1;33mFrames already extracted. Skipping extraction.\033[0m"
else
    echo -e "\033[1;33mExtracting frames...\033[0m"
    python3 pipeline/extract_frames.py
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to extract frames. Please check the script for errors.\033[0m"
        exit 1
    fi
    echo -e "\033[1;33mFrames extracted successfully.\033[0m"
fi

# Step 3: Clone sam2 repository and install requirements
# Step 3.1: Clone the SAM2 repository
echo -e "\033[1;33mCloning SAM2 repository...\033[0m"
if [ -d "sam2" ]; then
    echo -e "\033[1;33mSAM2 repository already exists. Skipping clone.\033[0m"
else
    git clone https://github.com/facebookresearch/sam2.git
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to clone SAM2 repository. Please check your internet connection or the repository URL.\033[0m"
        exit 1
    fi
fi

# Step 3.2: Download SAM2 model weights
echo -e "\033[1;33mDownloading SAM2 model weights...\033[0m"
if [ -f "sam2/sam2_weights/sam2.1_hiera_large.pth" ]; then
    echo -e "\033[1;33mSAM2 model weights already downloaded. Skipping download.\033[0m"
else
    cd sam2/checkpoints && \
    bash download_ckpts.sh
    if [ $? -ne 0 ]; then
        echo -e "\033[1;33mFailed to download SAM2 model weights. Please check your internet connection or the URL.\033[0m"
        exit 1
    fi
    cd ../..
    echo -e "\033[1;33mSAM2 model weights downloaded successfully.\033[0m"
fi

# Step 3.3: Build the SAM2 model
echo -e "\033[1;33mBuilding SAM2 model...\033[0m"
cd sam2
pip install -e .
if [ $? -ne 0 ]; then
    echo -e "\033[1;33mFailed to build SAM2 model. Please check the setup.\033[0m"
    exit 1
fi
echo -e "\033[1;33mSAM2 model built successfully.\033[0m"
cd ..

# Step 4: Run the bounding box body script
echo -e "\033[1;33mRunning bounding box body script...\033[0m"
python3 pipeline/bounding_box_body.py
if [ $? -ne 0 ]; then
    echo -e "\033[1;33mFailed to run bounding box body script. Please check the script for errors.\033[0m"
    exit 1
fi
echo -e "\033[1;33mBounding box body script executed successfully.\033[0m"