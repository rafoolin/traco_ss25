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
    conda create -n $conda_env python=3.8 -y
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
python3 pipeline/extract_frames.py
if [ $? -ne 0 ]; then
    echo -e "\033[1;33mFailed to extract frames. Please check the script for errors.\033[0m"
    exit 1
fi