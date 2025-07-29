import os
import subprocess
import sys

COLOR = "\033[1;33m"
RESET = "\033[0m"


def run_command(command, cwd=None):
    print(f"{COLOR}Running: {command}{RESET}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"{COLOR}Command failed: {command}{RESET}")
        sys.exit(1)


def clone_tracko_repo():
    if os.path.isdir("traco_2024"):
        print(f"{COLOR}Repository already exists. Skipping clone.{RESET}")
    else:
        print(f"{COLOR}Cloning Tracko repository...{RESET}")
        run_command("git clone https://github.com/ankilab/traco_2024.git")


def setup_conda_env(env_name="traco_env"):
    if os.path.isdir("env"):
        print(f"{COLOR}Conda environment already exists. Skipping creation.{RESET}")
    else:
        print(f"{COLOR}Creating conda environment...{RESET}")
        run_command(f"conda create -n {env_name} python=3.10 -y")

    print(f"{COLOR}Activating conda environment and installing packages...{RESET}")
    if os.path.isfile("requirements.txt"):
        run_command(f"conda run -n {env_name} pip install -r requirements.txt")
    else:
        print(f"{COLOR}requirements.txt not found.{RESET}")
        sys.exit(1)


def extract_frames():
    if os.path.isdir("data/frames"):
        print(f"{COLOR}Frames already extracted. Skipping.{RESET}")
    else:
        print(f"{COLOR}Extracting frames...{RESET}")
        run_command("python3 pipeline/extract_frames.py")


def clone_sam2_repo():
    if os.path.isdir("sam2"):
        print(f"{COLOR}SAM2 repository already exists. Skipping clone.{RESET}")
    else:
        print(f"{COLOR}Cloning SAM2 repository...{RESET}")
        run_command("git clone https://github.com/facebookresearch/sam2.git")


def download_sam2_weights():
    weights_path = "sam2/sam2_weights/sam2.1_hiera_large.pt"
    if os.path.isfile(weights_path):
        print(f"{COLOR}SAM2 model weights already downloaded. Skipping.{RESET}")
    else:
        print(f"{COLOR}Downloading SAM2 model weights...{RESET}")
        run_command("bash download_ckpts.sh", cwd="sam2/checkpoints")


def build_sam2_model():
    print(f"{COLOR}Installing SAM2 model...{RESET}")
    run_command("pip install -e .", cwd="sam2")


def run_bounding_box_body_script():
    print(f"{COLOR}Running bounding box body script...{RESET}")
    run_command("python3 pipeline/bounding_box_body.py")


def run_bounding_box_head_script():
    print(f"{COLOR}Running bounding box head script...{RESET}")
    run_command("python3 pipeline/bounding_box_head.py")


def run_pipeline():
    # Set up projects
    clone_tracko_repo()
    clone_sam2_repo()
    # Set up conda environment and install dependencies
    setup_conda_env()
    # Download SAM2 weights
    download_sam2_weights()
    build_sam2_model()

    # Run the pipeline steps
    print(f"{COLOR}Starting pipeline...{RESET}")
    # Step 1: Extract frames from videos
    extract_frames()
    # Step 2: Run bounding box script for body detection
    run_bounding_box_body_script()
    # Step 3: Run bounding box script for head detection
    run_bounding_box_head_script()
    print(f"{COLOR}Pipeline completed successfully!{RESET}")


if __name__ == "__main__":
    run_pipeline()
