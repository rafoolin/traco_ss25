import os

import torch
from pipeline.utils.logger import setup_logger

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = setup_logger()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Note: This function is adapted from [sam notebooks](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)
def get_device() -> torch.device:
    """
    Determines and configures the best available PyTorch device (CUDA, MPS, or CPU) for computation.

    Returns:
        torch.device: The selected device object.
    """
    current_device = None
    if torch.cuda.is_available():
        current_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        current_device = torch.device("mps")
    else:
        current_device = torch.device("cpu")
    logger.info("using device: %s", current_device)
    if current_device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif current_device.type == "mps":
            logger.warning(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
    return 'cpu'


def load_sam2() -> SAM2ImagePredictor:
    """
    Loads the SAM2 model and returns an image predictor instance.

    Args:
        device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda').

    Returns:
        SAM2ImagePredictor: An instance of the image predictor initialized with the loaded SAM2 model.
    """
    checkpoint_path = "sam2/checkpoints/sam2.1_hiera_large.pt"
    config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(config_path, checkpoint_path, device=get_device())
    return SAM2ImagePredictor(sam2_model)
