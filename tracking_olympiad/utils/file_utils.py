import os
import shutil

from utils.logger import setup_logger

logger = setup_logger()


def reset_dir(path: str):
    """
    Deletes and recreates a directory.

    Args:
        path (str): Path to the directory.
    """
    if os.path.exists(path):
        logger.info("Deleting %s ...", path)
        shutil.rmtree(path)
    mkdir_safe(path)
    logger.info("Created %s", path)


def mkdir_safe(path: str, exist_ok: bool = True):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory.
        exist_ok (bool): If True, does not raise an error if the directory already exists.
    """

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=exist_ok)
        logger.info("Created directory: %s", path)
