import os
import shutil

from utils.logger import setup_logger

logger = setup_logger()


def reset_dir(path: str):
    """Deletes and recreates a directory."""
    if os.path.exists(path):
        logger.info("Deleting %s ...", path)
        shutil.rmtree(path)
    os.makedirs(path)
    logger.info("Created %s", path)
