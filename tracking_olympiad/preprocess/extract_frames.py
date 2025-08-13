import cv2
import os

from utils.file_utils import mkdir_safe
from utils.logger import setup_logger
from utils.media import open_video

logger = setup_logger()


def extract_frames(video_dir, output_dir):
    """
    Extracts frames from all `.mp4` videos in a directory and saves them as image files.

    Args:
        video_dir (str): Path to the directory containing `.mp4` video files.
        output_dir (str): Path to the directory where extracted frames will be saved.
            Frames are stored in subfolders named after each video (without extension).
    """
    mkdir_safe(output_dir)
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, filename)
        name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, name)
        mkdir_safe(save_path)

        cap = open_video(video_path)
        frame_num = 0
        success, frame = cap.read()

        while success:
            out_file = os.path.join(save_path, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(out_file, frame)
            frame_num += 1
            success, frame = cap.read()

        cap.release()
        logger.info("Extracted %s frames from %s", frame_num, filename)
