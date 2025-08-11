import cv2
import os

from utils.file_utils import mkdir_safe


def extract_frames(video_dir, output_dir):
    """
    Extract frames from a video file and save them as images.

    :param video_dir: Path to the directory containing video files.
    :param output_dir: Folder where extracted frames will be saved.
    """
    mkdir_safe(output_dir)
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, filename)
        name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, name)
        mkdir_safe(save_path)

        cap = cv2.VideoCapture(video_path)
        frame_num = 0
        success, frame = cap.read()

        while success:
            out_file = os.path.join(save_path, f"frame_{frame_num:05d}.jpg")
            cv2.imwrite(out_file, frame)
            frame_num += 1
            success, frame = cap.read()

        cap.release()
        print(f"Extracted {frame_num} frames from {filename}")

