import cv2
import os


def extract_frames(video_dir, output_dir):
    """
    Extract frames from a video file and save them as images.

    :param video_dir: Path to the directory containing video files.
    :param output_dir: Folder where extracted frames will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(video_dir):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, filename)
        name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, name)
        os.makedirs(save_path, exist_ok=True)

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


if __name__ == "__main__":

    video_dir = "traco_2024/training"
    output_dir = "data/frames"
    extract_frames(video_dir, output_dir)
