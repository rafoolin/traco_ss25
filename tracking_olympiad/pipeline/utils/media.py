import cv2


def open_video(video_path):
    """Open a video file and return the video capture object."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    return cap


def close_video(cap):
    """Release the video capture object and close all OpenCV windows."""
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
