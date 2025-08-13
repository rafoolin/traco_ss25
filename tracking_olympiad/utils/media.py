import cv2


def open_video(video_path):
    """
    Opens a video file for reading.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: OpenCV video capture object.

    Raises:
        AssertionError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    return cap


def close_video(cap):
    """
    Releases a video capture object and closes all OpenCV windows.

    Args:
        cap (cv2.VideoCapture): The OpenCV video capture object to release.
    """
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
