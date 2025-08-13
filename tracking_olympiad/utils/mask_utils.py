import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor


def _get_hexbug_points(hexbugs: list) -> list:
    """Extracts valid Hexbug points from a list of Hexbug data dictionaries."""
    hexbug_points = []
    for hexbug_data in hexbugs:
        x, y, hexbug = hexbug_data["x"], hexbug_data["y"], hexbug_data["hexbug"]
        if not np.isnan(x) and not np.isnan(y):
            hexbug_points.append({"x": int(x), "y": int(y), "hexbug": hexbug})
    return hexbug_points


def _get_center_points(hexbug_points: list) -> np.ndarray:
    return np.array([[point["x"], point["y"]] for point in hexbug_points])


def get_frame_masks(
    csv_data: dict,
    frame_index: int,
    sam2_predictor: SAM2ImagePredictor,
) -> list:
    """
    Generates object masks for a specific frame using the SAM2 predictor.

    Args:
        csv_data (dict): Frame data mapping frame indices to object detection info.
        frame_index (int): Index of the frame to process.
        sam2_predictor (SAM2ImagePredictor): SAM2 predictor instance for mask generation.

    Returns:
        list: List of dictionaries, each containing:
            - "bbox": Bounding box ((x_min, y_min), (x_max, y_max)) from mask.
            - "x" (float): Center x-coordinate of the object.
            - "y" (float): Center y-coordinate of the object.
            - "hexbug" (int/str): Hexbug identifier.
    """
    hexbug_points = _get_hexbug_points(csv_data[frame_index])
    center_points = _get_center_points(hexbug_points)
    # SAM
    input_points = np.array(center_points).reshape(-1, 1, 2)
    input_labels = np.array([1] * len(input_points)).reshape(-1, 1)

    # Predict bounding boxes
    masks, _, __ = sam2_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    return [
        {
            "bbox": get_mask_bounding_box(mask),
            "x": hexbug["x"],
            "y": hexbug["y"],
            "hexbug": hexbug["hexbug"],
        }
        for mask, hexbug in zip(masks, hexbug_points)
        if mask is not None
    ]


def get_mask_bounding_box(mask: np.ndarray) -> tuple:
    """
    Finds the bounding box of the non-zero region in a binary mask.

    Args:
        mask (np.ndarray): Binary mask with non-zero values for the object region.

    Returns:
        tuple: ((x_min, y_min), (x_max, y_max)) bounding box coordinates, or None if empty.
    """
    ys, xs = np.where(np.squeeze(mask))
    if ys.size == 0 or xs.size == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (x_min, y_min), (x_max, y_max)


def extract_xy(boxes, keypoints, i):
    """
    Gets object center coordinates from keypoints if available, otherwise from bbox center.

    Args:
        boxes: YOLO detection boxes object.
        keypoints: YOLO detection keypoints object.
        i (int): Index of the object.

    Returns:
        tuple: (x, y) coordinates as floats.
    """
    if keypoints is not None and getattr(keypoints, "xy", None) is not None:
        k = keypoints.xy[i]
        if k is not None and len(k) > 0:
            return float(k[0][0]), float(k[0][1])
    # fallback: bbox center (xywh)
    xywh = boxes.xywh[i]
    return float(xywh[0]), float(xywh[1])
