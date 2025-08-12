
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
    """Retrieves masks for a specific frame using the SAM2 predictor."""
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
    Calculates the bounding box coordinates of the non-zero region in a binary mask.

    Args:
        mask (np.ndarray): A binary mask array where non-zero values indicate the region of interest.

    Returns:
        tuple: A tuple containing two tuples ((x_min, y_min), (x_max, y_max)) representing the top-left and bottom-right
            coordinates of the bounding box. Returns None if the mask contains no non-zero elements.
    """
    ys, xs = np.where(np.squeeze(mask))
    if ys.size == 0 or xs.size == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (x_min, y_min), (x_max, y_max)


def extract_xy(boxes, keypoints, i):
    """
    Prefer the first keypoint (index 0) if available, otherwise bbox center (xywh).
    Returns (x, y) as floats.
    """
    if keypoints is not None and getattr(keypoints, "xy", None) is not None:
        k = keypoints.xy[i]
        if k is not None and len(k) > 0:
            return float(k[0][0]), float(k[0][1])
    # fallback: bbox center (xywh)
    xywh = boxes.xywh[i]
    return float(xywh[0]), float(xywh[1])