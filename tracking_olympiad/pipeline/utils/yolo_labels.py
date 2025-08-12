import os
from pipeline.utils.logger import setup_logger
from pipeline.utils.draw_utils import get_mask_bounding_box


logger = setup_logger()


def get_yolo_labels_body(mask_dict, image_size, frame_index):
    """Generates YOLO format labels for body masks."""

    bbox = get_mask_bounding_box(mask_dict["mask"])
    if bbox is None:
        return []
    image_width, image_height = image_size
    mask_x_min = bbox[0][0]
    mask_y_min = bbox[0][1]
    mask_x_max = bbox[1][0]
    mask_y_max = bbox[1][1]
    bbox_cx = (mask_x_min + mask_x_max) / 2 / image_width
    bbox_cy = (mask_y_min + mask_y_max) / 2 / image_height
    bbox_w = (mask_x_max - mask_x_min) / image_width
    bbox_h = (mask_y_max - mask_y_min) / image_height
    hexbug_cx, hexbug_cy = (
        mask_dict["x"] / image_width,
        mask_dict["y"] / image_height,
    )
    hexbug_id = mask_dict["hexbug"]
    return [
        frame_index,
        hexbug_id,
        bbox_cx,
        bbox_cy,
        bbox_w,
        bbox_h,
        hexbug_cx,
        hexbug_cy,
        2,
    ]


def get_label_head(csv_data, frame_index, w, h, box_size=100, shift_x=50, shift_y=50):
    """Generates YOLO format labels for head data."""
    if len(csv_data) == 0:
        return []

    cx = csv_data["x"]
    cy = csv_data["y"]
    hexbug_id = csv_data["hexbug"]

    x_min = max(0, cx - shift_x)
    y_min = max(0, cy - shift_y)
    x_max = min(w, x_min + box_size)
    y_max = min(h, y_min + box_size)

    bw = x_max - x_min
    bh = y_max - y_min

    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    bw /= w
    bh /= h
    n_cx = cx / w
    n_cy = cy / h

    return [
        frame_index,
        hexbug_id,
        x_center,
        y_center,
        bw,
        bh,
        n_cx,
        n_cy,
        2,
    ]


def save_yolo_labels(yolo_data: list, video_name: str, label_path: str):
    """
    Saves YOLO formatted labels to a text file.

    Args:
        yolo_data (list): List of YOLO formatted data.
        video_name (str): Name of the video for which labels are being saved.
        label_path (str): Path to the directory where the label file will be saved.
    """
    yolo_label_path = os.path.join(label_path, f"{video_name}.txt")
    with open(yolo_label_path, "w", encoding="utf-8") as f:
        for row in yolo_data:
            f.write(
                f"{row[0]} {row[1]} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f} {row[8]}\n"
            )
    logger.info("YOLO label saved to %s", yolo_label_path)
