import os

from PIL import Image, ImageDraw
from utils.file_utils import mkdir_safe
from utils.logger import setup_logger
from utils.mask_utils import get_mask_bounding_box

logger = setup_logger()


def draw_head_fixed_bbox(
    image,
    frame_index,
    csv_data,
    box_size=50,
    shift_x=25,
    shift_y=25,
):
    """
    Annotates the head part of the video frame with bounding boxes and center points.

    Args:
        image (Image): The image to annotate.
        frame_index (int): The index of the frame.
        csv_data (dict): The CSV data containing hexbug coordinates.
    """
    hexbugs_data = csv_data[frame_index]
    if not hexbugs_data:
        return image

    for hexbug_data in hexbugs_data:
        cx, cy, hexbug_id = (
            hexbug_data["x"],
            hexbug_data["y"],
            hexbug_data["hexbug"],
        )
        draw = ImageDraw.Draw(image)
        radius = 10
        draw.ellipse(
            [(cx - radius, cy - radius), (cx + radius, cy + radius)],
            fill="red",
            outline="red",
        )
        draw.text((cx, cy), str(hexbug_id), fill=(255, 255, 255, 128))
        # Bounding box
        w, h = image.size
        x_min = max(0, cx - shift_x)
        y_min = max(0, cy - shift_y)
        x_max = min(w, x_min + box_size)
        y_max = min(h, y_min + box_size)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="yellow", width=5)

    return image


def draw_mask_and_center(image: Image, masks: list):
    """
    Draws masks, center points, and bounding boxes for a list of detected objects on an image.

    Args:
        image (Image): The image on which to draw. Should be a PIL Image object.
        masks (list): A list of dictionaries, each containing:
            - "mask": The mask of the detected object (format expected by get_mask_bounding_box).
            - "hexbug": An identifier for the object.
            - "x": The x-coordinate of the object's center.
            - "y": The y-coordinate of the object's center.

    Returns:
        Image: The modified image with drawn masks, center points, bounding boxes, and labels.
    """
    draw = ImageDraw.Draw(image)
    for mask_dict in masks:
        if mask_dict is None:
            continue
        mask = mask_dict["mask"]
        hexbug = mask_dict["hexbug"]
        # Center point
        x, y = int(mask_dict["x"]), int(mask_dict["y"])
        radius = 10
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill="red",
            outline="red",
        )

        # Bounding box
        bbox = get_mask_bounding_box(mask)
        if bbox is None:
            logger.warning(" -No bounding box for hexbug %s at (%f, %f)", hexbug, x, y)
            continue
        (x_min, y_min), (x_max, y_max) = bbox
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="yellow", width=5)
        draw.text((x, y), str(hexbug), fill=(255, 255, 255, 128))
    return image


def save_annotated_image(annotation_dir_path, annotated_image, video_name, frame_index):
    """
    Saves the annotated image to the specified directory.

    Args:
        annotation_dir_path (str): Path to the directory where the annotated image will be saved.
        annotated_image (Image): The annotated image to be saved.
        video_name (str): The name of the video for which the image is being saved.
        frame_index (int): The index of the frame in the video.
    """
    output_image_path = os.path.join(
        annotation_dir_path,
        video_name,
        f"frame_{frame_index:05d}.jpg",
    )
    mkdir_safe(os.path.dirname(output_image_path))
    annotated_image.save(output_image_path)
