import os

from PIL import Image, ImageDraw
from utils.file_utils import mkdir_safe
from utils.logger import setup_logger

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
    Draws fixed-size head bounding boxes and center points.

    Args:
        image (Image): Image to annotate.
        frame_index (int): Frame index.
        csv_data (dict): Frame-wise hexbug coordinates.
        box_size (int, optional): Bounding box size. Defaults to 50.
        shift_x (int, optional): Horizontal shift. Defaults to 25.
        shift_y (int, optional): Vertical shift. Defaults to 25.

    Returns:
        Image: Annotated image.
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


def draw_head_and_bbox(image: Image, masks: list):
    """
    Draws center points and bounding boxes on an image.

    Args:
        image (Image): PIL Image to annotate.
        masks (list): List of dicts with:
            - "bbox": ((x_min, y_min), (x_max, y_max))
            - "hexbug": Object ID.
            - "x": X center coordinate.
            - "y": Y center coordinate.

    Returns:
        Image: Annotated image.
    """
    draw = ImageDraw.Draw(image)
    for mask_dict in masks:
        if mask_dict is None:
            continue
        bbox = mask_dict["bbox"]
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
        if bbox is None:
            logger.warning(" -No bounding box for hexbug %s at (%f, %f)", hexbug, x, y)
            continue
        (x_min, y_min), (x_max, y_max) = bbox
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="yellow", width=5)
        draw.text((x, y), str(hexbug), fill=(255, 255, 255, 128))
    return image


def save_annotated_image(annotation_dir_path, annotated_image, video_name, frame_index):
    """
    Saves an annotated image to disk.

    Args:
        annotation_dir_path (str): Directory to save the image.
        annotated_image (Image): PIL Image to save.
        video_name (str): Video name for folder structure.
        frame_index (int): Frame index for file naming.
    """
    output_image_path = os.path.join(
        annotation_dir_path,
        video_name,
        f"frame_{frame_index:05d}.jpg",
    )
    mkdir_safe(os.path.dirname(output_image_path))
    annotated_image.save(output_image_path)
