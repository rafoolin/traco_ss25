from dataclasses import dataclass
import os


import utils.frame_utils as frame_utils
import utils.draw_utils as draw_utils
import utils.yolo_labels as yolo_labels
import utils.mask_utils as mask_utils
from PIL import Image
from utils.file_utils import mkdir_safe
from utils.logger import setup_logger

from sam2.sam2_image_predictor import SAM2ImagePredictor


logger = setup_logger()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@dataclass(frozen=True)
class SAMConfig:
    """Configuration for SAM2 model."""

    device: str = "cpu"  # Default device
    sam2_predictor: SAM2ImagePredictor = None


class BoundingBox:
    """Generates bounding boxes for head & body parts in video frames using SAM2 model."""

    def __init__(
        self,
        frame_dir_path,
        csv_dir_path,
        data_dir_path,
        yolo_dir_path,
        sam_config: SAMConfig,
        box_size: int = 300,
        anchor_ratio: float = 0.25,
    ):
        self.box_size = box_size
        self.anchor_ratio = anchor_ratio
        self.shift_x = box_size * anchor_ratio
        self.shift_y = box_size * anchor_ratio
        self.frame_dir_path = frame_dir_path
        self.frame_data = []
        self.sam_config = sam_config
        self.csv_dir_path = csv_dir_path
        self.data_dir_path = data_dir_path
        self.annotation_dir_path_body = os.path.join(data_dir_path, "annotations/body")
        self.annotation_dir_path_head = os.path.join(data_dir_path, "annotations/head")
        self.raw_yolo_dir_path_body = os.path.join(yolo_dir_path, "body")
        self.raw_yolo_dir_path_head = os.path.join(yolo_dir_path, "head")
        mkdir_safe(self.annotation_dir_path_body)
        mkdir_safe(self.annotation_dir_path_head)
        mkdir_safe(self.raw_yolo_dir_path_body)
        mkdir_safe(self.raw_yolo_dir_path_head)

    def generate_bounding_boxes(self):
        """
        Generates bounding boxes for each video frame based on the provided CSV data
        and saves the results in YOLO format.

        The annotated images and YOLO labels are saved in specified directories.
        This will be done for both body and head parts.
        """
        # Walk through video directories
        all_videos = sorted(os.listdir(self.frame_dir_path))
        for video_name in ["training020"]:
            logger.info("Processing video: %s", video_name)
            frame_path = os.path.join(self.frame_dir_path, video_name)
            csv_path = os.path.join(self.csv_dir_path, f"{video_name}.csv")
            # CSV Data
            csv_data = frame_utils.read_csv_frames(csv_path)
            yolo_data_body = []
            yolo_data_head = []
            missing_data = []
            all_frames = sorted(os.listdir(frame_path))

            if not os.path.isdir(frame_path):
                continue
            for frame_img in all_frames:
                logger.info("Processing frame: %s", frame_img)
                if not frame_img.endswith(".jpg") and not frame_img.endswith(".png"):
                    continue

                frame_index = int(frame_img.split("_")[1].split(".")[0])
                image_path = os.path.join(frame_path, frame_img)
                # Image and draw setup
                image = Image.open(image_path).convert("RGB")
                self.sam_config.sam2_predictor.set_image(image)
                if frame_index not in csv_data:
                    missing_data.append(frame_index)
                    logger.info(
                        "No data for frame %d in CSV, skipping...",
                        frame_index,
                    )
                    continue

                # Body Part
                mask_list = mask_utils.get_frame_masks(
                    csv_data,
                    frame_index,
                    self.sam_config.sam2_predictor,
                )
                # Annotate Body
                annotated_image_body = draw_utils.draw_head_and_bbox(
                    image.copy(),
                    mask_list,
                )
                # Save annotated image for body
                draw_utils.save_annotated_image(
                    self.annotation_dir_path_body,
                    annotated_image_body,
                    video_name,
                    frame_index,
                )
                # YOLO labels Body
                for mask_dict in mask_list:
                    if mask_dict is None:
                        continue
                    yolo_data_body.append(
                        yolo_labels.get_yolo_labels_body(
                            mask_dict,
                            image.size,
                            frame_index,
                        )
                    )
                # Head Part
                # Annotate Head
                annotated_image_head = draw_utils.draw_head_fixed_bbox(
                    image.copy(),
                    frame_index,
                    csv_data,
                )
                # Save annotated image for head
                draw_utils.save_annotated_image(
                    self.annotation_dir_path_head,
                    annotated_image_head,
                    video_name,
                    frame_index,
                )
                # YOLO labels Head
                for row_data in csv_data[frame_index]:
                    if row_data is None:
                        continue
                    yolo_data_head.append(
                        yolo_labels.get_label_head(
                            row_data,
                            frame_index,
                            image.size[0],
                            image.size[1],
                            self.box_size,
                            self.shift_x,
                            self.shift_y,
                        )
                    )

            # Save missing frames
            if len(missing_data) > 0:
                frame_utils.save_missing_frames(
                    missing_data, video_name, self.data_dir_path
                )
            # Save YOLO labels for body
            if len(yolo_data_body) > 0:
                yolo_labels.save_yolo_labels(
                    yolo_data_body,
                    video_name,
                    self.raw_yolo_dir_path_body,
                )
            # Save YOLO labels for head
            if len(yolo_data_head) > 0:
                yolo_labels.save_yolo_labels(
                    yolo_data_head,
                    video_name,
                    self.raw_yolo_dir_path_head,
                )
