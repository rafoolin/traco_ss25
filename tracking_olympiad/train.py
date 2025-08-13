import argparse
import os

import yaml
from preprocess.compute_bounding_box import BoundingBox, SAMConfig
from preprocess.extract_frames import extract_frames
from preprocess.generate_yolo_dataset import create_yolo_dataset
from utils.logger import setup_logger
from utils.sam import get_device, load_sam2
from ultralytics import YOLO

logger = setup_logger()


class YoloDataset:
    """Class to handle YOLO dataset training and validation."""

    def _load_config(self, config_path):
        """Load configuration from a YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __init__(
        self,
        config_path,
        hyp_path,
        model_path="yolov8s-pose.pt",
        yolo_db_path="yolo_dataset",
    ):
        self.yolo_db_path = yolo_db_path
        self.config_path = config_path
        self.hyp_path = hyp_path
        config_data = self._load_config(config_path)
        self.hyp_data = dict(self._load_config(hyp_path))
        self.train_dir_path = config_data["train"]
        self.val_dir_path = config_data["val"]
        self.nc = config_data["nc"]
        self.names = config_data["names"]
        self.kpt_shape = config_data["kpt_shape"]
        self.model_path = model_path

    def train_and_validate(
        self,
        epochs=300,
        imgsz=640,
        batch=32,
        patience=50,
        augment=True,
        project="runs/pose",
        name="body",
    ):
        """Train and validate the YOLO model with the given configuration."""
        # Initialize model
        model = YOLO(f"{self.yolo_db_path}/{self.model_path}")

        # Train
        model.train(
            data=self.config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            augment=augment,
            project=f"{self.yolo_db_path}/{project}",
            name=f"{name}_train",
            **self.hyp_data,
        )

        # Validate best weights from this run
        best_weights = os.path.join(model.trainer.save_dir, "weights", "best.pt")
        model = YOLO(best_weights)
        model.val(
            data=self.config_path,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=f"{name}_val",
            **self.hyp_data,
        )

        logger.info("[OK] %s: best weights at %s", name, best_weights)


def run(
    video_dir: str,
    data_dir: str,
    csv_dir: str,
    yolo_db_dir: str,
    frame_dir_path: str,
    yolo_labels: str,
    body_config_path: str,
    head_config_path: str,
    hyp_path: str,
):
    """Run the training pipeline for YOLO models."""

    # Run the pipeline steps
    logger.info("Starting pipeline...")
    # Step 1: Extract frames from videos
    logger.info("Extracting frames...")
    if os.path.exists(frame_dir_path):
        logger.info("Frame directory already exists. Skipping extraction.")
    else:
        extract_frames(video_dir=video_dir, output_dir=frame_dir_path)
        logger.info("Frames extracted successfully!")
    # Step 2: Run bounding box script for body detection
    logger.info("Running bounding box script for body & head detection...")
    # Sam2 configuration
    device = get_device()
    sam2_predictor = load_sam2()
    bbox_body = BoundingBox(
        frame_dir_path=frame_dir_path,
        csv_dir_path=csv_dir,
        data_dir_path=data_dir,
        yolo_dir_path=yolo_labels,
        sam_config=SAMConfig(device=device, sam2_predictor=sam2_predictor),
    )
    bbox_body.generate_bounding_boxes()
    logger.info("Bounding box body script executed successfully!")
    # Step 3: Run Yolo dataset generator for Body
    create_yolo_dataset(
        frame_dir=frame_dir_path,
        label_dir=f"{yolo_labels}/body",
        img_out=f"{yolo_db_dir}/body/images",
        lbl_out=f"{yolo_db_dir}/body/label",
    )
    # Step 4: Run Yolo dataset generator for Head
    logger.info("Running Yolo dataset generator for Head...")
    create_yolo_dataset(
        frame_dir=frame_dir_path,
        label_dir=f"{yolo_labels}/head",
        img_out=f"{yolo_db_dir}/head/images",
        lbl_out=f"{yolo_db_dir}/head/label",
    )
    # Step 5: Train YOLO models for Body and Head
    logger.info("Training YOLO models for Body and Head...")

    body_dataset = YoloDataset(body_config_path, hyp_path)
    head_dataset = YoloDataset(head_config_path, hyp_path)

    body_dataset.train_and_validate(name="body")
    head_dataset.train_and_validate(name="head")
    logger.info("YOLO models trained successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos in a folder."
    )
    # Video
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Input directory with videos.",
    )
    # Output directory for frames
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Output directory for extracted and processed frames.",
    )
    # CSV directory
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Root directory for CSV files.",
    )

    # Yolo directory
    parser.add_argument(
        "--yolo_db_dir",
        type=str,
        required=True,
        help="Root directory for Yolo dataset folders.",
    )

    # Body and head config paths
    parser.add_argument(
        "--body_config_path",
        type=str,
        default="yolo_dataset/configs/hexbug_body.yaml",
        help="Path to the body YOLO config file.",
    )
    parser.add_argument(
        "--head_config_path",
        type=str,
        default="yolo_dataset/configs/hexbug_head.yaml",
        help="Path to the head YOLO config file.",
    )
    # Hyperparameters path
    parser.add_argument(
        "--hyp_path",
        type=str,
        default="yolo_dataset/hyp.yaml",
        help="Path to the hyperparameters file.",
    )
    args = parser.parse_args()
    frame_dir_path = f"{args.data_dir}/frames"
    yolo_db_dir = args.yolo_db_dir
    yolo_labels = f"{args.data_dir}/yolo_raw_labels"

    run(
        video_dir=args.video_dir,
        data_dir=args.data_dir,
        csv_dir=args.csv_dir,
        yolo_db_dir=yolo_db_dir,
        frame_dir_path=frame_dir_path,
        yolo_labels=yolo_labels,
        body_config_path=args.body_config_path,
        head_config_path=args.head_config_path,
        hyp_path=args.hyp_path,
    )
