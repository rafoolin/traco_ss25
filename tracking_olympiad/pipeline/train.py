import os
from ultralytics import YOLO
import yaml

from pipeline.utils.logger import setup_logger

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
        model = YOLO(self.model_path)

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
