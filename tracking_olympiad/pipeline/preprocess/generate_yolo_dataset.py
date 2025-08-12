"""Generate a YOLO dataset from video frames and raw Yolo labels."""

import os
import shutil
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from pipeline.utils.logger import setup_logger
from pipeline.utils.file_utils import reset_dir

logger = setup_logger()


def create_yolo_dataset(
    frame_dir: str,
    label_dir: str,
    img_out: str,
    lbl_out: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Create a YOLO dataset from frames and labels.

    Args:
        frame_dir (str): Directory containing video frames.
        label_dir (str): Directory containing YOLO labels.
        img_out (str): Output directory for images.
        lbl_out (str): Output directory for labels.
        val_ratio (float): Proportion of data to use for validation.
        seed (int): Random seed for reproducibility.
    """
    logger.info("Starting YOLO dataset preprocessing...")

    frame_to_labels = _collect_valid_labeled_frames(frame_dir, label_dir)
    train_keys, val_keys = train_test_split(
        list(frame_to_labels.keys()),
        test_size=val_ratio,
        random_state=seed,
    )

    # Remove the folders if already exist
    for split in ["train", "val"]:
        reset_dir(os.path.join(img_out, split))
        reset_dir(os.path.join(lbl_out, split))

    _write_split(train_keys, frame_to_labels, img_out, lbl_out, "train")
    _write_split(val_keys, frame_to_labels, img_out, lbl_out, "val")

    logger.info(
        "Done: %d training and %d validation images written.",
        len(train_keys),
        len(val_keys),
    )


def _collect_valid_labeled_frames(
    frame_dir: str, label_dir: str
) -> Dict[Tuple[str, str, str], List[str]]:
    """Maps (image_path, flat_img_name, flat_lbl_name) → list of YOLO label lines."""
    frame_to_labels = {}

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        video_name = label_file[:-4]
        video_img_dir = os.path.join(frame_dir, video_name)
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    logger.warning(
                        "Skipping line with unexpected format: %s",
                        line.strip(),
                    )
                    continue

                frame_idx = int(parts[0])
                frame_name = f"frame_{frame_idx:05d}.jpg"
                img_path = os.path.join(video_img_dir, frame_name)

                if not os.path.exists(img_path):
                    continue

                flat_img_name = f"{video_name}_{frame_name}"
                flat_lbl_name = flat_img_name.replace(".jpg", ".txt")
                key = (img_path, flat_img_name, flat_lbl_name)

                if key not in frame_to_labels:
                    frame_to_labels[key] = []

                frame_to_labels[key].append(" ".join(parts[1:]))

    valid = {k: v for k, v in frame_to_labels.items() if v}
    logger.info("Valid labeled frames collected: %d", len(valid))
    return valid


def _write_split(
    keys: List[Tuple[str, str, str]],
    frame_to_labels: Dict[Tuple[str, str, str], List[str]],
    img_out_dir: str,
    lbl_out_dir: str,
    split: str,
):
    for img_path, flat_img_name, flat_lbl_name in keys:
        shutil.copy(img_path, os.path.join(img_out_dir, split, flat_img_name))
        with open(
            os.path.join(lbl_out_dir, split, flat_lbl_name), "w", encoding="utf-8"
        ) as f:
            for label in frame_to_labels[(img_path, flat_img_name, flat_lbl_name)]:
                f.write(label + "\n")
