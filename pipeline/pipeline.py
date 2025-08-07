import argparse
import os
import logging

from preprocess.extract_frames import extract_frames
from preprocess.bounding_box_body import BoundingBoxBody
from utils.logger import setup_logger
from preprocess.generate_yolo_dataset import create_yolo_dataset

logger = setup_logger()


def run_pipeline(args: argparse.Namespace):
    frame_dir_path = f"{args.data_dir}/frames"
    yolo_db_dir = args.yolo_db_dir

    # Run the pipeline steps
    logger.info("Starting pipeline...")
    # Step 1: Extract frames from videos
    logger.info("Extracting frames...")
    if os.path.exists(frame_dir_path):
        logger.info("Frame directory already exists. Skipping extraction.")
    else:
        extract_frames(video_dir=args.video_dir, output_dir=frame_dir_path)
        logger.info("Frames extracted successfully!")
    # Step 2: Run bounding box script for body detection
    logger.info("Running bounding box body script...")
    yolo_labels = f"{args.data_dir}/yolo_raw_labels_body"
    bbox_body = BoundingBoxBody(
        frame_dir_path=frame_dir_path,
        csv_dir_path=args.csv_dir,
        data_dir_path=args.data_dir,
        yolo_dir_path=yolo_labels,
    )
    bbox_body.generate_bounding_boxes_body()
    logger.info("Bounding box body script executed successfully!")
    # Step 3: Run bounding box script for head detection

    # Step 4: Run Yolo dataset generator for Body
    create_yolo_dataset(
        frame_dir=frame_dir_path,
        label_dir=yolo_labels,
        img_out=f"{yolo_db_dir}/images_body",
        lbl_out=f"{yolo_db_dir}/label_body",
    )
    logger.info("Pipeline completed successfully!")


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
    args = parser.parse_args()
    run_pipeline(args)
