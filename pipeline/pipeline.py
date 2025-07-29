import argparse
import os
from preprocess.extract_frames import extract_frames
from preprocess.bounding_box_body import BoundingBoxBody

COLOR = "\033[1;33m"
RESET = "\033[0m"


def run_pipeline(args: argparse.Namespace):
    frame_dir_path = f"{args.data_dir}/frames"
    # Run the pipeline steps
    print(f"{COLOR}Starting pipeline...{RESET}")
    # Step 1: Extract frames from videos
    print(f"{COLOR}Extracting frames...{RESET}")
    if os.path.exists(frame_dir_path):
        print(f"{COLOR}Frame directory already exists. Skipping extraction.{RESET}")
    else:
        extract_frames(video_dir=args.video_dir, output_dir=frame_dir_path)
        print(f"{COLOR}Frames extracted successfully!{RESET}")
    # Step 2: Run bounding box script for body detection
    print(f"{COLOR}Running bounding box body script...{RESET}")
    bbox_body = BoundingBoxBody(
        frame_dir_path=frame_dir_path,
        csv_dir_path=args.csv_dir,
        data_dir_path=args.data_dir,
        yolo_dir_path=f"{args.data_dir}/yolo_labels_body",
    )
    bbox_body.generate_bounding_boxes_body()
    print(f"{COLOR}Bounding box body script executed successfully!{RESET}")
    # Step 3: Run bounding box script for head detection
    print(f"{COLOR}Pipeline completed successfully!{RESET}")


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
    args = parser.parse_args()
    run_pipeline(args)
