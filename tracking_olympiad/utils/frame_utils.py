import csv
import os

import pandas as pd


def read_csv_frames(csv_path: str) -> dict:
    """
    Reads frame data from a CSV and groups rows by frame index.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        dict: Mapping of frame index (int) → list of dicts with keys:
            'row_index', 'hexbug', 'x', 'y'.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "row_index"})
    grouped = {}
    for t, group in df.groupby("t"):
        grouped[int(t)] = group[["row_index", "hexbug", "x", "y"]].to_dict(
            orient="records"
        )
    return grouped


def save_missing_frames(missing_data, video_name, annotation_dir_path):
    """
    Saves indices of missing frames to a text file.

    Args:
        missing_data (list): List of missing frame indices.
        video_name (str): Name of the video.
        annotation_dir_path (str): Directory where the file will be saved.
    """
    with open(
        f"{annotation_dir_path}/missing_frames.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"-{video_name}:\n\t{missing_data}\n")


def write_hexbug_head_to_csv(rows, save_path):
    """
    Writes hexbug head detection results to a CSV file.

    Args:
        rows (list): List of detection rows, each formatted as [row_index, frame_index, hexbug_id, x, y].
        save_path (str): Path where the CSV file will be saved.

    Returns:
        str: Path to the saved CSV file.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "t", "hexbug", "x", "y"])
        w.writerows(rows)
    return save_path
