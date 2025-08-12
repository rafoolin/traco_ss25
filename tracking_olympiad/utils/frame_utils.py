import csv
import os

import pandas as pd


def read_csv_frames(csv_path: str) -> dict:
    """
    Reads a CSV file containing frame data and groups the rows by the 't' column.

    Args:
        csv_path (str): Path to the CSV file to be read.

    Returns:
        dict: A dictionary where keys are frame indices and values are lists of dictionaries
            containing 'row_index', 'hexbug', 'x', and 'y' for each frame.
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
    """Save missing frame data to a text file."""
    with open(
        f"{annotation_dir_path}/missing_frames.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"-{video_name}:\n\t{missing_data}\n")


def write_hexbug_head_to_csv(rows, save_path):
    """Writes hexbug head detection results to a CSV file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "t", "hexbug", "x", "y"])
        w.writerows(rows)
    return save_path
