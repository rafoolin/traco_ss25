# track_assign_ids.py
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from utils.media import open_video
from utils.logger import setup_logger


logger = setup_logger()


class TrackerConfig:
    def __init__(
        self,
        patch_size=50,
        distance_weight=1.0,
        color_weight=0.1,
        max_cost=200.0,
        max_missing_frames=3,
        max_ids=11,
    ):
        self.patch_size = int(patch_size)
        self.distance_weight = float(distance_weight)
        self.color_weight = float(color_weight)
        self.max_cost = float(max_cost)
        self.max_missing_frames = int(max_missing_frames)
        self.max_ids = int(max_ids)


def _extract_color(frame, x, y, patch=50):
    """Mean BGR color in a square patch around (x, y)."""
    h, w = frame.shape[:2]
    x1 = max(int(x) - patch, 0)
    y1 = max(int(y) - patch, 0)
    x2 = min(int(x) + patch, w - 1)
    y2 = min(int(y) + patch, h - 1)
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return sub.mean(axis=(0, 1)).astype(np.float32)  # (B,G,R)


def _get_frame(cap, t, cache):
    """Random-access a specific frame index (t) with small cache."""
    if t not in cache:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read()
        if not ok:
            return None
        cache[t] = frame
    return cache[t]


def _assign_ids_for_video(
    video_path: str,
    detections_csv_path: str,
    output_csv_path: str,
    cfg: TrackerConfig,
):
    """
    Assigns consistent object IDs to detections across video frames using position and color similarity.

    Args:
        video_path (str): Path to the input video file.
        detections_csv_path (str): Path to the CSV file containing frame-by-frame detections (with x, y coordinates).
        output_csv_path (str): Path to save the updated CSV with assigned IDs.
        cfg (TrackerConfig): Tracker configuration containing patch size, weights, cost thresholds, and limits.

    Returns:
        str: Path to the saved output CSV file containing updated detection data with assigned IDs.
    """
    # Load detections
    df = pd.read_csv(detections_csv_path)
    if not {"t", "x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: t, x, y (and optionally hexbug).")
    if "hexbug" not in df.columns:
        df["hexbug"] = -1

    # Video + cache
    cap = open_video(video_path)
    frame_cache = {}

    next_id = 0
    active_tracks = {}  # id -> {'pos': (x,y), 'color': (BGR), 'last_seen_t': t}

    # Process in frame order
    for t in sorted(df["t"].unique()):
        frame = _get_frame(cap, t, frame_cache)
        if frame is None:
            continue

        detections = df[df["t"] == t].copy()
        if detections.empty:
            continue

        curr_pos = detections[["x", "y"]].values.astype(np.float32)
        curr_col = np.array(
            [_extract_color(frame, x, y, patch=cfg.patch_size) for x, y in curr_pos],
            dtype=np.float32,
        )

        assigned = [-1] * len(detections)

        # Split active vs inactive tracks by recency
        act_ids, act_pts, act_col = [], [], []
        inact_ids, inact_pts, inact_col = [], [], []

        for tid, tr in active_tracks.items():
            if t - tr["last_seen_t"] <= cfg.max_missing_frames:
                act_ids.append(tid)
                act_pts.append(tr["pos"])
                act_col.append(tr["color"])
            else:
                inact_ids.append(tid)
                inact_pts.append(tr["pos"])
                inact_col.append(tr["color"])

        all_ids = act_ids + inact_ids
        all_pts = (
            np.array(act_pts + inact_pts, dtype=np.float32)
            if (act_pts or inact_pts)
            else np.zeros((0, 2), np.float32)
        )
        all_col = (
            np.array(act_col + inact_col, dtype=np.float32)
            if (act_col or inact_col)
            else np.zeros((0, 3), np.float32)
        )

        # Primary assignment (Hungarian) using distance + color
        if len(all_pts) > 0 and len(curr_pos) > 0:
            d_pos = np.linalg.norm(
                curr_pos[:, None, :] - all_pts[None, :, :], axis=2
            )  # (N,M)
            d_col = np.linalg.norm(
                curr_col[:, None, :] - all_col[None, :, :], axis=2
            )  # (N,M)
            cost = cfg.distance_weight * d_pos + cfg.color_weight * d_col  # (N,M)

            rows, cols = linear_sum_assignment(cost)
            used_ids = set()
            for r, c in zip(rows, cols):
                if cost[r, c] < cfg.max_cost and all_ids[c] not in used_ids:
                    assigned[r] = all_ids[c]
                    used_ids.add(all_ids[c])
                    active_tracks[all_ids[c]] = {
                        "pos": curr_pos[r],
                        "color": curr_col[r],
                        "last_seen_t": int(t),
                    }

        # Create new tracks for remaining detections (respect max_ids)
        for i, aid in enumerate(assigned):
            if aid == -1 and next_id < cfg.max_ids:
                assigned[i] = next_id
                active_tracks[next_id] = {
                    "pos": curr_pos[i],
                    "color": curr_col[i],
                    "last_seen_t": int(t),
                }
                next_id += 1

        # Write back
        df.loc[detections.index, "hexbug"] = assigned

    cap.release()
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def run(
    video_path: str,
    csv_in: str,
    csv_out: str,
    patch_size: int,
    distance_weight: float,
    color_weight: float,
    max_cost: int,
    max_missing_frames: int,
    max_ids: int,
):
    """
    Runs the Hexbug ID assignment pipeline.

    Args:
        video_path (str): Path to the input video file.
        csv_in (str): Path to the detections CSV (with initial hexbug=-1).
        csv_out (str): Path to save the updated CSV with assigned IDs.
        patch_size (int): Patch size for mean color extraction.
        distance_weight (float): Weight for position distance in cost function.
        color_weight (float): Weight for color distance in cost function.
        max_cost (float): Maximum allowed matching cost.
        max_missing_frames (int): Frames to keep a track alive without detection.
        max_ids (int): Maximum number of unique IDs to assign.
    """
    cfg = TrackerConfig(
        patch_size=patch_size,
        distance_weight=distance_weight,
        color_weight=color_weight,
        max_cost=max_cost,
        max_missing_frames=max_missing_frames,
        max_ids=max_ids,
    )

    out = _assign_ids_for_video(
        video_path=video_path,
        detections_csv_path=csv_in,
        output_csv_path=csv_out,
        cfg=cfg,
    )
    logger.info("Saved to %s", out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Assign stable IDs to hexbug detections."
    )
    parser.add_argument(
        "--video_path", required=True, type=str, help="Path to the video."
    )
    parser.add_argument(
        "--csv_in",
        required=True,
        type=str,
        help="Path to detections CSV (hexbug may be -1).",
    )
    parser.add_argument(
        "--csv_out", required=True, type=str, help="Path to save CSV with assigned IDs."
    )
    parser.add_argument("--patch_size", type=int, default=50)
    parser.add_argument("--distance_weight", type=float, default=1.0)
    parser.add_argument("--color_weight", type=float, default=0.1)
    parser.add_argument("--max_cost", type=float, default=200.0)
    parser.add_argument("--max_missing_frames", type=int, default=3)
    parser.add_argument("--max_ids", type=int, default=11)
    args = parser.parse_args()

    run(
        video_path=args.video_path,
        color_weight=args.color_weight,
        csv_in=args.csv_in,
        csv_out=args.csv_out,
        distance_weight=args.distance_weight,
        max_cost=args.max_cost,
        max_ids=args.max_ids,
        max_missing_frames=args.max_missing_frames,
        patch_size=args.patch_size,
    )
