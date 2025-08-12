import os
import argparse
import cv2
from PIL import Image

from utils.frame_utils import write_hexbug_head_to_csv
from utils.mask_utils import extract_xy
from utils.media import open_video
from ultralytics import YOLO
from utils.logger import setup_logger
from utils.draw_utils import draw_head_and_bbox, save_annotated_image

logger = setup_logger()


def _run_yolo_model(model, frame, conf_thresh: float):
    """
    Run a YOLO model on a single frame.
    Returns a list of (x_center, y_center, conf, ((x1,y1),(x2,y2))) sorted by conf desc.
    """
    if model is None:
        return []
    res_list = model(frame, conf=conf_thresh, verbose=False)
    if not res_list:
        return []

    res = res_list[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    keypoints = getattr(res, "keypoints", None)
    confs = boxes.conf.cpu().numpy()
    out = []
    for i in range(len(boxes)):
        conf = float(confs[i])
        x_center, y_center = extract_xy(boxes, keypoints, i)
        x_min, y_min, x_max, y_max = boxes.xyxy[i].cpu().numpy()
        out.append(
            (
                x_center,
                y_center,
                conf,
                ((x_min, y_min), (x_max, y_max)),
            )
        )
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def _choose_detections(
    primary_dets,
    fallback_dets,
    primary_thresh: float,
    fallback_thresh: float,
):
    """
    Choose detections based on confidence thresholds.
    Each detection is expected as:
        (x_center, y_center, conf, bbox)
    where bbox = ((x_min, y_min), (x_max, y_max))
    """
    max_p = primary_dets[0][2] if primary_dets else 0.0
    max_f = fallback_dets[0][2] if fallback_dets else 0.0

    if max_p >= primary_thresh:
        return [
            (x, y, c, bbox) for (x, y, c, bbox) in primary_dets if c >= primary_thresh
        ]
    if max_f >= max_p:
        return [
            (x, y, c, bbox) for (x, y, c, bbox) in fallback_dets if c >= fallback_thresh
        ]
    return [(x, y, c, bbox) for (x, y, c, bbox) in primary_dets if c >= primary_thresh]


def _process_video(
    video_path: str,
    primary_model,
    fallback_model,
    annotate_dir: str,
    primary_thresh: float = 0.55,
    fallback_thresh: float = 0.45,
    annotate_results: bool = False,
):
    """Process a video to extract hexbug head positions.
    Returns a list of rows [row_id, frame_id, -1, x, y].
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = open_video(video_path)
    rows = []
    row_id = 0
    frame_id = 0

    while True:
        logger.info("Processing frame %d...", frame_id)
        ok, frame = cap.read()
        if not ok:
            break

        p_dets = _run_yolo_model(primary_model, frame, conf_thresh=primary_thresh)
        f_dets = []
        if (not p_dets) or (p_dets and p_dets[0][2] < primary_thresh):
            f_dets = (
                _run_yolo_model(fallback_model, frame, conf_thresh=fallback_thresh)
                if fallback_model
                else []
            )

        chosen = _choose_detections(p_dets, f_dets, primary_thresh, fallback_thresh)
        for x, y, _, bbox in chosen:
            rows.append([row_id, frame_id, -1, x, y, bbox])
            row_id += 1
        if annotate_results:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            annotated_image = draw_head_and_bbox(
                image=image,
                masks=[
                    {
                        "bbox": bbox,
                        "x": x,
                        "y": y,
                        "hexbug": "-",
                    }
                    for x, y, _, bbox in chosen
                    if bbox is not None
                ],
            )
            save_annotated_image(
                annotation_dir_path=annotate_dir,
                annotated_image=annotated_image,
                video_name=video_name,
                frame_index=frame_id,
            )
        frame_id += 1
    cap.release()

    return rows[:-1]


def detect_hexbugs_head(
    video_path: str,
    primary_model_path: str,
    csv_out_dir: str,
    annotate_dir: str,
    primary_thresh: float = 0.55,
    fallback_model_path: str = None,
    fallback_thresh: float = 0.45,
    annotate_results: bool = False,
):
    """Detect hexbug heads in a video using YOLO models.
    Saves results to a CSV file in the specified output directory.
    """
    primary = YOLO(primary_model_path)
    fallback = YOLO(fallback_model_path)

    rows = _process_video(
        video_path=video_path,
        primary_model=primary,
        fallback_model=fallback,
        primary_thresh=primary_thresh,
        fallback_thresh=fallback_thresh,
        annotate_results=annotate_results,
        annotate_dir=annotate_dir,
    )
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(csv_out_dir, f"predicted_{video_name}.csv")
    write_hexbug_head_to_csv(rows, save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detect hexbug heads in a video.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument(
        "--primary_model_path", type=str, help="Path to the primary YOLO model."
    )
    parser.add_argument(
        "--csv_out_dir",
        type=str,
        help="Directory to save the output CSV file.",
    )
    parser.add_argument(
        "--fallback_model_path",
        type=str,
        default=None,
        help="Path to the fallback YOLO model.",
    )
    parser.add_argument(
        "--primary_thresh",
        type=float,
        default=0.55,
        help="Confidence threshold for primary model detections.",
    )
    parser.add_argument(
        "--fallback_thresh",
        type=float,
        default=0.45,
        help="Confidence threshold for fallback model detections.",
    )
    parser.add_argument(
        "--annotate_results",
        action="store_true",
        help="Whether to annotate the video with detection results.",
    )
    parser.add_argument(
        "--annotate_dir",
        type=str,
        help="Where to save detection results annotations",
    )

    args = parser.parse_args()
    os.makedirs(args.csv_out_dir, exist_ok=True)
    logger.info("Processing video: %s", args.video_path)
    detect_hexbugs_head(
        video_path=args.video_path,
        annotate_results=args.annotate_results,
        annotate_dir=args.annotate_dir,
        csv_out_dir=args.csv_out_dir,
        fallback_model_path=args.fallback_model_path,
        fallback_thresh=args.fallback_thresh,
        primary_model_path=args.primary_model_path,
        primary_thresh=args.primary_thresh,
    )
    logger.info("Detection results saved to %s", args.csv_out_dir)
    logger.info("Detection completed successfully.")
