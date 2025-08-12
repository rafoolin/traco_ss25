# Tracking Olympiad

## Dataset

The dataset contains video frames and their corresponding annotations in CSV format.

- Each video is stored in its own directory under the dataset root.
- For every video directory, there is a matching CSV file (same name as the directory) containing the frame-by-frame annotations.

**NOTE:** Known Issue: Swapped X/Y Coordinates

Some training videos have the x and y coordinates swapped in their CSV annotations.
If these videos are still present in the dataset, you can correct them by running:

```bash
python scripts/fix_swapped_xy.py
```

The affected videos are:

```text
problematic_videos = [
        "training020",
        "training058",
        "training076",
        "training078",
        "training079",
        "training082",
        "training083",
        "training089",
    ]
```

## Setup environment

Before running training or detection, set up the environment and required packages by running:

```bash
bash run.sh
```

This will:

1. Clone all required external repositories (e.g., Traco, SAM2).
2. Create and configure the conda environment and activate it.
3. Install all Python dependencies from requirements.txt
4. Install the SAM2 model.
5. Extract frames from the training videos and save them in the `data` directory.
6. Prepare the YOLO dataset for training by extracting frames and annotations from the CSV files.
7. Create the necessary directories for YOLO dataset structure.

You’re now ready to proceed to the training or detection steps.

## Training (head & body)

Train both models (separate dataset YAMLs; same hyp).

```bash
python tracking_olympiad/train.py \
--video_dir "./traco_2024/training" \
--csv_dir "./traco_2024/training" \
--data_dir "./data" \
--yolo_db_dir "./yolo_dataset/datasets" \
--body_config_path "./yolo_dataset/configs/hexbug_body.yaml" \
--head_config_path "./yolo_dataset/configs/hexbug_head.yaml" \
--hyp_path "./yolo_dataset/hyp.yaml"
```

- Uses: `configs/hexbug_head.yaml`, `configs/hexbug_body.yaml`, `hyp.yaml`
- Produces weights in `runs/pose/{head,body}_train/weights/best.pt`

## Detection (head & body)

Export detected Hexbugs for each frame with `primary` & `fallback` model selection:

- If `primary` has any detection above `primary_thresh`, use it;
- Otherwise use whichever model has higher max confidence, each filtered by its own threshold.
- Detector output `CSV` sets `hexbug=-1` (we do ID assignment later).

To run detection, use:

```bash
python tracking_olympiad/detector.py \
--video_path "./traco_2024/leaderboard_data/test001.mp4" \
--primary_model_path "./yolo_dataset/runs/pose/body_train/weights/best.pt" \
--fallback_model_path "./yolo_dataset/runs/pose/head_train/weights/best.pt" \
--primary_thresh 0.5 \
--fallback_thresh 0.5 \
--csv_out_dir "./output/detector_csv_files" \
--annotate_results \
--annotate_dir "./output/annotations"
```


## Hexbug Tracker — ID Assignment

This step assigns stable IDs to detected Hexbug positions in each frame of a video.
It is intended to run after the detection step in the pipeline and will read the detection CSV (with hexbug=-1), use appearance + position matching to keep consistent IDs across frames, and output an updated CSV.

**Input:**

- A .mp4 video file.
- A CSV file with detection results from the detection stage.

The CSV must contain:

```text
t,x,y,hexbug
```

- t: frame index
- x,y: detection coordinates
- hexbug: initially -1 for all rows

**Output:**

- A CSV file with the same columns, but hexbug replaced by stable IDs in the range [0, max_ids-1].

**How It Works**:

- Hungarian algorithm assigns detections to existing tracks based on:
- Euclidean distance between positions.
- Mean BGR color extracted from a square patch around each detection.
- Tracks are kept alive for a few frames (max_missing_frames) to handle short occlusions.
- If a detection does not match an existing track and there are free IDs available, a new track is created.
- Color helps distinguish Hexbugs that are spatially close.

```bash
python ./tracking_olympiad/tracker.py \
--video_path ./traco_2024/leaderboard_data/test001.mp4 \
--csv_in ./output/detector_csv_files/predicted_test001.csv \
--csv_out output/detector_csv_files/predicted_test001_ids.csv \
--patch_size 50 \
--distance_weight 1.0 \
--color_weight 0.1 \
--max_cost 200 \
--max_missing_frames 3 \
--max_ids 3
```