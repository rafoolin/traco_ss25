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
python tracking_olympiad/pipeline/train.py \
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
python tracking_olympiad/pipeline/detector.py \
--video_path "./traco_2024/leaderboard_data/test001.mp4" \
--primary_model_path "./yolo_dataset/runs/pose/body_train/weights/best.pt" \
--fallback_model_path "./yolo_dataset/runs/pose/head_train/weights/best.pt" \
--primary_thresh 0.5 \
--fallback_thresh 0.5 \
--csv_out_dir "./output/detector_csv_files" \
--annotate_results \
--annotate_dir "./output/annotations"
```