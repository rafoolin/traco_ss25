![Python](https://img.shields.io/badge/python-3.10-blue)  

# Tracking Olympiad

This project is part of the [Tracking Olympiad seminar](https://traco.anki.xyz/) at FAU.

This repository contains the full pipeline for detecting and tracking multiple Hexbugs in videos.  
The process is split into **modular steps** so you can run or debug each stage independently.

## Implementation Details

Training and evaluation were performed on the **tinyGPU FAU HPC cluster** using an NVIDIA A100-SXM4-40GB GPU with Slurm job scheduling.  

### Hardware

- GPU: NVIDIA A100-SXM4-40GB  
- Runtime (body model): ~3h 45m  
- Runtime (head model): ~2h 20m  
- Max GPU memory usage: 11 GB (body), 2 GB (head)  
- System RAM usage: up to 19 GB  

## 📂 Dataset

The dataset contains video frames and their corresponding annotations in CSV format from [traco_2024](https://github.com/ankilab/traco_2024) repository.

- Each video is stored under the dataset root.
- For every video, there is a matching CSV file (same name as the video) containing the frame-by-frame annotations.

There is a bash file that clones these repositories and set the projects up.

### ⚠️ Known Issue — Swapped X/Y

Some videos have swapped coordinates in their CSVs.  
If you still have these in your dataset, first set the project up and then fix them by running:

```bash
python scripts/fix_swapped_xy.py
```

Affected videos:

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

## ⚙️ Environment Setup

Before training or running detection, set up the environment:

```bash
bash run.sh
```

This script will:

1. **Clone** required repositories (e.g., Traco, SAM2).
2. **Create & activate** the conda environment.
3. **Install** all Python dependencies from `requirements.txt`.
4. **Download & install** the SAM2 model(For segmentation in order to detect bounding box around HexBugs).
5. **Extract frames** from training videos.
6. **Prepare** the YOLO images and labels datasets.

After this, you are ready for training or detection.

⚠️ **CAUTION:** ⚠️

Make sure you have correctly activated the conda environment called `traco_env`.

## 🎯 Bounding Box Generation

This step uses the **SAM2 segmentation model** to generate bounding boxes for Hexbug **bodies** and **heads** from annotated CSV coordinates.

**How it works:**

- Reads the CSV annotations for each frame each training videos.
- Uses SAM2 to generate precise masks for Hexbugs.
- Converts masks into bounding boxes for **body** and fixed-size boxes for **head**.
- Saves annotated images and YOLO-format label files for later YOLO training.

## 🏋️ Training (Head & Body Models)

We train **two separate YOLOv8-pose models**:

- **Body model**: detects Hexbug bodies.
- **Head model**: detects Hexbug heads.

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

**This will**:

- **Use** `configs/hexbug_head.yaml`, `configs/hexbug_body.yaml`, `hyp.yaml`
- **Produce** weights in `runs/pose/{head,body}_train/weights/best.pt`

**Trained Models:**

I have included the trained YOLO models in `yolo_dataset/models` so you can run detection directly without retraining.  
Just make sure to update the `--primary_model_path` and `--fallback_model_path` arguments in the detection command above to point to the correct model files.

## 🔍 Detection

This step detects Hexbugs in each frame.

Uses **primary** and **fallback** models:

- If `primary` finds a detection ≥ `primary_thresh`, use it.

- Otherwise, use whichever model has the highest confidence.

**Run detection**:

```bash
python tracking_olympiad/detector.py \
--video_path "./traco_2024/leaderboard_data/test001.mp4" \
--primary_model_path "./yolo_dataset/runs/pose/body_train/weights/best.pt" \
--fallback_model_path "./yolo_dataset/runs/pose/head_train/weights/best.pt" \
--primary_thresh 0.40 \
--fallback_thresh 0.42 \
--csv_out_dir "./output/detector_csv_files" \
--annotate_results \
--annotate_dir "./output/annotations"
```

**Output**:

- CSV file with:

```text
  t, hexbug, x, y
```

(IDs are set to `-1` here —> ID assignment happens in the next step)

- Optional annotated images in `--annotate_dir`.

## 🎯 Tracking & ID Assignment

This script assigns **consistent IDs** to Hexbug detections across video frames.  
It reads a detection CSV (from the detector stage) and the corresponding video,  
then uses **position** and **color similarity** to match detections over time.

**How it works:**

- For each frame, extract `(x, y)` positions from the CSV and mean BGR color from the video.
- Match detections to previous tracks using the **Hungarian algorithm** with a cost based on distance and color.
- Keep tracks alive for a few frames to handle short occlusions.
- Assign new IDs when needed, never exceeding the `max_ids` limit.

**Run tracker**:

```bash
python ./tracking_olympiad/tracker.py \
--video_path ./traco_2024/leaderboard_data/test001.mp4 \
--csv_in ./output/detector_csv_files/predicted_test001.csv \
--csv_out output/detector_csv_files/predicted_test001_ids.csv \
--patch_size 250 \
--distance_weight 0.7 \
--color_weight 0.1 \
--max_cost 300.0 \
--max_missing_frames 3 \
--max_ids 3
```

**Output**:

- CSV with same columns as detection CSV, but `hexbug` now contains **stable IDs** `[0 .. max_ids-1]`.

## ⚠️ Important Note on Tracking Algorithm

The **training** and **preprocessing** steps for detecting bounding boxes (SAM2 + YOLO detector) are solid and work reliably.

However, the **tracking** algorithm provided here is **not highly accurate**.  
I experimented with both my custom tracker (based on Hungarian matching with position & color features) and existing trackers like **DeepSORT** and **ByteTrack**, but the results were still not robust enough for competitive performance.

**Recommendation:**  
If you plan to build upon this repository, consider replacing or improving the tracking stage with a more advanced, competition-ready tracker.

## Note

You can modify the tracking parameters (e.g., `patch_size`, `distance_weight`, `color_weight`) in the `tracker.py` script to better suit your specific video and detection characteristics. This is also true for training parameters in `train.py`.

## 📌 Full Pipeline

1. **Setup** → `bash run.sh`
2. **Train models** → `train.py`
3. **Detect** → `detector.py`
4. **Track / Assign IDs** → `tracker.py`

## 📝 Annotate Results

```bash
python scripts/annotate_result.py \
--video_path ./traco_2024/leaderboard_data/test001.mp4 \
--csv_path output/detector_csv_files/predicted_test001_ids.csv \
--output_dir output/cvs_annotations
```

## 📚 References

- **Tracking Olympiad (TRACO)**: [https://traco.anki.xyz/](https://traco.anki.xyz/)
- **YOLOv8**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)  
- **Segment Anything Model 2 (SAM2)**: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)  
- **DeepSORT**: [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort)  
- **ByteTrack**: [https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)  
- **Hungarian Algorithm (linear_sum_assignment)**: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)  
