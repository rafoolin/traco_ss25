import os
import pandas as pd
from PIL import Image

# === Config ===
FRAME_ROOT = "./data/frames"
CSV_ROOT = "./traco_2024/training"
LABEL_ROOT = "./data/yolo_labels_head"
os.makedirs(LABEL_ROOT, exist_ok=True)

# Config
BOX_SIZE = 300
ANCHOR_RATIO = 0.25  # head should be at 25% from top-left corner

# Shift calculation
SHIFT_X = BOX_SIZE * ANCHOR_RATIO
SHIFT_Y = BOX_SIZE * ANCHOR_RATIO

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


# === Iterate through videos ===
for video_name in sorted(os.listdir(FRAME_ROOT)):
    video_path = os.path.join(FRAME_ROOT, video_name)
    csv_path = os.path.join(CSV_ROOT, f"{video_name}.csv")
    label_path = os.path.join(LABEL_ROOT, video_name)
    os.makedirs(label_path, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"⚠️ CSV not found for {video_name}, skipping...")
        continue

    print(f"🔄 Processing {video_name}")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "row_index"})  # If needed
    grouped = df.groupby("t")
    dx = dy = 0

    for frame_file in sorted(os.listdir(video_path)):
        if not frame_file.endswith(".jpg"):
            continue

        frame_index = int(frame_file.split("_")[1].split(".")[0])
        frame_path = os.path.join(video_path, frame_file)
        image = Image.open(frame_path)
        w, h = image.size

        label_lines = []

        if frame_index not in grouped.groups:
            print(f"  ⚠️ Frame {frame_index} not in CSV, skipping...")
            continue

        frame_data = grouped.get_group(frame_index)

        for _, row in frame_data.iterrows():
            cx, cy = row["x"], row["y"]
            if video_name in problematic_videos:
                cx, cy = row["y"], row["x"]
            hexbug_id = int(row["hexbug"])

            # Calculate bounding box coordinates
            x_min = max(0, cx - SHIFT_X)
            y_min = max(0, cy - SHIFT_Y)
            x_max = min(w, x_min + BOX_SIZE)
            y_max = min(h, y_min + BOX_SIZE)

            # Now recompute actual width and height in case box got cut at the edges
            actual_bw = x_max - x_min
            actual_bh = y_max - y_min

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / w
            y_center = (y_min + y_max) / 2 / h
            bw = actual_bw / w
            bh = actual_bh / h

            n_cx = cx / w
            n_cy = cy / h

            label_lines.append(
                f"{hexbug_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f} {n_cx:.6f} {n_cy:.6f} 2"
            )

        # Save label file
        out_label_file = os.path.join(label_path, frame_file.replace(".jpg", ".txt"))
        with open(out_label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

    print(f"✅ Done: {video_name}")
