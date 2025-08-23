import os
import cv2
import pandas as pd
import argparse


def annotate_video(video_path, csv_path, output_dir):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create subdirectory inside output_dir using video_name
    output_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: " "})
    print("📄 CSV columns:", df.columns.tolist())
    df["t"] = df["t"].astype(int)

    # Load video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"❌ Could not open video: {video_path}"

    frame_id = 0
    frame_map = {}
    for t, group in df.groupby("t"):
        frame_map[t] = group.to_dict(orient="records")

    # Annotate frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_map:
            hexbugs = frame_map[frame_id]
            for item in hexbugs:
                x, y = int(float(item["x"])), int(float(item["y"]))
                hexbug_id = int(item["hexbug"])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"id {hexbug_id}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        out_path = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_id += 1

    cap.release()
    print("✅ All annotated frames saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate video frames using CSV data."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Full path to the video file"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Full path to the CSV annotation file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save annotated frames"
    )

    args = parser.parse_args()

    annotate_video(
        video_path=args.video_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
    )