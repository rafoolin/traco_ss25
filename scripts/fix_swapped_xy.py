import os
import pandas as pd


def swap_xy_in_csv(src_path):
    if not os.path.exists(src_path):
        print(f"[skip] not found: {src_path}")
        return False

    df = pd.read_csv(src_path)
    if "x" not in df.columns or "y" not in df.columns:
        print(f"[skip] missing x/y columns: {src_path}")
        return False

    # swap columns
    df["x"], df["y"] = df["y"], df["x"]
    df.to_csv(src_path, index=False)
    return True


if __name__ == "__main__":
    csv_dir = "./traco_2024/training"

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
    for name in problematic_videos:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        swap_xy_in_csv(csv_path)
        print("Successfully swapped x/y in:", csv_path)
    print("All problematic CSV files processed.")
