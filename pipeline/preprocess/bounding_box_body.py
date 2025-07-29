import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


class BoundingBoxBody:
    def __init__(self, frame_dir_path, csv_dir_path, data_dir_path, yolo_dir_path):
        self.frame_dir_path = frame_dir_path
        self.frame_data = []
        self.device = self._get_device()
        self.sam2_predictor = self._load_sam2()
        self.csv_dir_path = csv_dir_path
        self.annotation_dir_path = os.path.join(data_dir_path, "annotations_body")
        self.yolo_dir_path = yolo_dir_path
        os.makedirs(self.annotation_dir_path, exist_ok=True)
        os.makedirs(self.yolo_dir_path, exist_ok=True)

    # Note: This function is adapted from [sam notebooks](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)
    def _get_device(self) -> torch.device:
        """
        Determines and configures the best available PyTorch device (CUDA, MPS, or CPU) for computation.

        Returns:
            torch.device: The selected device object.
        """
        current_device = None
        if torch.cuda.is_available():
            current_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            current_device = torch.device("mps")
        else:
            current_device = torch.device("cpu")
        print(f"using device: {current_device}")

        if current_device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            elif current_device.type == "mps":
                print(
                    "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                    "give numerically different outputs and sometimes degraded performance on MPS. "
                    "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
                )
        return current_device

    def _load_sam2(self) -> SAM2ImagePredictor:
        """
        Loads the SAM2 model and returns an image predictor instance.

        Args:
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda').

        Returns:
            SAM2ImagePredictor: An instance of the image predictor initialized with the loaded SAM2 model.
        """
        checkpoint_path = "sam2/checkpoints/sam2.1_hiera_large.pt"
        config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
        return SAM2ImagePredictor(sam2_model)

    def _read_csv_frames(self, csv_path: str) -> dict:
        """
        Reads a CSV file containing frame data and groups the rows by the 't' column.

        Args:
            csv_path (str): Path to the CSV file to be read.

        Returns:
            dict: A dictionary where each key is an integer frame index ('t'), and each value is a list of dictionaries.
                Each dictionary in the list contains the keys 'row_index', 'hexbug', 'x', and 'y' corresponding to a row in that frame.
        """
        df = pd.read_csv(csv_path)
        df = df.rename(columns={df.columns[0]: "row_index"})
        grouped = {}
        for t, group in df.groupby("t"):
            grouped[int(t)] = group[["row_index", "hexbug", "x", "y"]].to_dict(
                orient="records"
            )
        return grouped

    def _get_hexbug_points(self, hexbugs: list) -> list:
        """
        Extracts valid Hexbug points from a list of Hexbug data dictionaries.

        Args:
            hexbugs (list): A list of dictionaries, each containing "x", "y", and "hexbug" keys.

        Returns:
            list: A list of dictionaries with integer "x" and "y" coordinates and the associated "hexbug" value,
                excluding entries where "x" or "y" are NaN.
        """
        hexbug_points = []
        for hexbug_data in hexbugs:
            x, y, hexbug = hexbug_data["y"], hexbug_data["x"], hexbug_data["hexbug"]
            if not np.isnan(x) and not np.isnan(y):
                hexbug_points.append({"x": int(x), "y": int(y), "hexbug": hexbug})
        return hexbug_points

    def _get_center_points(self, hexbug_points: list) -> np.ndarray:
        return np.array([[point["x"], point["y"]] for point in hexbug_points])

    def _get_frame_masks(
        self,
        csv_data: dict,
        frame_index: int,
    ) -> list:
        """
        Generates segmentation masks for each detected object in a video frame using a SAM2 image predictor.

        Args:
            csv_data (dict): Dictionary containing per-frame data with object coordinates.
            frame_index (int): Index of the frame to process.

        Returns:
            list: A list of dictionaries, each containing:
                - "mask": The predicted mask for the object.
                - "x": The x-coordinate of the object.
                - "y": The y-coordinate of the object.
                - "hexbug": The identifier or label for the object.
        """
        hexbug_points = self._get_hexbug_points(csv_data[frame_index])
        center_points = self._get_center_points(hexbug_points)
        # SAM
        input_points = np.array(center_points).reshape(-1, 1, 2)
        input_labels = np.array([1] * len(input_points)).reshape(-1, 1)

        # Predict bounding boxes
        masks, _, __ = self.sam2_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,
        )
        return [
            {
                "mask": mask,
                "x": hexbug["x"],
                "y": hexbug["y"],
                "hexbug": hexbug["hexbug"],
            }
            for mask, hexbug in zip(masks, hexbug_points)
            if mask is not None
        ]

    def _get_mask_bounding_box(self, mask: np.ndarray) -> tuple:
        """
        Calculates the bounding box coordinates of the non-zero region in a binary mask.

        Args:
            mask (np.ndarray): A binary mask array where non-zero values indicate the region of interest.

        Returns:
            tuple: A tuple containing two tuples ((x_min, y_min), (x_max, y_max)) representing the top-left and bottom-right
                coordinates of the bounding box. Returns None if the mask contains no non-zero elements.
        """
        ys, xs = np.where(np.squeeze(mask))
        if ys.size == 0 or xs.size == 0:
            return None
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return (x_min, y_min), (x_max, y_max)

    def _draw_mask_and_center(self, image: Image, masks: list):
        """
        Draws masks, center points, and bounding boxes for a list of detected objects on an image.

        Args:
            image (Image): The image on which to draw. Should be a PIL Image object.
            masks (list): A list of dictionaries, each containing:
                - "mask": The mask of the detected object (format expected by get_mask_bounding_box).
                - "hexbug": An identifier for the object.
                - "x": The x-coordinate of the object's center.
                - "y": The y-coordinate of the object's center.

        Returns:
            Image: The modified image with drawn masks, center points, bounding boxes, and labels.
        """
        draw = ImageDraw.Draw(image)
        for mask_dict in masks:
            if mask_dict is None:
                continue
            mask = mask_dict["mask"]
            hexbug = mask_dict["hexbug"]
            # Center point
            x, y = int(mask_dict["x"]), int(mask_dict["y"])
            radius = 10
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill="red",
                outline="red",
            )

            # Bounding box
            bbox = self._get_mask_bounding_box(mask)
            if bbox is None:
                print(f" -No bounding box for hexbug {hexbug} at ({x}, {y})")
                continue
            (x_min, y_min), (x_max, y_max) = bbox
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="yellow", width=5)
            draw.text((x, y), str(hexbug), fill=(255, 255, 255, 128))
        return image

    def generate_bounding_boxes_body(self):
        # Walk through video directories
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

        all_videos = sorted(os.listdir(self.frame_dir_path))
        for video_name in all_videos:
            print(f"Processing video: {video_name}")
            frame_path = os.path.join(self.frame_dir_path, video_name)
            csv_path = os.path.join(self.csv_dir_path, f"{video_name}.csv")
            yolo_data = []
            missing_data = []
            all_frames = sorted(os.listdir(frame_path))

            if not os.path.isdir(frame_path):
                continue
            for frame_img in all_frames:
                print(f"  -Processing frame: {frame_img}")
                if not frame_img.endswith(".jpg"):
                    continue

                frame_index = int(frame_img.split("_")[1].split(".")[0])
                image_path = os.path.join(frame_path, frame_img)
                # Image and draw setup
                image = Image.open(image_path).convert("RGB")
                self.sam2_predictor.set_image(image)
                # CSV Data
                csv_data = self._read_csv_frames(csv_path)
                if frame_index not in csv_data:
                    missing_data.append(frame_index)
                    print(f"  -No data for frame {frame_index} in CSV, skipping...")
                    continue
                mask_list = self._get_frame_masks(csv_data, frame_index)
                # print(f"  -Number of masks predicted: {len(mask_list)}")
                # Draw bounding boxes and center on the image
                annotated_image = self._draw_mask_and_center(image, mask_list)

                # Save the annotated image
                image_dir = os.path.join(self.annotation_dir_path, video_name)
                os.makedirs(image_dir, exist_ok=True)
                output_image_path = os.path.join(
                    self.annotation_dir_path,
                    video_name,
                    f"frame_{frame_index:05d}.jpg",
                )
                annotated_image.save(output_image_path)
                # print(f"  -Annotated image saved to {output_image_path}")

                # YOLO
                for mask_dict in mask_list:
                    bbox = self._get_mask_bounding_box(mask_dict["mask"])
                    if bbox is None:
                        continue
                    image_width, image_height = annotated_image.size
                    mask_w, mask_h = bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]
                    mask_x_min = bbox[0][0]
                    mask_y_min = bbox[0][1]
                    mask_x_max = bbox[1][0]
                    mask_y_max = bbox[1][1]
                    bbox_cx = (mask_x_min + mask_x_max) / 2 / image_width
                    bbox_cy = (mask_y_min + mask_y_max) / 2 / image_height
                    bbox_w = (mask_x_max - mask_x_min) / image_width
                    bbox_h = (mask_y_max - mask_y_min) / image_height
                    hexbug_cx, hexbug_cy = (
                        mask_dict["x"] / image_width,
                        mask_dict["y"] / image_height,
                    )
                    hexbug_id = mask_dict["hexbug"]
                    yolo_data.append(
                        [
                            frame_index,
                            hexbug_id,
                            bbox_cx,
                            bbox_cy,
                            bbox_w,
                            bbox_h,
                            hexbug_cx,
                            hexbug_cy,
                            2,
                        ]
                    )

            # Save YOLO labels
            yolo_label_path = os.path.join(self.yolo_dir_path, f"{video_name}.txt")
            with open(yolo_label_path, "w", encoding="utf-8") as f:
                for row in yolo_data:
                    f.write(
                        f"{row[0]} {row[1]} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f} {row[6]:.6f} {row[7]:.6f} {row[8]}\n"
                    )
            print(f"  -YOLO label saved to {yolo_label_path}")
            # Save missing frames
            if len(missing_data) > 0:
                with open(
                    "../../data/original_data/missing_frames.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(f"-{video_name}:\n\t{missing_data}\n")
