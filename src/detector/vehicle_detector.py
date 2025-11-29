from utils import save_stub, read_stub, get_device, setup_model_for_gpu, get_optimal_batch_size, optimize_inference_settings
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from dataclasses import dataclass
import sys
import os
sys.path.append("../")


@dataclass
class vehicle_detector:
    vehicle_model_path: str
    video_path: str
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45

    def __post_init__(self):
        """Initialize model and optimize for inference"""
        self.model = YOLO(self.vehicle_model_path)

        # Optimize inference settings
        optimize_inference_settings()

        # Setup device and move model to GPU if available
        device = get_device()
        self.model.to(device)

        # Set optimal batch size
        self.batch_size = get_optimal_batch_size(device.type, 'medium')

        print(f"Vehicle detector initialized on {device}")
        print(f"Batch size: {self.batch_size}")

    def process_frame(self, frame: np.ndarray, frame_number: int = 0) -> list:
        """
        Process a single frame and return vehicle detections

        Args:
            frame: Input frame as numpy array
            frame_number: Frame number for tracking

        Returns:
            list: List of dictionaries with bbox info
        """
        # Run inference on frame
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'frame_number': frame_number,
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                    detections.append(detection)

        return detections

    def process_video_frames(self, max_frames: int = None) -> list:
        """
        Process video frame by frame and return all detections

        Args:
            max_frames: Maximum number of frames to process (None for all)

        Returns:
            list: List of all detections across all processed frames
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        all_detections = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process current frame
                frame_detections = self.process_frame(frame, frame_count)
                all_detections.extend(frame_detections)

                frame_count += 1

                # Stop if max_frames reached
                if max_frames is not None and frame_count >= max_frames:
                    break

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, {len(all_detections)} vehicles detected")

        finally:
            cap.release()

        print(f"Total processed: {frame_count} frames, {len(all_detections)} vehicles detected")
        return all_detections

    def detector(self):
        """
        Legacy method for backward compatibility
        Returns detections from entire video
        """
        return self.process_video_frames()

    def get_detections_with_bbox(self) -> list:
        """
        Get detections with bounding box coordinates

        Returns:
            list: List of dictionaries with bbox info
        """
        return self.process_video_frames()

    def filter_vehicles_by_size(self, detections: list, min_width: int = 50, min_height: int = 50) -> list:
        """
        Filter detections by minimum size to remove small false positives

        Args:
            detections: List of detection dictionaries
            min_width: Minimum width threshold
            min_height: Minimum height threshold

        Returns:
            list: Filtered detections
        """
        filtered = []
        for detection in detections:
            if detection['width'] >= min_width and detection['height'] >= min_height:
                filtered.append(detection)
        return filtered

    def get_vehicle_count_by_frame(self) -> dict:
        """
        Get vehicle count for each frame

        Returns:
            dict: {frame_number: vehicle_count}
        """
        detections = self.process_video_frames()
        frame_counts = {}

        for detection in detections:
            frame_num = detection['frame_number']
            frame_counts[frame_num] = frame_counts.get(frame_num, 0) + 1

        return frame_counts