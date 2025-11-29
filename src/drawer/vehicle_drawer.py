import cv2
import numpy as np
from dataclasses import dataclass
import sys
sys.path.append("../")


@dataclass
class vehicle_drawer:
    """Drawer class for vehicle detection bounding boxes on video frames"""

    bbox_color: tuple = (0, 255, 0)  # Green color for vehicle boxes (BGR format)
    bbox_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    show_confidence: bool = True
    show_frame_number: bool = True

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw vehicle detection bounding boxes on video frame

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from vehicle_detector

        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        # Make a copy of the frame to avoid modifying the original
        annotated_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            frame_number = detection['frame_number']
            class_id = detection['class_id']

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                self.bbox_color,
                self.bbox_thickness
            )

            # Prepare label text
            label_parts = []

            # Add class name if available
            if class_id == 0:
                class_name = "Car"
            elif class_id == 1:
                class_name = "Truck"
            elif class_id == 2:
                class_name = "Bus"
            elif class_id == 3:
                class_name = "Motorcycle"
            else:
                class_name = f"Vehicle_{class_id}"

            label_parts.append(class_name)

            # Add confidence if enabled
            if self.show_confidence:
                label_parts.append(f"{confidence:.2f}")

            # Add frame number if enabled
            if self.show_frame_number:
                label_parts.append(f"Frame:{frame_number}")

            label = " ".join(label_parts)

            # Calculate text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, self.font_scale, self.font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                self.bbox_color,
                -1  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )

        return annotated_frame

    def draw_with_custom_style(self, frame: np.ndarray, detections: list,
                              color: tuple = None, thickness: int = None,
                              show_labels: bool = True) -> np.ndarray:
        """
        Draw with custom styling options

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from vehicle_detector
            color: Custom color in BGR format (optional)
            thickness: Custom line thickness (optional)
            show_labels: Whether to show labels (optional)

        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        # Temporarily override settings
        original_color = self.bbox_color
        original_thickness = self.bbox_thickness
        original_show_confidence = self.show_confidence

        if color is not None:
            self.bbox_color = color
        if thickness is not None:
            self.bbox_thickness = thickness

        self.show_confidence = show_labels

        # Draw detections
        annotated_frame = self.draw(frame, detections)

        # Restore original settings
        self.bbox_color = original_color
        self.bbox_thickness = original_thickness
        self.show_confidence = original_show_confidence

        return annotated_frame

    def draw_vehicle_count(self, frame: np.ndarray, detections: list,
                          position: tuple = (10, 30)) -> np.ndarray:
        """
        Draw total vehicle count on frame

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from vehicle_detector
            position: Position to draw the count (x, y)

        Returns:
            np.ndarray: Frame with vehicle count
        """
        annotated_frame = frame.copy()

        # Count unique vehicles (by grouping overlapping detections)
        vehicle_count = len(detections)

        # Draw background for text
        text = f"Vehicles: {vehicle_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, self.font_scale, self.font_thickness
        )

        cv2.rectangle(
            annotated_frame,
            (position[0], position[1] - text_height - baseline),
            (position[0] + text_width, position[1] + baseline),
            (0, 0, 0),  # Black background
            -1
        )

        # Draw text
        cv2.putText(
            annotated_frame,
            text,
            position,
            font,
            self.font_scale,
            (255, 255, 255),  # White text
            self.font_thickness
        )

        return annotated_frame