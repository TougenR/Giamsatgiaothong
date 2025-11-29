import cv2
import numpy as np
from dataclasses import dataclass
import sys
sys.path.append("../")


@dataclass
class license_plate_drawer:
    """Drawer class for license plate detection bounding boxes on video frames"""

    bbox_color: tuple = (255, 0, 0)  # Blue color for license plate boxes (BGR format)
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    show_confidence: bool = True
    show_frame_number: bool = False  # Usually not needed for license plates
    label_prefix: str = "LP"

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw license plate detection bounding boxes on video frame

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from license_plate_detector

        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        # Make a copy of the frame to avoid modifying the original
        annotated_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            frame_number = detection['frame_number']

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
            label_parts = [self.label_prefix]

            # Add confidence if enabled
            if self.show_confidence:
                label_parts.append(f"{confidence:.2f}")

            # Add frame number if enabled
            if self.show_frame_number:
                label_parts.append(f"F:{frame_number}")

            label = " ".join(label_parts)

            # Calculate text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, self.font_scale, self.font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 3),
                (x1 + text_width, y1),
                self.bbox_color,
                -1  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 1),
                font,
                self.font_scale,
                (255, 255, 255),  # White text
                self.font_thickness
            )

        return annotated_frame

    def draw_with_custom_style(self, frame: np.ndarray, detections: list,
                              color: tuple = None, thickness: int = None,
                              show_labels: bool = True, label: str = None) -> np.ndarray:
        """
        Draw with custom styling options

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from license_plate_detector
            color: Custom color in BGR format (optional)
            thickness: Custom line thickness (optional)
            show_labels: Whether to show labels (optional)
            label: Custom label prefix (optional)

        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        # Temporarily override settings
        original_color = self.bbox_color
        original_thickness = self.bbox_thickness
        original_show_confidence = self.show_confidence
        original_label_prefix = self.label_prefix

        if color is not None:
            self.bbox_color = color
        if thickness is not None:
            self.bbox_thickness = thickness
        if label is not None:
            self.label_prefix = label

        self.show_confidence = show_labels

        # Draw detections
        annotated_frame = self.draw(frame, detections)

        # Restore original settings
        self.bbox_color = original_color
        self.bbox_thickness = original_thickness
        self.show_confidence = original_show_confidence
        self.label_prefix = original_label_prefix

        return annotated_frame

    def draw_license_plate_count(self, frame: np.ndarray, detections: list,
                               position: tuple = (10, 30)) -> np.ndarray:
        """
        Draw total license plate count on frame

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from license_plate_detector
            position: Position to draw the count (x, y)

        Returns:
            np.ndarray: Frame with license plate count
        """
        annotated_frame = frame.copy()

        # Count license plates
        plate_count = len(detections)

        # Draw background for text
        text = f"License Plates: {plate_count}"
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

    def draw_enhanced_boxes(self, frame: np.ndarray, detections: list,
                          corner_length: int = 15, corner_thickness: int = 3) -> np.ndarray:
        """
        Draw license plate boxes with corner enhancement for better visibility

        Args:
            frame: Input video frame as numpy array
            detections: List of detection dictionaries from license_plate_detector
            corner_length: Length of corner lines
            corner_thickness: Thickness of corner lines

        Returns:
            np.ndarray: Frame with enhanced corner boxes
        """
        annotated_frame = frame.copy()

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Draw corners instead of full rectangle
            # Top-left corner
            cv2.line(annotated_frame, (x1, y1), (x1 + corner_length, y1),
                    self.bbox_color, corner_thickness)
            cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_length),
                    self.bbox_color, corner_thickness)

            # Top-right corner
            cv2.line(annotated_frame, (x2 - corner_length, y1), (x2, y1),
                    self.bbox_color, corner_thickness)
            cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_length),
                    self.bbox_color, corner_thickness)

            # Bottom-left corner
            cv2.line(annotated_frame, (x1, y2 - corner_length), (x1, y2),
                    self.bbox_color, corner_thickness)
            cv2.line(annotated_frame, (x1, y2), (x1 + corner_length, y2),
                    self.bbox_color, corner_thickness)

            # Bottom-right corner
            cv2.line(annotated_frame, (x2 - corner_length, y2), (x2, y2),
                    self.bbox_color, corner_thickness)
            cv2.line(annotated_frame, (x2, y2 - corner_length), (x2, y2),
                    self.bbox_color, corner_thickness)

        return annotated_frame