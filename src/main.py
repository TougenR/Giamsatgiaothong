#!/usr/bin/env python3
"""
Main processing script for vehicle and license plate detection with OCR pipeline
"""
import cv2
import os
import sys
import glob
from detector.vehicle_detector import vehicle_detector
from detector.license_plate_detector import license_plate_detector
from drawer.vehicle_drawer import vehicle_drawer
from drawer.license_plate_drawer import license_plate_drawer
from License_plate_ocr.license_plate_ocr import LicensePlateOCR
from utils.logger import get_logger
sys.path.append("../")


def create_output_directory():
    """Create output_predict_video directory if it doesn't exist"""
    output_dir = "./output_predict_video"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def get_next_video_number():
    """Get the next available video number in the output directory"""
    output_dir = create_output_directory()

    # Find all existing numbered videos
    pattern = os.path.join(output_dir, "pipeline_output_*.mp4")
    existing_files = glob.glob(pattern)

    if not existing_files:
        return 1

    # Extract numbers from existing files and find the max
    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Extract number from "pipeline_output_X.mp4"
        try:
            number = int(filename.replace("pipeline_output_", "").replace(".mp4", ""))
            numbers.append(number)
        except ValueError:
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def get_pipeline_output_filename():
    """Generate a unique output filename for pipeline processing"""
    video_number = get_next_video_number()
    output_dir = create_output_directory()
    filename = f"pipeline_output_{video_number}.mp4"
    return os.path.join(output_dir, filename)


def list_available_videos():
    """List all available videos in test_video directory"""
    test_video_dir = "./test_video"

    if not os.path.exists(test_video_dir):
        print(f"❌ Test video directory not found: {test_video_dir}")
        return []

    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []

    for ext in video_extensions:
        pattern = os.path.join(test_video_dir, ext)
        video_files.extend(glob.glob(pattern))

    # Sort and get just filenames
    video_files.sort()
    video_names = [os.path.basename(f) for f in video_files]

    return video_names


def select_video():
    """Interactive video selection from test_video directory"""
    videos = list_available_videos()

    if not videos:
        print("❌ No videos found in test_video directory")
        return None

    print("\n📹 Available Videos:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")

    print("  0. Enter custom path")

    while True:
        try:
            choice = input(f"\nSelect video (0-{len(videos)}): ").strip()

            if choice == "0":
                custom_path = input("Enter custom video path: ").strip()
                if os.path.exists(custom_path):
                    return custom_path
                else:
                    print("❌ File not found. Please try again.")
                    continue

            choice_num = int(choice)
            if 1 <= choice_num <= len(videos):
                selected_video = videos[choice_num - 1]
                return os.path.join("./test_video", selected_video)
            else:
                print(f"❌ Please enter a number between 0 and {len(videos)}")

        except ValueError:
            print("❌ Please enter a valid number")
        except EOFError:
            # Handle EOF gracefully
            if videos:
                selected_video = videos[0]
                print(f"\nUsing first available video: {selected_video}")
                return os.path.join("./test_video", selected_video)
            return None


class TrafficDetectionSystem:
    """Main class for traffic detection and visualization with OCR pipeline"""

    def __init__(self, vehicle_model_path: str, license_plate_model_path: str):
        """
        Initialize detection system

        Args:
            vehicle_model_path: Path to YOLO vehicle detection model
            license_plate_model_path: Path to YOLO license plate detection model
        """
        # Initialize drawers
        self.vehicle_drawer = vehicle_drawer(
            bbox_color=(0, 255, 0),  # Green for vehicles
            bbox_thickness=2,
            show_confidence=True
        )

        self.license_plate_drawer = license_plate_drawer(
            bbox_color=(255, 0, 0),  # Blue for license plates
            bbox_thickness=2,
            show_confidence=True
        )

        # Initialize OCR
        self.ocr = LicensePlateOCR()

        # Initialize logger for debugging
        self.logger = get_logger(
            log_file="traffic_detection_pipeline",
            debug_images_dir="./debug",
            save_debug_images=False,
            verbose_console=True
        )
        self.logger.info("TrafficDetectionSystem initialized with OCR pipeline")

        # Store model paths
        self.vehicle_model_path = vehicle_model_path
        self.license_plate_model_path = license_plate_model_path

    def process_frame_with_pipeline(self, frame, frame_number=0):
        """
        Complete pipeline: Vehicle detection → crop → license plate detection → OCR → visualization

        Args:
            frame: Input frame
            frame_number: Frame number for tracking

        Returns:
            tuple: (processed_frame, vehicle_with_ocr_info)
        """
        self.logger.start_timer(f"frame_{frame_number}_pipeline")
        self.logger.log_pipeline_step("Starting pipeline", frame_number)

        # Step 1: Detect vehicles
        self.logger.start_timer("vehicle_detection")
        vehicle_detector_instance = vehicle_detector(
            vehicle_model_path=self.vehicle_model_path,
            video_path="",  # Not used for single frame
            conf_threshold=0.6,
            iou_threshold=0.4
        )

        vehicle_detections = vehicle_detector_instance.process_frame(frame, frame_number)
        self.logger.end_timer("vehicle_detection")
        self.logger.info(f"Step 1 - Vehicle Detection: Found {len(vehicle_detections)} vehicles in frame {frame_number}")

        # Step 2: For each vehicle, crop and detect license plates
        license_plate_detector_instance = license_plate_detector(
            license_plate_model_path=self.license_plate_model_path,
            video_path="",  # Not used for single frame
            conf_threshold=0.5,  # Lower confidence for more detections
            iou_threshold=0.3
        )

        enhanced_vehicle_detections = []

        # Fallback: Detect license plates in full frame
        self.logger.start_timer("license_plate_detection_full_frame")
        full_frame_license_plates = license_plate_detector_instance.process_frame(frame, frame_number)
        self.logger.end_timer("license_plate_detection_full_frame")
        self.logger.info(f"Full frame license plate detections: {len(full_frame_license_plates)}")

        for i, vehicle in enumerate(vehicle_detections):
            self.logger.log_pipeline_step(f"Processing vehicle {i}", frame_number, {"bbox": vehicle['bbox']})
            self.logger.start_timer(f"vehicle_{i}_processing")

            bbox = vehicle['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            # Step 2: Crop vehicle region
            vehicle_crop = frame[y1:y2, x1:x2]
            self.logger.debug(f"Vehicle {i}: bbox {bbox}, crop shape {vehicle_crop.shape}")

            # Save vehicle crop for debugging
            self.logger.save_vehicle_image(vehicle_crop, i, frame_number)

            # Step 3: Detect license plates in vehicle crop
            self.logger.start_timer(f"vehicle_{i}_lp_detection")
            license_plate_detections = license_plate_detector_instance.process_frame(vehicle_crop, frame_number)
            self.logger.end_timer(f"vehicle_{i}_lp_detection")
            self.logger.info(f"Vehicle {i}: found {len(license_plate_detections)} license plates in vehicle crop")

            # Fallback: Find license plates that overlap with vehicle bbox
            if len(license_plate_detections) == 0:
                self.logger.warning(f"Vehicle {i}: No plates in vehicle crop, checking full frame detections...")
                vehicle_license_plates = []

                for lp in full_frame_license_plates:
                    lp_bbox = lp['bbox']
                    lp_x1, lp_y1, lp_x2, lp_y2 = [int(coord) for coord in lp_bbox]
                    lp_center_x = (lp_x1 + lp_x2) // 2
                    lp_center_y = (lp_y1 + lp_y2) // 2

                    # Check if license plate center is within vehicle bbox (with margin)
                    margin = 20
                    if (x1 - margin <= lp_center_x <= x2 + margin and
                        y1 - margin <= lp_center_y <= y2 + margin):
                        self.logger.debug(f"  Found overlapping license plate at {lp_bbox}")
                        vehicle_license_plates.append(lp)

                license_plate_detections = vehicle_license_plates
                self.logger.info(f"Vehicle {i}: found {len(license_plate_detections)} license plates from full frame")

            # Step 4: Extract OCR text from license plates
            ocr_texts = []
            for j, lp in enumerate(license_plate_detections):
                self.logger.log_pipeline_step(f"Processing license plate {j} for vehicle {i}", frame_number, {"bbox": lp['bbox']})

                lp_bbox = lp['bbox']
                lp_x1, lp_y1, lp_x2, lp_y2 = [int(coord) for coord in lp_bbox]

                # Determine if this is from vehicle crop or full frame
                if lp_x2 < vehicle_crop.shape[1] and lp_y2 < vehicle_crop.shape[0]:
                    # From vehicle crop
                    self.logger.debug(f"License plate {j}: Processing from vehicle crop")

                    # Ensure coordinates are within bounds
                    lp_x1 = max(0, lp_x1)
                    lp_y1 = max(0, lp_y1)
                    lp_x2 = min(vehicle_crop.shape[1], lp_x2)
                    lp_y2 = min(vehicle_crop.shape[0], lp_y2)

                    if lp_x2 > lp_x1 and lp_y2 > lp_y1:
                        # Crop license plate from vehicle crop
                        license_plate_crop = vehicle_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                        self.logger.debug(f"License plate {j}: crop shape {license_plate_crop.shape}")
                    else:
                        continue
                else:
                    # From full frame - convert to vehicle crop coordinates
                    self.logger.debug(f"License plate {j}: Processing from full frame")

                    # Convert full frame coordinates to vehicle crop coordinates
                    lp_x1_vehicle = max(0, lp_x1 - x1)
                    lp_y1_vehicle = max(0, lp_y1 - y1)
                    lp_x2_vehicle = min(vehicle_crop.shape[1], lp_x2 - x1)
                    lp_y2_vehicle = min(vehicle_crop.shape[0], lp_y2 - y1)

                    lp_x1, lp_y1, lp_x2, lp_y2 = lp_x1_vehicle, lp_y1_vehicle, lp_x2_vehicle, lp_y2_vehicle

                    if lp_x2 > lp_x1 and lp_y2 > lp_y1:
                        # Crop license plate from vehicle crop
                        license_plate_crop = vehicle_crop[lp_y1:lp_y2, lp_x1:lp_x2]
                        self.logger.debug(f"License plate {j}: crop shape {license_plate_crop.shape} (converted)")
                    else:
                        continue

                # Save license plate crop for debugging
                self.logger.save_license_plate_image(license_plate_crop, i, j, frame_number)

                # Extract OCR text
                self.logger.start_timer(f"license_plate_{i}_{j}_ocr")
                try:
                    ocr_result = self.ocr.extract_text_from_array(license_plate_crop)
                    self.logger.debug(f"License plate {j}: OCR result: {ocr_result}")

                    if ocr_result:
                        # Get the text with highest confidence
                        best_result = max(ocr_result, key=lambda x: x['confidence'])
                        ocr_text = best_result['text']
                        confidence = best_result['confidence']
                        self.logger.info(f"License plate {j}: Text '{ocr_text}' (confidence: {confidence:.2f})")
                        ocr_texts.append(ocr_text)
                    else:
                        self.logger.warning(f"License plate {j}: No OCR result")
                except Exception as e:
                    self.logger.error(f"License plate {j}: OCR error: {e}")
                finally:
                    self.logger.end_timer(f"license_plate_{i}_{j}_ocr")

            # Add OCR info to vehicle detection
            enhanced_vehicle = vehicle.copy()
            enhanced_vehicle['ocr_texts'] = ocr_texts
            enhanced_vehicle_detections.append(enhanced_vehicle)

            self.logger.info(f"Vehicle {i}: Final OCR texts: {ocr_texts}")
            self.logger.end_timer(f"vehicle_{i}_processing")

        # Log frame statistics
        self.logger.log_detection_stats(frame_number, enhanced_vehicle_detections, full_frame_license_plates, enhanced_vehicle_detections)
        self.logger.end_timer(f"frame_{frame_number}_pipeline")

        # Return both vehicle detections with OCR and license plate detections for drawing
        return enhanced_vehicle_detections, full_frame_license_plates

    def draw_with_ocr(self, frame, vehicle_detections, license_plate_detections=None):
        """
        Draw vehicles with OCR results and license plates on bounding boxes

        Args:
            frame: Input frame
            vehicle_detections: Vehicle detections with OCR info
            license_plate_detections: License plate detections (optional)

        Returns:
            Processed frame with OCR results
        """
        annotated_frame = frame.copy()

        # Draw license plate detections first (so they appear behind vehicles)
        if license_plate_detections:
            annotated_frame = self.license_plate_drawer.draw(annotated_frame, license_plate_detections)

        for detection in vehicle_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            frame_number = detection['frame_number']
            class_id = detection['class_id']
            ocr_texts = detection.get('ocr_texts', [])

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # Green for vehicles
                2
            )

            # Prepare label text
            label_parts = []

            # Add class name
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

            # Add confidence
            label_parts.append(f"{confidence:.2f}")

            # Add OCR texts if available
            if ocr_texts:
                ocr_text = " | ".join(ocr_texts)
                label_parts.append(f"LP: {ocr_text}")

            label = " | ".join(label_parts)

            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),  # Green background for vehicles
                -1  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),  # White text
                font_thickness
            )

        return annotated_frame

    def process_video_with_pipeline(self, video_path: str, output_path: str = None) -> str:
        """
        Process entire video with the complete pipeline: Vehicle detection → crop → license plate detection → OCR → visualization

        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)

        Returns:
            str: Path to output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video writer
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_with_ocr_pipeline.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video with OCR pipeline: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Output will be saved to: {output_path}")

        frame_count = 0
        total_vehicles = 0
        total_license_plates = 0
        total_ocr_texts = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame with complete pipeline
                vehicle_detections_with_ocr, license_plate_detections = self.process_frame_with_pipeline(frame, frame_count)

                # Draw results with OCR
                final_frame = self.draw_with_ocr(frame, vehicle_detections_with_ocr, license_plate_detections)

                # Write frame to output
                out.write(final_frame)

                # Update counters
                frame_count += 1
                total_vehicles += len(vehicle_detections_with_ocr)

                for vehicle in vehicle_detections_with_ocr:
                    ocr_texts = vehicle.get('ocr_texts', [])
                    total_license_plates += len(ocr_texts)
                    total_ocr_texts += len(ocr_texts)

                # Print progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames), "
                          f"Vehicles: {total_vehicles}, OCR texts: {total_ocr_texts}")

        finally:
            # Release resources
            cap.release()
            out.release()

        # Print summary
        print(f"\nPipeline processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Total license plates detected: {total_license_plates}")
        print(f"Total OCR texts extracted: {total_ocr_texts}")
        print(f"Output saved to: {output_path}")

        return output_path

    def process_single_frame_with_pipeline(self, image_path: str, output_path: str = None) -> str:
        """
        Process single image with the complete pipeline

        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)

        Returns:
            str: Path to output image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Process frame with complete pipeline
        vehicle_detections_with_ocr, license_plate_detections = self.process_frame_with_pipeline(frame, 0)

        # Draw results with OCR
        final_frame = self.draw_with_ocr(frame, vehicle_detections_with_ocr, license_plate_detections)

        # Save output
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_with_ocr_pipeline.jpg"

        cv2.imwrite(output_path, final_frame)

        print(f"Single frame pipeline processing complete!")
        print(f"Vehicles detected: {len(vehicle_detections_with_ocr)}")

        for i, vehicle in enumerate(vehicle_detections_with_ocr):
            ocr_texts = vehicle.get('ocr_texts', [])
            print(f"  Vehicle {i+1}: {len(ocr_texts)} license plates, OCR: {ocr_texts}")

        print(f"Output saved to: {output_path}")

        return output_path

    def process_video(self, video_path: str, output_path: str = None) -> str:
        """
        Process entire video with detection and drawing

        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)

        Returns:
            str: Path to output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Initialize detectors
        vehicle_detector_instance = vehicle_detector(
            vehicle_model_path=self.vehicle_model_path,
            video_path=video_path,
            conf_threshold=0.6,
            iou_threshold=0.4
        )

        license_plate_detector_instance = license_plate_detector(
            license_plate_model_path=self.license_plate_model_path,
            video_path=video_path,
            conf_threshold=0.8,
            iou_threshold=0.3
        )

        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video writer
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_with_detections.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        print(f"Output will be saved to: {output_path}")

        frame_count = 0
        total_vehicles = 0
        total_license_plates = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect vehicles and license plates in current frame
                vehicle_detections = vehicle_detector_instance.process_frame(frame, frame_count)
                license_plate_detections = license_plate_detector_instance.process_frame(frame, frame_count)

                # Draw detections
                frame_with_vehicles = self.vehicle_drawer.draw(frame, vehicle_detections)
                frame_with_all = self.license_plate_drawer.draw(frame_with_vehicles, license_plate_detections)

                # Add counts
                final_frame = self.vehicle_drawer.draw_vehicle_count(
                    frame_with_all, vehicle_detections, position=(10, 30)
                )
                final_frame = self.license_plate_drawer.draw_license_plate_count(
                    final_frame, license_plate_detections, position=(10, 60)
                )

                # Write frame to output
                out.write(final_frame)

                # Update counters
                frame_count += 1
                total_vehicles += len(vehicle_detections)
                total_license_plates += len(license_plate_detections)

                # Print progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

        finally:
            # Release resources
            cap.release()
            out.release()

        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Total license plates detected: {total_license_plates}")
        print(f"Output saved to: {output_path}")

        return output_path

    def process_single_frame(self, image_path: str, output_path: str = None) -> str:
        """
        Process single image with detection and drawing

        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)

        Returns:
            str: Path to output image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Initialize detectors with single frame (video_path not used but required)
        vehicle_detector_instance = vehicle_detector(
            vehicle_model_path=self.vehicle_model_path,
            video_path=image_path,  # Use image path as placeholder
            conf_threshold=0.6,
            iou_threshold=0.4
        )

        license_plate_detector_instance = license_plate_detector(
            license_plate_model_path=self.license_plate_model_path,
            video_path=image_path,  # Use image path as placeholder
            conf_threshold=0.8,
            iou_threshold=0.3
        )

        # Detect objects
        vehicle_detections = vehicle_detector_instance.process_frame(frame, 0)
        license_plate_detections = license_plate_detector_instance.process_frame(frame, 0)

        # Draw detections
        frame_with_vehicles = self.vehicle_drawer.draw(frame, vehicle_detections)
        frame_with_all = self.license_plate_drawer.draw(frame_with_vehicles, license_plate_detections)

        # Add counts
        final_frame = self.vehicle_drawer.draw_vehicle_count(
            frame_with_all, vehicle_detections, position=(10, 30)
        )
        final_frame = self.license_plate_drawer.draw_license_plate_count(
            final_frame, license_plate_detections, position=(10, 60)
        )

        # Save output
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_with_detections.jpg"

        cv2.imwrite(output_path, final_frame)

        print(f"Single frame processing complete!")
        print(f"Vehicles detected: {len(vehicle_detections)}")
        print(f"License plates detected: {len(license_plate_detections)}")
        print(f"Output saved to: {output_path}")

        return output_path


def main():
    """Main function for the traffic detection system"""
    print("🚦 Traffic Detection System")
    print("=" * 60)

    # Create output directory at startup
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir}")

    # Paths to your existing files
    VEHICLE_MODEL_PATH = "models/vehicle_model.pt"
    LICENSE_PLATE_MODEL_PATH = "models/License_plate_model.pt"

    # Check if model files exist
    print("📁 Checking files:")
    print(f"  🚗 Vehicle model: {VEHICLE_MODEL_PATH} - {'✅' if os.path.exists(VEHICLE_MODEL_PATH) else '❌'}")
    print(f"  🪪 License plate model: {LICENSE_PLATE_MODEL_PATH} - {'✅' if os.path.exists(LICENSE_PLATE_MODEL_PATH) else '❌'}")

    if not all([os.path.exists(VEHICLE_MODEL_PATH), os.path.exists(LICENSE_PLATE_MODEL_PATH)]):
        print("\n❌ Model files are missing. Please check the paths above.")
        print("Make sure you have:")
        print("  - models/vehicle_model.pt")
        print("  - models/License_plate_model.pt")
        return

    print("\n🎯 Processing options:")
    print("1. Process first 100 frames of selected video (quick test)")
    print("2. Process entire selected video (full test)")
    print("3. Process custom video/image")
    print("4. Pipeline: Process with OCR (Vehicle → LP Detection → OCR)")
    print("5. Exit")

    try:
        choice = input("Enter choice (1, 2, 3, 4, or 5): ").strip()
    except EOFError:
        print("\nRunning quick test with first 100 frames...")
        choice = "1"

    if choice in ["1", "2", "4"]:
        # For options 1, 2, and 4, let user select a video
        selected_video = select_video()
        if selected_video is None:
            print("❌ No video selected. Exiting.")
            return
        print(f"📹 Selected video: {selected_video}")

    if choice == "1":
        # Quick test with first 100 frames
        quick_test_frames(VEHICLE_MODEL_PATH, LICENSE_PLATE_MODEL_PATH, selected_video)
    elif choice == "2":
        # Full video processing
        full_video_test(VEHICLE_MODEL_PATH, LICENSE_PLATE_MODEL_PATH, selected_video)
    elif choice == "3":
        # Custom processing
        custom_processing(VEHICLE_MODEL_PATH, LICENSE_PLATE_MODEL_PATH)
    elif choice == "4":
        # Pipeline processing with OCR
        pipeline_processing(VEHICLE_MODEL_PATH, LICENSE_PLATE_MODEL_PATH, selected_video)
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def quick_test_frames(vehicle_model, license_plate_model, video_path):
    """Process first 100 frames for quick testing"""
    print("\n🚀 Quick Test: Processing first 100 frames")
    print(f"📹 Video: {video_path}")

    try:
        # Initialize system
        detection_system = TrafficDetectionSystem(
            vehicle_model_path=vehicle_model,
            license_plate_model_path=license_plate_model
        )

        # Initialize detectors
        vehicle_detector_instance = vehicle_detector(
            vehicle_model_path=vehicle_model,
            video_path=video_path,
            conf_threshold=0.6,
            iou_threshold=0.4
        )

        license_plate_detector_instance = license_plate_detector(
            license_plate_model_path=license_plate_model,
            video_path=video_path,
            conf_threshold=0.8,
            iou_threshold=0.3
        )

        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output
        output_path = "quick_test_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        max_frames = 100
        total_vehicles = 0
        total_license_plates = 0

        print(f"Processing {max_frames} frames...")

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            vehicle_detections = vehicle_detector_instance.process_frame(frame, frame_count)
            license_plate_detections = license_plate_detector_instance.process_frame(frame, frame_count)

            # Draw detections
            frame_with_vehicles = detection_system.vehicle_drawer.draw(frame, vehicle_detections)
            frame_with_all = detection_system.license_plate_drawer.draw(frame_with_vehicles, license_plate_detections)

            # Add counts
            final_frame = detection_system.vehicle_drawer.draw_vehicle_count(
                frame_with_all, vehicle_detections, position=(10, 30)
            )
            final_frame = detection_system.license_plate_drawer.draw_license_plate_count(
                final_frame, license_plate_detections, position=(10, 60)
            )

            # Write frame
            out.write(final_frame)

            # Update counters
            frame_count += 1
            total_vehicles += len(vehicle_detections)
            total_license_plates += len(license_plate_detections)

            # Print progress
            if frame_count % 25 == 0:
                print(f"  Frame {frame_count}/{max_frames}, Vehicles: {len(vehicle_detections)}, LPs: {len(license_plate_detections)}")

        # Cleanup
        cap.release()
        out.release()

        print(f"\n✅ Quick test complete!")
        print(f"📊 Results:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Total vehicles detected: {total_vehicles}")
        print(f"  Total license plates detected: {total_license_plates}")
        print(f"  Output saved to: {output_path}")
        print(f"\n🎬 Play the output video:")
        print(f"  mpv {output_path}")
        print(f"  vlc {output_path}")
        print(f"  or double-click the file")

    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        import traceback
        traceback.print_exc()


def full_video_test(vehicle_model, license_plate_model, video_path):
    """Process entire video"""
    print("\n🚀 Full Test: Processing entire video")
    print(f"📹 Video: {video_path}")

    try:
        detection_system = TrafficDetectionSystem(
            vehicle_model_path=vehicle_model,
            license_plate_model_path=license_plate_model
        )

        output_path = detection_system.process_video(video_path, "full_test_output.mp4")
        print(f"\n✅ Full test complete!")
        print(f"📹 Output saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error during full test: {e}")
        import traceback
        traceback.print_exc()


def custom_processing(vehicle_model, license_plate_model):
    """Process custom video/image"""
    print("\n🎯 Custom Processing")

    media_type = input("Choose media type (1=video, 2=image): ").strip()

    if media_type == "1":
        # Video processing
        video_path = input("Enter video path: ").strip()
        output_path = input("Enter output path (leave blank for default): ").strip()

        if not output_path:
            output_path = None

        try:
            detection_system = TrafficDetectionSystem(
                vehicle_model_path=vehicle_model,
                license_plate_model_path=license_plate_model
            )
            detection_system.process_video(video_path, output_path)
        except Exception as e:
            print(f"❌ Error processing video: {e}")

    elif media_type == "2":
        # Image processing
        image_path = input("Enter image path: ").strip()
        output_path = input("Enter output path (leave blank for default): ").strip()

        if not output_path:
            output_path = None

        try:
            detection_system = TrafficDetectionSystem(
                vehicle_model_path=vehicle_model,
                license_plate_model_path=license_plate_model
            )
            detection_system.process_single_frame(image_path, output_path)
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    else:
        print("Invalid choice. Please enter 1 or 2.")


def pipeline_processing(vehicle_model, license_plate_model, video_path):
    """Process with complete pipeline: Vehicle detection → crop → license plate detection → OCR"""

    try:
        # Initialize system
        detection_system = TrafficDetectionSystem(
            vehicle_model_path=vehicle_model,
            license_plate_model_path=license_plate_model
        )

        detection_system.logger.info("=== PIPELINE PROCESSING STARTED ===")
        detection_system.logger.info(f"Video: {video_path}")

        print("\n🔥 Pipeline Processing: Vehicle → LP Detection → OCR")
        print(f"📹 Video: {video_path}")
        print("\n🎯 Pipeline options:")
        print("1. Quick pipeline test (first 50 frames)")
        print("2. Full pipeline video processing")
        print("3. Pipeline processing on custom video/image")

        try:
            pipeline_choice = input("Enter pipeline choice (1, 2, or 3): ").strip()
        except EOFError:
            print("\nRunning quick pipeline test with first 50 frames...")
            pipeline_choice = "1"

        if pipeline_choice == "1":
            # Quick pipeline test with first 50 frames
            detection_system.logger.info("Starting quick pipeline test (first 50 frames)")
            print("\n🚀 Quick Pipeline Test: Processing first 50 frames")

            # Setup video capture
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            detection_system.logger.info(f"Video properties: {width}x{height} @ {fps} FPS")

            # Setup output with incremental numbering
            output_path = get_pipeline_output_filename().replace(".mp4", "_quick_test.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            max_frames = 50
            total_vehicles = 0
            total_ocr_texts = 0
            successful_ocr_vehicles = 0

            detection_system.logger.info(f"Processing {max_frames} frames with pipeline...")
            print(f"Processing {max_frames} frames with pipeline...")

            detection_system.start_timer("quick_test_total")

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame with complete pipeline
                vehicle_detections_with_ocr, license_plate_detections = detection_system.process_frame_with_pipeline(frame, frame_count)

                # Draw results with OCR
                final_frame = detection_system.draw_with_ocr(frame, vehicle_detections_with_ocr, license_plate_detections)

                # Write frame
                out.write(final_frame)

                # Update counters
                frame_count += 1
                total_vehicles += len(vehicle_detections_with_ocr)

                for vehicle in vehicle_detections_with_ocr:
                    ocr_texts = vehicle.get('ocr_texts', [])
                    total_ocr_texts += len(ocr_texts)
                    if ocr_texts:
                        successful_ocr_vehicles += 1

                # Print progress
                if frame_count % 10 == 0:
                    ocr_rate = (successful_ocr_vehicles / max(1, total_vehicles)) * 100
                    print(f"  Frame {frame_count}/{max_frames}, Vehicles: {len(vehicle_detections_with_ocr)}, OCR success: {ocr_rate:.1f}%")
                    detection_system.logger.info(f"Progress: Frame {frame_count}/{max_frames}, Vehicles: {len(vehicle_detections_with_ocr)}, OCR success: {ocr_rate:.1f}%")

            total_time = detection_system.end_timer("quick_test_total")

            # Cleanup
            cap.release()
            out.release()

            # Log final statistics
            detection_system.logger.log_system_summary(frame_count, total_vehicles, total_ocr_texts, total_ocr_texts, total_time)

            print(f"\n✅ Quick pipeline test complete!")
            print(f"📊 Results:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Total vehicles detected: {total_vehicles}")
            print(f"  Vehicles with OCR results: {successful_ocr_vehicles}")
            print(f"  Total OCR texts extracted: {total_ocr_texts}")
            print(f"  OCR success rate: {(successful_ocr_vehicles/max(1,total_vehicles)*100):.1f}%")
            print(f"  Processing time: {total_time:.2f}s")
            print(f"  Average FPS: {frame_count/max(1,total_time):.2f}")
            print(f"  Output saved to: {output_path}")
            print(f"\n🎬 Play the output video:")
            print(f"  mpv {output_path}")
            print(f"  vlc {output_path}")
            print(f"  or double-click the file")
            print(f"\n📁 Check debug folder for detailed logs and images:")
            print(f"  - Debug logs: debug/traffic_detection_pipeline_*.log")
            print(f"  - Vehicle crops: debug/vehicle_*.jpg")
            print(f"  - License plate crops: debug/plate_*.jpg")

        elif pipeline_choice == "2":
            # Full pipeline video processing
            print("\n🚀 Full Pipeline Video Processing")
            output_path = detection_system.process_video_with_pipeline(video_path, get_pipeline_output_filename())
            print(f"\n✅ Full pipeline processing complete!")
            print(f"📹 Output saved to: {output_path}")

        elif pipeline_choice == "3":
            # Custom pipeline processing
            print("\n🎯 Custom Pipeline Processing")

            media_type = input("Choose media type (1=video, 2=image): ").strip()

            if media_type == "1":
                # Video processing with pipeline
                video_path = input("Enter video path: ").strip()
                custom_output = input("Enter output path (leave blank for default): ").strip()

                if not custom_output:
                    output_path = get_pipeline_output_filename()
                    print(f"Using default output path: {output_path}")
                else:
                    output_path = custom_output

                detection_system.process_video_with_pipeline(video_path, output_path)

            elif media_type == "2":
                # Image processing with pipeline
                image_path = input("Enter image path: ").strip()
                custom_output = input("Enter output path (leave blank for default): ").strip()

                if not custom_output:
                    # For images, use similar numbering but with .jpg extension
                    output_dir = create_output_directory()
                    video_number = get_next_video_number()
                    output_path = os.path.join(output_dir, f"pipeline_output_{video_number}.jpg")
                    print(f"Using default output path: {output_path}")
                else:
                    output_path = custom_output

                detection_system.process_single_frame_with_pipeline(image_path, output_path)

            else:
                print("Invalid choice. Please enter 1 or 2.")

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    except Exception as e:
        print(f"❌ Error during pipeline processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
