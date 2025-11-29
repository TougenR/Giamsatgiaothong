#!/usr/bin/env python3
"""
Simple logging utilities for debugging and monitoring
"""
import cv2
import os
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json


@dataclass
class Logger:
    """
    Simple logging utility for debugging and monitoring system performance
    """

    log_file: Optional[str] = None
    debug_images_dir: str = "debug_images"
    save_debug_images: bool = True
    verbose_console: bool = True

    def __post_init__(self):
        """Initialize logging system"""
        # Create debug images directory
        if self.save_debug_images and not os.path.exists(self.debug_images_dir):
            os.makedirs(self.debug_images_dir)
            print(f"Created debug images directory: {self.debug_images_dir}")

        # Initialize log file
        if self.log_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"{self.log_file}_{timestamp}.log"
            with open(self.log_file, 'w') as f:
                f.write(f"=== Traffic Detection System Log - {timestamp} ===\n\n")

    def log(self, message: str, level: str = "INFO"):
        """Log a message to console and file"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"

        # Console output
        if self.verbose_console:
            print(formatted_message)

        # File output
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")

    def info(self, message: str):
        """Log info message"""
        self.log(message, "INFO")

    def warning(self, message: str):
        """Log warning message"""
        self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message"""
        self.log(message, "ERROR")

    def debug(self, message: str):
        """Log debug message"""
        self.log(message, "DEBUG")

    def save_image(self, image: Any, filename: str, description: str = ""):
        """Save debug image with timestamp"""
        if not self.save_debug_images:
            return

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename_with_timestamp = f"{timestamp}_{filename}"
        filepath = os.path.join(self.debug_images_dir, filename_with_timestamp)

        try:
            cv2.imwrite(filepath, image)
            if description:
                self.debug(f"Saved debug image: {description} -> {filename_with_timestamp}")
            else:
                self.debug(f"Saved debug image: {filename_with_timestamp}")
        except Exception as e:
            self.error(f"Failed to save debug image {filename}: {e}")

    def save_vehicle_image(self, vehicle_image: Any, vehicle_id: int, frame_number: int):
        """Save vehicle crop with standard naming"""
        filename = f"vehicle_{frame_number}_{vehicle_id}.jpg"
        self.save_image(vehicle_image, filename, f"Vehicle {vehicle_id} from frame {frame_number}")

    def save_license_plate_image(self, plate_image: Any, vehicle_id: int, plate_id: int, frame_number: int):
        """Save license plate crop with standard naming"""
        filename = f"plate_{frame_number}_{vehicle_id}_{plate_id}.jpg"
        self.save_image(plate_image, filename, f"License plate {plate_id} from vehicle {vehicle_id}")

    def log_detection_stats(self, frame_number: int, vehicles: List[Dict], license_plates: List[Dict], ocr_results: List[Dict]):
        """Log detection statistics in structured format"""
        stats = {
            "frame_number": frame_number,
            "timestamp": datetime.datetime.now().isoformat(),
            "vehicle_count": len(vehicles),
            "license_plate_count": len(license_plates),
            "ocr_results_count": len([r for r in ocr_results if r.get('ocr_texts')]),
            "vehicles": []
        }

        for i, vehicle in enumerate(vehicles):
            vehicle_stats = {
                "vehicle_id": i,
                "bbox": vehicle.get('bbox'),
                "confidence": vehicle.get('confidence'),
                "class_id": vehicle.get('class_id'),
                "ocr_texts": vehicle.get('ocr_texts', []),
                "license_plates_in_vehicle": 0
            }
            stats["vehicles"].append(vehicle_stats)

        # Log to file if available
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"FRAME {frame_number} STATS:\n")
                f.write(json.dumps(stats, indent=2) + "\n\n")

        # Log summary to console
        ocr_count = sum(1 for v in vehicles if v.get('ocr_texts'))
        self.info(f"Frame {frame_number}: {len(vehicles)} vehicles, {len(license_plates)} plates, {ocr_count} with OCR")

    def log_pipeline_step(self, step: str, frame_number: int, details: Dict = None):
        """Log specific pipeline steps"""
        message = f"Frame {frame_number} - {step}"
        if details:
            message += f": {details}"
        self.debug(message)

    def start_timer(self, timer_name: str):
        """Start a named timer for performance measurement"""
        if not hasattr(self, 'timers'):
            self.timers = {}
        self.timers[timer_name] = datetime.datetime.now()
        self.debug(f"Started timer: {timer_name}")

    def end_timer(self, timer_name: str):
        """End a named timer and log duration"""
        if hasattr(self, 'timers') and timer_name in self.timers:
            start_time = self.timers[timer_name]
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.info(f"Timer {timer_name}: {duration:.3f} seconds")
            return duration
        else:
            self.warning(f"Timer {timer_name} was not started")
            return None

    def log_system_summary(self, total_frames: int, total_vehicles: int, total_plates: int, total_ocr_texts: int, processing_time: float):
        """Log final system summary"""
        summary = {
            "total_frames": total_frames,
            "total_vehicles": total_vehicles,
            "total_license_plates": total_plates,
            "total_ocr_texts": total_ocr_texts,
            "vehicles_per_frame": total_vehicles / max(1, total_frames),
            "plates_per_vehicle": total_plates / max(1, total_vehicles),
            "ocr_success_rate": total_ocr_texts / max(1, total_plates) * 100,
            "processing_fps": total_frames / max(1, processing_time),
            "avg_time_per_frame": processing_time / max(1, total_frames)
        }

        self.info("=== SYSTEM SUMMARY ===")
        self.info(f"Total frames processed: {total_frames}")
        self.info(f"Total vehicles detected: {total_vehicles}")
        self.info(f"Total license plates detected: {total_plates}")
        self.info(f"Total OCR texts extracted: {total_ocr_texts}")
        self.info(f"Average vehicles per frame: {summary['vehicles_per_frame']:.2f}")
        self.info(f"Average plates per vehicle: {summary['plates_per_vehicle']:.2f}")
        self.info(f"OCR success rate: {summary['ocr_success_rate']:.1f}%")
        self.info(f"Processing speed: {summary['processing_fps']:.2f} FPS")
        self.info(f"Average time per frame: {summary['avg_time_per_frame']:.3f}s")

        # Save summary to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write("=== FINAL SUMMARY ===\n")
                f.write(json.dumps(summary, indent=2) + "\n")


# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger(log_file: str = None, debug_images_dir: str = "debug_images",
               save_debug_images: bool = True, verbose_console: bool = True) -> Logger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(
            log_file=log_file,
            debug_images_dir=debug_images_dir,
            save_debug_images=save_debug_images,
            verbose_console=verbose_console
        )
    return _global_logger


def log_info(message: str):
    """Quick access to global logger info"""
    get_logger().info(message)


def log_warning(message: str):
    """Quick access to global logger warning"""
    get_logger().warning(message)


def log_error(message: str):
    """Quick access to global logger error"""
    get_logger().error(message)


def log_debug(message: str):
    """Quick access to global logger debug"""
    get_logger().debug(message)