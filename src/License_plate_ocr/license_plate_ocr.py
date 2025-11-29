import easyocr
import cv2
import numpy as np


class LicensePlateOCR:
    def __init__(self):
        """Initialize EasyOCR reader for license plate recognition"""
        print("Initializing EasyOCR reader...")
        # Initialize EasyOCR reader with English language
        self.reader = easyocr.Reader(['vi'], gpu=True)
        print("EasyOCR reader initialized successfully")

    def extract_text(self, image_path):
        """
        Extract text from license plate image

        Args:
            image_path: Path to license plate image file

        Returns:
            List of detected texts with confidence scores
        """
        try:
            # Read text from image
            result = self.reader.readtext(
                image_path,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # License plate characters only
                detail=1,  # Return bounding boxes and confidence
                paragraph=False
            )

            # Format results
            texts = []
            for bbox, text, confidence in result:
                texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })

            return texts

        except Exception as e:
            print(f"Error extracting text: {e}")
            return []

    def extract_text_simple(self, image_path):
        """
        Simple text extraction returning only text strings

        Args:
            image_path: Path to license plate image file

        Returns:
            List of detected text strings
        """
        try:
            # Get only text without bounding boxes
            result = self.reader.readtext(
                image_path,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                detail=0  # Return only text
            )
            return result

        except Exception as e:
            print(f"Error extracting text: {e}")
            return []

    def extract_text_from_array(self, image_array):
        """
        Extract text from numpy array image

        Args:
            image_array: Image as numpy array

        Returns:
            List of detected texts with confidence scores
        """
        try:
            result = self.reader.readtext(
                image_array,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                detail=1,
                paragraph=False
            )

            texts = []
            for bbox, text, confidence in result:
                texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })

            return texts

        except Exception as e:
            print(f"Error extracting text: {e}")
            return []


