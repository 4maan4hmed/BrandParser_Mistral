from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        self.text_frequencies = defaultdict(int)
        self.best_confidences = {}
        self.accumulated_text = ""
        self.min_confidence = 0.7  # Configurable confidence threshold
        self.max_results = 10  # Configurable max results to store
        self.data_file = Path("data.json")  # Path to the JSON file
        self.item_data = self.load_item_data()  # Load item data from JSON

    def load_item_data(self):
        """Load item data from the JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading item data: {e}")
                return []
        else:
            print(f"File {self.data_file} does not exist. Creating an empty list.")
            return []

    def process_ocr_result(self, result):
        """Process OCR result and update frequencies and confidences."""
        if not result or not isinstance(result, list) or not result[0]:
            return

        try:
            for detection in result[0]:
                if not isinstance(detection, (list, tuple)) or len(detection) != 2:
                    continue

                box, text_info = detection
                if not isinstance(text_info, (list, tuple)) or len(text_info) != 2:
                    continue

                text, confidence = text_info

                # Skip low confidence detections
                if confidence < self.min_confidence:
                    continue

                # Clean and validate text
                text = text.strip()
                if not text or len(text) < 2:  # Skip very short or empty text
                    continue

                # Update frequency and confidence
                self.text_frequencies[text] += 1
                if text not in self.best_confidences or confidence > self.best_confidences[text]:
                    self.best_confidences[text] = confidence

                # Update accumulated text with only high-frequency items
                if self.text_frequencies[text] > 1:
                    if text not in self.accumulated_text:
                        self.accumulated_text += f"{text} "

        except Exception as e:
            print(f"Error processing detection: {e}")
            return None

    def process_image(self, image):
        """Process an image frame and return accumulated text."""
        if image is None or not isinstance(image, np.ndarray):
            print("Invalid image input")
            return None

        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.ocr.ocr(img_rgb, cls=True)
            if result:
                self.process_ocr_result(result)
                self.save_results()
                return self.accumulated_text.strip()
            return None

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def identify_item(self):
        """Identify the item based on the accumulated OCR text."""
        if not self.accumulated_text:
            print("No accumulated text to identify item.")
            return None

        # Debugging: Print accumulated text
        print(f"Accumulated Text: {self.accumulated_text}")

        # Compare accumulated text with item data
        best_match = None
        best_score = 0

        for item in self.item_data:
            ocr_text = item.get("ocr_text", "")
            if not ocr_text:
                continue

            # Debugging: Print item OCR text
            print(f"Item OCR Text: {ocr_text}")

            # Calculate a simple matching score (e.g., number of overlapping words)
            item_words = set(ocr_text.split())
            detected_words = set(self.accumulated_text.split())
            overlap = len(item_words.intersection(detected_words))

            # Debugging: Print matching score
            print(f"Matching Score for {item['item_name']}: {overlap}")

            if overlap > best_score:
                best_score = overlap
                best_match = item

        # Debugging: Print best match
        if best_match:
            print(f"Best Match: {best_match['item_name']}")
        else:
            print("No matching item found.")

        return best_match

    def reset(self):
        """Reset all OCR processing data."""
        self.text_frequencies.clear()
        self.best_confidences.clear()
        self.accumulated_text = ""
        try:
            with open('output_ocr_text.txt', 'w', encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            print(f"Error clearing output file: {e}")