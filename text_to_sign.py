"""
File: text_to_sign.py
Description:
    Converts Text → Sign images/animations.
    - Displays sign images or avatar animations for interviewer text.
    - Supports replaying the last signs.
    - Helps Deaf user understand spoken input visually.
"""

import os
import cv2
import numpy as np

IMAGE_FOLDER = "sign_images"

class SignDisplay:

    def __init__(self, display_width=400, display_height=350, delay=400):

        self.last_text = ""            # Last sentence shown
        self.last_signs = []           # List of (word, image) tuples
        self.current_idx = 0           # Index of current word being shown
        self.showing = False           # True if a sentence is currently being displayed
        self.delay = delay             # Time (ms) to show each word
        self.last_time = 0             # Last tick time used for timing
        self.display_width = display_width
        self.display_height = display_height

    def create_frame(self, text="", sign_img=None):

        frame = np.zeros((400, 900, 3), dtype=np.uint8)

        # Text overlay (optional)
        cv2.putText(frame, "Interviewer says:", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, text, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Sign image area
        if sign_img is not None:
            sign_resized = cv2.resize(sign_img, (300, 300))
            frame[50:350, 550:850] = sign_resized

        return frame

    def show_sentence(self, text, delay=None):

        self.last_text = text
        self.last_signs = []
        self.current_idx = 0
        self.showing = True
        self.last_time = cv2.getTickCount()
        if delay is not None:
            self.delay = delay

        for word in text.split():
            file_path = os.path.join(IMAGE_FOLDER, f"{word.lower()}.jpg")
            if os.path.exists(file_path):
                img = cv2.imread(file_path)
            else:
                img = None  # No image for this word
            self.last_signs.append((word, img))

    def replay_last(self):
        """Replay the last sentence."""
        if self.last_text:
            self.show_sentence(self.last_text, delay=self.delay)

    def update(self):

        if not self.showing or not self.last_signs:
            return None

        now = cv2.getTickCount()
        elapsed = (now - self.last_time) / cv2.getTickFrequency() * 1000  # in milliseconds

        if elapsed >= self.delay:
            self.last_time = now
            if self.current_idx < len(self.last_signs):
                word, img = self.last_signs[self.current_idx]
                self.current_idx += 1
                return self.create_frame(self.last_text, img)
            else:
                self.showing = False
        return None
