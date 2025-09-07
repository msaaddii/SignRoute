# Import libraries
import os
import time
import numpy as np
import cv2  # OpenCV for webcam capture and display
import mediapipe as mp  # Hand detection and landmarks
from tensorflow.keras.models import load_model  # Load trained LSTM model
from collections import deque  # For voting buffer
import re  # Regex for GrammarAI corrections
from typing import List  # Type hints

# === GRAMMAR AI CLASS ===
class GrammarAI:
    """
    Simple grammar AI to clean up raw predicted words from the model
    and convert them into natural English sentences.
    """
    def __init__(self):
        # Map from raw model labels to normalized words
        self.word_map = {
            "ThankYou": "Thank you",
            "Please": "Please",
            "Sorry": "Sorry"
        }

        # Direct mappings for common phrases
        self.phrase_map = {
            "hello": "Hello!",
            "thank you": "Thank you.",
            "please": "Please.",
            "sorry": "Sorry.",
            "yes": "Yes.",
            "no": "No."
        }

        # Regex-based pattern corrections for simple sentence structures
        self.patterns = [
            (re.compile(r"\bHello My Name (?:Is )?([A-Za-z]+)\b", re.I), r"Hello, my name is \1"),
            (re.compile(r"\bMy Name (?:Is )?([A-Za-z]+)\b", re.I), r"My name is \1"),
            (re.compile(r"\bI Want Job\b", re.I), "I want this job"),
            (re.compile(r"\bI Can Work Team\b", re.I), "I can work in a team"),
            (re.compile(r"\bI Can\b", re.I), "I can"),
            (re.compile(r"\bI Want\b", re.I), "I want"),
            (re.compile(r"\bI Work Experience\b", re.I), "I have work experience"),
            (re.compile(r"\bI No Experience\b", re.I), "I do not have experience"),
            (re.compile(r"\bWhat Experience\b", re.I), "What experience do you have?"),
            (re.compile(r"\bWhat Skill\b", re.I), "What are your skills?"),
            (re.compile(r"\bI Skill ([A-Za-z]+)\b", re.I), r"I have skills in \1"),
            (re.compile(r"\bI Like ([A-Za-z]+)\b", re.I), r"I like \1"),
            (re.compile(r"\bI Love ([A-Za-z]+)\b", re.I), r"I love \1"),
            (re.compile(r"\bWhy Company\b", re.I), "Why this company?"),
            (re.compile(r"\bI Like Company\b", re.I), "I like this company"),
            (re.compile(r"\bI Love Company\b", re.I), "I love this company"),
            (re.compile(r"\bI Work Team\b", re.I), "I work well in a team"),
            (re.compile(r"\bMy Strong\b", re.I), "My strength is"),
            (re.compile(r"\bMy Weak\b", re.I), "My weakness is"),
            (re.compile(r"\bI Good\b", re.I), "I am good"),
            (re.compile(r"\bI Bad\b", re.I), "I am not well"),
            (re.compile(r"\bWhere You Work\b", re.I), "Where do you work?"),
            (re.compile(r"\bHow You Work Team\b", re.I), "How do you work in a team?"),
            (re.compile(r"\bWhy You Want Job\b", re.I), "Why do you want this job?"),
        ]

    # Normalize raw model tokens using word_map
    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        return [self.word_map.get(t, t) for t in tokens]

    # Remove consecutive duplicate tokens
    def _collapse_dupes(self, tokens: List[str]) -> List[str]:
        if not tokens: return tokens
        out = [tokens[0]]
        for t in tokens[1:]:
            if t != out[-1]:
                out.append(t)
        return out

    # Capitalize first letter and add period if missing
    def _polish(self, text: str) -> str:
        text = text.strip()
        if not text: return text
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
        return text

    # Fix raw token stream into clean sentence
    def fix(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        tokens = self._collapse_dupes(self._normalize_tokens(tokens))
        sentence = " ".join(tokens).strip()
        low = sentence.lower()
        if low in self.phrase_map:
            return self.phrase_map[low]
        fixed = sentence
        for pat, repl in self.patterns:
            fixed = pat.sub(repl, fixed)
        return self._polish(fixed)


# === SETTINGS AND MODEL LOADING ===
MODEL_PATH = "sign_model.h5"
CLASSES_PATH = "label_classes.npy"
CAM_INDEX = 0  # Default webcam
CONF_THRESH = 0.70  # Minimum probability to accept prediction
PREDICT_EVERY = 0.08  # Seconds between predictions
VOTE_WINDOW = 5  # Number of predictions for majority vote
EMIT_STABILITY = 3  # How many consistent votes to emit word

# Load trained LSTM model and classes
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# === UTILITY FUNCTIONS ===
def extract_two_hands_126(rgb_frame):
    """
    Extract 126 features (x, y, z) for both hands from a frame.
    Returns combined features and MediaPipe results.
    """
    results = hands.process(rgb_frame)
    left = [0.0] * 63
    right = [0.0] * 63
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hlm in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            coords = []
            for lm in hlm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left = coords
            else:
                right = coords
    return left + right, results

def majority_vote(labels):
    """Return the label with the most occurrences."""
    if not labels: return None
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]

# === MAIN LOOP ===
def main():
    ai = GrammarAI()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    pred_buf = deque(maxlen=VOTE_WINDOW)  # Buffer for voting
    last_emitted = None
    stable_count = 0
    words_stream = []  # All emitted words
    last_t = 0.0

    print("Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        if now - last_t >= PREDICT_EVERY:
            feats, results = extract_two_hands_126(rgb)
            if any(v != 0.0 for v in feats):
                x = np.array(feats, dtype=np.float32).reshape(1, 1, 126)
                probs = model.predict(x, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])
                best_label = classes[best_idx]

                # Display label and probability
                cv2.putText(frame, f"{best_label} ({best_prob:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Process prediction if confidence high
                if best_prob >= CONF_THRESH:
                    pred_buf.append(best_label)
                    voted = majority_vote(list(pred_buf))

                    if voted == last_emitted:
                        stable_count = 0
                    else:
                        if voted is not None and list(pred_buf).count(voted) >= (VOTE_WINDOW // 2 + 1):
                            stable_count += 1
                            if stable_count >= EMIT_STABILITY:
                                words_stream.append(voted)
                                cleaned = ai.fix(words_stream)
                                last_emitted = voted
                                stable_count = 0
                                pred_buf.clear()
                                print("RAW:", words_stream)
                                print("CLEANED:", cleaned)
            last_t = now

        # Draw hand landmarks
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # Display raw and cleaned text on frame
        disp_raw = " ".join(words_stream[-8:])
        cv2.putText(frame, f"Raw: {disp_raw}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)

        cleaned = ai.fix(words_stream)
        cv2.putText(frame, f"Clean: {cleaned}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Sign Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal RAW words:", words_stream)
    print("Final CLEANED sentence:", ai.fix(words_stream))

if __name__ == "__main__":
    main()
