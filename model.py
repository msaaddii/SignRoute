import os
import time
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# === SETTINGS ===
MODEL_PATH = "sign_model.h5"
CLASSES_PATH = "label_classes.npy"
CAM_INDEX = 0
CONF_THRESH = 0.70
PREDICT_EVERY = 0.08  # seconds
VOTE_WINDOW = 5       # majority vote over last N predictions
EMIT_STABILITY = 3    # need same voted word this many times in a row to emit

# === LOAD MODEL + LABELS ===
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
classes = classes.tolist() if isinstance(classes, np.ndarray) else classes

# === MEDIAPIPE HANDS ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_two_hands_126(rgb_frame):
    results = hands.process(rgb_frame)
    left = [0.0] * 63
    right = [0.0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hlm in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            coords = []
            for lm in hlm.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            if label == "Left":
                left = coords
            else:
                right = coords
    return left + right, results

def majority_vote(labels):
    if not labels: return None
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ Could not open webcam"); return

    pred_buf = deque(maxlen=VOTE_WINDOW)
    last_emitted = None
    stable_count = 0
    words_stream = []  # collected words you’ll pass to Grammar AI
    last_t = 0.0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.time()
        if now - last_t >= PREDICT_EVERY:
            feats, results = extract_two_hands_126(rgb)

            # only predict if we have at least one hand (non-zero somewhere)
            if any(v != 0.0 for v in feats):
                x = np.array(feats, dtype=np.float32).reshape(1, 1, 126)  # [batch, timesteps=1, features]
                probs = model.predict(x, verbose=0)[0]
                best_idx = int(np.argmax(probs))
                best_prob = float(probs[best_idx])
                best_label = classes[best_idx]

                # show overlay
                cv2.putText(frame, f"{best_label} ({best_prob:.2f})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if best_prob >= CONF_THRESH:
                    pred_buf.append(best_label)
                    voted = majority_vote(list(pred_buf))

                    if voted == last_emitted:
                        stable_count = 0  # already emitted this word
                    else:
                        # check stability
                        if voted is not None and list(pred_buf).count(voted) >= (VOTE_WINDOW // 2 + 1):
                            stable_count += 1
                            if stable_count >= EMIT_STABILITY:
                                words_stream.append(voted)
                                last_emitted = voted
                                stable_count = 0
                                pred_buf.clear()
                                print("→ WORD:", voted)
                else:
                    pred_buf.clear()
                    stable_count = 0
            last_t = now

        # draw hands for visualization
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # display current collected words
        disp = " ".join(words_stream[-8:])
        cv2.putText(frame, f"Words: {disp}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)

        cv2.imshow("Live Sign Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollected words stream:", words_stream)

if __name__ == "__main__":
    main()
