# Import necessary libraries
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for hand landmark detection
import csv  # To save hand landmark data in CSV files
import os  # To create directories and handle file paths
import time  # For countdowns and controlling save intervals

# Directory to save collected sign language data
DATA_DIR = "sign_language_data"

# List of sign labels to collect
SIGNS = [
    "Hello", "My", "Name", "Is", "I", "You", "Work", "Job", "Experience", "Skill",
    "Yes", "No", "ThankYou", "Please", "Sorry", "Good", "Bad", "Like", "Want", "Can",
    "Why", "What", "Where", "How", "Strong", "Weak", "Learn", "Team", "Company", "Love"
]

# Number of samples to collect per sign
SAMPLES_PER_SIGN = 400

# Minimum time interval (in seconds) between saving consecutive samples
SAVE_INTERVAL = 0.05

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Utility to draw hand landmarks
hands = mp_hands.Hands(
    static_image_mode=False,  # Video feed mode
    max_num_hands=2,  # Track up to 2 hands
    min_detection_confidence=0.5  # Minimum detection confidence
)

# Start video capture (webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

print("Press 'q' to quit anytime.")  # Instruction for user

current_idx = 0  # Index for iterating through SIGNS

# Main loop to collect data for each sign
while current_idx < len(SIGNS):
    current_label = SIGNS[current_idx]  # Current sign label
    csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")  # Path to save CSV

    # Skip sign if CSV already exists
    if os.path.isfile(csv_path):
        print(f"Skipping {current_label} (already recorded).")
        current_idx += 1
        continue

    # Countdown before starting recording
    for i in range(5, 0, -1):
        print(f"{current_label} starts in {i} sec...")
        time.sleep(1)
        # Allow quitting during countdown
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Setup CSV file with header
    header = ["label"]  # First column is label
    # Add columns for left hand landmarks (21 points × 3 coordinates)
    header += [f"L_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    # Add columns for right hand landmarks
    header += [f"R_{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    sample_count = 0  # Track how many samples have been recorded
    last_save_time = time.time()  # Track last save time

    # Loop to capture samples for current sign
    while sample_count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()  # Read a frame from webcam
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror image for user-friendly view

        # Allow quitting early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = hands.process(rgb_frame)  # Process frame with MediaPipe

        landmark_row = [current_label]  # Start row with label
        left_hand = [0.0] * 63  # Initialize left hand landmarks
        right_hand = [0.0] * 63  # Initialize right hand landmarks

        # If hands are detected, extract landmarks
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                if label == "Left":
                    left_hand = coords
                else:
                    right_hand = coords

                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Add landmark coordinates to row
        landmark_row.extend(left_hand)
        landmark_row.extend(right_hand)

        # Save row if interval has passed and hand data exists
        if time.time() - last_save_time > SAVE_INTERVAL and any(v != 0.0 for v in landmark_row[1:]):
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(landmark_row)
            sample_count += 1
            last_save_time = time.time()

        # Display current label and sample count
        cv2.putText(
            frame,
            f"Label: {current_label} | Samples: {sample_count} / {SAMPLES_PER_SIGN}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.imshow("Sign Language Data Collection", frame)  # Show video feed

        time.sleep(0.01)  # Small delay to reduce CPU load

    print(f"Finished {current_label} ({sample_count} samples)")
    current_idx += 1  # Move to next sign

# Release resources when done
cap.release()
cv2.destroyAllWindows()
