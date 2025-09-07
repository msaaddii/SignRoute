# Import custom modules
from speech_module import SpeechModule  # Handles microphone speech recognition
from sign_displayer import SignDisplay  # Handles showing sentences with sign images
import cv2  # For displaying frames and capturing keypresses

# === INITIALIZE MODULES ===
speech = SpeechModule(model_path="vosk_model")  # Path to your Vosk model folder
sign_display = SignDisplay()  # Sign display window

# Callback function called when speech is recognized
def handle_text(text):
    print(f"Recognized: {text}")  # Print recognized text to console
    sign_display.show_sentence(text, delay=400)  # Show sentence with sign images

# Start speech recognition
speech.start_listening(handle_text)

print("Controls: 's' = stop listening | 'l' = start listening | 'r' = replay signs | 'q' = quit")

# === MAIN LOOP ===
while True:
    sign_display.update()  # Update the sign display window
    key = cv2.waitKey(50) & 0xFF  # Wait 50ms for key press
    if key == ord('s'):
        speech.stop_listening()  # Stop microphone listening
    elif key == ord('l'):
        speech.start_listening(handle_text)  # Start microphone listening
    elif key == ord('r'):
        sign_display.replay_last()  # Replay last sentence
    elif key == ord('q'):
        # Stop everything and exit
        speech.stop_listening()
        sign_display.stop()
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
