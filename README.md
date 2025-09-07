# SignBridge

SignBridge is a real-time communication tool for deaf and hearing users. It translates sign language into text, and speech into sign images, enabling smooth conversation during interviews or chats.

---

## Features

- **Real-time Sign Recognition**: Converts hand signs to text using MediaPipe + TensorFlow.
- **Speech-to-Sign**: Converts spoken words to corresponding sign images.
- **Grammar AI**: Cleans and corrects recognized text automatically.
- **Interactive Chat**: Deaf and interviewer can confirm and send messages.
- **Customizable Inputs**: Select camera and microphone devices.

---

## Technologies Used

- **Python 3**
- **Tkinter** (GUI)
- **OpenCV** (Camera & Image Handling)
- **MediaPipe** (Hand Tracking)
- **TensorFlow / Keras** (Sign Recognition Model)
- **Vosk** (Speech Recognition)
- **Pillow** (Image display in Tkinter)
- **NumPy**

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/msaaddii/SignBridge.git
cd SignBridge
Create a virtual environment:

bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Ensure the following are in your project root:

vosk_model/ folder

sign_model.h5

label_classes.npy

Usage
Run the UI:

bash
Copy code
python main_ui.py
Use the buttons to start/stop the camera or microphone.

Recognized text appears in the Deaf Side, and sign images display for the interviewer.

Confirm messages to send them to the chat box.

License
This project is released under the MIT License.
