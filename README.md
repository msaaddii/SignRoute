# SignRoute

SignRoute is a real-time communication tool for deaf and hearing users.  
It translates **sign language into text**, and **speech into sign images**, enabling smooth conversation during interviews or chats.

---

## ✨ Features
- **Real-time Sign Recognition** → Converts hand signs to text using MediaPipe + TensorFlow.  
- **Speech-to-Sign** → Converts spoken words to corresponding sign images.  
- **Grammar AI** → Cleans and corrects recognized text automatically.  
- **Interactive Chat** → Deaf and interviewer can confirm and send messages.  
- **Customizable Inputs** → Select camera and microphone devices.  

---

## 🛠 Technologies Used
- Python 3  
- Tkinter (GUI)  
- OpenCV (Camera & Image Handling)  
- MediaPipe (Hand Tracking)  
- TensorFlow / Keras (Sign Recognition Model)  
- Vosk (Speech Recognition)  
- Pillow (Image display in Tkinter)  
- NumPy  
- scikit-learn  

---

## 📥 Installation (ZIP Users)

1. Go to the [SignRoute GitHub page](https://github.com/msaaddii/SignRoute).  
2. Click the green **Code** button → **Download ZIP**.  
3. Extract the ZIP file on your computer (e.g., Desktop).  
4. Open **Command Prompt (cmd)** inside the extracted folder.  

---

## ⚙️ Setup

1. Make sure you have **Python 3.9+** installed.  
2. (Optional) Create a virtual environment:  

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:  

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following are in your project folder:  
   - `vosk_model/` folder (download from [Vosk Models](https://alphacephei.com/vosk/models), e.g. `vosk-model-small-en-us-0.15`)  
   - `sign_model.h5` (trained model file)  
   - `label_classes.npy` (label mappings)  

---

## 🚀 Usage

Run the main UI:

```bash
python signroute_ui.py
```

- Use the buttons to start/stop the camera or microphone.  
- Recognized text appears in the **Deaf Side**, and sign images display for the **Interviewer Side**.  
- Confirm messages to send them to the chat box.  

---

## 📌 Notes
- If the camera **blinks** or the microphone has **no sound**, that’s just a demo/video issue.  
  ✅ The software itself works fine.  
- All code is fully commented for beginners who want to learn how it works.  

---

## 📄 License
This project is released under the **MIT License**.