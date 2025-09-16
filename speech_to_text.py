"""
File: speech_to_text.py
Description:
    Converts Speech → Text for the Interviewer Side.
    - Uses microphone input with Vosk model.
    - Listens and transcribes spoken words in real time.
    - Sends recognized text to the UI for chat + sign display.
"""

# Import necessary libraries
import queue  # Thread-safe queue to store audio chunks
import sounddevice as sd  # Capture microphone audio
import vosk  # Offline speech recognition library
import json  # Parse Vosk recognition results
import threading  # Run recognition in background thread

# === SPEECH MODULE ===
class SpeechModule:
    """
    Handles live speech recognition using Vosk and streams results via a callback.
    """
    def __init__(self, model_path="vosk_model", samplerate=16000):
        # Load Vosk model
        self.model = vosk.Model(model_path)
        self.samplerate = samplerate  # Audio sampling rate
        self.q = queue.Queue()  # Queue to store incoming audio frames
        self.running = False  # Flag to control recognition loop
        self.callback = None  # User-defined function to receive recognized text
        self.thread = None  # Background recognition thread
        self.stream = None  # Audio input stream

    # Callback for sounddevice stream: pushes raw audio into the queue
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status, flush=True)
        self.q.put(bytes(indata))  # Convert numpy array to bytes

    # Background loop for continuous recognition
    def _recognition_loop(self):
        rec = vosk.KaldiRecognizer(self.model, self.samplerate)
        while self.running:
            try:
                data = self.q.get(timeout=1)  # Wait for audio chunk
            except queue.Empty:
                continue
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text and self.callback:  # Emit text if valid
                    self.callback(text)

    # Start listening in a separate thread and call 'callback' with recognized text
    def start_listening(self, callback):
        if self.running:
            return  # Already running
        self.callback = callback
        self.running = True
        # Start recognition thread
        self.thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.thread.start()
        # Open microphone input stream
        self.stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()

    # Stop listening and clean up resources
    def stop_listening(self):
        if self.running:
            self.running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
