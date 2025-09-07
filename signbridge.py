import tkinter as tk
from tkinter import ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
import time
from collections import deque
import numpy as np

from deaf_text import GrammarAI, extract_two_hands_126, model, classes, CONF_THRESH, PREDICT_EVERY, VOTE_WINDOW, EMIT_STABILITY
from sign_displayer import SignDisplay
from speech_module import SpeechModule

class SignBridgeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge Software")
        self.root.geometry("1200x750")


        self.ai = GrammarAI()
        self.sign_display = SignDisplay()
        self.speech = SpeechModule(model_path="vosk_model")
        self.pred_buf = deque(maxlen=VOTE_WINDOW)
        self.words_stream = []
        self.last_emitted = None
        self.stable_count = 0
        self.last_t = 0.0
        self.camera_running = False
        self.cam_index = 0
        self.cap = None


        left_frame = tk.Frame(root, bd=2, relief="groove", padx=10, pady=10)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        tk.Label(left_frame, text="✋ Deaf Side", font=("Arial", 16, "bold")).pack()

        self.deaf_cam_area = tk.Label(left_frame, text="[Camera Feed]", bg="black", fg="white")
        self.deaf_cam_area.pack(pady=10)
        self.cam_width, self.cam_height = 640, 480

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="▶️ Start Cam", command=self.start_camera).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="⏹ Stop Cam", command=self.stop_camera).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="🎤 Set Mic Input", command=self.set_mic).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame, text="📷 Set Camera Input", command=self.set_camera).grid(row=0, column=3, padx=5)

        tk.Label(left_frame, text="Deaf Text (recognized):").pack(anchor="w")
        self.deaf_text = tk.Text(left_frame, height=4, width=50)
        self.deaf_text.pack(pady=5)
        ttk.Button(left_frame, text="✅ Confirm Deaf Text", command=self.confirm_deaf_text).pack(pady=5)

        tk.Label(left_frame, text="Deaf Typing:").pack(anchor="w")
        self.deaf_input = tk.Entry(left_frame, width=50)
        self.deaf_input.pack(pady=5)

        # ---------------- Interviewer Side ----------------
        right_frame = tk.Frame(root, bd=2, relief="groove", padx=10, pady=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        tk.Label(right_frame, text="👂 Interviewer Side", font=("Arial", 16, "bold")).pack()

        self.sign_display_area = tk.Label(right_frame, text="[Sign Images/Avatar]", bg="gray", fg="white")
        self.sign_display_area.pack(pady=10)
        self.sign_width, self.sign_height = 400, 350

        btn_frame2 = tk.Frame(right_frame)
        btn_frame2.pack(pady=5)
        ttk.Button(btn_frame2, text="▶️ Start Listening", command=self.start_speech).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame2, text="⏹ Stop Listening", command=self.stop_speech).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame2, text="🔄 Replay Signs", command=self.sign_display.replay_last).grid(row=0, column=2, padx=5)
        ttk.Button(btn_frame2, text="✅ Confirm Speech Text", command=self.confirm_interviewer_text).grid(row=0, column=3, padx=5)

        tk.Label(right_frame, text="Interviewer Text (recognized):").pack(anchor="w")
        self.interviewer_text = tk.Text(right_frame, height=4, width=50)
        self.interviewer_text.pack(pady=5)

        tk.Label(right_frame, text="Interviewer Typing:").pack(anchor="w")
        self.interviewer_input = tk.Entry(right_frame, width=50)
        self.interviewer_input.pack(pady=5)

        # ---------------- Chat Box ----------------
        chat_frame = tk.Frame(root, bd=2, relief="sunken", padx=10, pady=10)
        chat_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        tk.Label(chat_frame, text="💬 Chat Box", font=("Arial", 14, "bold")).pack(anchor="w")
        self.chat_log = tk.Text(chat_frame, height=8, width=120, state="disabled")
        self.chat_log.pack()
        chat_btn_frame = tk.Frame(chat_frame)
        chat_btn_frame.pack(pady=5)
        ttk.Button(chat_btn_frame, text="Send Deaf →", command=self.send_deaf_message).grid(row=0, column=0, padx=5)
        ttk.Button(chat_btn_frame, text="← Send Interviewer", command=self.send_interviewer_message).grid(row=0, column=1, padx=5)

        root.grid_rowconfigure(0, weight=3)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self.update_sign_display()

    # ---------------- Camera ----------------
    def camera_loop(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.camera_running = True
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now = time.time()
            if now - self.last_t >= PREDICT_EVERY:
                feats, results = extract_two_hands_126(rgb_frame)
                if any(v != 0.0 for v in feats):
                    x = np.array(feats, dtype=np.float32).reshape(1,1,126)
                    probs = model.predict(x, verbose=0)[0]
                    best_idx = int(np.argmax(probs))
                    best_prob = float(probs[best_idx])
                    best_label = classes[best_idx]
                    if best_prob >= CONF_THRESH:
                        self.pred_buf.append(best_label)
                        voted = self.majority_vote(list(self.pred_buf))
                        if voted != self.last_emitted and list(self.pred_buf).count(voted) >= (VOTE_WINDOW//2 + 1):
                            self.stable_count += 1
                            if self.stable_count >= EMIT_STABILITY:
                                self.words_stream.append(voted)
                                cleaned = self.ai.fix(self.words_stream)
                                self.deaf_text.delete(1.0, tk.END)
                                self.deaf_text.insert(tk.END, cleaned)
                                self.last_emitted = voted
                                self.stable_count = 0
                                self.pred_buf.clear()
                self.last_t = now

            # Display in Tkinter
            img = Image.fromarray(rgb_frame)
            img = img.resize((self.cam_width, self.cam_height))
            imgtk = ImageTk.PhotoImage(image=img)
            self.deaf_cam_area.imgtk = imgtk
            self.deaf_cam_area.config(image=imgtk)

            time.sleep(0.03)
        self.cap.release()

    def start_camera(self):
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_camera(self):
        self.camera_running = False

    def set_camera(self):
        idx = simpledialog.askinteger("Camera Input", "Enter camera index (0,1,2...):", parent=self.root)
        if idx is not None:
            self.cam_index = idx


    def start_speech(self):
        self.speech.start_listening(self.handle_speech)

    def stop_speech(self):
        self.speech.stop_listening()

    def set_mic(self):
        idx = simpledialog.askinteger("Mic Input", "Enter microphone device index:", parent=self.root)
        if idx is not None:
            self.speech.stop_listening()
            self.speech.samplerate = 16000
            # You can implement advanced mic selection in SpeechModule if needed
            print(f"Mic set to device index {idx}")

    def handle_speech(self, text):
        self.interviewer_text.delete(1.0, tk.END)
        self.interviewer_text.insert(tk.END, text)
        self.sign_display.show_sentence(text)

    def update_sign_display(self):

        frame = self.sign_display.update()  # get next frame if available
        if frame is not None and self.sign_display.last_signs:
            # Get the current sign image
            word, img = self.sign_display.last_signs[self.sign_display.current_idx - 1]  # last shown
            if img is not None:
                img_resized = cv2.resize(img, (self.sign_width, self.sign_height))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(img_pil)

                # Update Tkinter label
                self.sign_display_area.imgtk = imgtk
                self.sign_display_area.config(image=imgtk)

        # schedule next update
        self.root.after(50, self.update_sign_display)

    def send_deaf_message(self):
        msg = self.deaf_input.get().strip()
        if msg:
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"✋ Deaf: {msg}\n")
            self.chat_log.config(state="disabled")
            self.deaf_input.delete(0, tk.END)

    def send_interviewer_message(self):
        msg = self.interviewer_input.get().strip()
        if msg:
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"👂 Interviewer: {msg}\n")
            self.chat_log.config(state="disabled")
            self.interviewer_input.delete(0, tk.END)

    def confirm_deaf_text(self):
        text = self.deaf_text.get(1.0, tk.END).strip()
        if text:
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"✋ Deaf: {text}\n")
            self.chat_log.config(state="disabled")
            self.deaf_text.delete(1.0, tk.END)

    def confirm_interviewer_text(self):
        text = self.interviewer_text.get(1.0, tk.END).strip()
        if text:
            self.chat_log.config(state="normal")
            self.chat_log.insert(tk.END, f"👂 Interviewer: {text}\n")
            self.chat_log.config(state="disabled")
            self.interviewer_text.delete(1.0, tk.END)


    def majority_vote(self, labels):
        if not labels: return None
        vals, counts = np.unique(labels, return_counts=True)
        return vals[np.argmax(counts)]

if __name__ == "__main__":
    root = tk.Tk()
    app = SignBridgeUI(root)
    root.mainloop()
