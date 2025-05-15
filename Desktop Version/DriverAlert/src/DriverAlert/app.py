import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER, LEFT
import cv2
import threading
import numpy as np
import face_recognition
from scipy.spatial import distance as dist
from playsound import playsound
from pathlib import Path

# Constants
MIN_EAR = 0.30
EYE_AR_CONSEC_FRAMES = 10
COUNTER = 0
ALARM_ON = False

# Alarm function
def sound_alarm():
    sound_path = Path(__file__).parent / "assets" / "alert.mp3"
    playsound(str(sound_path.resolve()))

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

class DriverDrowsy(toga.App):
    def startup(self):
        self.running = False

        # Header
        header = toga.Label(
            "Driver Drowsiness Detector",
            style=Pack(font_size=20, font_weight="bold", text_align=CENTER, padding=(10, 0))
        )
        sub_header = toga.Label(
            "Monitor drowsiness levels in real time.",
            style=Pack(font_size=12, text_align=CENTER, padding=(0, 0, 20, 0), color="#666")
        )

        # EAR label
        self.ear_label = toga.Label("EAR: --", style=Pack(padding=10, font_size=14, text_align=LEFT))
        self.status_label = toga.Label("Press Start to begin drowsiness detection.",
                                       style=Pack(padding=10, font_size=12, color="#007bff"))

        # Buttons
        self.start_button = toga.Button(
            "Start",
            on_press=self.start_detection,
            style=Pack(padding=10, background_color='#28a745', color='white', flex=1)
        )

        self.stop_button = toga.Button(
            "Stop",
            on_press=self.stop_detection,
            style=Pack(padding=10, background_color='#dc3545', color='white', flex=1)
        )

        button_box = toga.Box(children=[self.start_button, self.stop_button],
                              style=Pack(direction=ROW, alignment=CENTER, padding=10))

        # Card-style Box (simplified to avoid unsupported styles)
        card_box = toga.Box(style=Pack(direction=COLUMN, padding=20, background_color="#f8f9fa"))
        card_box.add(self.status_label)
        card_box.add(self.ear_label)
        card_box.add(button_box)

        # Main layout
        main_box = toga.Box(style=Pack(direction=COLUMN, alignment=CENTER, padding=20))
        main_box.add(header)
        main_box.add(sub_header)
        main_box.add(card_box)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def start_detection(self, widget):
        if not self.running:
            self.running = True
            self.set_status("Drowsiness detection started...", "#007bff")
            threading.Thread(target=self.drowsiness_detection, daemon=True).start()

    def stop_detection(self, widget):
        self.running = False
        self.set_status("Drowsiness detection stopped.", "#6c757d")

    def set_status(self, message, color="#007bff"):
        self.main_window.app.loop.call_soon_threadsafe(setattr, self.status_label, "text", message)
        self.main_window.app.loop.call_soon_threadsafe(setattr, self.status_label.style, "color", color)

    def set_ear(self, value, color="#28a745"):
        self.main_window.app.loop.call_soon_threadsafe(setattr, self.ear_label, "text", f"EAR: {value:.2f}")
        self.main_window.app.loop.call_soon_threadsafe(setattr, self.ear_label.style, "color", color)

    def drowsiness_detection(self):
        global COUNTER, ALARM_ON
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            for face_landmark in face_landmarks_list:
                leftEye = face_landmark['left_eye']
                rightEye = face_landmark['right_eye']
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                ear_color = "#28a745" if ear >= MIN_EAR else "#dc3545"
                self.set_ear(ear, ear_color)

                cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 0), 2)
                cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 0), 2)

                if ear < MIN_EAR:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            threading.Thread(target=sound_alarm, daemon=True).start()
                        cv2.putText(frame, "ALERT! You are feeling sleepy!", (50, 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                        self.set_status("⚠️ ALERT! You are feeling sleepy!", "#dc3545")
                else:
                    COUNTER = 0
                    ALARM_ON = False

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.set_status("Detection stopped.", "#6c757d")

def main():
    return DriverDrowsy()
