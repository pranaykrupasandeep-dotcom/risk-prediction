"""
FINAL DROWSINESS DETECTION CODE (with visual alert)
---------------------------------------------------
• Always opens webcam
• Always runs in LIVE mode
• No simulate mode
• No arguments
• Alerts with beep + flashing text
• Shows EAR on screen
• Clean + stable + simple
---------------------------------------------------
Before running, make sure this file exists:
shape_predictor_68_face_landmarks.dat
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread

# Sound alert
try:
    import winsound
    HAVE_WINSOUND = True
except:
    HAVE_WINSOUND = False


# ------------ EAR CALCULATION ------------
def eye_aspect_ratio(eye):
    """Compute eye aspect ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ------------ BEEP SOUND ------------
def play_beep():
    if HAVE_WINSOUND:
        winsound.Beep(1000, 500)
    else:
        print("[ALERT SOUND]")


# ------------ MAIN DROWSINESS FUNCTION ------------
def start_drowsiness_detection():

    predictor_path = "shape_predictor_68_face_landmarks.dat"

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    except:
        print("\nERROR: shape_predictor_68_face_landmarks.dat NOT FOUND!")
        print("Download it from:")
        print("https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    # Faster webcam init on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Webcam not found!")
        return

    EAR_THRESHOLD = 0.25
    FRAME_LIMIT = 20
    counter = 0
    alert_on = False

    print("\nWebcam detection started... Press 'Q' to exit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read webcam frame.")
                break

            frame = cv2.resize(frame, (900, 600))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                points = np.zeros((68, 2), dtype=int)

                for i in range(68):
                    points[i] = (landmarks.part(i).x, landmarks.part(i).y)

                left_eye = points[36:42]
                right_eye = points[42:48]

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2.0

                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= FRAME_LIMIT:
                        print("ALERT: DROWSINESS DETECTED!")
                        Thread(target=play_beep).start()
                        alert_on = True
                        counter = 0
                else:
                    counter = 0
                    alert_on = False

            # Visual flashing alert overlay
            if alert_on:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 0, 255), 4)

            cv2.imshow("Real-Time Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    cap.release()
    cv2.destroyAllWindows()


# ------------ START THE PROGRAM ------------
if __name__ == "__main__":
    start_drowsiness_detection()
