"""
alarm.py - Clean, single-file drowsiness alarm with simulation fallback.

Usage:
  python alarm.py --simulate
  python alarm.py --live --predictor path/to/shape_predictor_68_face_landmarks.dat
"""

import argparse
import time
import os

# Optional dependencies: attempt to import, but tolerate absence
HAVE_CV2 = HAVE_DLIB = HAVE_NUMPY = HAVE_SCIPY = False
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None

try:
    import dlib
    HAVE_DLIB = True
except Exception:
    dlib = None

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    np = None

try:
    from scipy.spatial import distance as dist
    HAVE_SCIPY = True
except Exception:
    dist = None

# winsound is used as a simple Windows beep fallback
try:
    import winsound
    HAVE_WINSOUND = True
except Exception:
    winsound = None
    HAVE_WINSOUND = False


def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio for a 6-point eye."""
    try:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
    except Exception:
        # fallback without scipy
        def _euclid(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        A = _euclid(eye[1], eye[5])
        B = _euclid(eye[2], eye[4])
        C = _euclid(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def play_beep():
    if HAVE_WINSOUND and winsound is not None:
        try:
            winsound.Beep(1000, 350)
            return
        except Exception:
            pass
    print("[ALERT SOUND]")


def simulate_run(frames=70, thresh=0.3, consec=5, delay=0.02):
    EAR_sequence = [0.32] * 30 + [0.25] * 10 + [0.32] * 20 + [0.25] * 10
    counter = 0
    print("Running simulation (no camera required). Ctrl+C to stop.")
    try:
        for i, ear in enumerate(EAR_sequence[:frames], start=1):
            print(f"Frame {i}: EAR={ear:.2f}")
            if ear < thresh:
                counter += 1
                if counter >= consec:
                    print("ALERT: simulated driver drowsy")
                    play_beep()
                    counter = 0
            else:
                counter = 0
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    print("Simulation finished")


def live_run(predictor_path=None, eye_ar_thresh=0.25, eye_ar_consec_frames=48, camera_index=0):
    missing = []
    if not HAVE_CV2:
        missing.append('opencv-python (cv2)')
    if not HAVE_DLIB:
        missing.append('dlib')
    if not HAVE_NUMPY:
        missing.append('numpy')
    if dist is None:
        missing.append('scipy')

    if missing:
        print("Live mode requires additional packages:", ", ".join(missing))
        print("Install them or run with --simulate. Example: pip install opencv-python dlib numpy scipy imutils")
        return

    try:
        import imutils
        from imutils import face_utils
    except Exception:
        print("Please install 'imutils' for live mode: pip install imutils")
        return

    if predictor_path is None:
        predictor_path = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.isfile(predictor_path):
        print(f"Predictor file not found: {predictor_path}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print(f"Failed to initialize dlib detector/predictor: {e}")
        return

    # Use CAP_DSHOW on Windows for slightly better camera compatibility
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Unable to open camera index {camera_index}")
        return

    print("Starting live capture. Press 'q' in the window to quit.")
    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; exiting")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # imutils/face_utils uses (left, right) indexing; keep consistent
                leftEye = shape[42:48]
                rightEye = shape[36:42]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < eye_ar_thresh:
                    counter += 1
                    if counter >= eye_ar_consec_frames:
                        print("ALERT: driver drowsy (live)")
                        play_beep()
                        counter = 0
                else:
                    counter = 0

            cv2.imshow('DrowsinessDetection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Live capture stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description='Drowsiness alarm (simulate or live)')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--simulate', action='store_true', help='Run simulation mode (default)')
    group.add_argument('--live', action='store_true', help='Attempt live camera + dlib mode')
    p.add_argument('--predictor', type=str, default=None, help='Path to dlib 68-point predictor .dat')
    p.add_argument('--thresh', type=float, default=0.25, help='EAR threshold')
    p.add_argument('--consec', type=int, default=48, help='Consecutive frames threshold')
    p.add_argument('--camera', type=int, default=0, help='Camera index for cv2.VideoCapture')
    return p.parse_args()


def main():
    args = parse_args()
    if args.live:
        live_run(predictor_path=args.predictor, eye_ar_thresh=args.thresh, eye_ar_consec_frames=args.consec, camera_index=args.camera)
    else:
        simulate_run()


if __name__ == '__main__':
    main()
