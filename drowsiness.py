#!/usr/bin/env python3
"""
Cleaned, defensive drowsiness detection script.

Modes:
  --simulate : deterministic simulation (works without OpenCV/dlib)
  --live     : live camera detection (requires opencv, dlib, imutils, numpy)

This version:
 - fixes syntax errors
 - robustly opens camera (tries int index, numeric-string, file path)
 - computes EAR and a simple risk score
 - non-blocking alert playback with cooldown
 - helpful console messages when dependencies are 
from threading import Thread
import time
import argparse
import os
import sys
import math
import platform

# Optional imports (fail gracefully)
HAVE_CV2 = HAVE_DLIB = HAVE_IMUTILS = HAVE_NUMPY = HAVE_SCIPY = False
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
    import imutils
    from imutils import face_utils
    HAVE_IMUTILS = True
except Exception:
    imutils = None
    face_utils = None

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

# winsound fallback for Windows beep
try:
    import winsound
    HAVE_WINSOUND = True
except Exception:
    winsound = None
    HAVE_WINSOUND = False

# ----------------- helpers -----------------
def euclidean(a, b):
    """Robust euclidean distance between two 2D points (iterable)."""
    try:
        # prefer numpy if available
        if HAVE_NUMPY and np is not None:
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            return float(np.linalg.norm(a - b))
        # otherwise math.hypot
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except Exception:
        # last-resort manual calculation
        try:
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            return (dx * dx + dy * dy) ** 0.5
        except Exception:
            return 0.0


def eye_aspect_ratio(eye):
    """Compute EAR for an eye represented by 6 (x,y) points."""
    try:
        if eye is None:
            return 0.0
        # ensure we can index
        pts = list(eye)
        if len(pts) < 6:
            return 0.0
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0


def play_beep():
    """Non-blocking beep: on Windows use winsound, otherwise print a marker."""
    def _beep():
        try:
            if HAVE_WINSOUND and winsound:
                winsound.Beep(1000, 350)
            else:
                # try system bell first
                sys.stdout.write("\a")
                sys.stdout.flush()
                # as fallback, print tag
                print("[ALERT SOUND]")
        except Exception:
            print("[ALERT SOUND]")

    # run in background thread so UI loop isn't blocked
    Thread(target=_beep, daemon=True).start()


def shape_to_np_manual(shape):
    """Convert dlib shape to numpy-like list of (x,y). Works without imutils."""
    try:
        # if shape has num_parts attribute
        num = getattr(shape, "num_parts", None)
        if num is not None:
            pts = []
            for i in range(int(num)):
                p = shape.part(i)
                pts.append((p.x, p.y))
            return pts
        # fallback to iterable .parts()
        parts = list(shape.parts())
        return [(p.x, p.y) for p in parts]
    except Exception:
        raise


def get_eye_indices():
    """
    Return (lStart, lEnd), (rStart, rEnd) for 68-point model.
    If imutils.face_utils provides mapping, use it.
    """
    # Common imutils attribute names
    if face_utils is not None:
        for attr in ("FACIAL_LANDMARKS_IDXS", "FACE_LANDMARKS_IDXS"):
            try:
                if hasattr(face_utils, attr):
                    mapping = getattr(face_utils, attr)
                    if isinstance(mapping, dict) and "left_eye" in mapping and "right_eye" in mapping:
                        ls, le = map(int, mapping["left_eye"])
                        rs, re = map(int, mapping["right_eye"])
                        return (ls, le), (rs, re)
            except Exception:
                pass
    # fallback to standard 68-point indices (left_eye: 42-48, right_eye: 36-42)
    return (42, 48), (36, 42)


def try_open_capture(camera_arg, tries=3, delay=0.25):
    """
    Try to open VideoCapture from camera_arg. Accept int, numeric-string, file path or URL.
    Returns cv2.VideoCapture object if opened, else None.
    """
    if cv2 is None:
        return None

    candidates = []
    if camera_arg is None:
        candidates.append(0)
    else:
        # if it's an int, try it
        try:
            if isinstance(camera_arg, int):
                candidates.append(camera_arg)
            elif isinstance(camera_arg, str) and camera_arg.strip().isdigit():
                candidates.append(int(camera_arg.strip()))
            elif isinstance(camera_arg, str):
                # try as file path or device string first
                candidates.append(camera_arg)
        except Exception:
            pass

    # always add 0 as a last resort
    if 0 not in candidates:
        candidates.append(0)

    for cand in candidates:
        for attempt in range(tries):
            try:
                cap = cv2.VideoCapture(cand)
                time.sleep(delay)  # let backend initialize
                if cap is not None and cap.isOpened():
                    # sometimes we need to grab a frame to confirm it works
                    ret, _ = cap.read()
                    if ret:
                        return cap
                    else:
                        # release and continue trying
                        try:
                            cap.release()
                        except Exception:
                            pass
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception:
                pass
    return None


# ----------------- Risk model -----------------
class RiskModel:
    def __init__(self, ear_weight=0.6, drowsy_weight=0.4):
        self.ear_weight = ear_weight
        self.drowsy_weight = drowsy_weight

    def predict(self, ear, drowsy):
        try:
            ear = float(ear)
        except Exception:
            ear = 0.2
        ear_risk = max(0.0, min(1.0, (0.3 - ear) / 0.3))
        drowsy_risk = 1.0 if bool(drowsy) else 0.0
        score = (self.ear_weight * ear_risk) + (self.drowsy_weight * drowsy_risk)
        return round(float(score), 2)


# ----------------- Simulation mode -----------------
def simulate_run(frames=70, thresh=0.3, consec=5, delay=0.02):
    """Runs a deterministic EAR simulation without camera."""
    seq = [0.32] * 30 + [0.25] * 10 + [0.32] * 20 + [0.25] * 10 + [0.32] * 10
    counter = 0
    risk_model = RiskModel()
    print("Simulation running. Ctrl+C to stop.\n")
    try:
        for i, ear in enumerate(seq[:frames], start=1):
            drowsy = False
            print(f"Frame {i}: EAR={ear:.2f}", end="")
            if ear < thresh:
                counter += 1
                if counter >= consec:
                    drowsy = True
                    print("  >> ALERT: Simulated drowsiness detected", end="")
                    play_beep()
                    counter = 0
                else:
                    print("", end="")
            else:
                counter = 0

            risk = risk_model.predict(ear, drowsy)
            print(f"  Risk={risk:.2f}")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    print("Simulation finished\n")


# ----------------- Live mode -----------------
def live_run(predictor_path=None, camera_index=0, eye_thresh=0.3, consec_frames=30, alert_cooldown=6.0):
    """Live camera detection mode."""
    missing = []
    if not HAVE_CV2:
        missing.append("opencv-python (cv2)")
    if not HAVE_DLIB:
        missing.append("dlib")
    if not HAVE_NUMPY:
        missing.append("numpy")
    # imutils is optional; we can operate without it

    if missing:
        print("Cannot run live mode. Missing:", ", ".join(missing))
        print("Install required packages or run --simulate.")
        return

    if predictor_path is None:
        predictor_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.isfile(predictor_path):
        print(f"Predictor not found: {predictor_path}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return

    # initialize dlib
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print("Failed to initialize dlib detector/predictor:", e)
        return

    # try to open camera robustly
    cap = try_open_capture(camera_index)
    if cap is None:
        print("Could not open any camera/source. Tried index/path:", camera_index)
        print("If using a laptop, try camera index 0 or close other apps that might use the camera.")
        return

    # get eye indices
    (lStart, lEnd), (rStart, rEnd) = get_eye_indices()

    counter = 0
    last_alert_time = 0.0
    risk_model = RiskModel()
    print("Live mode started. Press 'q' in the video window to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame capture failed. Stopping.")
                break

            # optional resize for performance
            try:
                if imutils is not None:
                    frame = imutils.resize(frame, width=640)
                else:
                    h, w = frame.shape[:2]
                    if w > 640:
                        new_h = int(h * (640.0 / w))
                        frame = cv2.resize(frame, (640, new_h))
            except Exception:
                pass

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            ear = 0.0
            drowsy = False

            for rect in rects:
                try:
                    shape = predictor(gray, rect)
                    # convert shape to list of points
                    try:
                        # prefer imutils.shape_to_np if available
                        if HAVE_IMUTILS and face_utils is not None and hasattr(face_utils, "shape_to_np"):
                            shape_np = face_utils.shape_to_np(shape)
                            # shape_np will be numpy array of shape (68,2)
                            # convert to list of tuples for our euclidean function
                            shape_pts = [(int(x), int(y)) for (x, y) in shape_np]
                        else:
                            shape_pts = shape_to_np_manual(shape)
                    except Exception:
                        # fallback to manual
                        shape_pts = shape_to_np_manual(shape)

                    # guard indices
                    if (lEnd > len(shape_pts)) or (rEnd > len(shape_pts)):
                        # unexpected mapping for current shape; skip this face
                        continue

                    leftEye = shape_pts[lStart:lEnd]
                    rightEye = shape_pts[rStart:rEnd]

                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0

                    if ear < eye_thresh:
                        counter += 1
                        if counter >= consec_frames:
                            now = time.time()
                            if now - last_alert_time >= alert_cooldown:
                                print(f"\n[!] Drowsiness detected (EAR={ear:.3f}). Alerting.")
                                play_beep()
                                last_alert_time = now
                            drowsy = True
                            counter = 0
                        else:
                            drowsy = False
                    else:
                        counter = 0
                        drowsy = False

                except Exception as e:
                    # if one face processing fails, continue to next detected face
                    print("Warning: failed to process detected face:", e)
                    continue
                # only process first face (remove break to handle multiple faces)
                break

            # compute risk and display
            risk_score = risk_model.predict(ear, drowsy)
            if risk_score < 0.3:
                risk_level = "LOW"
                color = (0, 255, 0)
            elif risk_score < 0.6:
                risk_level = "MEDIUM"
                color = (0, 255, 255)
            else:
                risk_level = "HIGH"
                color = (0, 0, 255)

            # overlay text
            try:
                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Risk: {risk_level} ({risk_score:.2f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                state_text = "DROWSY" if drowsy else "ATTENTIVE"
                cv2.putText(frame, state_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception:
                pass

            cv2.imshow("Drowsiness / Risk Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested by user.")
                break

    except KeyboardInterrupt:
        print("Stopped by user (KeyboardInterrupt).")

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Camera and windows released.")


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Drowsiness detection (simulate or live)")
    p.add_argument("--simulate", action="store_true", help="Run deterministic simulation (no camera required)")
    p.add_argument("--live", action="store_true", help="Run live camera detection (requires dependencies)")
    p.add_argument("--predictor", type=str, default="shape_predictor_68_face_landmarks.dat", help="Path to dlib predictor (.dat)")
    p.add_argument("--camera", type=str, default="0", help="Camera index (0) or video file path")
    p.add_argument("--thresh", type=float, default=0.3, help="EAR threshold for eye-closed detection")
    p.add_argument("--frames", type=int, default=30, help="Consecutive frames to trigger drowsiness")
    p.add_argument("--cooldown", type=float, default=6.0, help="Alert cooldown (seconds)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.simulate:
        simulate_run(frames=200, thresh=args.thresh, consec=5, delay=0.02)
        sys.exit(0)

    if args.live:
        cam_arg = None
        # parse camera arg as int where possible
        try:
            if args.camera is not None and args.camera.strip().isdigit():
                cam_arg = int(args.camera.strip())
            else:
                cam_arg = args.camera
        except Exception:
            cam_arg = args.camera
        live_run(predictor_path=args.predictor, camera_index=cam_arg, eye_thresh=args.thresh,
                 consec_frames=args.frames, alert_cooldown=args.cooldown)
        sys.exit(0)

    # default behaviour: print usage
    print("No mode specified. Use --simulate or --live. Example:")
    print("  python drowsiness.py --simulate")
    print("  python drowsiness.py --live --camera 0 --predictor shape_predictor_68_face_landmarks.dat")
