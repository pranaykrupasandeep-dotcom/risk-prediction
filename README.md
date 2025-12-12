🚗 AI-Based Real-Time Accident Risk Prediction System
Using Driver Drowsiness (EAR), Eye Landmarks & Real-Time Risk Scoring

This project detects driver drowsiness using eye aspect ratio (EAR), predicts accident risk, and triggers audio alerts when the driver is sleepy. It uses OpenCV, Dlib, Pygame, and a custom RiskModel.

📌 Features

✔ Real-time eye landmark detection
✔ EAR-based drowsiness detection
✔ Risk level calculation (LOW / MEDIUM / HIGH)
✔ Audio alert using Pygame
✔ Works with any USB / Laptop webcam
✔ Modular project structure
✔ Easy to run in VS Code or Python terminal

📁 Project Structure
AI-RISK-PREDICTION/
│── main.py                 # Main webcam detection script
│── alert.py                # Pygame-based alert module
│── risk_model.py           # Risk prediction logic (optional)
│── utils/                  # Extra helper files (optional)
│── assets/
│     └── alarm.wav         # Alarm sound file
│── shape_predictor_68_face_landmarks.dat
│── README.md

🛠 Technologies Used

Python 3.8+

OpenCV

Dlib (68 landmark model)

Imutils

NumPy

Pygame

📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/AI-Risk-Prediction.git
cd AI-Risk-Prediction

2️⃣ Install Dependencies
Windows
pip install opencv-python numpy pygame imutils
pip install dlib


If dlib fails → install CMake + Visual Studio build tools.

Linux
sudo apt install build-essential cmake
pip install opencv-python numpy pygame imutils dlib

Mac
brew install cmake
pip install opencv-python numpy pygame imutils dlib

3️⃣ Download Dlib Predictor File

Download this file manually:

🔗 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract it and place inside the project folder.

File needed:

shape_predictor_68_face_landmarks.dat

4️⃣ Add Alarm Sound

Place your sound file at:

assets/alarm.wav

▶️ How to Run

Run the main detection script:

python main.py


Press Q to quit the program.

⚙️ How It Works

Detects face using Dlib

Extracts eye landmarks

Calculates EAR (Eye Aspect Ratio)

If EAR < threshold → detects drowsiness

Risk score = EAR + Drowsiness weighted formula

Shows REAL-TIME:

EAR value

Risk Score

Risk Level (LOW/MEDIUM/HIGH)

Red / Yellow / Green color indicators

Plays alarm if drowsy for multiple frames

📊 Risk Model Explanation
risk_score = 0.6 * EAR_RISK + 0.4 * DROWSY_RISK


EAR_RISK → HIGH when eyes are closing

DROWSY_RISK → 1 if continuously sleepy

Risk = 0 to 1

Values rounded to 2 decimals

🛑 Common Errors & Fixes
❌ ModuleNotFoundError: dlib

Install CMake + Build Tools
or download prebuilt wheel for your Python version.

❌ pygame.error: mixer not initialized

Your system has no audio device.

Fix:

pygame.mixer.init(frequency=22050)

❌ FileNotFoundError: 'shape_predictor_68_face_landmarks.dat'

Download file → Put in project folder.

🖼️ Screenshots (Add After Running Project)
📷 Add your webcam detection screenshot here
⚠️ Add risk level display example
🔊 Add alarm demo image

🤝 Contributors

Your Name (Team Lead)

Team Member 1

Team Member 2

Team Member 3

⭐ Support the Project

If you like this project:

⭐ Star this repository


It helps a lot!

