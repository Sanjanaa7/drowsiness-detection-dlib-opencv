# --- drowsy_detector.py ---
import cv2
import dlib
import numpy as np
import pandas as pd
import os
import kagglehub
import shutil
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

# ===============================================
# 1️⃣ AUTO-DOWNLOAD MODEL IF NOT PRESENT
# ===============================================
def ensure_model():
    model_file = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_file):
        print("⚙️ Model file not found. Downloading from Kaggle...")
        path = kagglehub.dataset_download("sergiovirahonda/shape-predictor-68-face-landmarksdat")
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".dat"):
                    source = os.path.join(root, file)
                    shutil.copy(source, model_file)
                    print(f"✅ Model downloaded and saved as: {model_file}")
    else:
        print("✅ Model file already exists.")
    return model_file

# ===============================================
# 2️⃣ HELPER FUNCTIONS
# ===============================================
def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = euclidean_dist(mouth[2], mouth[10])  # 51,59
    B = euclidean_dist(mouth[4], mouth[8])   # 53,57
    C = euclidean_dist(mouth[0], mouth[6])   # 49,55
    return (A + B) / (2.0 * C)

# ===============================================
# 3️⃣ INITIAL SETUP
# ===============================================
model_path = ensure_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

video_path = "driver_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("❌ Could not open the video. Check the file name and path!")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"🎥 Video loaded: {frame_count} frames, {fps:.2f} FPS, {duration:.1f} sec")

# Writer for saving annotated video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps,
                      (int(cap.get(3)), int(cap.get(4))))

# ===============================================
# 4️⃣ CONSTANTS & BUFFERS
# ===============================================
EAR_THRESH = 0.25  # Eye Aspect Ratio threshold
MAR_THRESH = 0.7   # Mouth Aspect Ratio threshold
CONSEC_FRAMES = int(fps * 1.5)  # consecutive frames threshold
ear_history = deque(maxlen=150)
mar_history = deque(maxlen=150)
frame_idx = 0
drowsy_frames = 0
events = []

# ===============================================
# 5️⃣ MAIN LOOP (with tqdm progress bar)
# ===============================================
pbar = tqdm(total=frame_count, desc="Analyzing video", unit="frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    pbar.update(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = landmarks[36:42]
        rightEye = landmarks[42:48]
        mouth = landmarks[48:68]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        ear_history.append(ear)
        mar_history.append(mar)

        # draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Drowsiness logic
        if ear < EAR_THRESH or mar > MAR_THRESH:
            drowsy_frames += 1
            cv2.putText(frame, "⚠️ DROWSY!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if drowsy_frames == CONSEC_FRAMES:
                timestamp = frame_idx / fps
                events.append({"Time (sec)": round(timestamp, 2), "Event": "Drowsy"})
                print(f"[!] Drowsy event at {timestamp:.2f}s")
        else:
            drowsy_frames = 0

        cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    out.write(frame)

    # Optional live window
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pbar.close()
cap.release()
out.release()
cv2.destroyAllWindows()

# ===============================================
# 6️⃣ SAVE RESULTS
# ===============================================
if events:
    df = pd.DataFrame(events)
    df.to_csv("drowsy_log.csv", index=False)
    print("📝 Saved events to drowsy_log.csv")
else:
    print("✅ No drowsy events detected.")

# ===============================================
# 7️⃣ SHOW GRAPHS
# ===============================================
plt.figure(figsize=(10, 4))
plt.plot(ear_history, label="EAR (Eyes)")
plt.plot(mar_history, label="MAR (Mouth)")
plt.title("Eye and Mouth Aspect Ratios Over Time")
plt.xlabel("Frames")
plt.ylabel("Ratio")
plt.legend()
plt.show()
