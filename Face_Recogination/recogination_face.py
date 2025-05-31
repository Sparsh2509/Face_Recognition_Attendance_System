import cv2
import face_recognition
import pickle
import numpy as np
import os
import mediapipe as mp

# Load encodings
ENCODINGS_PATH = 'encodings/encodings.pickle'
if not os.path.exists(ENCODINGS_PATH):
    raise FileNotFoundError(f"[ERROR] Run register_face.py first to create {ENCODINGS_PATH}")

with open(ENCODINGS_PATH, 'rb') as f:
    data_list = pickle.load(f)

# Initialize MediaPipe segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Recognizing... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Convert frame to RGB for face_recognition and MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_rgb = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)

    # Detect faces
    face_locations = face_recognition.face_locations(small_rgb)
    face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

    # Segment background from full-size frame
    result = segmentor.process(rgb)
    mask = result.segmentation_mask > 0.5
    background_only = np.where(mask[..., None], 0, frame)

    # Compute average background color
    bg_pixels = background_only[background_only.sum(axis=2) > 0]
    avg_bg = np.mean(bg_pixels, axis=0) if bg_pixels.size > 0 else np.array([0, 0, 0])

    names = []

    for encoding in face_encodings:
        name = "Unknown"
        for person in data_list:
            known_face = np.array(person["encoding"])
            known_bg = np.array(person["bg_encoding"])

            face_distance = np.linalg.norm(known_face - encoding)
            bg_distance = np.linalg.norm(known_bg - avg_bg)

            # Thresholds: tune as needed
            if face_distance < 0.5 and bg_distance < 50:
                name = person["name"]
                break  # Accept first match with both OK

        names.append(name)

    # Draw boxes and names
    for ((top, right, bottom, left), name) in zip(face_locations, names):
        # Scale up from 1/4 frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
