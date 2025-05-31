import face_recognition
import mediapipe as mp
import cv2
import numpy as np
import os
import pickle

# Folder containing person photos (face + background together)
KNOWN_FACES_DIR = r'd:\Sparsh\ML_Projects\face_recognition\Face_Recogination\known_faces'

# Path to save the encodings
ENCODINGS_FILE = 'encodings/encodings.pickle'
os.makedirs('encodings', exist_ok=True)

# MediaPipe setup
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Load existing data if any
all_data = []
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        try:
            all_data = pickle.load(f)
        except:
            pass

# Process each image in the folder

print(f"[DEBUG] Scanning folder: {KNOWN_FACES_DIR}")
files = os.listdir(KNOWN_FACES_DIR)
print(f"[DEBUG] Files found: {files}")

for filename in files:
# for filename in os.listdir(KNOWN_FACES_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    print(f"[INFO] Processing {filename}...")

    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Segment background
    result = segmentor.process(image_rgb)
    mask = result.segmentation_mask > 0.5
    background_only = np.where(mask[..., None], 0, image_bgr)

    # Get face encodings
    face_locations = face_recognition.face_locations(image_rgb)
    if len(face_locations) == 0:
        print(f"[WARNING] No face found in {filename}")
        continue

    face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0]

    # Get average background color
    bg_pixels = background_only[background_only.sum(axis=2) > 0]
    if bg_pixels.size == 0:
        print(f"[WARNING] No background pixels found in {filename}")
        continue

    avg_bg = np.mean(bg_pixels, axis=0).tolist()

    # Use filename (without extension) as the name
    name = os.path.splitext(filename)[0]

    # Store data
    person_data = {
        "name": name,
        "encoding": face_encoding.tolist(),
        "bg_encoding": avg_bg
    }

    all_data.append(person_data)
    print(f"[SUCCESS] Registered {name}.")

# Save encodings
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump(all_data, f)

print(f"[INFO] All encodings saved to {ENCODINGS_FILE}")
