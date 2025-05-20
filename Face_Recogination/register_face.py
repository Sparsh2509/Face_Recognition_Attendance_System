# import face_recognition
import os
import pickle

# Folder containing known face images
# KNOWN_FACES_DIR = r'd:\Sparsh\ML_Projects\Face_Recognition\Face_Recognition\known_faces'

# File to save encodings
ENCODINGS_FILE = 'encodings/encodings.pickle'

# Lists to hold encodings and names
known_encodings = []
known_names = []

# Loop through each person’s folder/image
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"Processing {filename}...")

        # Load image
        image = face_recognition.load_image_file(image_path)

        # Get face encodings (assuming one face per image)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            # Use filename (without extension) as the person’s name
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face found in {filename}!")

# Save encodings and names to file
os.makedirs('encodings', exist_ok=True)
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print(f"Encodings saved to {ENCODINGS_FILE}")