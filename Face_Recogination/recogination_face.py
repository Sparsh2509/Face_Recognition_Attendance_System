import cv2
import face_recognition
import pickle
import os
import dlib

# Load known faces and embeddings
ENCODINGS_PATH = "d:\Sparsh\ML_Projects\Face\encodings\encodings.pickle"

if not os.path.exists(ENCODINGS_PATH):
    raise FileNotFoundError(f"Encoding file not found at {ENCODINGS_PATH}. Run the registration script first.")

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

print("[INFO] Starting video stream. Press 'q' to quit.")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Convert locations to dlib rectangles
    # dlib_rects = [dlib.rectangle(left, top, right, bottom)
    #               for (top, right, bottom, left) in face_locations]
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Compute face encodings
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # Use the known face with the smallest distance if match is found
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    # Draw results
    for ((top, right, bottom, left), name) in zip(face_locations, names):
        # Scale back up face locations since the frame we used was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.9, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
