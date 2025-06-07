# # import face_recognition
# import numpy as np
# import requests
# from PIL import Image
# from io import BytesIO

# # Trial image URL (no .jpg — will be auto-added if missing)
# image_url = "https://res.cloudinary.com/dzcwomu3h/image/upload/v1748717751/Sparsh_2311143_rmszww.jpg"

# # Force JPEG extension if needed
# if not image_url.endswith(".jpg"):
#     image_url += ".jpg"

# print("[INFO] Downloading image from:", image_url)

# response = requests.get(image_url)
# content_type = response.headers.get("Content-Type", "")

# print("[INFO] Status:", response.status_code)
# print("[INFO] Content-Type:", content_type)

# # Validate content
# if response.status_code != 200 or "image" not in content_type or "gif" in content_type:
#     print("[ERROR] Invalid image or GIF returned. Aborting.")
#     exit()

# try:
#     # Open and convert image to RGB
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#     img_np = np.array(img)
#     print("[INFO] Image loaded. Shape:", img_np.shape)

#     # Face detection
#     face_locations = face_recognition.face_locations(img_np)
#     if not face_locations:
#         print("[❌] No face detected.")
#     else:
#         print(f"[✅] Detected {len(face_locations)} face(s) at:", face_locations)
#         encoding = face_recognition.face_encodings(img_np, face_locations)[0]
#         print("[INFO] Encoding (first 5 values):", encoding[:5])

# except Exception as e:
#     print("[ERROR] Failed to process image:", e)


from deepface import DeepFace
from PIL import Image
import numpy as np
import requests
from io import BytesIO

image_url = "https://res.cloudinary.com/dzcwomu3h/image/upload/v1748717751/Sparsh_2311143_rmszww.jpg"

# Ensure .jpg ending
if not image_url.endswith(".jpg"):
    raise ValueError("Only .jpg images are accepted")

# Download and convert to RGB array
response = requests.get(image_url)
img = Image.open(BytesIO(response.content)).convert("RGB")
img_array = np.array(img)

# Run DeepFace
result = DeepFace.represent(img_path=img_array, model_name="Facenet")

print("Encoding successful. Length:", len(result[0]["embedding"]))

