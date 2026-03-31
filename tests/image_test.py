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

