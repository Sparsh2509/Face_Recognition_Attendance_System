import requests
import os

os.makedirs("models", exist_ok=True)

url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
save_path = "models/face_recognition_sface_2021dec.onnx"

print(f"Downloading model to {save_path}...")
with open(save_path, "wb") as f:
    f.write(requests.get(url).content)

print("Model downloaded successfully.")

