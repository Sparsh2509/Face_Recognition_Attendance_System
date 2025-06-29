import os
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Cached ONNX model
sface_model = None

def load_sface_model():
    """
    Loads and returns the ONNX SFace model only once (cached).
    """
    global sface_model
    if sface_model is None:
        model_path = "models/face_recognition_sface_2021dec.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError("SFace ONNX model not found in 'models/' directory.")
        sface_model = cv2.dnn.readNetFromONNX(model_path)
        print("[INFO] SFace ONNX model loaded.")
    return sface_model

def get_face_embedding(image: np.ndarray, model) -> np.ndarray:
    """
    Runs the SFace ONNX model to get a 512D face embedding from the image.
    """
    blob = cv2.dnn.blobFromImage(
        image,
        1.0 / 255,
        (112, 112),
        (0, 0, 0),
        swapRB=True,
        crop=False
    )
    model.setInput(blob)
    embedding = model.forward()
    return embedding.flatten()


def decode_base64_image(image_base64: str) -> np.ndarray:
    """
    Converts a base64 image string to a NumPy array (RGB).
    """
    if "," in image_base64:
        _, data = image_base64.split(",", 1)
    else:
        data = image_base64
    image_data = base64.b64decode(data)
    img = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(img)


def cosine_similarity(a, b) -> float:
    """
    Computes cosine similarity between two vectors.
    """
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def color_distance(bg1, bg2) -> float:
    """
    Computes Euclidean distance between two RGB vectors.
    """
    return np.linalg.norm(np.array(bg1) - np.array(bg2))