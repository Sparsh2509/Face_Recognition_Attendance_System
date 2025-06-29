import os
import cv2
import numpy as np

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
