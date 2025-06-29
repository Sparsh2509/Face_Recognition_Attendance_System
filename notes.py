# import os
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import cv2
# import mediapipe as mp
# import json
# from database import AsyncSessionLocal, UserFace
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError

# # Load SFace ONNX model once
# sface_model = None
# def load_sface_model():
#     global sface_model
#     if sface_model is None:
#         model_path = "models/face_recognition_sface_2021dec.onnx"
#         if not os.path.exists(model_path):
#             raise FileNotFoundError("SFace ONNX model not found in 'models/' directory.")
#         sface_model = cv2.dnn.readNetFromONNX(model_path)
#         print("[INFO] SFace ONNX model loaded.")
#     return sface_model

# def get_sface_embedding(image: np.ndarray, model) -> np.ndarray:
#     blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
#     model.setInput(blob)
#     embedding = model.forward()
#     return embedding.flatten()

# async def register_face(user_id: str, name: str, image_url: str) -> bool:
#     print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

#     if not image_url.endswith(".jpg"):
#         raise ValueError("Only JPEG images are accepted.")

#     print("[INFO] Fetching image from URL...")
#     response = requests.get(image_url)
#     if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
#         raise ValueError("Invalid image URL or format.")

#     try:
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img_np = np.array(img)

#         if max(img_np.shape[0], img_np.shape[1]) > 800:
#             img = img.resize((800, 800))
#             img_np = np.array(img)

#         print("[INFO] Performing SFace ONNX encoding...")
#         model = load_sface_model()
#         face_embedding = get_sface_embedding(img_np, model)

#         print("[INFO] Performing background encoding...")
#         mp_face_detection = mp.solutions.face_detection
#         with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#             results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#             if not results.detections:
#                 raise ValueError("Face not detected for background processing.")

#             detection = results.detections[0]
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = img_np.shape
#             x = int(bboxC.xmin * iw)
#             y = int(bboxC.ymin * ih)
#             w = int(bboxC.width * iw)
#             h = int(bboxC.height * ih)

#             mask = np.ones(img_np.shape[:2], dtype=bool)
#             mask[y:y + h, x:x + w] = False
#             bg_pixels = img_np[mask]
#             avg_bg_color = np.mean(bg_pixels, axis=0)
#             avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

#         encoding_str = json.dumps(face_embedding.tolist())

#         async with AsyncSessionLocal() as session:
#             try:
#                 query = select(UserFace).where(UserFace.user_id == user_id)
#                 result = await session.execute(query)
#                 user_face = result.scalars().first()

#                 if user_face:
#                     user_face.name = name
#                     user_face.encoding = encoding_str
#                     user_face.avg_bg_color = avg_bg_color
#                 else:
#                     user_face = UserFace(
#                         user_id=user_id,
#                         name=name,
#                         encoding=encoding_str,
#                         avg_bg_color=avg_bg_color
#                     )
#                     session.add(user_face)

#                 await session.commit()
#                 print(f"[SUCCESS] Registered user {name} in DB")
#                 return True

#             except SQLAlchemyError as e:
#                 await session.rollback()
#                 print(f"[DB ERROR] Failed to save user {name}: {e}")
#                 return False

#     except Exception as e:
#         print(f"[ERROR] No face found or encoding failed for user {name}: {e}")
#         return False

