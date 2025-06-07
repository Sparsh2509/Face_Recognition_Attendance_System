# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from deepface import DeepFace
# import cv2
# import mediapipe as mp
# from database import AsyncSessionLocal, UserFace
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError
# import json

# # Preload Facenet model once to avoid download delay on first request
# def preload_facenet_model():
#     print("[INFO] Preloading Facenet model...")
#     dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#     DeepFace.represent(img_path=dummy, model_name="Facenet", enforce_detection=False)
#     print("[INFO] Facenet model preloaded.")


# async def register_face(user_id: str, name: str, image_url: str) -> bool:
#     print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

#     if not image_url.endswith(".jpg"):
#         raise ValueError("Only JPEG images are accepted.")

#     response = requests.get(image_url)
#     if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
#         raise ValueError("Invalid image URL or format.")

#     try:
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img_np = np.array(img)

#         # 1. Face encoding using DeepFace
#         result = DeepFace.represent(img_path=img_np, model_name="Facenet")
#         face_embedding = result[0]["embedding"]

#         # 2. Background encoding using average color via MediaPipe
#         mp_face_detection = mp.solutions.face_detection
#         with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#             results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#             if not results.detections:
#                 raise ValueError("Face not detected for background processing.")

#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, _ = img_np.shape
#                 x = int(bboxC.xmin * iw)
#                 y = int(bboxC.ymin * ih)
#                 w = int(bboxC.width * iw)
#                 h = int(bboxC.height * ih)

#                 # Crop outside the face bounding box for background avg color
#                 bg_img = img_np.copy()
#                 bg_img[y:y + h, x:x + w] = 0
#                 avg_bg_color = bg_img[bg_img != 0].mean()

#         # Convert face embedding list to JSON string (for DB storage)
#         encoding_str = json.dumps(face_embedding)

#         # Save to DB using your AsyncSessionLocal and UserFace model
#         async with AsyncSessionLocal() as session:
#             try:
#                 query = select(UserFace).where(UserFace.user_id == user_id)
#                 result = await session.execute(query)
#                 user_face = result.scalars().first()

#                 if user_face:
#                     user_face.name = name
#                     user_face.encoding = encoding_str
#                     user_face.avg_bg_color = avg_bg_color.tolist() if isinstance(avg_bg_color, np.generic) else avg_bg_color
#                 else:
#                     user_face = UserFace(
#                         user_id=user_id,
#                         name=name,
#                         encoding=encoding_str,
#                         avg_bg_color=avg_bg_color.tolist() if isinstance(avg_bg_color, np.generic) else avg_bg_color
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


# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from deepface import DeepFace
# import cv2
# import mediapipe as mp
# from database import AsyncSessionLocal, UserFace
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError
# import json

# # Preload Facenet model once to avoid download delay on first request
# def preload_facenet_model():
#     print("[INFO] Preloading Facenet model...")
#     dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#     DeepFace.represent(img_path=dummy, model_name="Facenet", enforce_detection=False)
#     print("[INFO] Facenet model preloaded.")


# async def register_face(user_id: str, name: str, image_url: str) -> bool:
#     print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

#     if not image_url.endswith(".jpg"):
#         raise ValueError("Only JPEG images are accepted.")

#     response = requests.get(image_url)
#     if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
#         raise ValueError("Invalid image URL or format.")

#     try:
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img_np = np.array(img)

#         # 1. Face encoding using DeepFace
#         result = DeepFace.represent(img_path=img_np, model_name="Facenet")
#         face_embedding = result[0]["embedding"]

#         # 2. Background encoding using average color via MediaPipe
#         mp_face_detection = mp.solutions.face_detection
#         with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#             results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#             if not results.detections:
#                 raise ValueError("Face not detected for background processing.")

#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, _ = img_np.shape
#                 x = int(bboxC.xmin * iw)
#                 y = int(bboxC.ymin * ih)
#                 w = int(bboxC.width * iw)
#                 h = int(bboxC.height * ih)

#                 # Mask out face area
#                 bg_img = img_np.copy()
#                 bg_img[y:y + h, x:x + w] = 0

#                 # Compute average of non-zero background pixels
#                 mask = np.any(bg_img != 0, axis=2)
#                 bg_pixels = bg_img[mask]
#                 avg_bg_color = np.mean(bg_pixels, axis=0)

#                 # 🔵 ROUND TO 3 DECIMAL PLACES
#                 avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

#         # Convert face embedding list to JSON string (for DB storage)
#         encoding_str = json.dumps(face_embedding)

#         # Save to DB using your AsyncSessionLocal and UserFace model
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


# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from deepface import DeepFace
# import cv2
# import mediapipe as mp
# from database import AsyncSessionLocal, UserFace
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError
# import json

# async def register_face(user_id: str, name: str, image_url: str) -> bool:
#     print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

#     if not image_url.endswith(".jpg"):
#         raise ValueError("Only JPEG images are accepted.")
    

#     print("[INFO] Fetching image from URL...")
    
#     response = requests.get(image_url)
#     if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
#         raise ValueError("Invalid image URL or format.")
    
#     print("[INFO] Performing DeepFace encoding...")

#     try:
#         img = Image.open(BytesIO(response.content)).convert("RGB")
#         img_np = np.array(img)

#         # 1. Face encoding using DeepFace with SFace model
#         result = DeepFace.represent(img_path=img_np, model_name="SFace")
#         face_embedding = result[0]["embedding"]

#         # 2. Background encoding using average color via MediaPipe
#         mp_face_detection = mp.solutions.face_detection
#         with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#             results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#             if not results.detections:
#                 raise ValueError("Face not detected for background processing.")

#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 ih, iw, _ = img_np.shape
#                 x = int(bboxC.xmin * iw)
#                 y = int(bboxC.ymin * ih)
#                 w = int(bboxC.width * iw)
#                 h = int(bboxC.height * ih)

#                 # Mask out face area
#                 bg_img = img_np.copy()
#                 bg_img[y:y + h, x:x + w] = 0

#                 # Compute average of non-zero background pixels
#                 mask = np.any(bg_img != 0, axis=2)
#                 bg_pixels = bg_img[mask]
#                 avg_bg_color = np.mean(bg_pixels, axis=0)

#                 # Round to 3 decimal places
#                 avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

#         # Convert face embedding list to JSON string (for DB storage)
#         encoding_str = json.dumps(face_embedding)

#         # Save to DB using your AsyncSessionLocal and UserFace model
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


# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from deepface import DeepFace
# import cv2
# import mediapipe as mp
# from database import AsyncSessionLocal, UserFace
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError
# import json

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

#         # Resize if needed to reduce memory usage
#         if max(img_np.shape[0], img_np.shape[1]) > 800:
#             img = img.resize((800, 800))
#             img_np = np.array(img)

#         print("[INFO] Performing DeepFace encoding...")
#         result = DeepFace.represent(img_path=img_np, model_name="SFace")
#         face_embedding = result[0]["embedding"]

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

#             # Mask the face region and extract background pixels only
#             mask = np.ones(img_np.shape[:2], dtype=bool)
#             mask[y:y + h, x:x + w] = False
#             bg_pixels = img_np[mask]
#             avg_bg_color = np.mean(bg_pixels, axis=0)
#             avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

#         encoding_str = json.dumps(face_embedding)

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

import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import mediapipe as mp
import json
from database import AsyncSessionLocal, UserFace
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError

# Load SFace ONNX model once
sface_model = None
def load_sface_model():
    global sface_model
    if sface_model is None:
        model_path = "d:\Sparsh\ML_Projects\Face_Recogination\models"
        if not os.path.exists(model_path):
            raise FileNotFoundError("SFace ONNX model not found in 'models/' directory.")
        sface_model = cv2.dnn.readNetFromONNX(model_path)
        print("[INFO] SFace ONNX model loaded.")
    return sface_model

def get_sface_embedding(image: np.ndarray, model) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (112, 112), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    embedding = model.forward()
    return embedding.flatten()

async def register_face(user_id: str, name: str, image_url: str) -> bool:
    print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

    if not image_url.endswith(".jpg"):
        raise ValueError("Only JPEG images are accepted.")

    print("[INFO] Fetching image from URL...")
    response = requests.get(image_url)
    if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
        raise ValueError("Invalid image URL or format.")

    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        if max(img_np.shape[0], img_np.shape[1]) > 800:
            img = img.resize((800, 800))
            img_np = np.array(img)

        print("[INFO] Performing SFace ONNX encoding...")
        model = load_sface_model()
        face_embedding = get_sface_embedding(img_np, model)

        print("[INFO] Performing background encoding...")
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            if not results.detections:
                raise ValueError("Face not detected for background processing.")

            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_np.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            mask = np.ones(img_np.shape[:2], dtype=bool)
            mask[y:y + h, x:x + w] = False
            bg_pixels = img_np[mask]
            avg_bg_color = np.mean(bg_pixels, axis=0)
            avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

        encoding_str = json.dumps(face_embedding.tolist())

        async with AsyncSessionLocal() as session:
            try:
                query = select(UserFace).where(UserFace.user_id == user_id)
                result = await session.execute(query)
                user_face = result.scalars().first()

                if user_face:
                    user_face.name = name
                    user_face.encoding = encoding_str
                    user_face.avg_bg_color = avg_bg_color
                else:
                    user_face = UserFace(
                        user_id=user_id,
                        name=name,
                        encoding=encoding_str,
                        avg_bg_color=avg_bg_color
                    )
                    session.add(user_face)

                await session.commit()
                print(f"[SUCCESS] Registered user {name} in DB")
                return True

            except SQLAlchemyError as e:
                await session.rollback()
                print(f"[DB ERROR] Failed to save user {name}: {e}")
                return False

    except Exception as e:
        print(f"[ERROR] No face found or encoding failed for user {name}: {e}")
        return False
