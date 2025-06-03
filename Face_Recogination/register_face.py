import requests
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import cv2
import mediapipe as mp
from database import AsyncSessionLocal, UserFace
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError


async def register_face(user_id: str, name: str, image_url: str) -> bool:
    print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

    if not image_url.endswith(".jpg"):
        raise ValueError("Only JPEG images are accepted.")

    response = requests.get(image_url)
    if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
        raise ValueError("Invalid image URL or format.")

    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        # 1. Face encoding using DeepFace
        result = DeepFace.represent(img_path=img_np, model_name="Facenet")
        face_embedding = result[0]["embedding"]

        # 2. Background encoding using average color via MediaPipe
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            if not results.detections:
                raise ValueError("Face not detected for background processing.")

            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img_np.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Crop outside the face bounding box for background avg color
                bg_img = img_np.copy()
                bg_img[y:y + h, x:x + w] = 0
                avg_bg_color = bg_img[bg_img != 0].mean()

        # Convert face embedding list to JSON string (for DB storage)
        import json
        encoding_str = json.dumps(face_embedding)

        # Save to DB using your AsyncSessionLocal and UserFace model
        async with AsyncSessionLocal() as session:
            try:
                query = select(UserFace).where(UserFace.user_id == user_id)
                result = await session.execute(query)
                user_face = result.scalars().first()

                if user_face:
                    user_face.name = name
                    user_face.encoding = encoding_str
                    user_face.avg_bg_color = avg_bg_color.tolist() if isinstance(avg_bg_color, np.generic) else avg_bg_color
                else:
                    user_face = UserFace(
                        user_id=user_id,
                        name=name,
                        encoding=encoding_str,
                        avg_bg_color=avg_bg_color.tolist() if isinstance(avg_bg_color, np.generic) else avg_bg_color
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
