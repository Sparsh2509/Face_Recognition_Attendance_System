import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import mediapipe as mp
import json
from fastapi import HTTPException
from database import AsyncSessionLocal, UserFace
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from shared_code import load_sface_model, get_face_embedding


# Main registration function
async def register_face(user_id: str, name: str, image_url: str) -> bool:
    print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

    # Validate image URL
    if not str(image_url.endswith((".jpg", ".jpeg", ".png"))):
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are accepted.")

    print("[INFO] Fetching image from URL...")
    response = requests.get(str(image_url))
    if response.status_code != 200 or "image" not in response.headers.get("Content-Type", ""):
        raise HTTPException(status_code=400, detail="Invalid image URL or format.")

    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        # Resize if too large
        if max(img_np.shape[0], img_np.shape[1]) > 800:
            img = img.resize((800, 800))
            img_np = np.array(img)

        # Face embedding
        print("[INFO] Performing SFace ONNX encoding...")
        model = load_sface_model()

        print("[INFO] Performing Face encoding...")
        face_embedding = get_face_embedding(img_np, model)

        # Background color sampling
        print("[INFO] Performing background encoding...")
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            if not results.detections:
                raise HTTPException(status_code=400, detail="Face not detected for background sampling.")

            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_np.shape

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Sample background 50px above face, 100px to left
            bg_x1 = max(x - 100, 0)
            bg_y1 = max(y - 50, 0)
            bg_x2 = min(bg_x1 + 100, iw)
            bg_y2 = min(bg_y1 + 100, ih)

            bg_crop = img_np[bg_y1:bg_y2, bg_x1:bg_x2]
            avg_bg_color = np.mean(bg_crop.reshape(-1, 3), axis=0)
            avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

        # Prepare data for DB
        encoding_str = json.dumps(face_embedding.tolist())

        # Store in DB
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
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except HTTPException:
        raise  # Pass along any HTTPExceptions raised above

    except Exception as e:
        print(f"[ERROR] No face found or encoding failed for user {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
