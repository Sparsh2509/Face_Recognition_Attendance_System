
from deepface import DeepFace
import mediapipe as mp
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from database import AsyncSessionLocal, UserFace
import asyncio

# Setup MediaPipe segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Async function to register face
async def register_face(user_id: str, name: str, image_url: str):
    try:
        print(f"[INFO] Registering {name} ({user_id}) with image: {image_url}")

        # 1. Download image from Cloudinary URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        # 2. Segment background using MediaPipe
        result = segmentor.process(img_np)
        mask = result.segmentation_mask > 0.5
        background_only = np.where(mask[..., None], 0, img_np)

        # 3. Face detection and encoding
        try:
            embedding_obj = DeepFace.represent(img_path=img_np, model_name="Facenet", enforce_detection=True)[0]
            face_encoding = embedding_obj["embedding"]
        except Exception as e:
            print(f"[ERROR] No face found or encoding failed for user {name}: {e}")
            return False

        # 4. Calculate average background color
        bg_pixels = background_only[background_only.sum(axis=2) > 0]
        if bg_pixels.size == 0:
            print(f"[WARNING] No background pixels found in image for user {name}")
            avg_bg = [0, 0, 0]
        else:
            avg_bg = np.mean(bg_pixels, axis=0).tolist()

        # 5. Save to database
        async with AsyncSessionLocal() as session:
            new_user = UserFace(
                user_id=user_id,
                name=name,
                encoding=str(face_encoding.tolist()),
                avg_bg_color=avg_bg 
             
            )
            session.add(new_user)
            await session.commit()
            print(f"[SUCCESS] Registered {name} ({user_id}) successfully.")
            return True

    except Exception as e:
        print(f"[ERROR] Exception while registering {name}: {e}")
        return False

# # Example for testing standalone
# if __name__ == "__main__":
#     test_user_id = "test123"
#     test_name = "SparshTest"
#     test_image_url = "https://res.cloudinary.com/.../your_image.jpg"  # replace with a real one
#     asyncio.run(register_face(test_user_id, test_name, test_image_url))
