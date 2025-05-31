import face_recognition
import mediapipe as mp
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from database import AsyncSessionLocal, UserFace
# import asyncio

# === Setup MediaPipe Selfie Segmentation ===
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# === Main Async Function to Register Face ===
async def register_face(user_id: str, name: str, image_url: str):
    try:
        print(f"\n[INFO] Registering user: {name} ({user_id})")
        print(f"[INFO] Downloading image from: {image_url}")

        # === 1. Download Image ===
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        # === 2. MediaPipe Background Segmentation ===
        result = segmentor.process(img_np)
        mask = result.segmentation_mask > 0.5
        background_only = np.where(mask[..., None], 0, img_np)

        # === 3. Face Detection and Encoding ===
        face_locations = face_recognition.face_locations(img_np)
        if not face_locations:
            print(f"[ERROR] ❌ No face found for user {name}.")
            return False

        face_encoding = face_recognition.face_encodings(img_np, face_locations)[0]

        # === 4. Background Encoding (Average RGB) ===
        bg_pixels = background_only[background_only.sum(axis=2) > 0]
        if bg_pixels.size == 0:
            print(f"[WARN] ⚠️ No background pixels found, using [0,0,0].")
            avg_bg = [0, 0, 0]
        else:
            avg_bg = np.mean(bg_pixels, axis=0).tolist()

        # === 5. Save to Neon DB ===
        async with AsyncSessionLocal() as session:
            new_user = UserFace(
                user_id=user_id,
                name=name,
                encoding=face_encoding.tolist(),
            )
            session.add(new_user)
            await session.commit()

        print(f"[SUCCESS] ✅ {name} ({user_id}) registered successfully.\n")
        return True

    except Exception as e:
        print(f"[EXCEPTION] 💥 Error while registering {name}: {e}\n")
        return False

# === Test Hook for Local Debugging ===
# if __name__ == "__main__":
#     test_user_id = "test123"
#     test_name = "SparshTest"
#     test_image_url = "https://res.cloudinary.com/dzcwomu3h/image/upload/v1748717751/Sparsh_2311143_rmszww.jpg"  # Replace with real Cloudinary URL
#     asyncio.run(register_face(test_user_id, test_name, test_image_url))
