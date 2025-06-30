# import json
# import numpy as np
# from datetime import datetime
# from sqlalchemy.future import select
# from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy import and_
# from fastapi import HTTPException

# from database import AsyncSessionLocal, UserFace, AttendanceLog
# from shared_code import (
#     decode_base64_image,
#     load_sface_model,
#     get_face_embedding,
#     cosine_similarity,
#     color_distance
# )

# import cv2
# import mediapipe as mp


# async def mark_attendance(user_id: str, name: str, timestamp: datetime, intended_mode: str) -> str:
#     date_today = timestamp.date()
#     current_time = timestamp.time()
#     async with AsyncSessionLocal() as session:
#         try:
#             query = select(AttendanceLog).where(
#                 and_(
#                     AttendanceLog.user_id == user_id,
#                     AttendanceLog.date == date_today
#                 )
#             )
#             result = await session.execute(query)
#             log = result.scalars().first()

#             if log is None:
#                 if intended_mode == "out":
#                     return "invalid_out"

#                 log = AttendanceLog(
#                     user_id=user_id,
#                     name=name,
#                     date=date_today,
#                     in_time=current_time,
#                     in_status="present",
#                     out_status="absent"
#                 )
#                 session.add(log)
#                 mode = "in"

#             elif intended_mode == "in":
#                 if log.in_time is None:
#                     log.in_time = current_time
#                     log.in_status = "present"
#                     mode = "in"
#                 else:
#                     mode = "already_in"

#             elif intended_mode == "out":
#                 if log.out_time is None:
#                     log.out_time = current_time
#                     log.out_status = "present"
#                     mode = "out"
#                 else:
#                     mode = "already_out"

#             await session.commit()
#             return mode

#         except SQLAlchemyError as e:
#             await session.rollback()
#             print(f"[DB ERROR] Attendance mark failed for {name}: {e}")
#             return "error"


# async def recognize_face(image_base64: str, intended_mode: str) -> dict:
#     try:
#         # Step 1: Decode + preprocess
#         img_np = decode_base64_image(image_base64)
#         model = load_sface_model()

#         # Step 2: Face detection
#         mp_face_detection = mp.solutions.face_detection
#         with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
#             results = detector.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

#             if not results.detections:
#                 return {"status": "absent", "reason": "Face not detected"}

#             detection = results.detections[0]
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = img_np.shape

#             x = int(bboxC.xmin * iw)
#             y = int(bboxC.ymin * ih)
#             w = int(bboxC.width * iw)
#             h = int(bboxC.height * ih)

#             bg_x1 = max(x - 100, 0)
#             bg_y1 = max(y - 50, 0)
#             bg_x2 = min(bg_x1 + 100, iw)
#             bg_y2 = min(bg_y1 + 100, ih)

#             bg_crop = img_np[bg_y1:bg_y2, bg_x1:bg_x2]
#             avg_bg_color = np.mean(bg_crop.reshape(-1, 3), axis=0)
#             avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

#         # Step 3: Face embedding
#         face_embedding = get_face_embedding(img_np, model)

#         # Step 4: Match against DB
#         async with AsyncSessionLocal() as session:
#             query = select(UserFace)
#             result = await session.execute(query)
#             users = result.scalars().all()

#             for user in users:
#                 db_embedding = json.loads(user.encoding)
#                 sim = cosine_similarity(face_embedding, db_embedding)
#                 bg_dist = color_distance(avg_bg_color, user.avg_bg_color)

#                 if sim >= 0.6 and bg_dist <= 30:
#                     timestamp = datetime.now()
#                     mode = await mark_attendance(user.user_id, user.name, timestamp, intended_mode)

#                     if mode == "invalid_out":
#                         return {"status": "invalid", "message": "Please mark IN before marking OUT."}

#                     return {
#                         "status": "present",
#                         "mode": mode,
#                         "time": timestamp.strftime("%H:%M"),
#                         "date": timestamp.strftime("%Y-%m-%d"),
#                         "user_id": user.user_id,
#                         "name": user.name
#                     }

#         return {"status": "absent", "reason": "No matching face or background"}

#     except Exception as e:
#         print(f"[ERROR] {e}")
#         return {"status": "error", "reason": str(e)}


import json
import numpy as np
from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_
from fastapi import HTTPException

from database import AsyncSessionLocal, UserFace, AttendanceLog
from shared_code import (
    decode_base64_image,
    load_sface_model,
    get_face_embedding,
    cosine_similarity,
    color_distance
)

import cv2
import mediapipe as mp


async def mark_attendance(user_id: str, name: str, timestamp: datetime, intended_mode: str) -> str:
    date_today = timestamp.date()
    current_time = timestamp.time()
    async with AsyncSessionLocal() as session:
        try:
            query = select(AttendanceLog).where(
                and_(
                    AttendanceLog.user_id == user_id,
                    AttendanceLog.date == date_today
                )
            )
            result = await session.execute(query)
            log = result.scalars().first()

            if log is None:
                if intended_mode == "out":
                    return "invalid_out"

                log = AttendanceLog(
                    user_id=user_id,
                    name=name,
                    date=date_today,
                    in_time=current_time,
                    in_status="present",
                    out_status="absent"
                )
                session.add(log)
                mode = "in"

            elif intended_mode == "in":
                if log.in_time is None:
                    log.in_time = current_time
                    log.in_status = "present"
                    mode = "in"
                else:
                    mode = "already_in"

            elif intended_mode == "out":
                if log.out_time is None:
                    log.out_time = current_time
                    log.out_status = "present"
                    mode = "out"
                else:
                    mode = "already_out"

            await session.commit()
            return mode

        except SQLAlchemyError as e:
            await session.rollback()
            print(f"[DB ERROR] Attendance mark failed for {name}: {e}")
            return "error"


async def recognize_face(image_base64: str, intended_mode: str) -> dict:
    try:
        # Step 1: Decode base64 image (auto saves debug image inside decode_base64_image)
        img_np = decode_base64_image(image_base64)

        # Step 2: Load ONNX face model
        model = load_sface_model()

        # Step 3: Face detection
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as detector:
            results = detector.process(img_np)

            if not results.detections:
                return {"status": "absent", "reason": "Face not detected"}

            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img_np.shape

            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Sample background region (top-left of face box)
            bg_x1 = max(x - 100, 0)
            bg_y1 = max(y - 50, 0)
            bg_x2 = min(bg_x1 + 100, iw)
            bg_y2 = min(bg_y1 + 100, ih)

            bg_crop = img_np[bg_y1:bg_y2, bg_x1:bg_x2]
            avg_bg_color = np.mean(bg_crop.reshape(-1, 3), axis=0)
            avg_bg_color = [round(float(c), 3) for c in avg_bg_color]

        # Step 4: Extract face embedding
        face_embedding = get_face_embedding(img_np, model)

        # Step 5: Match against all stored users
        async with AsyncSessionLocal() as session:
            query = select(UserFace)
            result = await session.execute(query)
            users = result.scalars().all()

            for user in users:
                db_embedding = json.loads(user.encoding)
                sim = cosine_similarity(face_embedding, db_embedding)
                bg_dist = color_distance(avg_bg_color, user.avg_bg_color)

                print(f"[DEBUG] Matching with {user.name} (ID: {user.user_id}) → sim={sim:.4f}, bg_dist={bg_dist:.2f}")

                if sim >= 0.6 and bg_dist <= 30:
                    timestamp = datetime.now()
                    mode = await mark_attendance(user.user_id, user.name, timestamp, intended_mode)

                    if mode == "invalid_out":
                        return {"status": "invalid", "message": "Please mark IN before marking OUT."}

                    return {
                        "status": "present",
                        "mode": mode,
                        "time": timestamp.strftime("%H:%M"),
                        "date": timestamp.strftime("%Y-%m-%d"),
                        "user_id": user.user_id,
                        "name": user.name
                    }

        # No match found
        return {"status": "absent", "reason": "No matching face or background"}

    except Exception as e:
        print(f"[ERROR] {e}")
        return {"status": "error", "reason": str(e)}
