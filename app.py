from fastapi import FastAPI, HTTPException , Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl , constr
from typing import Optional
from typing import Annotated
from register_face import register_face
from recogination_face import recognize_face
from database import AsyncSessionLocal, AttendanceLog
from sqlalchemy.future import select
from enum import Enum


app = FastAPI()


class RegisterRequest(BaseModel):
    user_id: Annotated[str, constr(strip_whitespace=True, min_length=1)]
    name: Annotated[str, constr(strip_whitespace=True, min_length=1, max_length=50)]
    image_url: HttpUrl

class ModeEnum(str, Enum):
    in_ = "in"
    out = "out"

class RecognizeRequest(BaseModel):
    image_base64: Annotated[str, constr(strip_whitespace=True, min_length=100)]
    mode: ModeEnum




@app.get("/")
async def root():
    return {"message": "Face Attendance API is running."}

# User Registration via Cloudinary image
@app.post("/register/")
async def register_user(req: RegisterRequest):
    try:
        success = await register_face(req.user_id, req.name, req.image_url)

        if success:
            return {
                "status": "success",
                "message": f"{req.name} registered successfully.",
                "user_id": req.user_id,
                "name": req.name
            }
        else:
            raise HTTPException(status_code=500, detail="User registration failed.")

    except Exception as e:
        print("Unexpected error during registration:", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Unexpected error: {str(e)}"})
    

# Real-time Recognition (base64 + mode)

@app.post("/recognize/")
async def recognize_user(req: RecognizeRequest):
    try:
        result = await recognize_face(req.image_base64, req.mode.value)  # use .value from Enum

        if result["status"] == "present":
            return {
                "status": "success",
                "message": "Attendance marked",
                "data": result
            }
        elif result["status"] == "absent":
            return {
                "status": "absent",
                "message": result.get("reason", "Face not recognized"),
                "data": None
            }
        elif result["status"] == "invalid":
            return {
                "status": "invalid",
                "message": result.get("message", "Invalid flow"),
                "data": None
            }
        else:
            return {
                "status": "error",
                "message": result.get("reason", "Unknown error"),
                "data": None
            }

    except Exception as e:
        print("Unexpected error during recognition:", e)
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "data": None
        }

# Get attendance logs
@app.get("/attendance-log/")
# async def get_attendance_log(
    
#     user_id: Annotated[str, Query(min_length=1, description="User ID")],
#     date: Optional[Annotated[str, Query(pattern=r"^\d{4}-\d{2}-\d{2}$")]] = None
# ):
#     """
#     Returns attendance log for a given user_id.
#     If date is provided, it must be in YYYY-MM-DD format.
#     """
#     try:
#         async with AsyncSessionLocal() as session:
#             if date:
#                 query = select(AttendanceLog).where(
#                     AttendanceLog.user_id == user_id,
#                     AttendanceLog.date == date
#                 )
#             else:
#                 query = select(AttendanceLog).where(
#                     AttendanceLog.user_id == user_id
#                 ).order_by(AttendanceLog.date.desc())

#             result = await session.execute(query)
#             logs = result.scalars().all()

#             if not logs:
#                 return {
#                     "status": "not_found",
#                     "message": "No attendance records found.",
#                     "data": []
#                 }

#             return {
#                 "status": "success",
#                 "data": [
#                     {
#                         "user_id": log.user_id,
#                         "name": log.name,
#                         "date": log.date.isoformat(),
#                         "in_time": str(log.in_time) if log.in_time else None,
#                         "out_time": str(log.out_time) if log.out_time else None,
#                         "in_status": log.in_status,
#                         "out_status": log.out_status
#                     }
#                     for log in logs
#                 ]
#             }

#     except Exception as e:
#         print(f"[ERROR] Failed to fetch attendance log: {e}")
#         return {
#             "status": "error",
#             "message": "Internal server error",
#             "data": None
#         }

async def get_attendance_log(
    user_id: Annotated[str, Query(min_length=1, description="User ID must not be empty")]
):
    """
    Returns all attendance logs for the given user_id.
    """
    try:
        async with AsyncSessionLocal() as session:
            query = select(AttendanceLog).where(
                AttendanceLog.user_id == user_id
            ).order_by(AttendanceLog.date.desc())

            result = await session.execute(query)
            logs = result.scalars().all()

            if not logs:
                return {
                    "status": "not_found",
                    "message": "No attendance records found.",
                    "data": []
                }

            return {
                "status": "success",
                "data": [
                    {
                        "user_id": log.user_id,
                        "name": log.name,
                        "date": log.date.isoformat(),
                        "in_time": str(log.in_time) if log.in_time else None,
                        "out_time": str(log.out_time) if log.out_time else None,
                        "in_status": log.in_status,
                        "out_status": log.out_status
                    }
                    for log in logs
                ]
            }

    except Exception as e:
        print(f"[ERROR] Failed to fetch attendance log: {e}")
        return {
            "status": "error",
            "message": "Internal server error",
            "data": None
        }