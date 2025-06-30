from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl , constr
from typing import Optional
from typing import Annotated
from register_face import register_face
from recogination_face import recognize_face
from database import AsyncSessionLocal, AttendanceLog
from sqlalchemy.future import select

app = FastAPI()


class RegisterRequest(BaseModel):
    user_id: Annotated[str, constr(strip_whitespace=True, min_length=1)]
    name: Annotated[str, constr(strip_whitespace=True, min_length=1, max_length=50)]
    image_url: HttpUrl

class RecognizeRequest(BaseModel):
    image_base64: str
    mode: str  # "in" or "out"


@app.get("/")
async def root():
    return {"message": "Face Attendance API is running."}

# ✅ User Registration via Cloudinary image
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
    

# ✅ Real-time Recognition (base64 + mode)
@app.post("/recognize/")
async def recognize_user(req: RecognizeRequest):
    try:
        result = await recognize_face(req.image_base64, req.mode)
        return result

    except Exception as e:
        print("Unexpected error during recognition:", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ✅ Get attendance logs
@app.get("/attendance-log/")
async def get_attendance_log(user_id: str, date: Optional[str] = None):
    """
    Returns attendance log for a given user_id.
    If date is provided, filters to that specific date.
    """
    try:
        async with AsyncSessionLocal() as session:
            if date:
                query = select(AttendanceLog).where(
                    AttendanceLog.user_id == user_id,
                    AttendanceLog.date == date
                )
            else:
                query = select(AttendanceLog).where(
                    AttendanceLog.user_id == user_id
                ).order_by(AttendanceLog.date.desc())

            result = await session.execute(query)
            logs = result.scalars().all()

            if not logs:
                return {"status": "not_found", "message": "No attendance records found."}

            return {
                "status": "success",
                "records": [
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
        raise HTTPException(status_code=500, detail="Internal server error")
