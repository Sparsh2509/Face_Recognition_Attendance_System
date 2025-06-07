from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from register_face import register_face

app = FastAPI()

class RegisterRequest(BaseModel):
    user_id: str
    name: str
    image_url: str

@app.get("/")
async def root():
    return {"message": "Face Registration API is running."}

@app.post("/register/")
async def register_user(req: RegisterRequest):
    try:
        success = await register_face(req.user_id, req.name, req.image_url)

        if success:
            return {"status": "success", "message": f"{req.name} registered successfully."}
        else:
            raise HTTPException(status_code=500, detail="User registration failed.")

    except Exception as e:
        print("Unexpected error:", e)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

