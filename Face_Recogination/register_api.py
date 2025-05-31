from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from register_face import register_user_face

app = FastAPI()

class RegisterUserRequest(BaseModel):
    user_id: str
    name: str
    image_url: str

@app.post("/register/")
async def register_user(data: RegisterUserRequest):
    success = await register_user_face(data.user_id, data.name, data.image_url)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to register user")
    return {"status": "success", "user_id": data.user_id, "name": data.name}
