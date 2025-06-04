# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# # from register_face import register_face

# try:
#     from register_face import register_face
# except Exception as e:
#     print("[CRITICAL ERROR] Failed to import register_face:", e)
#     raise

# import sys
# import asyncio

# # Fix silent exception on Windows + asyncio
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# app = FastAPI()

# # Request body schema
# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# @app.post("/register/")
# async def register_user(data: RegisterRequest):
#     try:
#         success = await register_face(
#             user_id=data.user_id,
#             name=data.name,
#             image_url=data.image_url
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

#     if success:
#         return {"status": "success", "message": f"{data.name} registered."}
#     else:
#         raise HTTPException(status_code=400, detail="Failed to register user")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# from register_face import register_face
# import requests

# app = FastAPI()

# # ✅ Add your UserFace model here
# class UserFace(BaseModel):
#     name: str
#     encoding: List[float]

# # ✅ Model for /register/ request
# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# # ✅ Simulated in-memory face DB
# known_faces: List[UserFace] = []

# @app.post("/register/")
# async def register_user(data: RegisterRequest):
#     try:
#         # Step 1: Download image from URL
#         response = requests.get(data.image_url)
#         if response.status_code != 200:
#             raise HTTPException(status_code=400, detail="Could not download image.")

#         # Step 2: Extract face encoding using dlib/face_recognition (you replace this with your real encoding logic)
#         encoding = [0.1, 0.2, 0.3]  # placeholder

#         # Step 3: Create UserFace object
#         user_face = UserFace(name=data.name, encoding=encoding)

#         # Step 4: Save to DB or memory
#         known_faces.append(user_face)

#         return {
#             "status": "success",
#             "message": f"{data.name} registered."
#         }

#     except Exception as e:
#         print(f"[ERROR] Exception while registering {data.name}: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
    


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from register_face import register_face  # Your real logic

# app = FastAPI()

# @app.get("/message/")
# async def root():
#     return {"message": "Face Recognition API is running."}

# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# @app.post("/register/")
# async def register_user(data: RegisterRequest):
#     try:
#         success = await register_face(
#             user_id=data.user_id,
#             name=data.name,
#             image_url=data.image_url
#         )

#         if success:
#             return {"status": "success", "message": f"{data.name} registered."}
#         else:
#             raise HTTPException(status_code=400, detail="Face not detected or failed to register.")

#     except Exception as e:
#         print(f"[ERROR] Exception while registering {data.name}: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
    




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from register_face import register_face, preload_facenet_model

app = FastAPI()

# Call this once at app startup to preload Facenet weights
@app.on_event("startup")
async def startup_event():
    preload_facenet_model()



@app.get("/message/")
async def root():
    return {"message": "Face Recognition API is running."}


# Request body schema
class RegisterRequest(BaseModel):
    user_id: str
    name: str
    image_url: str


@app.post("/register/")
async def register_user(req: RegisterRequest):
    try:
        success = await register_face(req.user_id, req.name, req.image_url)
        if success:
            return {"status": "success", "message": f"{req.name} registered."}
        else:
            raise HTTPException(status_code=500, detail="Failed to register user.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

