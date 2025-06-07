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
    
# import numpy as np
# from deepface import DeepFace
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# # from register_face import register_face
# import os

# try:
#     from register_face import register_face
# except Exception as e:
#     print("[IMPORT ERROR] in register_face.py:", e)
#     import sys
#     sys.exit(1)


# app = FastAPI()

# # def preload_sface_model():
# #     print("[INFO] Preloading SFace model...")
# #     dummy = np.zeros((160, 160, 3), dtype=np.uint8)
# #     DeepFace.represent(img_path=dummy, model_name="SFace", enforce_detection=False)
# #     print("[INFO] SFace model preloaded.")


# # Call this once at app startup to preload Facenet weights
# @app.on_event("startup")
# async def startup_event():
#     preload_sface_model()



# @app.get("/message/")
# async def root():
#     return {"message": "Face Recognition API is running."}


# # Request body schema
# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str


# # @app.post("/register/")
# # async def register_user(req: RegisterRequest):

# #     print("[DEBUG] /register/ POST route hit") 
     
# #     try:
# #         print(f"Received register request: {req.dict()}")
# #         success = await register_face(req.user_id, req.name, req.image_url)
# #         if success:
# #             return {"status": "success", "message": f"{req.name} registered."}
# #         else:
# #             raise HTTPException(status_code=500, detail="Failed to register user.")
# #     except ValueError as ve:
# #         print("ValueError:", ve)
# #         raise HTTPException(status_code=400, detail=str(ve))
# #     except Exception as e:
# #         print("Unexpected Exception:", e)
# #         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# model_loaded = False  # global variable

# @app.post("/register/")
# async def register_user(req: RegisterRequest):
#     global model_loaded

#     print("[DEBUG] /register/ POST route hit")

#     try:
#         if not model_loaded:
#             print("[INFO] Lazy loading SFace model...")
#             dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#             DeepFace.represent(img_path=dummy, model_name="SFace", enforce_detection=False)
#             model_loaded = True
#             print("[INFO] SFace model loaded.")

#         # Proceed with registration
#         print(f"Received register request: {req.dict()}")
#         success = await register_face(req.user_id, req.name, req.image_url)
#         if success:
#             return {"status": "success", "message": f"{req.name} registered."}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to register user.")
    
#     except ValueError as ve:
#         print("ValueError:", ve)
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         print("Unexpected Exception:", e)
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     print(f"Running on port {port}")
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=port)

# import numpy as np
# from deepface import DeepFace
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import os

# # Import your register_face function
# try:
#     from register_face import register_face
# except Exception as e:
#     print("[IMPORT ERROR] in register_face.py:", e)
#     import sys
#     sys.exit(1)

# app = FastAPI()

# model_loaded = False  # global flag

# @app.get("/")
# async def root():
#     return {"message": "Face Recognition API is running."}

# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# @app.post("/register/")
# async def register_user(req: RegisterRequest):
#     global model_loaded
#     print("[DEBUG] /register/ POST route hit")

#     try:
#         if not model_loaded:
#             print("[INFO] Lazy loading SFace model...")
#             dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#             DeepFace.represent(img_path=dummy, model_name="SFace", enforce_detection=False)
#             model_loaded = True
#             print("[INFO] SFace model loaded.")

#         print(f"Received register request: {req.dict()}")
#         success = await register_face(req.user_id, req.name, req.image_url)
#         if success:
#             return {"status": "success", "message": f"{req.name} registered."}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to register user.")

#     except ValueError as ve:
#         print("ValueError:", ve)
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         print("Unexpected Exception:", e)
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     print(f"Running on port {port}")
#     uvicorn.run(app, host="0.0.0.0", port=port)

# import os
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from deepface import DeepFace

# # Import your register_face function
# try:
#     from register_face import register_face
# except Exception as e:
#     print("[IMPORT ERROR] in register_face.py:", e)
#     import sys
#     sys.exit(1)

# app = FastAPI()

# # Global model cache
# sface_model = None

# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# @app.get("/")
# async def root():
#     return {"message": "Face Recognition API is running."}

# @app.post("/register/")
# async def register_user(req: RegisterRequest):
#     global sface_model

#     print("[DEBUG] /register/ POST route hit")

#     try:
#         if sface_model is None:
#             print("[INFO] Lazy loading SFace model from local weights...")
#             # Monkey-patch DeepFace's weight path
#             from deepface.commons import functions
#             functions.get_deepface_home = lambda: os.path.abspath("models")

#             # Load model with monkey-patched path
#             sface_model = DeepFace.build_model("SFace")
#             print("[INFO] SFace model loaded from local ONNX.")

#         # Dummy call to ensure model is initialized
#         dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#         DeepFace.represent(img_path=dummy, model_name="SFace", model=sface_model, enforce_detection=False)

#         # Call your custom registration
#         success = await register_face(req.user_id, req.name, req.image_url)
#         if success:
#             return {"status": "success", "message": f"{req.name} registered."}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to register user.")

#     except Exception as e:
#         print("Unexpected Exception:", e)
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# import os
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from deepface import DeepFace

# # Keep your register_face import as you had it
# try:
#     from register_face import register_face
# except Exception as e:
#     print("[IMPORT ERROR] in register_face.py:", e)
#     import sys
#     sys.exit(1)

# app = FastAPI()

# # Global model cache
# sface_model = None

# class RegisterRequest(BaseModel):
#     user_id: str
#     name: str
#     image_url: str

# @app.get("/")
# async def root():
#     return {"message": "Face Recognition API is running."}

# @app.post("/register/")
# async def register_user(req: RegisterRequest):
#     global sface_model

#     print("[DEBUG] /register/ POST route hit")

#     try:
#         if sface_model is None:
#             print("[INFO] Loading SFace model from local weights...")

#             # Import SFace model directly (make sure deepface version supports this)
#             from deepface.basemodels.sface import SFace

#             # Provide path to your local ONNX weights file
#             weight_path = os.path.abspath("models/face_recognition_sface_2021dec.onnx")

#             sface_model = SFace.loadModel(weight_path)
#             print("[INFO] SFace model loaded from local ONNX.")

#         # Dummy call to ensure model is initialized
#         dummy = np.zeros((160, 160, 3), dtype=np.uint8)
#         DeepFace.represent(img_path=dummy, model_name="SFace", model=sface_model, enforce_detection=False)

#         # Call your custom registration function (async)
#         success = await register_face(req.user_id, req.name, req.image_url)
#         if success:
#             return {"status": "success", "message": f"{req.name} registered."}
#         else:
#             raise HTTPException(status_code=500, detail="Failed to register user.")

#     except Exception as e:
#         print("Unexpected Exception:", e)
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



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

