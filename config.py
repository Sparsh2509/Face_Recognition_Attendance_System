import os
import cloudinary
from dotenv import load_dotenv

load_dotenv()


cloud = cloudinary.config(
    CLOUD_NAME=os.getenv('CLOUDINARY_CLOUD_NAME'),
    API_KEY= os.getenv('CLOUDINARY_API_KEY'),
    API_SECRET= os.getenv('CLOUDINARY_API_SECRET'),
)

print(cloud)



