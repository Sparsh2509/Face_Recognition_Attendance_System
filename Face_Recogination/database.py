from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String , JSON
from dotenv import load_dotenv
import os

load_dotenv()

# Construct the database URL using environment variables

DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
# print(DATABASE_URL)
if not DATABASE_URL:
    raise ValueError("Database configuration is incomplete.")

# Create the async engine
engine = create_async_engine(
    DATABASE_URL, 
    connect_args={"ssl": True},
    echo=True
)

# Create a sessionmaker for async sessions
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for ORM models
Base = declarative_base()



class UserFace(Base):
    __tablename__ = "user_faces"  # your actual table name

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True )
    name = Column(String)
    encoding = Column(String) 
    avg_bg_color = Column(JSON)


# Dependency to get the database session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


print("[DEBUG] DATABASE_URL:", DATABASE_URL)




##### Database showing commands in cmd
# psql -h ep-polished-poetry-a80gt1ci-pooler.eastus2.azure.neon.tech -p 5432 -U neondb_owner -d neondb
# DB_PASSWORD=npg_fjZs4M8DpFXG



