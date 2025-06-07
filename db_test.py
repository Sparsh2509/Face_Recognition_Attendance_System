import asyncio
from database import AsyncSessionLocal
from sqlalchemy import text

async def test_connection():
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            print("[SUCCESS] Connected to Neon DB!")
    except Exception as e:
        print("[ERROR] Failed to connect:", e)

if __name__ == "__main__":
    asyncio.run(test_connection())