from database import AsyncSessionLocal, UserFace, AttendanceLog
from sqlalchemy import delete
import asyncio

async def clear_database():
    async with AsyncSessionLocal() as session:
        try:
            # Delete from AttendanceLog first (because it depends on user)
            await session.execute(delete(AttendanceLog))
            print("[CLEAN] Cleared attendance_status ")

            # Then delete users
            await session.execute(delete(UserFace))
            print("[CLEAN] Cleared user_faces ")

            await session.commit()
            print("[DONE] Database reset complete ")

        except Exception as e:
            await session.rollback()
            print(f"[ERROR] Failed to reset database: {e}")

# Run it
if __name__ == "__main__":
    asyncio.run(clear_database())