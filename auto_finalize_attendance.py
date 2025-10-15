from datetime import datetime, timedelta, time
from sqlalchemy import select, and_
from sqlalchemy.exc import SQLAlchemyError
from database import AsyncSessionLocal, AttendanceLog, UserFace

async def auto_finalize_attendance():
    try:
        yesterday = datetime.now().date() - timedelta(days=1)
        default_out_time = time(23, 59)

        async with AsyncSessionLocal() as session:

            #Finalize users who marked IN but not OUT
            result = await session.execute(
                select(AttendanceLog).where(
                    and_(
                        AttendanceLog.date == yesterday,
                        AttendanceLog.in_status == "present",
                        AttendanceLog.out_time == None
                    )
                )
            )
            logs = result.scalars().all()
            print(f"[INFO] Found {len(logs)} users to finalize OUT for {yesterday}")

            for log in logs:
                log.out_time = default_out_time
                log.out_status = "present"
                print(f"[AUTO-FINALIZED] {log.name} marked OUT at 11:59 PM")



            await session.commit()
            print("[SUCCESS] Auto-finalization completed.")

    except SQLAlchemyError as e:
        print(f"[DB ERROR] Auto finalization failed: {e}")