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

            # # Mark ABSENT for users who never marked IN
            # all_users_result = await session.execute(select(UserFace))
            # all_users = all_users_result.scalars().all()
            # all_user_ids = {user.user_id: user.name for user in all_users}

            # present_result = await session.execute(
            #     select(AttendanceLog.user_id).where(AttendanceLog.date == yesterday)
            # )
            # marked_user_ids = {row[0] for row in present_result.fetchall()}

            # missing_users = [uid for uid in all_user_ids if uid not in marked_user_ids]
            # print(f"[INFO] Found {len(missing_users)} users who did NOT mark IN on {yesterday}")

            # for uid in missing_users:
            #     session.add(AttendanceLog(
            #         user_id=uid,
            #         name=all_user_ids[uid],
            #         date=yesterday,
            #         in_time=None,
            #         out_time=None,
            #         in_status="absent",
            #         out_status="absent"
            #     ))
            #     print(f"[MARKED ABSENT] {all_user_ids[uid]} marked ABSENT for {yesterday}")

            await session.commit()
            print("[SUCCESS] Auto-finalization completed.")

    except SQLAlchemyError as e:
        print(f"[DB ERROR] Auto finalization failed: {e}")