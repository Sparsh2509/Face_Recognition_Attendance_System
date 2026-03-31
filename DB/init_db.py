import asyncio
from DB.database import engine, Base

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "_main_":
    asyncio.run(init_db())

