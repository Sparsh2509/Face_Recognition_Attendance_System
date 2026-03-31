import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from scripts.auto_finalize_attendance import auto_finalize_attendance

if __name__ == "__main__":
    asyncio.run(auto_finalize_attendance())