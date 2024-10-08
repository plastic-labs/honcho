import asyncio
import argparse

import uvloop

from .queue import main

if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main())