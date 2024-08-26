import asyncio
import argparse

import uvloop

from .queue import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-timing', action='store_true', help='Enable timing measurements')
    args = parser.parse_args()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(main(enable_timing=args.enable_timing))