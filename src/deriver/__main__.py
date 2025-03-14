import asyncio

import uvloop

from .queue import main

if __name__ == "__main__":
    print("[DERIVER] Starting deriver queue processor")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        print("[DERIVER] Running main loop")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DERIVER] Shutdown initiated via KeyboardInterrupt")
    except Exception as e:
        print(f"[DERIVER] Error in main process: {str(e)}")
    finally:
        print("[DERIVER] Deriver process exiting")
