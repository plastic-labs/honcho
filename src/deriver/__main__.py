import asyncio
from dotenv import load_dotenv
import uvloop

from src.utils.logging import console, setup_rich_logging

from .queue import main

load_dotenv()

setup_rich_logging()

if __name__ == "__main__":
    console.print("[bold blue]🚀 [DERIVER][/] Starting deriver queue processor")
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        console.print("[bold green]▶️  [DERIVER][/] Running main loop")
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print(
            "[bold yellow]⏹️  [DERIVER][/] Shutdown initiated via KeyboardInterrupt"
        )
    except Exception as e:
        console.print(f"[bold red]❌ [DERIVER][/] Error in main process: {str(e)}")
    finally:
        console.print("[bold cyan]👋 [DERIVER][/] Deriver process exiting")
