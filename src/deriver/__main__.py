import asyncio
import logging
import os

import uvloop
from prometheus_client import start_http_server

from src.config import settings

from .queue_manager import main


def setup_logging():
    """
    Configure logging for the deriver process.
    """
    # Get log level from environment or settings
    log_level_str = os.getenv("LOG_LEVEL", settings.LOG_LEVEL).upper()

    log_levels = {
        "CRITICAL": logging.CRITICAL,  # 50
        "ERROR": logging.ERROR,  # 40
        "WARNING": logging.WARNING,  # 30
        "INFO": logging.INFO,  # 20
        "DEBUG": logging.DEBUG,  # 10
        "NOTSET": logging.NOTSET,  # 0
    }

    log_level = log_levels.get(log_level_str, logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Disable SQLAlchemy engine logging unless explicitly enabled
    if not settings.DB.SQL_DEBUG:
        logging.getLogger("sqlalchemy.engine.Engine").disabled = True

    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("groq._base_client").setLevel(logging.WARNING)


def start_metrics_server() -> None:
    """Start Prometheus metrics HTTP server on port 9090."""
    try:
        # Uses default REGISTRY from prometheus_client
        start_http_server(9090, addr="0.0.0.0")  # nosec B104
        print("[DERIVER] Starting Prometheus metrics server on port 9090")
    except Exception as e:
        print(f"[DERIVER] Failed to start Prometheus metrics server: {str(e)}")


if __name__ == "__main__":
    print("[DERIVER] Starting deriver queue processor")

    # Setup logging before starting the main loop
    setup_logging()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        print("[DERIVER] Running main loop")
        start_metrics_server()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DERIVER] Shutdown initiated via KeyboardInterrupt")
    except Exception as e:
        print(f"[DERIVER] Error in main process: {str(e)}")
    finally:
        print("[DERIVER] Deriver process exiting")
