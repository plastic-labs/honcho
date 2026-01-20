import asyncio
import logging
import os

import uvloop

from src.config import settings
from src.telemetry import initialize_telemetry, shutdown_otel_metrics

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


if __name__ == "__main__":
    print("[DERIVER] Starting deriver queue processor")

    # Setup logging before starting the main loop
    setup_logging()

    # Initialize OTel metrics if enabled
    initialize_telemetry()

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    try:
        print("[DERIVER] Running main loop")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DERIVER] Shutdown initiated via KeyboardInterrupt")
    except Exception as e:
        print(f"[DERIVER] Error in main process: {str(e)}")
    finally:
        shutdown_otel_metrics()
        print("[DERIVER] Deriver process exiting")
