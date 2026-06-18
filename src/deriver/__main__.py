import asyncio
import logging
import os
import signal
import sys
from prometheus_client import start_http_server

from honcho.deriver import Deriver

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()


def handle_sigterm(signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


def start_metrics_server():
    """Start Prometheus metrics server on configurable port."""
    # Use DERIVER_METRICS_PORT env var if set, otherwise default to 9091
    # (9090 often conflicts with Prometheus itself in production)
    metrics_port = int(os.getenv("DERIVER_METRICS_PORT", "9091"))
    start_http_server(metrics_port)
    logger.info(f"Metrics server started on port {metrics_port}")


async def main():
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    try:
        # Start metrics server
        start_metrics_server()
        
        # Initialize and run deriver
        deriver = Deriver()
        logger.info("Starting deriver queue processor")
        await deriver.run(shutdown_event=shutdown_event)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise
    finally:
        logger.info("Deriver process exiting")


if __name__ == "__main__":
    asyncio.run(main())
