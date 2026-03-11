"""
Structured logging configuration.

WHY STRUCTURED LOGGING: Regular logs look like:
    "INFO: Verified claim about Eiffel Tower in 45ms"

Structured logs look like:
    {"event": "claim_verified", "claim_id": "abc123", "status": "supported",
     "latency_ms": 45, "timestamp": "2024-01-15T10:30:00Z"}

The second form is machine-parseable — you can filter, aggregate, and alert on
specific fields in CloudWatch/Datadog/ELK without regex gymnastics.
"""

import logging
import sys

import structlog
from app.config import settings


def setup_logging():
    """Configure structlog for JSON output with context binding."""

    # Structlog processors — each transforms the log event dict in sequence
    processors = [
        # Adds log level as a field: {"level": "info"}
        structlog.stdlib.add_log_level,
        # Adds timestamp: {"timestamp": "2024-01-15T10:30:00Z"}
        structlog.processors.TimeStamper(fmt="iso"),
        # If an exception is attached, formats the traceback
        structlog.processors.format_exc_info,
        # Renders the final dict as JSON string
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging (for uvicorn, httpx, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance, optionally bound to a component name.

    Usage:
        logger = get_logger("vector_store")
        logger.info("index_loaded", num_vectors=50000)
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    return logger
