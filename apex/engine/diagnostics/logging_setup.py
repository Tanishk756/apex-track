"""
Logging & Diagnostics Setup
============================
Configures structlog for the entire engine.
Call configure_logging() once at startup before any other module.

Output formats:
    console — coloured, human-readable (development)
    json    — machine-readable structured logs (production/ingest)
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    level: str = "INFO",
    fmt: Literal["console", "json"] = "console",
) -> None:
    """
    Configure structlog and stdlib logging for the entire APEX-Track engine.
    Must be called once at application startup.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors applied to every log record
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "PIL", "asyncio", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
