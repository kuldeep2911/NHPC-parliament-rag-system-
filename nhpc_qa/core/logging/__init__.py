"""
Structured logging — one setup for the whole application.

Every pipeline run, watcher event, and delete/reactivate goes through here, so an operator
reading the journal sees one consistent format rather than four phases' worth of ad-hoc
prints.

    from nhpc_qa.core.logging import setup, get_logger
    setup()                       # once, at process start
    log = get_logger(__name__)

Text by default (readable in a terminal and in `journalctl`); set NHPC_LOG_JSON=1 for
one-JSON-object-per-line, which is what you want when the on-prem server ships logs to a
collector.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields passed via logger.info(..., extra={...})
    are merged in, so a watcher event can carry its doc_key/path without string-mangling."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "asctime", "message"}

    def format(self, record):
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self.RESERVED and not k.startswith("_"):
                out[k] = v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def setup(level=None, json_output=None):
    """Idempotent: safe to call from the CLI, the API and the watcher alike."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = level or os.environ.get("NHPC_LOG_LEVEL", "INFO").upper()
    if json_output is None:
        json_output = os.environ.get("NHPC_LOG_JSON", "").strip().lower() in {
            "1", "true", "yes", "on"}

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s: %(message)s",
            datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))

    # Third-party noise: these libraries log at INFO on every call and drown the signal.
    for noisy in ("httpx", "httpcore", "urllib3", "docling", "RapidOCR", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
