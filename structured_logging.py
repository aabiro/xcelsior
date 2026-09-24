"""B7.2 / DA§9.2 — Structured JSON logging and context propagation.

Configures structured logging across the platform using `structlog` and Python's
standard `logging` library. Every log record emitted includes:
    - timestamp (UTC ISO 8601)
    - severity (uppercase log level: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - environment (XCELSIOR_ENV, default: development)
    - service (XCELSIOR_SERVICE, default: control-plane)
    - build (XCELSIOR_BUILD or git commit, default: dev)
    - trace_id and span_id (W3C trace context from active OpenTelemetry span)
    - tenant_id (pseudonym where approved)
    - job_id, attempt_id, lease_id, command_id, action_id
    - error_code, retryable
    - event / message

All messages, kwargs, format strings, and tracebacks pass through the central
redaction library (log_pii_filter) to remove tokens, secrets, signed URLs,
authorization headers, connection strings, environment secrets, prompt bodies,
and private host addresses.

Console logging is attached first so read-only filesystems or unwritable log
files degrade safely to stdout without crashing startup (Track A invariant).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import sys
from typing import Any, Mapping

import structlog

from log_pii_filter import PIIScrubFilter, _scrub, install as _install_pii_scrub, scrub_data

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover
    trace = None  # type: ignore

# Ensure root & xcelsior logger trees have the PII scrub filter attached
_install_pii_scrub()


def _get_active_trace_context() -> tuple[str | None, str | None]:
    """Extract W3C trace_id and span_id from the active OpenTelemetry span."""
    if trace is None:
        return None, None
    try:
        span = trace.get_current_span()
        if span is not None and span.get_span_context().is_valid:
            ctx = span.get_span_context()
            return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    except Exception:
        pass
    return None, None


def add_standard_metadata(
    logger: logging.Logger | None,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Processor adding standard operational metadata required by DA§9.2."""
    # Timestamp: ISO-8601 in UTC
    if "timestamp" not in event_dict:
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Severity: uppercase log level name
    if "severity" not in event_dict:
        level = event_dict.get("level") or method_name
        event_dict["severity"] = str(level).upper()

    # Environment, service, and build
    if "environment" not in event_dict:
        event_dict["environment"] = os.environ.get("XCELSIOR_ENV", "development")
    if "service" not in event_dict:
        event_dict["service"] = os.environ.get("XCELSIOR_SERVICE", "control-plane")
    if "build" not in event_dict:
        event_dict["build"] = os.environ.get("XCELSIOR_BUILD", "dev")

    # W3C Trace Context
    trace_id, span_id = _get_active_trace_context()
    if trace_id and "trace_id" not in event_dict:
        event_dict["trace_id"] = trace_id
    if span_id and "span_id" not in event_dict:
        event_dict["span_id"] = span_id

    return event_dict


def pii_scrub_processor(
    logger: logging.Logger | None,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Processor running central redaction library on all event_dict values."""
    return scrub_data(event_dict)  # type: ignore[return-value]


def bind_log_context(**kwargs: Any) -> None:
    """Bind contextual identifiers to the current execution thread / async task."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_log_context() -> None:
    """Clear all context variables for the current execution thread / async task."""
    structlog.contextvars.clear_contextvars()


_CONFIGURED = False


def configure_structured_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
    json_format: bool | None = None,
) -> logging.Logger:
    """Configure structlog and standard logging for xcelsior.

    Degrades to console when log_file is unwritable.
    """
    global _CONFIGURED

    # Format decision: default to JSON in non-development or when explicitly requested
    if json_format is None:
        fmt_env = os.environ.get("XCELSIOR_LOG_FORMAT", "").lower()
        if fmt_env == "console":
            json_format = False
        elif fmt_env == "json":
            json_format = True
        else:
            json_format = True

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_standard_metadata,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        pii_scrub_processor,
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logger = logging.getLogger("xcelsior")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler: added first so unwritable file paths never cause silent failure
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.addFilter(PIIScrubFilter())
    logger.addHandler(ch)

    # File handler: optional permanent record
    if log_file:
        try:
            fh = logging.FileHandler(log_file)
        except OSError as exc:
            logger.warning(
                "file logging disabled (%s): %s — console only. "
                "Set XCELSIOR_LOG_FILE to a writable path (e.g. /data/xcelsior.log).",
                log_file,
                exc,
            )
        else:
            fh.setLevel(level)
            fh.setFormatter(formatter)
            fh.addFilter(PIIScrubFilter())
            logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "xcelsior") -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound to the given name."""
    if not _CONFIGURED:
        configure_structured_logging()
    return structlog.get_logger(name)
