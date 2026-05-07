"""
Centralized logging configuration for llm-swarm-desktop.

Security invariants (CLAUDE.md §6, docs/notes.md §4 and §6):
- device_token, device_code, device_signature, device_pubkey, private_key /
  privkey values MUST NOT appear in log output (files or stdout).
- peer_id is public (base58(sha256(pubkey))) — allowed in DEBUG to file,
  masked to a 6-char fingerprint at INFO level on stdout.
- The ``SensitiveDataFilter`` and ``RedactingFormatter`` enforce these rules
  unconditionally on every log record, regardless of where in the codebase the
  log call originates.

Usage::

    from app.logging_setup import setup_logging
    setup_logging(level=logging.DEBUG, log_file=Path("desktop.log"))
"""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Deny-list — field names whose values are always redacted
# ---------------------------------------------------------------------------

_DEFAULT_DENY_KEYS: list[str] = [
    "device_token",
    "device_code",
    "device_signature",
    "device_pubkey",
    "private_key",
    "privkey",
    "secret",
    "Authorization",
]

# Pre-compiled per-key patterns are built lazily and cached here.
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}

# Keys whose values may contain a single internal space (e.g. HTTP scheme prefix).
# Pattern: ``Authorization: Bearer eyJ...`` — the value is ``Bearer eyJ...``.
_MULTI_WORD_VALUE_KEYS: frozenset[str] = frozenset({"Authorization"})


def _pattern_for(key: str) -> re.Pattern[str]:
    """Return (cached) compiled regex that matches ``key=<value>`` or ``key: <value>``."""
    if key not in _PATTERN_CACHE:
        # Matches:   key = "value"   key='value'   key=value   key: value
        # Group 1 — the key + separator prefix (kept).
        # Group 2 — the actual secret value (replaced with <redacted>).
        escaped = re.escape(key)
        if key in _MULTI_WORD_VALUE_KEYS:
            # Allow a single word followed by a space and another word
            # (e.g. "Bearer eyJ..."), in addition to plain single-token values.
            value_pat = r"""(?:[^"',\s}\]]+(?:\s+[^"',\s}\]]+)*)"""
        else:
            value_pat = r"""[^"',\s}\]]+"""
        pattern = re.compile(
            r"(?i)(" + escaped + r"""["']?\s*[:=]\s*["']?)(""" + value_pat + r")",
            re.UNICODE,
        )
        _PATTERN_CACHE[key] = pattern
    return _PATTERN_CACHE[key]


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------


def redact(text: str | None, deny_keys: list[str] | None = None) -> str | None:
    """Replace sensitive values in *text* with ``<redacted>``.

    Idempotent — calling twice on already-redacted text is a no-op.
    Returns *text* unchanged (including ``None``) if it is not a string.

    Args:
        text: Arbitrary string that may contain sensitive key=value pairs.
              ``None`` is returned as-is.
        deny_keys: Override the default deny-list (merged, not replaced).

    Returns:
        A copy of *text* with all detected sensitive values replaced,
        or ``None`` if *text* was ``None``.
    """
    if text is None:
        return None
    keys = list(_DEFAULT_DENY_KEYS)
    if deny_keys:
        for k in deny_keys:
            if k not in keys:
                keys.append(k)
    for key in keys:
        text = _pattern_for(key).sub(r"\1<redacted>", text)
    return text


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class SensitiveDataFilter(logging.Filter):
    """``logging.Filter`` that scrubs sensitive values from every log record.

    Applied to both the StreamHandler and RotatingFileHandler so that no
    sensitive data can reach any sink, regardless of formatter.

    Processing order:
    1. ``record.msg`` (str) — in-place regex substitution for each deny key.
    2. ``record.args`` as ``dict`` — shallow copy with redacted values.
    3. ``record.args`` as ``tuple`` / f-string result — record is
       pre-formatted via ``getMessage()``, then stored back as
       ``(redacted_message, ())`` so ``Formatter.format()`` sees only the
       already-safe string.
    """

    def __init__(self, name: str = "", deny_keys: list[str] | None = None) -> None:
        super().__init__(name)
        self._keys: list[str] = list(_DEFAULT_DENY_KEYS)
        if deny_keys:
            for k in deny_keys:
                if k not in self._keys:
                    self._keys.append(k)

    def _redact(self, text: str) -> str:
        return redact(text, []) or ""  # already uses self._keys-equivalent default

    # ------------------------------------------------------------------

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        # --- 1. Redact msg string -------------------------------------------
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)

        # --- 2. Redact args ---------------------------------------------------
        if isinstance(record.args, dict):
            # Shallow copy; values that are strings get redacted.
            safe: dict[str, Any] = {}
            for k, v in record.args.items():  # type: ignore[union-attr]
                if k in self._keys:
                    safe[k] = "<redacted>"
                elif isinstance(v, str):
                    safe[k] = redact(v)
                else:
                    safe[k] = v
            record.args = safe  # type: ignore[assignment]

        elif record.args is not None:
            # Tuple args or any other type: pre-render the message and
            # collapse args so the Formatter just uses msg verbatim.
            try:
                rendered = record.getMessage()
            except Exception:  # noqa: BLE001
                rendered = str(record.msg)
            record.msg = redact(rendered)
            record.args = ()

        return True


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RedactingFormatter(logging.Formatter):
    """``logging.Formatter`` that post-processes the fully formatted line.

    This catches anything that slipped past ``SensitiveDataFilter``, most
    importantly exc_info / exception text, which is appended *after* the
    filter runs.
    """

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        return redact(line) or ""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_LOG_BACKUP_COUNT = 3


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure the root logger with redacting handlers.

    Safe to call multiple times — subsequent calls replace existing handlers
    on the root logger to avoid duplicate output.

    Args:
        level: Logging level for both handlers (default: ``INFO``).
        log_file: Optional path for a rotating file log.  If ``None``, only
            the stream handler is installed.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers added by earlier calls or by basicConfig().
    root.handlers.clear()

    fmt = RedactingFormatter(_DEFAULT_FMT)
    filt = SensitiveDataFilter()

    # --- stdout handler -----------------------------------------------------
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(filt)
    root.addHandler(stream_handler)

    # --- rotating file handler ----------------------------------------------
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        file_handler.addFilter(filt)
        root.addHandler(file_handler)
