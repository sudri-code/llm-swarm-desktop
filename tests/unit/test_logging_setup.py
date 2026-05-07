"""
Unit tests for app.logging_setup.

Covers:
- device_token in plain string message → redacted.
- dict-args: device_token redacted, peer_id preserved.
- exc_info containing device_signature → redacted in formatted output.
- Authorization header → redacted.
- redact() idempotence.
- StreamHandler output captured via capsys.
- tuple-args path (pre-render + collapse).
- Nested sensitive key in tuple-arg string.
"""

from __future__ import annotations

import logging
import sys

import pytest

from app.logging_setup import (
    RedactingFormatter,
    SensitiveDataFilter,
    redact,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(name: str) -> logging.Logger:
    """Return a logger with no inherited handlers (test-local)."""
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = False
    return lg


def _capture_handler(stream: object) -> tuple[logging.Logger, logging.StreamHandler]:  # type: ignore[type-arg]
    """Attach a RedactingFormatter + SensitiveDataFilter StreamHandler."""
    handler = logging.StreamHandler(stream)  # type: ignore[arg-type]
    handler.setFormatter(RedactingFormatter("%(message)s"))
    handler.addFilter(SensitiveDataFilter())
    return handler


# ---------------------------------------------------------------------------
# 1. Plain string message — device_token
# ---------------------------------------------------------------------------


def test_plain_msg_device_token_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.plain_token")
    logger.info("device_token=abc123def")
    out = capsys.readouterr().out
    assert "abc123def" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 2. dict-args — device_token redacted, peer_id preserved
# ---------------------------------------------------------------------------


def test_dict_args_token_redacted_peer_id_preserved(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.dict_args")
    logger.info("data: %s", {"device_token": "xyz_secret", "peer_id": "abc123"})
    out = capsys.readouterr().out
    assert "xyz_secret" not in out
    assert "<redacted>" in out
    # peer_id is a public identifier — must survive
    assert "abc123" in out


# ---------------------------------------------------------------------------
# 3. exc_info with device_signature in exception text
# ---------------------------------------------------------------------------


def test_exc_info_device_signature_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.exc_info")
    try:
        raise ValueError("device_signature: BASE64STRING==")
    except ValueError:
        logger.exception("caught error")
    out = capsys.readouterr().out
    assert "BASE64STRING==" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 4. Authorization header
# ---------------------------------------------------------------------------


def test_authorization_bearer_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.auth_header")
    logger.info("Authorization: Bearer eyJhbGciOiJFZERTQQ")
    out = capsys.readouterr().out
    assert "eyJhbGciOiJFZERTQQ" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 5. redact() idempotence and None handling
# ---------------------------------------------------------------------------


def test_redact_idempotent() -> None:
    raw = "device_token=supersecret"
    once = redact(raw)
    twice = redact(once)
    assert once == twice
    assert "supersecret" not in once


def test_redact_handles_none() -> None:
    """redact(None) должен возвращать None без исключений."""
    result = redact(None)
    assert result is None


# ---------------------------------------------------------------------------
# 6. StreamHandler captured via capsys (setup_logging path)
# ---------------------------------------------------------------------------


def test_setup_logging_stream_captured(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("test.stream_capsys")
    logger.info("hello world")
    out = capsys.readouterr().out
    assert "hello world" in out


# ---------------------------------------------------------------------------
# 7. Tuple-args path — sensitive value inside a %-formatted string
# ---------------------------------------------------------------------------


def test_tuple_args_sensitive_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.tuple_args")
    # % formatting with a plain string containing a sensitive key=value
    logger.debug("sending request with device_code=%s", "DCCODE99")
    out = capsys.readouterr().out
    assert "DCCODE99" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 8. privkey / private_key variants
# ---------------------------------------------------------------------------


def test_privkey_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.privkey")
    logger.warning("privkey=AAABBBCCC loaded")
    out = capsys.readouterr().out
    assert "AAABBBCCC" not in out
    assert "<redacted>" in out


def test_private_key_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.private_key")
    logger.warning("private_key: SECRETBYTES123")
    out = capsys.readouterr().out
    assert "SECRETBYTES123" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 9. device_pubkey redacted
# ---------------------------------------------------------------------------


def test_device_pubkey_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger("test.device_pubkey")
    logger.info("device_pubkey=b64urlpubkeyABC")
    out = capsys.readouterr().out
    assert "b64urlpubkeyABC" not in out
    assert "<redacted>" in out


# ---------------------------------------------------------------------------
# 10. SensitiveDataFilter standalone — does not suppress records
# ---------------------------------------------------------------------------


def test_filter_returns_true_always() -> None:
    filt = SensitiveDataFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="device_token=xyz",
        args=(),
        exc_info=None,
    )
    result = filt.filter(record)
    assert result is True
    assert "xyz" not in record.msg
    assert "<redacted>" in record.msg


# ---------------------------------------------------------------------------
# 11. RedactingFormatter — exc_text path
# ---------------------------------------------------------------------------


def test_redacting_formatter_exc_text() -> None:
    """RedactingFormatter must redact exception text appended by format()."""
    import io

    try:
        raise RuntimeError("device_token=should_be_gone")
    except RuntimeError:
        ei = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="oops",
        args=(),
        exc_info=ei,
    )
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    handler.addFilter(SensitiveDataFilter())
    handler.emit(record)
    output = buf.getvalue()
    assert "should_be_gone" not in output
    assert "<redacted>" in output
