# Xcelsior — Serverless endpoint env encryption + redaction

from __future__ import annotations

import json
import logging
from typing import Any

from privacy import redact_env_vars

log = logging.getLogger("xcelsior.serverless.env_secrets")

_ENC_PREFIX = "enc:"


def _is_sensitive_key(key: str) -> bool:
    key_upper = str(key).upper()
    if any(
        token in key_upper
        for token in ("SECRET", "PASSWORD", "TOKEN", "KEY", "CREDENTIAL", "API_KEY")
    ):
        return True
    return key_upper in {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
    }


def encrypt_env_for_storage(env: dict[str, Any] | None) -> dict[str, str]:
    """Encrypt sensitive env values before JSONB persistence."""
    if not env:
        return {}
    from security import encrypt_secret

    stored: dict[str, str] = {}
    for key, value in env.items():
        k = str(key).strip()
        if not k:
            continue
        raw = str(value) if value is not None else ""
        if _is_sensitive_key(k) and raw:
            stored[k] = _ENC_PREFIX + encrypt_secret(raw)
        else:
            stored[k] = raw
    return stored


def decrypt_env_for_worker(env: dict[str, Any] | None) -> dict[str, str]:
    """Decrypt env values for scheduler/worker injection."""
    if not env:
        return {}
    from security import decrypt_secret

    resolved: dict[str, str] = {}
    for key, value in env.items():
        k = str(key)
        raw = str(value) if value is not None else ""
        if raw.startswith(_ENC_PREFIX):
            try:
                resolved[k] = decrypt_secret(raw[len(_ENC_PREFIX) :])
            except Exception as e:
                log.warning("Failed to decrypt env key %s: %s", k, e)
                resolved[k] = ""
        else:
            resolved[k] = raw
    return resolved


def redact_env_for_api(env: dict[str, Any] | None) -> dict[str, str]:
    """Safe env copy for API responses (never returns decrypted secrets)."""
    if not env:
        return {}
    plain = {str(k): str(v) if not str(v).startswith(_ENC_PREFIX) else "***" for k, v in env.items()}
    return redact_env_vars(plain)


def payload_byte_size(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))