# Xcelsior — Serverless per-endpoint API keys

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.keys")

KEY_PREFIX = "xcel_"
RAW_KEY_BYTES = 32


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_raw_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(RAW_KEY_BYTES)


def create_endpoint_key(
    repo: ServerlessRepo,
    owner_id: str,
    *,
    endpoint_id: str | None = None,
    name: str = "default",
    scopes: str = "inference:write",
    rate_limit_rpm: int = 60,
) -> tuple[str, dict]:
    """Return (raw_key, key_row). Raw key shown once to the customer."""
    raw = generate_raw_key()
    prefix = raw[:12]
    key_hash = _hash_key(raw)
    row = repo.create_api_key(
        owner_id,
        prefix,
        key_hash,
        endpoint_id=endpoint_id,
        name=name,
        scopes=scopes,
        rate_limit_rpm=rate_limit_rpm,
    )
    return raw, row


def validate_key(repo: ServerlessRepo, raw_key: str) -> dict | None:
    if not raw_key or not raw_key.startswith(KEY_PREFIX):
        return None
    row = repo.get_api_key_by_hash(_hash_key(raw_key))
    if row:
        repo.touch_api_key(str(row["key_id"]))
    return row


def key_has_scope(key_row: dict, scope: str) -> bool:
    scopes = str(key_row.get("scopes") or "")
    if scopes in ("full-access", "*"):
        return True
    return scope in {s.strip() for s in scopes.split(",") if s.strip()}