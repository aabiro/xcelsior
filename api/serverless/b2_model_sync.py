# B2 model sync credentials — same env contract as pxl-registry / ai-data-factory.
# Source: B2_MODEL_SYNC_BUCKET, B2_MODEL_SYNC_KEY_ID, B2_MODEL_SYNC_KEY
# Fallback: B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("xcelsior.serverless.b2_model_sync")

_DEFAULT_BUCKET = "pixelenhance-models"


@dataclass(frozen=True)
class B2ModelSyncConfig:
    bucket: str
    key_id: str | None
    app_key: str | None


def b2_model_sync_config() -> B2ModelSyncConfig:
    """Read B2 model-sync env (pxl-registry .env.example contract)."""
    return B2ModelSyncConfig(
        bucket=os.environ.get("B2_MODEL_SYNC_BUCKET", _DEFAULT_BUCKET).strip() or _DEFAULT_BUCKET,
        key_id=(
            os.environ.get("B2_MODEL_SYNC_KEY_ID")
            or os.environ.get("B2_APPLICATION_KEY_ID")
            or None
        ),
        app_key=(
            os.environ.get("B2_MODEL_SYNC_KEY")
            or os.environ.get("B2_APPLICATION_KEY")
            or None
        ),
    )


def default_b2_bucket() -> str:
    return b2_model_sync_config().bucket


def ensure_b2_authorized() -> bool:
    """Authorize b2 CLI using B2_MODEL_SYNC_* when not already logged in."""
    b2 = shutil.which("b2")
    if not b2:
        log.debug("b2 CLI not found")
        return False

    cfg = b2_model_sync_config()
    try:
        probe = subprocess.run(
            [b2, "account", "get"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode == 0:
            try:
                info = json.loads(probe.stdout)
                allowed = [b["name"] for b in info.get("allowed", {}).get("buckets", [])]
                if not allowed or cfg.bucket in allowed:
                    return True
            except json.JSONDecodeError:
                return True
    except (subprocess.TimeoutExpired, OSError) as e:
        log.debug("b2 account get failed: %s", e)

    if not cfg.key_id or not cfg.app_key:
        return False

    try:
        auth = subprocess.run(
            [b2, "account", "authorize", cfg.key_id, cfg.app_key],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if auth.returncode == 0:
            log.info("B2 authorized via B2_MODEL_SYNC_* for bucket %s", cfg.bucket)
            return True
        log.warning("B2 authorize failed: %s", (auth.stderr or auth.stdout or "").strip())
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("B2 authorize error: %s", e)
    return False


def download_b2_object(bucket: str, key: str, dest: Path) -> bool:
    """Download one B2 object to dest using the model-sync credentials."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True

    if not ensure_b2_authorized():
        return False

    b2 = shutil.which("b2")
    if not b2:
        return False

    bkt = bucket or default_b2_bucket()
    uri = f"b2://{bkt}/{key.lstrip('/')}"
    for cmd in (
        [b2, "file", "download", uri, str(dest)],
        [b2, "download-file-by-name", bkt, key, str(dest)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
            if dest.is_file() and dest.stat().st_size > 0:
                return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.debug("b2 download failed %s: %s", cmd[:3], e)
    return False