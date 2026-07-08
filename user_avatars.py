"""User profile avatar storage in object storage (S3-compatible).

Industry-standard pattern: image bytes live in object storage; the user record
holds a stable storage key plus cache-bust timestamp in ``preferences``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_AVATAR_PREFIX = "user_profile"
_LEGACY_AVATAR_DIR = Path(os.environ.get("XCELSIOR_AVATAR_DIR", "/opt/xcelsior/data/avatars"))
_MAX_BYTES = 2 * 1024 * 1024
_EXTENSIONS = ("webp", "jpg", "jpeg", "png")
_MIME_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def sniff_image_ext(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def avatar_storage_key(user_id: str, ext: str) -> str:
    uid = str(user_id or "").strip()
    return f"{_AVATAR_PREFIX}/{uid}/avatar.{ext}"


def avatar_asset_exists(prefs: dict | None, user_id: str) -> bool:
    """True when a profile photo is present in Postgres, storage, or legacy disk."""
    uid = str(user_id or "").strip()
    # Primary store: Postgres (durable across redeploys).
    try:
        from db import AvatarStore

        if AvatarStore.exists(uid):
            return True
    except Exception:
        pass

    # Legacy fallbacks for avatars uploaded before the Postgres migration.
    prefs = prefs if isinstance(prefs, dict) else {}
    key = (prefs.get("avatar_key") or "").strip()
    if key:
        if local_avatar_path_for_key(key):
            return True
        try:
            from artifacts import StorageBackend, get_artifact_manager

            mgr = get_artifact_manager()
            if mgr.primary.config.backend != StorageBackend.LOCAL and mgr.primary.head_object(key):
                return True
        except Exception:
            pass
    return _legacy_avatar_file_for_user(uid) is not None


def avatar_public_url(prefs: dict | None, user_id: str) -> str | None:
    if not avatar_asset_exists(prefs, user_id):
        return None
    prefs = prefs if isinstance(prefs, dict) else {}
    version = int(prefs.get("avatar_updated_at") or 0)
    if version <= 0:
        version = int(time.time())
    return f"/api/auth/me/avatar?v={version}"


def save_avatar(user_id: str, data: bytes, content_type: str | None = None) -> str:
    if len(data) > _MAX_BYTES:
        raise ValueError("Avatar exceeds 2 MB limit")
    ext = sniff_image_ext(data)
    if not ext and content_type:
        ext = _MIME_TO_EXT.get(content_type.split(";")[0].strip().lower())
    if not ext:
        raise ValueError("Unsupported image format — use JPEG, PNG, or WebP")

    uid = str(user_id or "").strip()
    mime = _EXT_TO_MIME.get(ext, "application/octet-stream")

    # Durable storage: image bytes live in Postgres so they survive redeploys.
    from db import AvatarStore

    AvatarStore.put(uid, mime, data, time.time())

    # Best-effort cleanup of any stale object-storage / legacy-disk copies.
    try:
        delete_avatar_objects(uid)
    except Exception:
        pass
    delete_legacy_avatar_files(uid)

    return avatar_storage_key(uid, ext)


def delete_avatar_objects(user_id: str, known_key: str | None = None) -> None:
    uid = str(user_id or "").strip()
    if not uid:
        return
    from artifacts import get_artifact_manager

    mgr = get_artifact_manager()
    keys = set()
    if known_key:
        keys.add(known_key.strip())
    prefix = f"{_AVATAR_PREFIX}/{uid}/"
    for obj in mgr.primary.list_objects(prefix):
        obj_key = (obj.get("key") or "").strip()
        if obj_key:
            keys.add(obj_key)
    for key in keys:
        mgr.primary.delete_object(key)


# ── Legacy disk storage (migration / dev fallback) ─────────────────────


def _legacy_avatar_file_for_user(user_id: str) -> Path | None:
    uid = str(user_id or "").strip()
    if not uid:
        return None
    for ext in _EXTENSIONS:
        path = _LEGACY_AVATAR_DIR / f"{uid}.{ext}"
        if path.is_file():
            return path
    return None


def delete_legacy_avatar_files(user_id: str) -> None:
    uid = str(user_id or "").strip()
    if not uid:
        return
    for ext in _EXTENSIONS:
        path = _LEGACY_AVATAR_DIR / f"{uid}.{ext}"
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def legacy_avatar_file_for_user(user_id: str) -> Path | None:
    """Return a legacy on-disk avatar if present (pre–object-storage uploads)."""
    return _legacy_avatar_file_for_user(user_id)


def local_avatar_path_for_key(key: str) -> Path | None:
    """Resolve a storage key to a local filesystem path (LOCAL backend only)."""
    from artifacts import StorageBackend, get_artifact_manager

    mgr = get_artifact_manager()
    if mgr.primary.config.backend != StorageBackend.LOCAL:
        return None
    base = Path(mgr.primary._local_dir())
    path = base / key.replace("/", os.sep)
    return path if path.is_file() else None