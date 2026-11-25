"""A model cache the platform provisions does not use up the user's quota.

Serverless presets get an NFS weight cache, created in the owner's name
(`serverless/cache.py`). It counted toward the owner's volume quota, so on
prod (100 GB limit) one 50 GB cache plus one 50 GB volume of their own left
the account at "0GB available". The user never asked for the cache.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_NFS_SERVER", "")

import pytest

import volumes as volumes_mod
from db import _get_pg_pool


@pytest.fixture
def engine(monkeypatch):
    e = volumes_mod.VolumeEngine()
    monkeypatch.setattr(e, "_provision_volume_storage", lambda vid, sz, **kw: True)
    monkeypatch.setattr(e, "_destroy_volume_storage", lambda vid, **kw: True)
    monkeypatch.setattr(e, "_emit_event", lambda *a, **kw: None)
    monkeypatch.setattr(volumes_mod, "MAX_TOTAL_STORAGE_GB", 100)
    return e


@pytest.fixture
def owner():
    owner_id = f"cache-quota-{uuid.uuid4().hex[:8]}"
    yield owner_id
    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM volumes WHERE owner_id = %s", (owner_id,))
        conn.commit()


def test_a_platform_cache_leaves_the_users_quota_untouched(engine, owner):
    engine.create_volume(owner, "slvr-cache-Qwen--Qwen3-8B-main-ca-on", 50, platform_cache=True)
    engine.create_volume(owner, "my-volume", 50)
    # 100 GB limit, 50 GB of it the user's own: 50 GB left, not 0.
    assert engine.check_new_volume(owner, "weights", 50) == "weights"
    with pytest.raises(ValueError, match="only 50GB available"):
        engine.check_new_volume(owner, "weights", 51)


def test_a_full_quota_does_not_block_the_platform_cache(engine, owner):
    engine.create_volume(owner, "mine", 100)
    created = engine.create_volume(owner, "slvr-cache-m-main", 50, platform_cache=True)
    assert created["volume_id"]


def test_users_cannot_take_the_reserved_prefix(engine, owner):
    with pytest.raises(ValueError, match="reserved for serverless model caches"):
        engine.check_new_volume(owner, "slvr-cache-sneaky", 500)
