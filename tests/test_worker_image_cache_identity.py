"""A requested digest must not evict its cached tag before launch."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import worker_agent
import worker_image_cache as cache

TAG = "example/workload:qualified"
DIGEST = "example/workload@sha256:" + "a" * 64
IDENTITY = "sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def isolated_index(monkeypatch):
    monkeypatch.setattr(cache, "_image_cache_index", {})
    monkeypatch.setattr(worker_agent, "IMAGE_CACHE_MAX_GB", 0.001)
    monkeypatch.setattr(worker_agent, "IMAGE_CACHE_EVICT_LOW_GB", 0)


def details():
    return {"id": IDENTITY, "size": 8 * 1024**3, "tags": [TAG], "digests": [DIGEST]}


def test_digest_alias_is_cached_and_counted_once(monkeypatch):
    cache.cache_track_pull(TAG, 8192)
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(details())),
    )
    assert cache.cache_image_available(DIGEST)
    assert cache._image_cache_index[DIGEST]["image_id"] == IDENTITY
    assert cache._total_cache_size_mb() == 8192
    assert cache._image_cache_index[TAG]["image_id"] == IDENTITY


def test_stale_index_does_not_claim_a_missing_image_is_cached(monkeypatch):
    cache.cache_track_pull(DIGEST, 8192)
    monkeypatch.setattr(
        cache.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
    )
    assert not cache.cache_image_available(DIGEST)
    assert DIGEST not in cache._image_cache_index


@pytest.mark.parametrize("running", [True, False])
def test_running_or_requested_digest_protects_tag_alias(monkeypatch, running):
    cache.cache_track_pull(TAG, 8192)
    removed = []

    def run(args, **kwargs):
        if args[1] == "ps":
            return SimpleNamespace(returncode=0, stdout=DIGEST if running else "")
        if args[1:3] == ["image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps(details()))
        removed.append(args)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(cache.subprocess, "run", run)
    assert cache.cache_evict_lru(exclude_images=set() if running else {DIGEST}) == 0
    assert not removed


@pytest.mark.parametrize("failure", ["inventory", "identity"])
def test_unknown_running_inventory_cannot_authorize_eviction(monkeypatch, failure):
    cache.cache_track_pull(TAG, 8192)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "ps":
            return SimpleNamespace(returncode=int(failure == "inventory"), stdout=DIGEST)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(cache.subprocess, "run", run)
    assert cache.cache_evict_lru() == 0
    assert not any(args[1] == "rmi" for args in calls)


def test_worker_startup_does_not_prune_global_docker_images(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda args, **k: (
            calls.append(args) or SimpleNamespace(returncode=0, stdout="<none>:<none> 8GB\n")
        ),
    )
    cache.cache_init()
    assert not cache._image_cache_index
    assert all("prune" not in args for args in calls)


def test_startup_inventory_does_not_authorize_removing_another_projects_image(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=0, stdout="another-project:rollback 8GB\n" if args[1] == "images" else ""
        )

    monkeypatch.setattr(cache.subprocess, "run", run)
    cache.cache_init()
    assert cache.cache_evict_lru() == 0
    assert not any("rmi" in args or "prune" in args for args in calls)
