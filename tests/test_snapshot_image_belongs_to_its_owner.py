"""A private snapshot in the platform registry can be launched only by its owner.

CVP-01 T-22 residual / test T13b. Workers pull from the platform registry with
one shared login, and a snapshot's namespace is derived from its owner's id, so
the API is the only thing standing between tenant B and tenant A's filesystem.
"""

import inspect

import pytest

import snapshot_image_access as access

REG = "registry.example.test/xcelsior"
A_NS = f"{REG}/alice-0123456789abcdef"
A_PRIVATE = f"{A_NS}/model:latest"
A_PUBLIC = f"{A_NS}/shared:v1"


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    monkeypatch.setenv("XCELSIOR_REGISTRY_URL", f"https://{REG}/")


def _lookup(namespace):
    assert namespace == f"{A_NS}/"
    return [
        ("alice@example.test", False, A_PRIVATE, "ready"),
        ("alice@example.test", True, A_PUBLIC, "ready"),
        ("alice@example.test", True, f"{A_NS}/building:v1", "pending"),
    ]


def _problem(image, *principals, lookup=_lookup):
    return access.foreign_snapshot_problem(image, principals, lookup=lookup)


def test_another_tenant_cannot_launch_a_private_snapshot():
    assert _problem(A_PRIVATE, "bob@example.test", "cus_bob") == access.REFUSAL


def test_the_owner_can():
    assert _problem(A_PRIVATE, "cus_team", "alice@example.test") is None


def test_a_public_ready_snapshot_is_open_to_everyone():
    assert _problem(A_PUBLIC, "bob@example.test") is None


def test_public_but_not_ready_is_refused():
    assert _problem(f"{A_NS}/building:v1", "bob@example.test") == access.REFUSAL


@pytest.mark.parametrize(
    "ref",
    [
        f"{A_NS}/model:other-tag",
        f"{A_NS}/model@sha256:" + "0" * 64,
        f"{A_NS}/never-pushed:latest",
        A_PRIVATE.replace("registry.example.test", "REGISTRY.example.test"),
        f"https://{A_PRIVATE}",
    ],
)
def test_no_other_spelling_inside_the_namespace_gets_through(ref):
    assert _problem(ref, "bob@example.test") == access.REFUSAL


def test_the_refusal_does_not_confirm_the_image_exists():
    assert "not found" in access.REFUSAL.lower()


def test_images_outside_the_registry_are_not_touched():
    def boom(_ns):
        raise AssertionError("no lookup for a public image")

    for ref in ("nvcr.io/nvidia/pytorch:24.12-py3", "pytorch/pytorch:2.4.0", "", None):
        assert _problem(ref, "bob", lookup=boom) is None
    assert _problem(f"{REG}/serverless-base:1", "bob", lookup=boom) is None


def test_a_namespace_with_no_snapshots_is_the_platforms_own():
    assert _problem(f"{REG}/platform/base:1", "bob", lookup=lambda ns: []) is None


def test_without_a_registry_nothing_is_a_snapshot(monkeypatch):
    monkeypatch.delenv("XCELSIOR_REGISTRY_URL")
    assert access.snapshot_namespace(A_PRIVATE) is None


def test_like_wildcards_in_a_reference_are_escaped(monkeypatch):
    """`%` or `_` in a typed namespace must not widen the lookup to other tenants."""
    seen = {}

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            seen["pattern"] = params[0]
            return self

        def fetchall(self):
            return []

    class _Pool:
        def connection(self):
            return _Conn()

    import db

    monkeypatch.setattr(db, "_get_pg_pool", lambda: _Pool())
    access.foreign_snapshot_problem(f"{REG}/a%_b/x:1", ["bob"])
    assert seen["pattern"] == f"{REG}/a\\%\\_b/%"


def test_a_failed_lookup_stops_the_launch():
    def down(_ns):
        raise RuntimeError("database unavailable")

    with pytest.raises(RuntimeError):
        _problem(A_PRIVATE, "bob", lookup=down)


def test_every_job_passes_through_the_check():
    """submit_job is the one job-creation authority; the check lives there."""
    import scheduler

    body = inspect.getsource(scheduler.submit_job)
    assert "snapshot_launch_decision(" in body
    assert body.index("snapshot_launch_decision(") < body.index("import canonicalize")
