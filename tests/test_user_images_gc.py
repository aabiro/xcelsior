"""P3/C2 — hard-delete GC for soft-deleted user_images."""
from pathlib import Path


def test_c2_gc_task_registered():
    """Task must be registered in the bg_worker tasks list."""
    src = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text()
    assert '"user_images_hard_delete_gc"' in src
    assert "_user_images_hard_delete_gc" in src


def test_c2_gc_uses_retention_env_var():
    src = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text()
    # Must be configurable, default 30 days.
    assert 'XCELSIOR_USER_IMAGES_GC_DAYS' in src
    assert '"30"' in src or "'30'" in src


def test_c2_gc_only_purges_already_soft_deleted_rows():
    """Source grep: WHERE clause must require deleted_at > 0."""
    src = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text()
    idx = src.find("def _user_images_hard_delete_gc")
    assert idx >= 0
    end = src.find("\n    tasks.append", idx)
    body = src[idx:end]
    assert "deleted_at > 0" in body, "GC must only touch rows already soft-deleted"
    assert "deleted_at < %s" in body, "GC must enforce retention window"
    assert "DELETE FROM user_images" in body


def test_c2_gc_runs_daily():
    src = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text()
    # 86400 seconds = daily cadence.
    assert '"user_images_hard_delete_gc", _user_images_hard_delete_gc, 86400' in src


def test_c2_gc_disabled_when_retention_zero():
    """Setting XCELSIOR_USER_IMAGES_GC_DAYS=0 must short-circuit."""
    src = (Path(__file__).resolve().parent.parent / "bg_worker.py").read_text()
    idx = src.find("def _user_images_hard_delete_gc")
    end = src.find("\n    tasks.append", idx)
    body = src[idx:end]
    assert "retention_days <= 0" in body
    assert "return" in body
