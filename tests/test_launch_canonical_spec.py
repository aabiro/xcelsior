"""Track B B2.2 — canonical spec, hash stability, and the worker drift guard.

The canonical spec hash is a *binding*: the scheduler stamps it on the
attempt and the worker recomputes it before starting a container. Two
properties must hold or the binding is meaningless:

1. **Stability.** The same logical request hashes identically regardless of
   input key order, surrounding whitespace, or whether an optional field was
   omitted or passed at its default.
2. **No drift.** The launch service (control-plane) and the worker compute
   the *same* hash for the same spec. They are deliberately independent
   implementations — the worker does not trust the control plane — so a
   guard must pin them together; otherwise one could change and silently
   start rejecting every launch.
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from control_plane.launch import canonicalize, spec_hash
from control_plane.launch.canonicalize import CANON_SPEC_VERSION
from control_plane.launch.quoting import price_moved_beyond_tolerance
from control_plane.launch.spend_policy import evaluate
from control_plane.launch.validation import validate_canonical_spec
from control_plane.scheduler.reservation import canonical_spec_hash


def _base_request() -> dict:
    return {
        "name": "train-run",
        "vram_needed_gb": 40,
        "num_gpus": 2,
        "gpu_model": "H100",
        "region": "ca-on",
        "image": "pytorch/pytorch:latest",
        "interactive": True,
        "command": "python train.py",
        "ssh_port": 2222,
        "pricing_mode": "on_demand",
        "volume_ids": ["vol-b", "vol-a"],
        "exposed_ports": [8080, 3000],
        "auto_launch": ["jupyter"],
    }


# ── Stability ────────────────────────────────────────────────────────


class TestCanonicalStability:
    def test_key_order_does_not_matter(self):
        req = _base_request()
        reordered = {k: req[k] for k in reversed(list(req.keys()))}
        assert canonicalize(req) == canonicalize(reordered)
        assert spec_hash(canonicalize(req)) == spec_hash(canonicalize(reordered))

    def test_surrounding_whitespace_is_stripped(self):
        req = _base_request()
        padded = dict(req)
        padded["name"] = "  train-run  "
        padded["gpu_model"] = " H100 "
        padded["command"] = "python train.py"
        assert canonicalize(req) == canonicalize(padded)

    def test_omitted_optional_equals_explicit_default(self):
        req = _base_request()
        # num_gpus default is 1; ssh_port default 22; interactive default True.
        minimal = {"name": "train-run"}
        explicit = {
            "name": "train-run",
            "num_gpus": 1,
            "ssh_port": 22,
            "interactive": True,
            "pricing_mode": "on_demand",
            "vram_needed_gb": 0,
        }
        assert canonicalize(minimal) == canonicalize(explicit)
        assert spec_hash(canonicalize(minimal)) == spec_hash(canonicalize(explicit))

    def test_port_and_volume_order_insensitive(self):
        a = dict(_base_request(), exposed_ports=[3000, 8080], volume_ids=["vol-a", "vol-b"])
        b = dict(_base_request(), exposed_ports=[8080, 3000], volume_ids=["vol-b", "vol-a"])
        assert spec_hash(canonicalize(a)) == spec_hash(canonicalize(b))

    def test_canonicalize_is_idempotent(self):
        spec = canonicalize(_base_request())
        assert canonicalize(spec) == spec

    def test_auto_launch_injects_its_port(self):
        spec = canonicalize(dict(_base_request(), auto_launch=["jupyter"], exposed_ports=[]))
        assert 8888 in spec["exposed_ports"]

    def test_spec_version_present(self):
        assert canonicalize(_base_request())["spec_version"] == CANON_SPEC_VERSION

    @settings(max_examples=200, deadline=None)
    @given(
        name=st.text(min_size=1, max_size=32),
        num_gpus=st.integers(min_value=1, max_value=64),
        vram=st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False),
        ports=st.lists(st.integers(min_value=1, max_value=65535), max_size=6),
        vols=st.lists(st.text(min_size=1, max_size=8), max_size=6),
        pad=st.sampled_from(["", " ", "  ", "\t"]),
    )
    def test_property_reorder_and_pad_are_stable(self, name, num_gpus, vram, ports, vols, pad):
        req = {
            "name": name,
            "num_gpus": num_gpus,
            "vram_needed_gb": vram,
            "exposed_ports": list(ports),
            "volume_ids": list(vols),
            "gpu_model": "A100",
        }
        padded = {
            "gpu_model": f"{pad}A100{pad}",
            "volume_ids": list(reversed(vols)),
            "exposed_ports": list(reversed(ports)),
            "vram_needed_gb": vram,
            "num_gpus": num_gpus,
            "name": f"{pad}{name}{pad}" if name.strip() == name else name,
        }
        # When padding would change the logical name (name has interior/edge
        # spaces already), skip the name equivalence for that case.
        h1 = spec_hash(canonicalize(req))
        h2 = spec_hash(canonicalize(padded))
        if name.strip() == name:
            assert h1 == h2
        # Regardless, the hash is well-formed and deterministic on re-run.
        assert h1 == spec_hash(canonicalize(req))
        assert h1.startswith("sha256:")


# ── Drift guard ──────────────────────────────────────────────────────


class TestNoHashDrift:
    def test_launch_matches_scheduler(self):
        spec = canonicalize(_base_request())
        assert spec_hash(spec) == canonical_spec_hash(spec)

    def test_launch_matches_worker(self):
        # The worker keeps an independent copy of the hash; it must agree.
        from worker_agent import compute_spec_hash

        spec = canonicalize(_base_request())
        assert spec_hash(spec) == compute_spec_hash(spec)

    @settings(max_examples=100, deadline=None)
    @given(
        gpu=st.sampled_from(["H100", "A100", "RTX 4090", ""]),
        n=st.integers(min_value=1, max_value=8),
        vram=st.integers(min_value=0, max_value=320),
    )
    def test_property_all_three_agree(self, gpu, n, vram):
        from worker_agent import compute_spec_hash

        spec = canonicalize({"name": "x", "gpu_model": gpu, "num_gpus": n, "vram_needed_gb": vram})
        h = spec_hash(spec)
        assert h == canonical_spec_hash(spec) == compute_spec_hash(spec)


# ── Validation ───────────────────────────────────────────────────────


class TestValidation:
    def test_clean_spec_has_no_problems(self):
        assert validate_canonical_spec(canonicalize(_base_request())) == []

    def test_missing_name_flagged(self):
        problems = validate_canonical_spec(canonicalize({"name": ""}))
        assert any(p.code == "name_required" for p in problems)

    def test_reserved_port_flagged(self):
        spec = canonicalize(_base_request())
        spec["exposed_ports"] = [22]
        assert any(p.code == "port_reserved" for p in validate_canonical_spec(spec))

    def test_bad_git_repo_flagged(self):
        spec = canonicalize(dict(_base_request(), git_repo="git@github.com:x/y.git"))
        assert any(p.code == "git_repo_invalid" for p in validate_canonical_spec(spec))


# ── Quoting tolerance ────────────────────────────────────────────────


class TestQuoteTolerance:
    def test_price_drop_never_violates(self):
        assert price_moved_beyond_tolerance(1_000_000, 500_000, 0) is False

    def test_within_tolerance_ok(self):
        # 4% increase, 5% tolerance -> allowed.
        assert price_moved_beyond_tolerance(1_000_000, 1_040_000, 500) is False

    def test_beyond_tolerance_blocks(self):
        # 6% increase, 5% tolerance -> blocked.
        assert price_moved_beyond_tolerance(1_000_000, 1_060_000, 500) is True

    def test_from_zero_any_increase_blocks(self):
        assert price_moved_beyond_tolerance(0, 1, 10_000) is True


# ── Spend policy ─────────────────────────────────────────────────────


class TestSpendPolicy:
    def test_no_policy_requires_human(self):
        d = evaluate(canonicalize(_base_request()), estimate_micros=1_000, policy=None)
        assert d.allowed and d.approval_mode == "human"

    def test_within_policy_auto_approves(self):
        policy = {
            "per_action_max_micros": 10_000_000,
            "allowed_gpu_models": ["H100"],
            "auto_approve": True,
        }
        d = evaluate(canonicalize(_base_request()), estimate_micros=1_000_000, policy=policy)
        assert d.allowed and d.approval_mode == "standing_policy"

    def test_over_ceiling_blocks_auto_approve(self):
        policy = {"per_action_max_micros": 500_000, "auto_approve": True}
        d = evaluate(canonicalize(_base_request()), estimate_micros=1_000_000, policy=policy)
        assert not d.allowed and d.approval_mode == "human"

    def test_disallowed_gpu_blocks(self):
        policy = {"allowed_gpu_models": ["A100"], "auto_approve": True}
        d = evaluate(canonicalize(_base_request()), estimate_micros=1_000, policy=policy)
        assert not d.allowed
        assert any("gpu_model" in r for r in d.reasons)
