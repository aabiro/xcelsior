"""The renter benchmark page says what xcelsior-bench does, not what it once did.

The page (api/fern/drafts/benchmark.mdx) repeats the signing key, the CLI flags,
the fault thresholds, the default sustained length and the field names a renter
compares. Each has one source in the code; these tests fail when the two drift.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "api" / "fern" / "drafts" / "benchmark.mdx"
BENCH = ROOT / "workers" / "xcelsior_bench.py"


@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("xcelsior_bench_for_docs", BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pem(text: str) -> str:
    match = re.search(r"-----BEGIN PUBLIC KEY-----\s*(\S+)\s*-----END PUBLIC KEY-----", text)
    assert match, "no public key block"
    return match.group(1)


def test_signing_key_is_the_installers(page: str) -> None:
    installer = (ROOT / "api" / "scripts" / "install.sh").read_text(encoding="utf-8")
    assert _pem(page) == _pem(installer)


def test_every_flag_on_the_page_exists(page: str, bench, monkeypatch) -> None:
    registered: set[str] = set()

    def capture(self, args=None, namespace=None):
        registered.update(opt for action in self._actions for opt in action.option_strings)
        raise SystemExit(0)

    # main() builds its parser inline; stop it at parse time, before any GPU work.
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        bench.main([])
    on_page = set(re.findall(r"`(--[a-z][a-z-]*)", page))
    assert on_page, "the page names no flags"
    assert on_page <= registered, on_page - registered


def test_thresholds_and_default_length(page: str, bench) -> None:
    flat = " ".join(page.split())  # prose wraps at any word
    share = f"{round(bench.THROTTLE_FAULT_SHARE * 100)}%"
    ratio = f"{round(bench.ALLOCATABLE_FAULT_RATIO * 100)}%"
    assert flat.count(f"more than {share} of the sustained phase") == 2
    assert f"under {ratio} of the memory CUDA can address" in flat
    reserve = f"{round(bench.DRIVER_RESERVE_MAX_RATIO * 100)}%"
    assert f"within {reserve} under the listing" in flat
    seconds = bench.DEFAULT_SUSTAINED_SECONDS
    assert f"default ({seconds})" in flat
    assert f"{seconds // 60} minutes by default" in flat
    assert f"A run shorter than {bench.MIN_SUSTAINED_SECONDS} seconds" in flat


def test_compared_fields_are_in_both_results(page: str) -> None:
    from host_evidence import bench_summary

    summary = bench_summary({}, {})
    bench_source = BENCH.read_text(encoding="utf-8")
    snippet = page.split("python3 - <<'EOF'", 1)[1].split("\nEOF", 1)[0]
    loops = re.findall(r"for key in \(([^)]*)\):\n\s+print\(([^\n]*)", snippet)
    assert len(loops) == 2
    for keys, body in loops:
        nested = "['sustained']" in body
        for key in re.findall(r'"([a-z0-9_]+)"', keys):
            assert key in (summary["sustained"] if nested else summary), key
            assert f'"{key}"' in bench_source, key


def test_fault_names_on_the_page_are_the_verdicts(page: str, bench) -> None:
    on_page = set(re.findall(r"\(`([a-z_]+)`\)", page))
    every_fault = bench.verdict(
        {"thermal_throttle_share": 1.0, "power_fault_share": 1.0}, 1.0, 12.0
    )["faults"]
    assert on_page == set(every_fault)
