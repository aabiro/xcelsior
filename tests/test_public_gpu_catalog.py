"""Hardware normalization must work in a checkout with no private billing data."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_gpu_hardware_catalog", ROOT / "api/scripts/export_gpu_hardware_catalog.py"
)
_export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_export)
hardware_catalog = _export.hardware_catalog


def test_hardware_import_needs_no_stripe_manifest(tmp_path):
    for name in ("host_metadata.py", "gpu_catalog.json"):
        shutil.copyfile(ROOT / "hosts" / name, tmp_path / name)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from host_metadata import normalize_gpu_model, default_vram_gb; "
            "assert normalize_gpu_model('NVIDIA GeForce RTX 4090') == 'RTX 4090'; "
            "assert default_vram_gb('A100') == 40",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_hardware_catalog_contains_only_names_and_vram():
    catalog = json.loads((ROOT / "hosts/gpu_catalog.json").read_text())
    assert len(catalog) >= 50
    assert all(isinstance(name, str) and type(vram) is int and vram > 0 for name, vram in catalog.items())


def test_public_hardware_matches_private_snapshot_when_available():
    path = ROOT / "billing/stripe_catalog.json"
    if not path.exists():
        pytest.skip("Private Stripe snapshot is deployment configuration")
    assert hardware_catalog(json.loads(path.read_text())) == json.loads(
        (ROOT / "hosts/gpu_catalog.json").read_text()
    )
