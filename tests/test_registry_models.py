"""pxl-registry preset model resolution (local paths only)."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from serverless.registry_models import (
    canonical_model_dir,
    find_local_model,
    hf_to_staging_dirname,
    model_usable,
    resolve_preset_model,
)


class TestRegistryModelResolve:
    def test_qwen3_8b_resolves_local_when_on_disk(self):
        resolved = resolve_preset_model("Qwen/Qwen3-8B")
        assert resolved.hf_ref == "Qwen/Qwen3-8B"
        assert resolved.registry_id == "store_llm_qwen_qwen3_8b"
        if find_local_model("Qwen/Qwen3-8B"):
            assert resolved.source == "local"
            assert os.path.isdir(resolved.local_path or "")
            assert model_usable(Path(resolved.local_path))
        else:
            assert resolved.source == "hf"
            assert resolved.launch_ref == "Qwen/Qwen3-8B"

    def test_unknown_model_passthrough(self):
        resolved = resolve_preset_model("acme/unknown-7b")
        assert resolved.launch_ref == "acme/unknown-7b"
        assert resolved.source == "hf"

    def test_canonical_staging_dirname(self):
        assert hf_to_staging_dirname("Qwen/Qwen3-8B") == "Qwen_Qwen3-8B"
        assert hf_to_staging_dirname("BAAI/bge-m3") == "BAAI_bge-m3"

    def test_find_local_via_canonical_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XCELSIOR_MODEL_STORE_ROOTS", str(tmp_path))
        target = canonical_model_dir("test/foo-7b")
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")
        (target / "model.safetensors").write_bytes(b"x" * 100)
        found = find_local_model("test/foo-7b")
        assert found == target