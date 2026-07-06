"""Speculative decoding acceptance-rate gate tests."""

from __future__ import annotations

import pytest

from serverless.service import _preset_startup_command
from serverless.speculative_gate import (
    MIN_ACCEPTANCE_RATE,
    MIN_SAMPLES,
    record_speculative_sample,
    reset_validation_store,
    speculative_startup_flags,
    validation_status,
)


@pytest.fixture(autouse=True)
def isolated_validation_store(tmp_path, monkeypatch):
    store = tmp_path / "eagle3.json"
    monkeypatch.setenv("XCELSIOR_EAGLE3_VALIDATION_PATH", str(store))
    monkeypatch.delenv("XCELSIOR_VLLM_EAGLE3_FORCE", raising=False)
    reset_validation_store()
    yield
    reset_validation_store()


class TestEagle3StartupGate:
    def test_default_startup_omits_eagle3(self, monkeypatch):
        monkeypatch.delenv("XCELSIOR_VLLM_EAGLE3", raising=False)
        cmd = _preset_startup_command("vllm", "Qwen/Qwen3-8B")
        assert "EAGLE3" not in cmd

    def test_eagle3_requires_validation_even_when_env_on(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_VLLM_EAGLE3", "1")
        cmd = _preset_startup_command("vllm", "Qwen/Qwen3-8B")
        assert "EAGLE3" not in cmd

    def test_eagle3_ships_by_default_after_validation(self, monkeypatch):
        """Compatible bases enable EAGLE-3 after validation without explicit env=1."""
        monkeypatch.delenv("XCELSIOR_VLLM_EAGLE3", raising=False)
        model = "Qwen/Qwen3-8B"
        for _ in range(MIN_SAMPLES):
            record_speculative_sample(
                model,
                acceptance_rate=0.82,
                tokens_per_sec=6400.0,
                baseline_tokens_per_sec=5000.0,
            )
        cmd = _preset_startup_command("vllm", model)
        assert "EAGLE3" in cmd
        assert "--speculative-model" in cmd

    def test_eagle3_opt_out_env_disables_after_validation(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_VLLM_EAGLE3", "0")
        model = "Qwen/Qwen3-8B"
        for _ in range(MIN_SAMPLES):
            record_speculative_sample(
                model,
                acceptance_rate=0.82,
                tokens_per_sec=6400.0,
                baseline_tokens_per_sec=5000.0,
            )
        assert speculative_startup_flags(model) == []
        assert "EAGLE3" not in _preset_startup_command("vllm", model)

    def test_eagle3_appended_after_validation_passes(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_VLLM_EAGLE3", "1")
        model = "Qwen/Qwen3-8B"
        for _ in range(MIN_SAMPLES):
            record_speculative_sample(
                model,
                acceptance_rate=0.80,
                tokens_per_sec=6200.0,
                baseline_tokens_per_sec=5000.0,
            )
        flags = speculative_startup_flags(model)
        assert "--speculative-algorithm" in flags
        cmd = _preset_startup_command("vllm", model)
        assert "EAGLE3" in cmd

    def test_low_acceptance_blocks_eagle3(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_VLLM_EAGLE3", "1")
        model = "Qwen/Qwen3-8B"
        for _ in range(MIN_SAMPLES):
            record_speculative_sample(
                model,
                acceptance_rate=0.55,
                tokens_per_sec=7000.0,
                baseline_tokens_per_sec=4000.0,
            )
        status = validation_status(model)
        assert status["mean_acceptance_rate"] < MIN_ACCEPTANCE_RATE
        assert status["validated"] is False
        assert speculative_startup_flags(model) == []


class TestEagle3ValidationMath:
    def test_throughput_must_beat_baseline(self):
        model = "Qwen/Qwen3-8B"
        for _ in range(MIN_SAMPLES):
            record_speculative_sample(
                model,
                acceptance_rate=0.90,
                tokens_per_sec=4000.0,
                baseline_tokens_per_sec=5000.0,
            )
        assert validation_status(model)["validated"] is False