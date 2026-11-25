"""A preset endpoint is refused on a GPU that cannot hold the model.

On prod, all three Qwen3-8B endpoints were in `error`. Two were on 12 GB RTX
3060s: the preset serves Qwen3-8B unquantized (about 16 GB of bf16 weights),
but `estimate_vram_gb` said 6 GB for any "8b" model, so creation and placement
both let it through and every worker boot failed. The estimate now follows the
weights' size and precision, and a GPU from the catalog that is too small is
refused before a plan is made.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest

import serverless.service as service_mod
from serverless.repo import EndpointCreate
from serverless.service import ServerlessService

estimate = ServerlessService.estimate_vram_gb


@pytest.mark.parametrize(
    ("model", "low", "high"),
    [
        ("Qwen/Qwen3-8B", 16, 22),  # bf16 weights alone are ~16 GB
        ("meta-llama/Llama-3.1-70B-Instruct", 140, 180),
        ("Qwen/Qwen2.5-1.5B-Instruct", 3, 5),
        ("Qwen/Qwen3-8B-AWQ", 4, 6),
        ("neuralmagic/Llama-3.1-8B-FP8", 8, 11),
    ],
)
def test_the_estimate_follows_the_weights(model, low, high):
    assert low <= estimate(model) <= high


def test_a_4bit_tag_is_not_read_as_a_4b_model():
    assert estimate("TheBloke/Llama-2-13B-chat-4bit") < estimate("meta-llama/Llama-2-13b-chat-hf")


def _spec(gpu_tier: str, gpu_count: int = 1, model: str = "Qwen/Qwen3-8B") -> EndpointCreate:
    return EndpointCreate(
        owner_id="cust-1",
        name="qwen",
        mode="preset",
        model_ref=model,
        gpu_tier=gpu_tier,
        gpu_count=gpu_count,
    )


@pytest.fixture
def catalog(monkeypatch):
    vram = {"RTX 3060": 12.0, "RTX 4090": 24.0, "A40": 48.0}
    monkeypatch.setattr(service_mod, "_catalog_vram_gb", lambda gpu: vram.get(gpu))


def test_qwen3_8b_is_refused_on_an_rtx_3060(catalog):
    with pytest.raises(ValueError, match=r"needs about .* GB .* RTX 3060 has 12 GB"):
        ServerlessService.check_gpu_fits(_spec("RTX 3060"))


def test_qwen3_8b_fits_a_24_gb_gpu(catalog):
    ServerlessService.check_gpu_fits(_spec("RTX 4090"))
    ServerlessService.check_gpu_fits(_spec("A40"))


def test_two_small_gpus_can_hold_it_together(catalog):
    ServerlessService.check_gpu_fits(_spec("RTX 3060", gpu_count=2))


def test_a_quantized_model_fits_the_small_gpu(catalog):
    ServerlessService.check_gpu_fits(_spec("RTX 3060", model="Qwen/Qwen3-8B-AWQ"))


def test_a_gpu_missing_from_the_catalog_is_left_to_placement(catalog):
    ServerlessService.check_gpu_fits(_spec("Some Future GPU"))


def test_the_catalog_lookup_reads_gpu_pricing():
    """The real lookup answers from the table the price list comes from."""
    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        row = conn.execute(
            "SELECT gpu_model, MAX(vram_gb) FROM gpu_pricing GROUP BY 1 LIMIT 1"
        ).fetchone()
    if not row:
        pytest.skip("gpu_pricing is empty in this database")
    gpu, vram = list(row.values()) if isinstance(row, dict) else row
    assert service_mod._catalog_vram_gb(gpu) == float(vram)
