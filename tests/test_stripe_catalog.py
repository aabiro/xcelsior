"""Tests for stripe_catalog invoice line enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stripe_catalog import (
    build_invoice_line_items,
    enrich_invoice_lines_with_catalog,
    resolve_platform_product_id,
)


@pytest.fixture
def manifest() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "stripe_catalog.json"
    if not path.exists():
        pytest.skip("stripe_catalog.json manifest not present")
    return json.loads(path.read_text())


def test_resolve_platform_product_ids(manifest: dict):
    assert resolve_platform_product_id("serverless", manifest)
    assert resolve_platform_product_id("storage", manifest)


def test_enrich_invoice_lines_gpu_serverless_storage(manifest: dict):
    lines = [
        {
            "line_type": "compute",
            "gpu_model": "RTX 4090",
            "trust_tier": "standard",
            "subtotal_cad": 12.5,
        },
        {
            "line_type": "serverless",
            "gpu_model": "RTX 4090",
            "subtotal_cad": 3.25,
        },
        {
            "line_type": "storage",
            "gpu_model": "storage",
            "subtotal_cad": 1.50,
        },
    ]
    enrich_invoice_lines_with_catalog(lines, manifest=manifest)
    assert lines[0].get("stripe_product_id")
    assert lines[1].get("stripe_product_id")
    assert lines[2].get("stripe_product_id")


def test_build_invoice_line_items_maps_all_types(manifest: dict):
    lines = [
        {
            "description": "RTX 4090 — On-Demand",
            "line_type": "compute",
            "gpu_model": "RTX 4090",
            "subtotal_cad": 10.0,
        },
        {
            "description": "Serverless — RTX 4090",
            "line_type": "serverless",
            "gpu_model": "RTX 4090",
            "subtotal_cad": 2.0,
        },
        {
            "description": "Storage — vol-1 (100 GB)",
            "line_type": "storage",
            "subtotal_cad": 0.75,
        },
    ]
    enrich_invoice_lines_with_catalog(lines, manifest=manifest)
    items = build_invoice_line_items(lines, manifest=manifest)
    assert len(items) == 3
    for item in items:
        assert item["quantity"] == 1
        pd = item["price_data"]
        assert pd["currency"] == "cad"
        assert pd["unit_amount"] >= 1
        assert "product" in pd