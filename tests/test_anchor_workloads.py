"""Anchor workload scripts — token metering integration with real repos on disk."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from serverless.metering import token_cost_metadata
from serverless.openai_proxy import accrue_proxy_token_usage


class TestPixelenhanceLabsAnchor:
    """pixelenhance-labs worker patterns call OpenAI-compatible embeddings."""

    def test_embedding_batch_token_cost(self):
        # Simulates a caption/embed pipeline batch (BGE-M3 class model).
        meta = token_cost_metadata(
            12_000,
            0,
            model_ref="BAAI/bge-m3",
            cached_tokens=4_000,
        )
        assert meta["total_token_cost_cad"] > 0
        assert meta["cached_tokens"] == 4_000

    def test_accrual_records_ledger(self):
        repo = MagicMock()
        repo.token_usage_already_recorded.return_value = False
        ep = {"endpoint_id": "ep-pel", "model_ref": "BAAI/bge-m3"}
        usage = {"input_tokens": 12000, "output_tokens": 0, "cached_tokens": 4000}
        result = accrue_proxy_token_usage(
            repo, ep, usage, idempotency_key="pel-caption-batch-1"
        )
        assert result["accrued"] is True
        repo.record_token_usage_idempotency.assert_called_once()


class TestPhantomTradesAnchor:
    """phantom-trades-mvp uses LLM for signal commentary — chat token path."""

    def test_chat_signal_commentary_cost(self):
        meta = token_cost_metadata(
            2_500,
            800,
            model_ref="Qwen/Qwen3-8B",
            cached_tokens=1_200,
        )
        assert meta["input_price_cad_per_m"] == 0.15
        assert meta["total_token_cost_cad"] > 0

    def test_idempotent_retry_same_key(self):
        repo = MagicMock()
        repo.token_usage_already_recorded.side_effect = [False, True]
        ep = {"endpoint_id": f"ep-pt-{uuid.uuid4().hex[:6]}", "model_ref": "Qwen/Qwen3-8B"}
        usage = {"input_tokens": 2500, "output_tokens": 800, "cached_tokens": 0}
        with patch.object(repo, "accrue_endpoint_token_cost") as accrue:
            r1 = accrue_proxy_token_usage(repo, ep, usage, idempotency_key="pt-signal-42")
            r2 = accrue_proxy_token_usage(repo, ep, usage, idempotency_key="pt-signal-42")
        assert r1["accrued"] is True
        assert r2["accrued"] is False
        accrue.assert_called_once()