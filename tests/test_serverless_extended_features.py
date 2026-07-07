"""Extended §10 features: semantic cache, batch API, prefix routing, hedge, prewarm, SLO, forecast."""

from __future__ import annotations

import os
import time
import uuid

import pytest

os.environ["XCELSIOR_ENV"] = "test"
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.autoscaler import AutoscalerInput, compute_desired_workers
from serverless.batch_api import apply_batch_token_discount, create_batch
from serverless.forecast_fallback import forecast_queue_depth_with_fallback
from serverless.hedged_requests import dispatch_hedge, hedge_p95_ms, should_hedge_job
from serverless.metering import pricing_for_endpoint
from serverless.prefix_routing import (
    prefix_hash_from_payload,
    rank_workers_for_prefix,
)
from serverless.prewarm_pool import apply_prewarm_to_desired
from serverless.repo import EndpointCreate, JOB_STATUS_IN_PROGRESS, ServerlessRepo
from serverless.semantic_cache import (
    extract_prompt_text,
    similarity,
    store_cache_entry,
    try_cache_hit,
)
from serverless.slo import enrich_pricing_with_ga, enrich_usage_with_slo, token_endpoint_is_ga


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repo():
    return ServerlessRepo()


@pytest.fixture
def preset_ep(repo: ServerlessRepo, owner_id: str):
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner_id,
            name="ext-feat",
            mode="preset",
            model_ref="Qwen/Qwen3-8B",
            managed_engine="vllm",
            gpu_tier="RTX 4090",
            region="ca-east",
            min_workers=0,
        )
    )
    yield ep
    repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)


class TestTokenGaAndSlo:
    def test_preset_endpoint_ga_pricing(self, preset_ep: dict):
        assert token_endpoint_is_ga(preset_ep)
        quote = enrich_pricing_with_ga(preset_ep, pricing_for_endpoint(preset_ep))
        assert quote.get("ga_status") == "ga"
        assert quote.get("slo_targets", {}).get("kv_cache_hit_rate_target") == pytest.approx(0.30)

    def test_usage_slo_enrichment(self):
        enriched = enrich_usage_with_slo(
            {"kv_cache_hit_rate": 0.35, "ttft_p95_ms": 2000, "tokens_per_sec": 80}
        )
        assert enriched["slo_status"]["kv_cache_hit_met"] is True


class TestSemanticCache:
    def test_similarity_threshold(self):
        a = extract_prompt_text({"messages": [{"role": "user", "content": "analyze trade signal"}]})
        b = extract_prompt_text({"messages": [{"role": "user", "content": "analyze trade signal"}]})
        assert similarity(a, b) == 1.0

    def test_cache_store_and_hit(self, repo: ServerlessRepo, preset_ep: dict):
        ep_id = preset_ep["endpoint_id"]
        body = {"messages": [{"role": "user", "content": "hello cache"}]}
        response = {"choices": [{"message": {"content": "hi"}}]}
        usage = {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}
        store_cache_entry(repo, ep_id, body, response, usage)
        hit = try_cache_hit(repo, ep_id, body)
        assert hit is not None
        assert hit["cache_hit"] is True


class TestBatchApi:
    def test_batch_discount(self):
        assert apply_batch_token_discount(1.0, discount_rate=0.5) == 0.5

    def test_create_and_enqueue_batch(self, repo: ServerlessRepo, preset_ep: dict, owner_id: str):
        batch = create_batch(
            repo,
            endpoint_id=preset_ep["endpoint_id"],
            owner_id=owner_id,
            requests=[{"custom_id": "1", "body": {"input": "a"}}],
        )
        assert batch["status"] == "validating"
        from serverless.batch_api import enqueue_batch_requests

        n = enqueue_batch_requests(repo, batch, preset_ep)
        assert n == 1
        got = repo.get_batch_job(batch["batch_id"], owner_id=owner_id)
        assert got is not None
        assert got["status"] == "in_progress"


class TestPrefixRouting:
    def test_prefix_hash_stable(self):
        p = {"messages": [{"role": "user", "content": "prefix test"}]}
        assert prefix_hash_from_payload(p) == prefix_hash_from_payload(p)

    def test_rank_prefers_affinity_worker(self):
        workers = [
            {"worker_id": "w1", "state": "ready"},
            {"worker_id": "w2", "state": "ready"},
        ]
        ranked = rank_workers_for_prefix(workers, "abc123", affinity={"abc123": "w2"})
        assert ranked[0]["worker_id"] == "w2"

    def test_record_affinity(self, repo: ServerlessRepo, preset_ep: dict):
        ep_id = preset_ep["endpoint_id"]
        repo.record_prefix_affinity(ep_id, "hash1", "w-affinity")
        aff = repo.get_prefix_affinities(ep_id)
        assert aff.get("hash1") == "w-affinity"


class TestHedgedRequests:
    def test_should_hedge_after_p95(self):
        job = {
            "status": JOB_STATUS_IN_PROGRESS,
            "started_at": time.time() - 10,
            "worker_id": "w1",
        }
        assert should_hedge_job(job, p95_ms=5000) is True

    def test_dispatch_hedge_assigns_secondary(
        self, repo: ServerlessRepo, preset_ep: dict, owner_id: str
    ):
        from serverless.dispatcher import ServerlessDispatcher

        ep_id = preset_ep["endpoint_id"]
        w1 = repo.create_worker(ep_id, scheduler_job_id="s1")
        w2 = repo.create_worker(ep_id, scheduler_job_id="s2")
        repo.update_worker(str(w1["worker_id"]), state="ready")
        repo.update_worker(str(w2["worker_id"]), state="ready")
        job = repo.enqueue_job(ep_id, owner_id, {"x": 1})
        repo.claim_next_job(ep_id, str(w1["worker_id"]))
        stale = time.time() - 10
        with repo._conn() as conn:
            conn.execute(
                "UPDATE serverless_jobs SET started_at = %s WHERE job_id = %s",
                (stale, job["job_id"]),
            )
        job = repo.get_job(job["job_id"], endpoint_id=ep_id)
        dispatcher = ServerlessDispatcher(repo)
        result = dispatch_hedge(
            repo, dispatcher, preset_ep, job, p95_ms=hedge_p95_ms(preset_ep, {})
        )
        assert result is not None
        assert result["hedge_worker_id"] == str(w2["worker_id"])


class TestPrewarmAndForecast:
    def test_prewarm_raises_desired_workers(self):
        desired = apply_prewarm_to_desired(
            0,
            min_workers=0,
            forecast_depth=8,
            max_concurrency=2,
            endpoint={"min_workers": 0},
            active_workers=0,
        )
        assert desired >= 1

    def test_toto_fallback_on_high_error(self):
        now = time.time()
        samples = [(now - 60 * i, (i % 3) * 5) for i in range(10)]
        fc = forecast_queue_depth_with_fallback(samples, underfit_threshold=0.01)
        assert fc["forecast_depth"] >= 0
        assert fc["model"] in ("chronos-ewma", "toto-2.0-sma-fallback")

    def test_autoscaler_uses_forecast(self):
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=2,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=3,
            max_queue_wait_sec=0,
            workers=[],
            queue_depth_samples=[(time.time() - 30, 1), (time.time(), 5)],
        )
        desired = compute_desired_workers(inp)
        assert desired >= 1