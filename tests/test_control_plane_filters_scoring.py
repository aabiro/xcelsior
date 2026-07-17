"""Stage C/D — pure filter and scoring determinism tests (§26.1)."""

from hypothesis import given, settings
from hypothesis import strategies as st

from control_plane.scheduler.filters import (
    FILTER_POLICY_VERSION,
    FilterContext,
    aggregate_reason,
    evaluate_host,
    filter_hosts,
)
from control_plane.scheduler.scoring import (
    deterministic_tie_break,
    rank_candidates,
    score_host,
)


def _host(**over):
    base = {
        "host_id": "h1",
        "status": "active",
        "administrative_state": "admitted",
        "gpu_model": "RTX 4090",
        "gpu_count": 2,
        "free_gpu_count": 2,
        "free_vram_gb": 24.0,
        "cost_per_hour": 1.0,
        "region": "ca-on",
        "inventory_generation": 1,
    }
    base.update(over)
    return base


def _job(**over):
    base = {
        "job_id": "j1",
        "gpu_model": "RTX 4090",
        "num_gpus": 1,
        "vram_needed_gb": 8.0,
        "region": "",
    }
    base.update(over)
    return base


class TestHardFilters:
    def test_eligible_host_has_no_reasons(self):
        assert evaluate_host(_job(), _host()) == []

    def test_every_failed_constraint_is_reported(self):
        reasons = evaluate_host(
            _job(gpu_model="H100", num_gpus=4, vram_needed_gb=80.0),
            _host(status="dead", administrative_state="draining"),
        )
        codes = {r.code for r in reasons}
        assert codes == {
            "host_not_admitted",
            "host_not_ready",
            "gpu_model_mismatch",
            "insufficient_gpus",
            "insufficient_vram",
        }

    def test_stale_host_rejected_via_context(self):
        ctx = FilterContext(stale_host_ids=frozenset({"h1"}))
        codes = {r.code for r in evaluate_host(_job(), _host(), ctx)}
        assert codes == {"host_observation_stale"}

    def test_region_and_price_constraints(self):
        codes = {
            r.code
            for r in evaluate_host(
                _job(region="ca-qc", max_price_per_hour=0.5),
                _host(region="ca-on", cost_per_hour=2.0),
            )
        }
        assert codes == {"region_mismatch", "price_exceeds_approved"}

    def test_filter_hosts_splits_and_aggregates(self):
        good = _host(host_id="good")
        bad = _host(host_id="bad", free_vram_gb=1.0)
        eligible, rejections = filter_hosts(_job(), [good, bad])
        assert [h["host_id"] for h in eligible] == ["good"]
        assert set(rejections) == {"bad"}
        agg = aggregate_reason(rejections)
        assert agg["policy_version"] == FILTER_POLICY_VERSION
        assert agg["failed_constraints"] == {"insufficient_vram": 1}


class TestScoring:
    def test_cheaper_host_ranks_first(self):
        cheap = _host(host_id="cheap", cost_per_hour=0.5)
        pricey = _host(host_id="pricey", cost_per_hour=3.0)
        job = _job(max_price_per_hour=4.0)
        ranked = rank_candidates(job, [pricey, cheap])
        assert [s.host_id for s in ranked][0] == "cheap"

    def test_tighter_packing_wins_at_equal_cost(self):
        snug = _host(host_id="snug", free_vram_gb=9.0)
        roomy = _host(host_id="roomy", free_vram_gb=48.0)
        ranked = rank_candidates(_job(vram_needed_gb=8.0), [roomy, snug])
        assert ranked[0].host_id == "snug"

    def test_breakdown_components_sum_to_total(self):
        s = score_host(_job(), _host())
        assert s.total == sum(s.components.values())

    def test_identical_hosts_order_by_tie_break_not_list_order(self):
        a = _host(host_id="host-a")
        b = _host(host_id="host-b")
        ranked_ab = rank_candidates(_job(), [a, b])
        ranked_ba = rank_candidates(_job(), [b, a])
        assert [s.host_id for s in ranked_ab] == [s.host_id for s in ranked_ba]

    def test_tie_break_is_stable_and_job_dependent(self):
        t1 = deterministic_tie_break("job-1", "host-a", 7)
        assert t1 == deterministic_tie_break("job-1", "host-a", 7)
        assert t1 != deterministic_tie_break("job-2", "host-a", 7)
        assert t1 != deterministic_tie_break("job-1", "host-a", 8)


_host_strategy = st.fixed_dictionaries(
    {
        "host_id": st.text(min_size=1, max_size=12),
        "status": st.sampled_from(["active", "dead", "unknown"]),
        "gpu_model": st.sampled_from(["RTX 4090", "H100", "A100"]),
        "free_gpu_count": st.integers(min_value=0, max_value=8),
        "gpu_count": st.integers(min_value=0, max_value=8),
        "free_vram_gb": st.floats(min_value=0, max_value=640, allow_nan=False),
        "cost_per_hour": st.floats(min_value=0, max_value=50, allow_nan=False),
        "region": st.sampled_from(["ca-on", "ca-qc", ""]),
        "inventory_generation": st.integers(min_value=0, max_value=10),
        "reliability_score": st.floats(min_value=0, max_value=1, allow_nan=False),
    }
)


class TestDeterminismProperties:
    @settings(max_examples=50, deadline=None)
    @given(hosts=st.lists(_host_strategy, min_size=1, max_size=8))
    def test_ranking_is_permutation_invariant(self, hosts):
        """Replica A and replica B must rank any snapshot identically,
        regardless of the order hosts arrived from their queries."""
        job = _job(vram_needed_gb=4.0)
        forward = [s.host_id for s in rank_candidates(job, hosts)]
        backward = [s.host_id for s in rank_candidates(job, list(reversed(hosts)))]
        assert forward == backward

    @settings(max_examples=50, deadline=None)
    @given(host=_host_strategy)
    def test_filters_are_pure(self, host):
        job = _job()
        first = [r.code for r in evaluate_host(job, host)]
        second = [r.code for r in evaluate_host(job, host)]
        assert first == second
