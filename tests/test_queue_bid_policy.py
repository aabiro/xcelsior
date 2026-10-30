"""Exact policy boundaries independent of PostgreSQL's moving clock."""

from hypothesis import given, strategies as st
from control_plane.queue_bids import ordering


def quote(tier="standard", bid=200000, gap=600000):
    return {"tier": tier, "queue_bid_micros": bid, "next_tier_gap_micros": gap}


def test_partial_fifteen_minute_intervals_add_no_credit():
    assert ordering(quote(), "899.999999")["aging_credit_micros"] == 0
    assert ordering(quote(), 900)["aging_credit_micros"] == 20000
    assert ordering(quote(), "1799.999999")["aging_credit_micros"] == 20000
    assert ordering(quote(), 1800)["aging_credit_micros"] == 40000


def test_promotion_at_the_exact_gap_and_one_tier_cap():
    assert ordering(quote(), 19 * 900)["effective_tier"] == 1
    assert ordering(quote(), 20 * 900)["effective_tier"] == 2
    assert ordering(quote(), 200000 * 900)["effective_tier"] == 2
    assert ordering(quote("premium"), 20 * 900)["effective_tier"] == 3
    assert ordering(quote("dedicated", gap=None), 200000 * 900)["effective_tier"] == 3


def test_promotion_time_is_when_bid_plus_credit_qualifies():
    assert ordering(quote(), 20 * 900)["promotion_after_seconds"] == 20 * 900
    assert ordering(quote(bid=600000), 0)["promotion_after_seconds"] == 0


def test_zero_bid_never_creates_aging_credit_or_free_promotion():
    policy = ordering(quote(bid=0), 10**9)
    assert policy["aging_credit_micros"] == 0
    assert policy["promoted"] is False


@given(bid=st.integers(1, 10**10), steps=st.integers(0, 1000000), gap=st.integers(1, 10**10))
def test_linear_integer_credit_and_exact_promotion_threshold(bid, steps, gap):
    frozen = quote(bid=bid, gap=gap)
    policy = ordering(frozen, steps * 900)
    assert policy["aging_credit_micros"] == steps * bid // 10
    assert policy["effective_tier"] in (1, 2)
    assert frozen["queue_bid_micros"] == bid
    if policy["promoted"]:
        threshold = policy["promotion_after_seconds"] // 900
        assert bid + threshold * bid // 10 >= gap
        assert threshold == 0 or bid + (threshold - 1) * bid // 10 < gap
