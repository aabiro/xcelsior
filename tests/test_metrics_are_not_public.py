"""Operational metrics must not be reachable from the public edge.

`GET /metrics` returns `get_metrics_snapshot()` — queue depth, job, host,
volume and notification counts. `GET /metrics/prometheus` exports the same
figures. Neither endpoint authenticates, by design: they are scraped over the
host-private API ports.

The only thing standing between that and the internet is a pair of `return
404` blocks in `nginx/xcelsior.conf`, added deliberately and documented there:

    # Metrics are scraped over the host-private API ports. Never expose either
    # the JSON snapshot or Prometheus exposition through the public edge.

Nothing pinned them. A reorganisation of that file, or a `location` block added
above them, publishes them with no second line of defence and no error
anywhere — the endpoints would simply start answering.

This asserts the edge, not the handler. Adding auth to `/metrics` instead would
break the scraper, which is the reason the protection lives in nginx.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONF = Path(__file__).resolve().parents[1] / "api" / "nginx" / "xcelsior.conf"


@pytest.fixture(scope="module")
def conf() -> str:
    assert CONF.exists(), f"{CONF} is missing; the edge config is what enforces this"
    return CONF.read_text(encoding="utf-8")


def _block_for(conf: str, location: str) -> str:
    """The body of a `location <location> {` block, brace-matched."""
    start = conf.find(location)
    assert start != -1, f"no `{location}` block in the nginx config"
    open_brace = conf.index("{", start)
    depth, i = 0, open_brace
    while i < len(conf):
        if conf[i] == "{":
            depth += 1
        elif conf[i] == "}":
            depth -= 1
            if depth == 0:
                return conf[open_brace : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces after `{location}`")


def test_the_json_snapshot_is_refused_at_the_edge(conf) -> None:
    body = _block_for(conf, "location = /metrics")
    assert re.search(r"\breturn\s+404\b", body), (
        "the exact-match /metrics block no longer returns 404 — "
        f"the metrics snapshot is now public:\n{body}"
    )


def test_the_prometheus_exposition_is_refused_at_the_edge(conf) -> None:
    body = _block_for(conf, "location ^~ /metrics/")
    assert re.search(r"\breturn\s+404\b", body), (
        "the /metrics/ prefix block no longer returns 404 — "
        f"the Prometheus exposition is now public:\n{body}"
    )


def test_neither_block_proxies_to_the_api(conf) -> None:
    """A `proxy_pass` added beside the 404 would take precedence over it."""
    for location in ("location = /metrics", "location ^~ /metrics/"):
        body = _block_for(conf, location)
        assert "proxy_pass" not in body, (
            f"`{location}` proxies to the API as well as returning 404; "
            f"whichever nginx applies, this is no longer a refusal:\n{body}"
        )

