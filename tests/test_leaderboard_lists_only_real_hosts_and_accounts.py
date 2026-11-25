"""The reputation leaderboard lists registered hosts, or real accounts.

`reputation_scores.entity_type` defaults to "host", so every account whose
score was ever computed was stored as a host, and test ids written straight
into the table were too. On prod on 2026-11-25 the host board led with
`test-curl` and four other ids that are not hosts, at score 50, above the two
real hosts, and listed nine customer and user ids. Membership now comes from
the hosts and users tables.
"""

from __future__ import annotations

import json
import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest

from db import _get_pg_pool
from reputation import ReputationEngine, ReputationScore

_RUN = uuid.uuid4().hex[:8]
HOST = f"lb-host-{_RUN}"
USER = f"user-lb-{_RUN}"
CUSTOMER = f"cust-lb-{_RUN}"
STRAY = f"test-curl-{_RUN}"


@pytest.fixture
def seeded():
    pool = _get_pg_pool()
    engine = ReputationEngine()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO hosts (host_id, status, registered_at, payload,
                                  admission_state, administrative_state, availability_state)
               VALUES (%s, 'active', %s, %s, 'admitted', 'admitted', 'ready')""",
            (HOST, time.time(), json.dumps({"host_id": HOST, "gpu_model": "RTX 3060"})),
        )
        conn.execute(
            "INSERT INTO users (email, user_id, customer_id, name, password_hash, salt, created_at) "
            "VALUES (%s, %s, %s, '', '', '', 0)",
            (f"lb-{_RUN}@example.test", USER, CUSTOMER),
        )
        conn.commit()
    # Every row stored the way prod stored them: as "host".
    for entity, score in ((HOST, 10), (USER, 20), (CUSTOMER, 30), (STRAY, 999)):
        engine.store.save_score(
            ReputationScore(entity_id=entity, entity_type="host", final_score=score)
        )
    yield engine
    with pool.connection() as conn:
        for entity in (HOST, USER, CUSTOMER, STRAY):
            conn.execute("DELETE FROM reputation_scores WHERE entity_id = %s", (entity,))
        conn.execute("DELETE FROM hosts WHERE host_id = %s", (HOST,))
        conn.execute("DELETE FROM users WHERE user_id = %s", (USER,))
        conn.commit()


def _ids(board: list) -> set[str]:
    return {row["entity_id"] for row in board}


def test_the_host_board_lists_only_registered_hosts(seeded):
    board = _ids(seeded.get_leaderboard("host", 1000))
    assert HOST in board
    assert not board & {USER, CUSTOMER, STRAY}


def test_the_host_board_carries_the_hosts_gpu(seeded):
    row = next(r for r in seeded.get_leaderboard("host", 1000) if r["entity_id"] == HOST)
    assert row["gpu_model"] == "RTX 3060"


def test_the_user_board_lists_only_real_accounts(seeded):
    board = _ids(seeded.get_leaderboard("user", 1000))
    assert {USER, CUSTOMER} <= board
    assert not board & {HOST, STRAY}
