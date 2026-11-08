"""The batched Slurm reconciler against real PostgreSQL and a scripted cluster.

The rules under test (workers/slurm_reconciler.py, plan §5a):

- one squeue per pass however many mappings are open, and sacct only for
  jobs that have left the controller, always bounded;
- a mapping closes only when its job has left the controller and accounting
  says it finished: a finished state in squeue is not final, because a
  requeued job comes back under the same id;
- silence is never a verdict: a job in neither source keeps its observation;
- writes happen only on change, guarded by version;
- a research-only cluster can never be marketplace or agent-reachable.
"""

from __future__ import annotations

import subprocess
import uuid

import psycopg
import pytest

import slurm_adapter as sa
import slurm_reconciler as rc
from db import pg_connection, pg_transaction


class FakeCluster:
    """A runner that answers squeue/sacct/scancel from scripted state."""

    remote = True
    ssh_host = "fake"

    def __init__(self, squeue=(), sacct=(), squeue_rc=0, sacct_rc=0, stderr=""):
        self.squeue = list(squeue)
        self.sacct = list(sacct)
        self.squeue_rc = squeue_rc
        self.sacct_rc = sacct_rc
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def run(self, argv, *, input_text=None, timeout=30, env=None):
        self.calls.append(list(argv))
        prog = argv[0]
        if prog == "squeue":
            out, rc_ = "\n".join(self.squeue) + "\n", self.squeue_rc
        elif prog == "sacct":
            out, rc_ = "\n".join(self.sacct) + "\n", self.sacct_rc
        else:
            out, rc_ = "", 0
        return subprocess.CompletedProcess(argv, rc_, out, self.stderr if rc_ else "")

    def count(self, prog):
        return sum(1 for c in self.calls if c[0] == prog)


def q(sid, state, name=None, reason="None"):
    """One squeue line in the reconciler's format: %i|%j|%T|%r|%M|%l|%D|%N."""
    return f"{sid}|{name or 'xcl-' + sid}|{state}|{reason}|0:05|10:00|1|tower"


def a(sid, state, submit=1794000000, start=1794000005, end=1794000065, elapsed=60, exit_code="0:0"):
    """One sacct -D -X line: JobIDRaw|JobName|State|Submit|Start|End|ElapsedRaw|ExitCode|AllocTRES|NodeList."""
    return (
        f"{sid}|xcl-{sid}|{state}|{submit}|{start}|{end}|{elapsed}|{exit_code}|cpu=1,node=1|tower"
    )


@pytest.fixture
def cluster():
    """A fresh cluster id with mappings registered on it; cleaned up after."""
    cid = f"rt-{uuid.uuid4().hex[:8]}"
    made: list[str] = []

    def add(slurm_id, **kw):
        xid = f"{cid}-{slurm_id}"
        sa.register_slurm_job(xid, str(slurm_id), cluster_id=cid)
        if kw:
            sets = ", ".join(f"{k} = %s" for k in kw)
            with pg_transaction() as conn:
                conn.execute(
                    f"UPDATE slurm_job_mappings SET {sets} WHERE xcelsior_job_id = %s",
                    (*kw.values(), xid),
                )
        made.append(xid)
        return xid

    add.cid = cid
    yield add
    with pg_transaction() as conn:
        conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
        conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = %s", (cid,))


def row(xid):
    with pg_connection() as conn:
        r = conn.execute(
            "SELECT observed_state, slurm_state, terminal_at IS NOT NULL, missing_since IS NOT NULL, "
            "version, last_observed_at, started_at IS NOT NULL, exit_code "
            "FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
            (xid,),
        ).fetchone()
    assert r is not None
    keys = (
        "observed",
        "slurm_state",
        "terminal",
        "missing",
        "version",
        "seen_at",
        "started",
        "exit_code",
    )
    return dict(zip(keys, r))


def run_pass(add, fake, **kw):
    return rc.reconcile_cluster(add.cid, "generic", runner=fake, **kw)


class TestOneCallPerPass:
    def test_many_open_jobs_cost_one_squeue_and_no_sacct(self, cluster):
        xids = [cluster(100 + i) for i in range(12)]
        fake = FakeCluster(squeue=[q(str(100 + i), "RUNNING") for i in range(12)])
        result = run_pass(cluster, fake)
        assert result.ok and result.in_controller == 12
        assert fake.count("squeue") == 1 and fake.count("sacct") == 0
        assert fake.calls[0] == ["squeue", "--me", "-h", "-t", "all", "-o", sa._SQUEUE_FORMAT]
        assert {row(x)["observed"] for x in xids} == {"running"}

    def test_unchanged_rows_are_not_rewritten(self, cluster):
        xid = cluster(200)
        fake = FakeCluster(squeue=[q("200", "PENDING", reason="Priority")])
        run_pass(cluster, fake)
        first = row(xid)
        second_pass = run_pass(cluster, fake)
        again = row(xid)
        assert second_pass.changed == []
        assert again["version"] == first["version"]  # no write
        assert again["seen_at"] > first["seen_at"]  # but freshness moved


class TestWhenAJobIsFinished:
    def test_finished_in_squeue_is_observed_but_not_settled(self, cluster):
        xid = cluster(300)
        run_pass(cluster, FakeCluster(squeue=[q("300", "COMPLETED")]))
        r = row(xid)
        assert r["observed"] == "completed" and not r["terminal"]

    def test_gone_from_squeue_and_finished_in_accounting_settles(self, cluster):
        xid = cluster(301)
        fake = FakeCluster(squeue=[], sacct=[a("301", "COMPLETED")])
        result = run_pass(cluster, fake)
        r = row(xid)
        assert r["observed"] == "completed" and r["terminal"] and r["started"]
        assert result.settled == [xid]
        sacct = next(c for c in fake.calls if c[0] == "sacct")
        assert sacct[:5] == ["sacct", "-D", "-X", "-n", "--parsable2"]
        since = sacct[sacct.index("-S") + 1]
        assert since.startswith("now-") and since.endswith("minutes")  # always bounded

    def test_a_requeued_job_stays_open(self, cluster):
        # Preempted, then requeued by Slurm under the same id: squeue shows it
        # pending again even though accounting already holds a PREEMPTED run.
        xid = cluster(302, observed_state="running")
        run_pass(
            cluster,
            FakeCluster(
                squeue=[q("302", "PENDING", reason="BeginTime")], sacct=[a("302", "PREEMPTED")]
            ),
        )
        r = row(xid)
        assert r["observed"] == "pending" and not r["terminal"]

    def test_latest_run_decides_after_a_requeue(self, cluster):
        xid = cluster(303)
        runs = [a("303", "PREEMPTED", submit=1794000000), a("303", "COMPLETED", submit=1794000100)]
        run_pass(cluster, FakeCluster(squeue=[], sacct=runs))
        r = row(xid)
        assert r["observed"] == "completed" and r["terminal"]

    def test_running_in_accounting_but_not_in_squeue_stays_open(self, cluster):
        xid = cluster(304)
        run_pass(cluster, FakeCluster(squeue=[], sacct=[a("304", "RUNNING", end="Unknown")]))
        r = row(xid)
        assert r["observed"] == "running" and not r["terminal"]

    @pytest.mark.parametrize(
        "state,observed",
        [
            ("SPECIAL_EXIT", "held"),
            ("REQUEUE_HOLD", "held"),
            ("SUSPENDED", "suspended"),
            ("NODE_FAIL", "failed"),
        ],
    )
    def test_holds_and_suspension_are_their_own_states(self, cluster, state, observed):
        xid = cluster(310)
        run_pass(cluster, FakeCluster(squeue=[q("310", state)]))
        r = row(xid)
        assert r["observed"] == observed and not r["terminal"]


class TestSilenceIsNotAVerdict:
    def test_missing_everywhere_keeps_its_observation(self, cluster):
        xid = cluster(400, observed_state="running")
        result = run_pass(cluster, FakeCluster(squeue=[], sacct=[]))
        r = row(xid)
        assert r["observed"] == "running" and r["missing"] and not r["terminal"]
        assert result.missing == ["400"]

    def test_long_missing_reads_unknown_but_stays_open(self, cluster):
        xid = cluster(401, observed_state="running")
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE slurm_job_mappings SET missing_since = clock_timestamp() - interval '2 hours' "
                "WHERE xcelsior_job_id = %s",
                (xid,),
            )
        run_pass(cluster, FakeCluster(squeue=[], sacct=[]))
        r = row(xid)
        assert r["observed"] == "unknown" and not r["terminal"]

    def test_reappearing_clears_missing(self, cluster):
        xid = cluster(402)
        run_pass(cluster, FakeCluster(squeue=[], sacct=[]))
        assert row(xid)["missing"]
        run_pass(cluster, FakeCluster(squeue=[q("402", "RUNNING")]))
        assert not row(xid)["missing"]


class TestFailures:
    def test_transport_failure_changes_nothing(self, cluster):
        xid = cluster(500, observed_state="running")
        before = row(xid)
        result = run_pass(
            cluster, FakeCluster(squeue_rc=255, stderr="ssh: connect to host fake: timed out")
        )
        assert not result.ok and result.error_code == "transport_error"
        assert row(xid)["version"] == before["version"]

    def test_accounting_failure_still_applies_what_squeue_saw(self, cluster):
        in_ctl = cluster(501)
        gone = cluster(502, observed_state="running")
        fake = FakeCluster(
            squeue=[q("501", "RUNNING")], sacct_rc=1, stderr="sacct: error: slurmdbd unreachable"
        )
        result = run_pass(cluster, fake)
        assert result.ok and result.accounting_error
        assert row(in_ctl)["observed"] == "running"
        g = row(gone)
        assert g["observed"] == "running" and not g["terminal"] and not g["missing"]

    def test_a_stale_version_loses_the_write(self, cluster):
        xid = cluster(503)
        [stale] = rc._open_mappings(cluster.cid)
        with pg_transaction() as conn:  # someone else writes first
            conn.execute(
                "UPDATE slurm_job_mappings SET version = version + 1 WHERE xcelsior_job_id = %s",
                (xid,),
            )
        obs = rc._observation(
            stale, {"503": {"state": "RUNNING", "reason": "None", "name": "xcl-503"}}, {}, 0.0
        )
        assert rc._write([(stale, obs)], []) == [xid]
        assert row(xid)["observed"] != "running"


class TestDesiredState:
    def test_cancel_is_sent_while_the_job_runs(self, cluster):
        cluster(600, desired_state="cancelled")
        fake = FakeCluster(squeue=[q("600", "RUNNING")])
        result = run_pass(cluster, fake)
        assert fake.calls[-1] == ["scancel", "-v", "600"]
        assert result.cancel_requested == ["600"]

    def test_no_cancel_once_finished(self, cluster):
        cluster(601, desired_state="cancelled")
        fake = FakeCluster(squeue=[q("601", "CANCELLED")])
        run_pass(cluster, fake)
        assert fake.count("scancel") == 0

    def test_compat_entry_point_cancels_only_through_its_callback(self, cluster):
        cluster(602, desired_state="cancelled")
        fake = FakeCluster(squeue=[q("602", "RUNNING")])
        run_pass(cluster, fake, cancel=False)
        assert fake.count("scancel") == 0


class TestOrphans:
    def test_adapter_jobs_without_a_mapping_are_reported(self, cluster):
        cluster(700)
        fake = FakeCluster(
            squeue=[
                q("700", "RUNNING"),
                q("701", "RUNNING"),
                q("702", "RUNNING", name="someone-else"),
            ]
        )
        result = run_pass(cluster, fake)
        assert result.orphans == ["701"]  # adapter-named, unmapped; 702 isn't ours


class TestRegistry:
    def test_alliance_profiles_register_research_only(self):
        info = rc.register_cluster("nibi", enabled=False)
        try:
            [c] = [c for c in rc.list_clusters() if c["cluster_id"] == "nibi"]
            assert c["research_only"] and not c["marketplace"] and not c["agent_access"]
            assert c["poll_interval_seconds"] == 120
            assert info["research_only"]
        finally:
            with pg_transaction() as conn:
                conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = 'nibi'")

    def test_the_schema_refuses_a_research_cluster_open_to_agents(self, cluster):
        with pytest.raises(psycopg.errors.CheckViolation):
            with pg_transaction() as conn:
                conn.execute(
                    "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, research_only, agent_access) "
                    "VALUES (%s, 'nibi', 'x', TRUE, TRUE)",
                    (cluster.cid,),
                )

    def test_a_failed_pass_opens_the_breaker_and_a_good_one_closes_it(self, cluster):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, poll_interval_seconds) "
                "VALUES (%s, 'generic', 'x', 30)",
                (cluster.cid,),
            )

        def state():
            with pg_connection() as conn:
                return conn.execute(
                    "SELECT consecutive_failures, breaker_open_until > clock_timestamp(), "
                    "transport_ok_at IS NOT NULL, last_error_code FROM slurm_clusters WHERE cluster_id = %s",
                    (cluster.cid,),
                ).fetchone()

        bad = rc.PassResult(
            cluster_id=cluster.cid, ok=False, error_code="transport_error", error="down"
        )
        rc._record_health(cluster.cid, bad, 30, 0)
        assert state() == (1, True, False, "transport_error")
        rc._record_health(cluster.cid, rc.PassResult(cluster_id=cluster.cid), 30, 1)
        assert state() == (0, None, True, None)

    def test_accounting_health_needs_a_sacct_that_ran(self, cluster):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name) VALUES (%s, 'generic', 'x')",
                (cluster.cid,),
            )

        def accounting_ok():
            with pg_connection() as conn:
                return conn.execute(
                    "SELECT accounting_ok_at IS NOT NULL FROM slurm_clusters WHERE cluster_id = %s",
                    (cluster.cid,),
                ).fetchone()[0]

        rc._record_health(cluster.cid, rc.PassResult(cluster_id=cluster.cid), 60, 0)
        assert not accounting_ok()  # nothing needed accounting this pass
        rc._record_health(
            cluster.cid, rc.PassResult(cluster_id=cluster.cid, accounting_checked=True), 60, 0
        )
        assert accounting_ok()
