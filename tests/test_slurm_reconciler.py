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

    def __init__(self, squeue=(), sacct=(), squeue_rc=0, sacct_rc=0, stderr="", outputs=None):
        self.squeue = list(squeue)
        self.sacct = list(sacct)
        self.squeue_rc = squeue_rc
        self.sacct_rc = sacct_rc
        self.stderr = stderr
        #: {(slurm id, stream): text} for `tail`; a missing file exits 1.
        self.outputs = dict(outputs or {})
        self.calls: list[list[str]] = []

    def run(self, argv, *, input_text=None, timeout=30, env=None):
        self.calls.append(list(argv))
        prog = argv[0]
        if prog == "squeue":
            out, rc_ = "\n".join(self.squeue) + "\n", self.squeue_rc
        elif prog == "sacct":
            out, rc_ = "\n".join(self.sacct) + "\n", self.sacct_rc
        elif prog == "tail":
            sid, stream = argv[-1].rsplit("-", 1)[1].split(".")
            text = self.outputs.get((sid, stream))
            out, rc_ = (text, 0) if text is not None else ("", 1)
            if rc_:
                return subprocess.CompletedProcess(argv, 1, "", f"tail: {argv[-1]}: No such file")
        else:
            out, rc_ = "", 0
        return subprocess.CompletedProcess(argv, rc_, out, self.stderr if rc_ else "")

    def count(self, prog):
        return sum(1 for c in self.calls if c[0] == prog)


def q(sid, state, name=None, reason="None"):
    """One squeue line in the reconciler's format: %i|%j|%T|%r|%M|%l|%D|%N."""
    return f"{sid}|{name or 'xcl-' + sid}|{state}|{reason}|0:05|10:00|1|tower"


def a(
    sid,
    state,
    submit=1794000000,
    start=1794000005,
    end=1794000065,
    elapsed=60,
    exit_code="0:0",
    name=None,
):
    """One sacct -D -X line: JobIDRaw|JobName|State|Submit|Start|End|ElapsedRaw|ExitCode|AllocTRES|NodeList."""
    return (
        f"{sid}|{name or 'xcl-' + sid}|{state}|{submit}|{start}|{end}|{elapsed}|{exit_code}"
        "|cpu=1,node=1|tower"
    )


@pytest.fixture
def cluster():
    """A fresh cluster id with mappings registered on it; cleaned up after.

    `cluster.q` and `cluster.a` write squeue and sacct lines named as the
    adapter names its jobs, ``xcl-<xcelsior job id>``, for the mapping that
    `cluster(slurm_id)` registers (or for `xid=` when one is given).
    """
    cid = f"rt-{uuid.uuid4().hex[:8]}"
    made: list[str] = []

    def add(slurm_id, *, xid=None, **kw):
        xid = xid or f"{cid}-{slurm_id}"
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

    def name(sid, xid=None):
        return f"xcl-{xid or f'{cid}-{sid}'}"

    add.cid = cid
    add.q = lambda sid, state, *, xid=None, name_=None, **kw: q(
        sid, state, name=name_ or name(sid, xid), **kw
    )
    add.a = lambda sid, state, *, xid=None, name_=None, **kw: a(
        sid, state, name=name_ or name(sid, xid), **kw
    )
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
        fake = FakeCluster(squeue=[cluster.q(str(100 + i), "RUNNING") for i in range(12)])
        result = run_pass(cluster, fake)
        assert result.ok and result.in_controller == 12
        assert fake.count("squeue") == 1 and fake.count("sacct") == 0
        assert fake.calls[0] == ["squeue", "--me", "-h", "-t", "all", "-o", sa._SQUEUE_FORMAT]
        assert {row(x)["observed"] for x in xids} == {"running"}

    def test_unchanged_rows_are_not_rewritten(self, cluster):
        xid = cluster(200)
        fake = FakeCluster(squeue=[cluster.q("200", "PENDING", reason="Priority")])
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
        run_pass(cluster, FakeCluster(squeue=[cluster.q("300", "COMPLETED")]))
        r = row(xid)
        assert r["observed"] == "completed" and not r["terminal"]

    def test_gone_from_squeue_and_finished_in_accounting_settles(self, cluster):
        xid = cluster(301)
        fake = FakeCluster(squeue=[], sacct=[cluster.a("301", "COMPLETED")])
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
                squeue=[cluster.q("302", "PENDING", reason="BeginTime")],
                sacct=[cluster.a("302", "PREEMPTED")],
            ),
        )
        r = row(xid)
        assert r["observed"] == "pending" and not r["terminal"]

    def test_latest_run_decides_after_a_requeue(self, cluster):
        xid = cluster(303)
        runs = [
            cluster.a("303", "PREEMPTED", submit=1794000000),
            cluster.a("303", "COMPLETED", submit=1794000100),
        ]
        run_pass(cluster, FakeCluster(squeue=[], sacct=runs))
        r = row(xid)
        assert r["observed"] == "completed" and r["terminal"]

    def test_running_in_accounting_but_not_in_squeue_stays_open(self, cluster):
        xid = cluster(304)
        run_pass(
            cluster, FakeCluster(squeue=[], sacct=[cluster.a("304", "RUNNING", end="Unknown")])
        )
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
        run_pass(cluster, FakeCluster(squeue=[cluster.q("310", state)]))
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
        run_pass(cluster, FakeCluster(squeue=[cluster.q("402", "RUNNING")]))
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
            squeue=[cluster.q("501", "RUNNING")],
            sacct_rc=1,
            stderr="sacct: error: slurmdbd unreachable",
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
            stale, {"503": {"state": "RUNNING", "reason": "None", "name": f"xcl-{xid}"}}, {}, 0.0
        )
        assert rc._write([(stale, obs)], []) == [xid]
        assert row(xid)["observed"] != "running"


class TestDesiredState:
    def test_cancel_is_sent_while_the_job_runs(self, cluster):
        cluster(600, desired_state="cancelled")
        fake = FakeCluster(squeue=[cluster.q("600", "RUNNING")])
        result = run_pass(cluster, fake)
        assert fake.calls[-1] == ["scancel", "-v", "600"]
        assert result.cancel_requested == ["600"]

    def test_no_cancel_once_finished(self, cluster):
        cluster(601, desired_state="cancelled")
        fake = FakeCluster(squeue=[cluster.q("601", "CANCELLED")])
        run_pass(cluster, fake)
        assert fake.count("scancel") == 0

    def test_compat_entry_point_cancels_only_through_its_callback(self, cluster):
        cluster(602, desired_state="cancelled")
        fake = FakeCluster(squeue=[cluster.q("602", "RUNNING")])
        run_pass(cluster, fake, cancel=False)
        assert fake.count("scancel") == 0


class TestOrphans:
    def test_adapter_jobs_without_a_mapping_are_reported(self, cluster):
        cluster(700)
        fake = FakeCluster(
            squeue=[
                cluster.q("700", "RUNNING"),
                cluster.q("701", "RUNNING"),
                cluster.q("702", "RUNNING", name_="someone-else"),
            ]
        )
        result = run_pass(cluster, fake)
        assert result.orphans == ["701"]  # adapter-named, unmapped; 702 isn't ours


class TestJobIdReuse:
    """A Slurm job id is unique on one cluster, and only for a while.

    Two clusters hand out the same numbers. And one cluster starts numbering
    again from FirstJobId when its controller state is lost (a rebuild, a
    wiped StateSaveLocation), and wraps at MaxJobId. The adapter names every
    job ``xcl-<xcelsior job id>``, and that name is what tells two jobs under
    one number apart.
    """

    def test_the_same_id_on_two_clusters_never_crosses(self, cluster):
        here = cluster(800)
        other = f"{cluster.cid}-b"
        there = f"{other}-800"
        sa.register_slurm_job(there, "800", cluster_id=other)
        try:
            before = row(there)
            run_pass(cluster, FakeCluster(squeue=[cluster.q("800", "RUNNING")]))
            assert row(here)["observed"] == "running"
            assert row(there)["version"] == before["version"]
            assert sa.read_observed_status("800")["error_code"] == "ambiguous_job_id"
            assert sa.read_observed_status("800", cluster_id=other)["xcelsior_job_id"] == there
        finally:
            with pg_transaction() as conn:
                conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (other,))

    def test_a_settled_id_can_be_handed_out_again(self, cluster):
        old = cluster(801)
        run_pass(cluster, FakeCluster(squeue=[], sacct=[cluster.a("801", "COMPLETED")]))
        assert row(old)["terminal"]
        # The controller is rebuilt and numbering starts over: a new job, a
        # new Xcelsior id, the same Slurm id.
        new = cluster(801, xid=f"{cluster.cid}-801-again")
        result = run_pass(cluster, FakeCluster(squeue=[cluster.q("801", "RUNNING", xid=new)]))
        assert row(new)["observed"] == "running" and not row(new)["terminal"]
        assert row(old)["observed"] == "completed"
        assert result.orphans == []
        status = sa.read_observed_status("801", cluster_id=cluster.cid)
        assert status["xcelsior_job_id"] == new and not status["settled"]

    def test_a_newer_job_under_an_open_mappings_id_is_not_taken_for_it(self, cluster):
        # The old job left the controller while accounting was down, so its
        # mapping is still open, and then its id was handed out again.
        old = cluster(802, observed_state="running")
        newer = f"xcl-{cluster.cid}-someone-else"
        result = run_pass(
            cluster, FakeCluster(squeue=[cluster.q("802", "COMPLETED", name_=newer)], sacct=[])
        )
        r = row(old)
        assert r["observed"] == "running" and r["missing"] and not r["terminal"]
        assert result.in_controller == 0
        # And the newer job is an Xcelsior job no open mapping accounts for.
        assert result.orphans == ["802"]

    def test_a_newer_jobs_accounting_does_not_settle_the_old_mapping(self, cluster):
        old = cluster(803, observed_state="running")
        newer = f"xcl-{cluster.cid}-newer"
        own_run = cluster.a("803", "FAILED", submit=1794000000, exit_code="1:0")
        newer_run = cluster.a("803", "COMPLETED", submit=1794000500, name_=newer)
        run_pass(cluster, FakeCluster(squeue=[], sacct=[own_run, newer_run]))
        r = row(old)
        assert r["observed"] == "failed" and r["terminal"] and r["exit_code"] == "1:0"

    def test_only_a_newer_jobs_accounting_leaves_the_old_mapping_open(self, cluster):
        old = cluster(804, observed_state="running")
        newer_run = cluster.a("804", "COMPLETED", name_=f"xcl-{cluster.cid}-newer")
        run_pass(cluster, FakeCluster(squeue=[], sacct=[newer_run]))
        r = row(old)
        assert r["observed"] == "running" and r["missing"] and not r["terminal"]

    def test_a_job_the_adapter_did_not_submit_is_matched_by_id(self, cluster):
        # Registered for a job submitted some other way: its name says nothing
        # about Xcelsior, so the id is all there is to go on.
        xid = cluster(805)
        run_pass(cluster, FakeCluster(squeue=[cluster.q("805", "RUNNING", name_="train.sh")]))
        assert row(xid)["observed"] == "running"


class TestOutputCache:
    """Readers read slurm_job_outputs; only the reconciler pass fetches.

    Plan item 1.7: output is fetched at most once per interval per job,
    however many readers there are.
    """

    def test_many_readers_cost_one_fetch(self, cluster):
        xid = cluster(900, observed_state="running")
        squeue = [cluster.q("900", "RUNNING")]
        first = [sa.read_cached_output(xid) for _ in range(5)]
        assert all(r["refresh_pending"] and r["content"] == "" for r in first)

        fake = FakeCluster(squeue=squeue, outputs={("900", "out"): "epoch 1\nepoch 2\n"})
        result = run_pass(cluster, fake)
        assert fake.count("tail") == 1 and result.outputs_fetched == [(xid, "out")]
        tail = next(c for c in fake.calls if c[0] == "tail")
        assert tail[:3] == ["tail", "-c", str(sa.OUTPUT_TAIL_BYTES)]

        after = [sa.read_cached_output(xid) for _ in range(5)]
        assert {r["content"] for r in after} == {"epoch 1\nepoch 2\n"}
        # Fresh within the poll interval: no reader files another request,
        # so the next pass fetches nothing.
        assert not any(r["refresh_pending"] for r in after)
        again = FakeCluster(squeue=squeue, outputs={("900", "out"): "epoch 3\n"})
        run_pass(cluster, again)
        assert again.count("tail") == 0

    def test_a_stale_copy_is_requested_again(self, cluster):
        xid = cluster(901, observed_state="running")
        squeue = [cluster.q("901", "RUNNING")]
        sa.read_cached_output(xid)
        run_pass(cluster, FakeCluster(squeue=squeue, outputs={("901", "out"): "a\n"}))
        with pg_transaction() as conn:  # a poll interval passes
            conn.execute(
                "UPDATE slurm_job_outputs SET fetched_at = fetched_at - interval '2 minutes', "
                "attempted_at = attempted_at - interval '2 minutes' WHERE xcelsior_job_id = %s",
                (xid,),
            )
        assert sa.read_cached_output(xid)["refresh_pending"]
        run_pass(cluster, FakeCluster(squeue=squeue, outputs={("901", "out"): "a\nb\n"}))
        assert sa.read_cached_output(xid)["content"] == "a\nb\n"

    def test_a_settled_job_gets_one_final_copy(self, cluster):
        xid = cluster(902, observed_state="running")
        sa.read_cached_output(xid, "err")
        run_pass(
            cluster, FakeCluster(squeue=[cluster.q("902", "RUNNING")])
        )  # tail fails: no file yet
        assert sa.read_cached_output(xid, "err")["fetch_error_code"] == "output_unavailable"

        # The job ends. The pass that settles it takes one final copy,
        # unasked, because the file can't change any more.
        done = FakeCluster(
            squeue=[],
            sacct=[cluster.a("902", "FAILED", exit_code="1:0")],
            outputs={("902", "err"): "Traceback: boom\n"},
        )
        result = run_pass(cluster, done)
        assert result.settled == [xid] and result.outputs_fetched == [(xid, "err")]
        final = sa.read_cached_output(xid, "err")
        assert final["final"] and final["content"] == "Traceback: boom\n"
        assert not final["refresh_pending"] and final["fetch_error_code"] is None

        later = FakeCluster(squeue=[], outputs={("902", "err"): "changed?\n"})
        sa.read_cached_output(xid, "err")
        run_pass(cluster, later)
        assert later.count("tail") == 0

    def test_fetches_per_pass_are_capped(self, cluster):
        xids = [cluster(910 + i, observed_state="running") for i in range(12)]
        for x in xids:
            sa.read_cached_output(x)
        squeue = [cluster.q(str(910 + i), "RUNNING") for i in range(12)]
        outputs = {(str(910 + i), "out"): f"job {i}\n" for i in range(12)}
        first = FakeCluster(squeue=squeue, outputs=outputs)
        run_pass(cluster, first)
        assert first.count("tail") == rc.OUTPUT_FETCHES_PER_PASS
        second = FakeCluster(squeue=squeue, outputs=outputs)
        run_pass(cluster, second)
        assert second.count("tail") == 12 - rc.OUTPUT_FETCHES_PER_PASS

    def test_output_is_stored_whatever_bytes_the_job_printed(self, cluster):
        xid = cluster(903, observed_state="running")
        sa.read_cached_output(xid)
        noisy = "progress\x00\x00 50% \ufffd\n"
        run_pass(
            cluster,
            FakeCluster(squeue=[cluster.q("903", "RUNNING")], outputs={("903", "out"): noisy}),
        )
        assert sa.read_cached_output(xid)["content"] == "progress 50% \ufffd\n"

    def test_reading_never_contacts_the_cluster(self, cluster, monkeypatch):
        xid = cluster(904)

        def refuse(*_a, **_kw):
            raise AssertionError("a read contacted the cluster")

        monkeypatch.setattr(sa, "get_slurm_job_output", refuse)
        monkeypatch.setattr(sa.SlurmRunner, "run", refuse)
        for stream in ("out", "err", "out"):
            assert sa.read_cached_output(xid, stream)["source"] == "reconciler"
        assert sa.read_cached_output(xid, "both")["error_code"] == "invalid_request"
        assert sa.read_cached_output("no-such-job")["error_code"] == "not_found"


class TestRunnerDecoding:
    def test_bytes_that_are_not_utf8_do_not_raise(self, tmp_path):
        # `tail -c` usually starts inside a multi-byte character.
        f = tmp_path / "out"
        f.write_bytes("é".encode()[1:] + b"ok\xff\n")
        r = sa.SlurmRunner().run(["cat", str(f)])
        assert r.returncode == 0 and r.stdout.endswith("ok\ufffd\n")


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
