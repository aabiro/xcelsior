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

import contextlib
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
        elif prog == "scontrol" and argv[1:] == ["ping"]:
            out, rc_ = "Slurmctld(primary) at tower is UP\n", 0
        elif prog == "scontrol" and argv[1:] == ["show", "config"]:
            out, rc_ = CONFIG_TEXT, 0
        elif prog in ("tail", "cat", "ls"):
            sid, stream = argv[-1].rsplit("-", 1)[1].split(".")
            text = self.outputs.get((sid, stream))
            if text is None:
                return subprocess.CompletedProcess(argv, 1, "", f"{prog}: {argv[-1]}: No such file")
            out, rc_ = text, 0
            if prog == "ls":
                out = f"-rw-r----- 1 xclslurm xcl {len(text.encode())} Nov  8 12:00 {argv[-1]}\n"
        else:
            out, rc_ = "", 0
        return subprocess.CompletedProcess(argv, rc_, out, self.stderr if rc_ else "")

    def count(self, prog):
        return sum(1 for c in self.calls if c[0] == prog)


#: What `scontrol show config` prints, trimmed to the facts kept and one more.
CONFIG_TEXT = (
    "Configuration data as of 2026-11-08T12:00:00\n"
    + "".join(f"{k:<24}= v-{k}\n" for k in sa.CONFIG_FACTS)
    + "SlurmUser               = slurm(64030)\n"
)


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
    derived="0:0",
):
    """One sacct -D -X line: JobIDRaw|JobName|State|Submit|Start|End|ElapsedRaw|ExitCode|AllocTRES|NodeList|DerivedExitCode."""
    return (
        f"{sid}|{name or 'xcl-' + sid}|{state}|{submit}|{start}|{end}|{elapsed}|{exit_code}"
        f"|cpu=1,node=1|tower|{derived}"
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

    def test_registering_onto_a_lost_open_mapping_closes_it(self, cluster):
        # The controller lost its state while accounting had no outcome for
        # 806, so its mapping is still open; then sbatch hands 806 out again.
        old = cluster(806, observed_state="running")
        new = f"{cluster.cid}-806-new"
        # Without proof the id is fresh from sbatch, it is refused as before.
        with pytest.raises(psycopg.errors.UniqueViolation):
            sa.register_slurm_job(new, "806", cluster_id=cluster.cid)
        assert row(old)["observed"] == "running" and not row(old)["terminal"]
        assert sa.register_slurm_job(new, "806", cluster_id=cluster.cid, just_submitted=True) == [
            old
        ]
        with pg_connection() as conn:
            reason, superseded_by = conn.execute(
                "SELECT state_reason, metadata->>'superseded_by' FROM slurm_job_mappings "
                "WHERE xcelsior_job_id = %s",
                (old,),
            ).fetchone()
        r = row(old)
        assert r["observed"] == "unknown" and r["terminal"]
        assert reason == "slurm_job_id_reused" and superseded_by == new
        # A retried registration of the same job closes nothing.
        assert sa.register_slurm_job(new, "806", cluster_id=cluster.cid, just_submitted=True) == []
        result = run_pass(cluster, FakeCluster(squeue=[cluster.q("806", "RUNNING", xid=new)]))
        assert row(new)["observed"] == "running" and not row(new)["terminal"]
        assert result.orphans == []

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


class TestSuccessVerdict:
    @pytest.mark.parametrize(
        "state,exit_code,derived,settled,verdict",
        [
            ("completed", "0:0", "0:0", True, "succeeded"),
            ("completed", "0:0", "1:0", True, "failed"),  # a step failed, the script exited 0
            ("completed", "1:0", "0:0", True, "failed"),
            ("completed", "0:0", None, True, "unknown"),
            ("failed", "1:0", "1:0", True, "failed"),
            ("preempted", "0:0", "0:0", True, "failed"),
            ("cancelled", "0:15", "0:0", True, "cancelled"),
            ("unknown", None, None, True, "unknown"),
            ("completed", "0:0", "0:0", False, None),  # open: no verdict yet
        ],
    )
    def test_the_rule(self, state, exit_code, derived, settled, verdict):
        assert sa.success_verdict(state, exit_code, derived, settled) == verdict

    def test_the_derived_exit_code_is_recorded_at_settle(self, cluster):
        xid = cluster(950, observed_state="running")
        run_pass(cluster, FakeCluster(sacct=[cluster.a("950", "COMPLETED", derived="2:0")]))
        with pg_connection() as conn:
            got = conn.execute(
                "SELECT observed_state, exit_code, derived_exit_code, terminal_at IS NOT NULL "
                "FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
                (xid,),
            ).fetchone()
        assert got == ("completed", "0:0", "2:0", True)
        assert sa.success_verdict(*got) == "failed"


class TestSchedulerProgress:
    def test_time_limit_and_start_come_from_squeue(self, cluster):
        xid = cluster(951)
        run_pass(cluster, FakeCluster(squeue=[cluster.q("951", "PENDING", reason="Priority")]))
        line = "951|xcl-{}|RUNNING|None|1:30|1:00:00|1|tower".format(xid)
        run_pass(cluster, FakeCluster(squeue=[line]))
        with pg_connection() as conn:
            limit, ran = conn.execute(
                "SELECT time_limit_seconds, EXTRACT(EPOCH FROM clock_timestamp() - started_at) "
                "FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
                (xid,),
            ).fetchone()
        assert limit == 3600
        assert 90 <= float(ran) < 120  # started 1:30 before it was seen running


class TestAwaitingAccounting:
    def _left(self, xid):
        with pg_connection() as conn:
            return conn.execute(
                "SELECT left_controller_at IS NOT NULL FROM slurm_job_mappings "
                "WHERE xcelsior_job_id = %s",
                (xid,),
            ).fetchone()[0]

    def test_stamped_when_gone_even_if_accounting_fails(self, cluster):
        xid = cluster(960, observed_state="running")
        run_pass(cluster, FakeCluster(squeue=[cluster.q("960", "RUNNING")]))
        assert not self._left(xid)
        down = FakeCluster(squeue=[], sacct_rc=1, stderr="sacct: error: slurmdbd unreachable")
        result = run_pass(cluster, down)
        assert result.accounting_error and result.accounting_error_code == "sacct_failed"
        assert self._left(xid) and not row(xid)["terminal"]

    def test_cleared_if_the_job_comes_back(self, cluster):
        xid = cluster(961, observed_state="running")
        run_pass(cluster, FakeCluster(squeue=[], sacct=[]))
        assert self._left(xid)
        run_pass(cluster, FakeCluster(squeue=[cluster.q("961", "PENDING")]))
        assert not self._left(xid)


class TestFailureCounts:
    def test_counted_by_stage_and_code(self, cluster):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name) "
                "VALUES (%s, 'generic', 'x')",
                (cluster.cid,),
            )
        down = rc.PassResult(cluster_id=cluster.cid, ok=False, error_code="host_key_mismatch")
        rc._record_health(cluster.cid, down, 30, 0)
        rc._record_health(cluster.cid, down, 30, 1)
        partial = rc.PassResult(
            cluster_id=cluster.cid,
            accounting_error="slurmdbd unreachable",
            accounting_error_code="sacct_failed",
            output_errors=[("j", "out", "output_unavailable")],
        )
        rc._record_health(cluster.cid, partial, 30, 2)
        with pg_connection() as conn:
            counts = conn.execute(
                "SELECT failure_counts FROM slurm_clusters WHERE cluster_id = %s", (cluster.cid,)
            ).fetchone()[0]
        assert counts == {
            "poll:host_key_mismatch": 2,
            "accounting:sacct_failed": 1,
            "output:output_unavailable": 1,
        }


class TestReportedProgress:
    def test_none_until_the_job_has_written_one(self, cluster):
        xid = cluster(970, observed_state="running")
        assert sa.read_reported_progress(xid) is None  # files the request
        run_pass(cluster, FakeCluster(squeue=[cluster.q("970", "RUNNING")]))  # no file yet
        assert sa.read_reported_progress(xid) is None

    def test_a_report_per_the_contract(self, cluster):
        xid = cluster(971, observed_state="running")
        sa.read_reported_progress(xid)
        import time as _t

        report = '{"fraction": 0.5, "message": "epoch 5 of 10", "updated_at": %d}' % _t.time()
        fake = FakeCluster(
            squeue=[cluster.q("971", "RUNNING")], outputs={("971", "progress"): report}
        )
        run_pass(cluster, fake)
        tail = next(c for c in fake.calls if c[0] == "tail")
        assert tail[2] == str(sa.PROGRESS_MAX_BYTES) and tail[-1].endswith(".progress")
        got = sa.read_reported_progress(xid)
        assert got["valid"] and got["source"] == "job" and got["fraction"] == 0.5
        assert got["message"] == "epoch 5 of 10" and not got["stale"]

    @pytest.mark.parametrize(
        "content",
        ['{"fraction": 0.5}', '{"fraction": 2, "updated_at": 1}', "not json", "[1, 2]"],
    )
    def test_a_file_that_breaks_the_contract_says_so(self, cluster, content):
        xid = cluster(972, observed_state="running")
        sa.read_reported_progress(xid)
        run_pass(
            cluster,
            FakeCluster(
                squeue=[cluster.q("972", "RUNNING")], outputs={("972", "progress"): content}
            ),
        )
        got = sa.read_reported_progress(xid)
        assert got["valid"] is False and "contract" in got["error"]

    def test_an_old_report_on_a_running_job_is_stale(self, cluster):
        xid = cluster(973, observed_state="running")
        sa.read_reported_progress(xid)
        old = '{"updated_at": 1794000000}'
        run_pass(
            cluster,
            FakeCluster(squeue=[cluster.q("973", "RUNNING")], outputs={("973", "progress"): old}),
        )
        assert sa.read_reported_progress(xid)["stale"]

    def test_the_script_tells_the_job_where(self):
        script, _meta = sa.xcelsior_job_to_sbatch(
            {"job_id": "p-1", "name": "train", "num_gpus": 0, "command": "true"}, "generic"
        )
        assert 'export XCELSIOR_PROGRESS_FILE="' in script
        assert (
            script.split("XCELSIOR_PROGRESS_FILE=")[1]
            .split("\n")[0]
            .endswith('/xcl-p-1-${SLURM_JOB_ID}.progress"')
        )


class TestRegistryTransport:
    def _register(self, cluster, **kw):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, "
                "poll_interval_seconds, ssh_host, known_hosts_line, credential_ref) "
                "VALUES (%s, 'generic', 'x', 30, %s, %s, %s)",
                (
                    cluster.cid,
                    kw.get("ssh_host", "203.0.113.7"),
                    kw.get("known_hosts_line", "h ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA"),
                    kw.get("credential_ref", "file:/etc/xcelsior/slurm/lab.key"),
                ),
            )

    def test_the_schema_refuses_an_unpinned_or_keyless_host(self, cluster):
        for missing in ("known_hosts_line", "credential_ref"):
            with pytest.raises(psycopg.errors.CheckViolation):
                with pg_transaction() as conn:
                    conn.execute(
                        f"INSERT INTO slurm_clusters (cluster_id, profile_key, display_name, "
                        f"ssh_host, {'credential_ref' if missing == 'known_hosts_line' else 'known_hosts_line'}) "
                        "VALUES (%s, 'generic', 'x', 'lab', 'file:/k')",
                        (cluster.cid,),
                    )
        with pytest.raises(psycopg.errors.CheckViolation):
            self._register(cluster, credential_ref="vault:lab")

    def test_registration_refuses_an_unusable_pin(self, cluster):
        with pytest.raises(sa.SlurmAdapterError) as e:
            rc.register_cluster(
                "generic",
                cluster_id=cluster.cid,
                ssh_host="203.0.113.7",
                known_hosts_line="@cert-authority * ssh-ed25519 AAAA",
                credential_ref="file:/k",
            )
        assert e.value.code == "invalid_host_key"

    def test_a_registered_cluster_is_polled_over_its_registry_transport(self, cluster, monkeypatch):
        self._register(cluster)
        xid = cluster(980)
        fake = FakeCluster(squeue=[cluster.q("980", "RUNNING")])
        used = []

        @contextlib.contextmanager
        def runner_from_registry(row):
            used.append(row["ssh_host"])
            yield fake

        monkeypatch.setattr(rc.st, "cluster_runner", runner_from_registry)
        monkeypatch.setattr(rc, "_due_clusters", lambda: [rc._cluster_row(cluster.cid)])
        [result] = rc.reconcile_due_clusters()
        assert used == ["203.0.113.7"] and result["ok"] and fake.count("squeue") == 1
        assert row(xid)["observed"] == "running"

    def test_a_connection_check_stores_the_config_facts(self, cluster, monkeypatch):
        self._register(cluster)
        fake = FakeCluster()

        @contextlib.contextmanager
        def runner_from_registry(row):
            yield fake

        monkeypatch.setattr(rc.st, "cluster_runner", runner_from_registry)
        got = rc.check_cluster_connection(cluster.cid)
        assert got["ok"] and got["missing"] == [] and got["controller"]["controller"] == "up"
        with pg_connection() as conn:
            facts = conn.execute(
                "SELECT facts FROM slurm_clusters WHERE cluster_id = %s", (cluster.cid,)
            ).fetchone()[0]
        assert set(facts["config"]) == set(sa.CONFIG_FACTS)
        assert facts["config"]["MinJobAge"] == "v-MinJobAge" and facts["config_captured_at"]
        assert "SlurmUser" not in facts["config"]

    def test_a_check_of_an_unknown_cluster_reports_it(self):
        assert rc.check_cluster_connection("no-such-cluster")["error_code"] == "not_found"


class TestFactsRefresh:
    def test_due_when_missing_old_or_unreadable(self):
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        assert rc._facts_due({}) and rc._facts_due(None)
        assert rc._facts_due({"config_captured_at": "yesterday-ish"})
        assert rc._facts_due({"config_captured_at": "2026-11-08T12:00:00"})  # no zone
        old = (now - dt.timedelta(days=8)).isoformat()
        assert rc._facts_due({"config_captured_at": old})
        assert not rc._facts_due({"config_captured_at": (now - dt.timedelta(days=1)).isoformat()})

    def test_a_due_cluster_captures_them_on_its_pass(self, cluster, monkeypatch):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name) "
                "VALUES (%s, 'generic', 'x')",
                (cluster.cid,),
            )
        fake = FakeCluster()

        @contextlib.contextmanager
        def runner_for(row):
            yield fake

        monkeypatch.setattr(rc, "_runner_for", runner_for)
        monkeypatch.setattr(rc, "_due_clusters", lambda: [rc._cluster_row(cluster.cid)])
        [first] = rc.reconcile_due_clusters()
        assert first["facts_refreshed"] and fake.calls[-1] == ["scontrol", "show", "config"]
        [second] = rc.reconcile_due_clusters()  # captured a moment ago: not due
        assert not second["facts_refreshed"]
        assert sum(1 for c in fake.calls if c[:2] == ["scontrol", "show"]) == 1

    def test_a_failed_capture_is_counted_not_fatal(self, cluster, monkeypatch):
        with pg_transaction() as conn:
            conn.execute(
                "INSERT INTO slurm_clusters (cluster_id, profile_key, display_name) "
                "VALUES (%s, 'generic', 'x')",
                (cluster.cid,),
            )

        class NoConfig(FakeCluster):
            def run(self, argv, **kw):
                if argv[:2] == ["scontrol", "show"]:
                    return subprocess.CompletedProcess(argv, 1, "", "scontrol: error: denied")
                return super().run(argv, **kw)

        @contextlib.contextmanager
        def runner_for(row):
            yield NoConfig()

        monkeypatch.setattr(rc, "_runner_for", runner_for)
        monkeypatch.setattr(rc, "_due_clusters", lambda: [rc._cluster_row(cluster.cid)])
        [result] = rc.reconcile_due_clusters()
        assert result["ok"] and result["facts_error_code"] and not result["facts_refreshed"]
        with pg_connection() as conn:
            counts = conn.execute(
                "SELECT failure_counts FROM slurm_clusters WHERE cluster_id = %s", (cluster.cid,)
            ).fetchone()[0]
        assert any(k.startswith("facts:") for k in counts)


@pytest.fixture
def cat_profile(tmp_path, monkeypatch):
    """A profile whose wrapper allows cat and ls, not tail (as Alliance's)."""
    import json as _json

    path = tmp_path / "profiles.json"
    path.write_text(
        _json.dumps(
            {
                "profiles": {
                    "catlike": {
                        "name": "cat-only wrapper",
                        "partition_gpu": "g",
                        "partition_cpu": "c",
                        "output_reader": "cat",
                        "scratch_dir": "/scratch/{user}",
                    }
                }
            }
        )
    )
    monkeypatch.setenv(sa.PROFILES_FILE_ENV, str(path))
    return "catlike"


class TestOutputSizeCheck:
    def _pass(self, cluster, profile, fake):
        return rc.reconcile_cluster(cluster.cid, profile, runner=fake)

    def _expire(self, xid):
        with pg_transaction() as conn:  # a poll interval passes
            conn.execute(
                "UPDATE slurm_job_outputs SET fetched_at = fetched_at - interval '5 minutes', "
                "attempted_at = attempted_at - interval '5 minutes' WHERE xcelsior_job_id = %s",
                (xid,),
            )

    def test_an_unchanged_file_is_not_read_again(self, cluster, cat_profile):
        xid = cluster(990, observed_state="running")
        squeue = [cluster.q("990", "RUNNING")]
        sa.read_cached_output(xid)
        first = FakeCluster(squeue=squeue, outputs={("990", "out"): "epoch 1\n"})
        self._pass(cluster, cat_profile, first)
        assert [c[0] for c in first.calls if c[0] in ("ls", "cat", "tail")] == ["ls", "cat"]
        self._expire(xid)
        assert sa.read_cached_output(xid)["refresh_pending"]
        again = FakeCluster(squeue=squeue, outputs={("990", "out"): "epoch 1\n"})
        result = self._pass(cluster, cat_profile, again)
        assert [c[0] for c in again.calls if c[0] in ("ls", "cat", "tail")] == ["ls"]
        assert result.outputs_unchanged == [(xid, "out")]
        got = sa.read_cached_output(xid)
        assert got["content"] == "epoch 1\n" and not got["refresh_pending"]

    def test_a_grown_file_is_read(self, cluster, cat_profile):
        xid = cluster(991, observed_state="running")
        squeue = [cluster.q("991", "RUNNING")]
        sa.read_cached_output(xid)
        self._pass(
            cluster, cat_profile, FakeCluster(squeue=squeue, outputs={("991", "out"): "a\n"})
        )
        self._expire(xid)
        sa.read_cached_output(xid)
        grown = FakeCluster(squeue=squeue, outputs={("991", "out"): "a\nb\n"})
        self._pass(cluster, cat_profile, grown)
        assert [c[0] for c in grown.calls if c[0] in ("ls", "cat")] == ["ls", "cat"]
        assert sa.read_cached_output(xid)["content"] == "a\nb\n"

    def test_past_the_cap_it_is_never_catted(self, cluster, cat_profile):
        xid = cluster(992, observed_state="running")
        sa.read_cached_output(xid)
        huge = "x" * (sa.OUTPUT_CAT_MAX_BYTES + 1)
        fake = FakeCluster(squeue=[cluster.q("992", "RUNNING")], outputs={("992", "out"): huge})
        result = self._pass(cluster, cat_profile, fake)
        assert fake.count("cat") == 0 and result.output_errors == [(xid, "out", "output_too_large")]
        assert sa.read_cached_output(xid)["fetch_error_code"] == "output_too_large"

    def test_a_tail_cluster_needs_no_size_check(self, cluster):
        xid = cluster(993, observed_state="running")
        sa.read_cached_output(xid)
        fake = FakeCluster(squeue=[cluster.q("993", "RUNNING")], outputs={("993", "out"): "a\n"})
        run_pass(cluster, fake)
        assert fake.count("ls") == 0 and fake.count("tail") == 1

    def test_a_settled_unchanged_file_becomes_final(self, cluster, cat_profile):
        xid = cluster(994, observed_state="running")
        sa.read_cached_output(xid)
        self._pass(
            cluster,
            cat_profile,
            FakeCluster(squeue=[cluster.q("994", "RUNNING")], outputs={("994", "out"): "done\n"}),
        )
        settled = FakeCluster(
            sacct=[cluster.a("994", "COMPLETED")], outputs={("994", "out"): "done\n"}
        )
        result = self._pass(cluster, cat_profile, settled)
        assert result.settled == [xid] and result.outputs_unchanged == [(xid, "out")]
        assert sa.read_cached_output(xid)["final"]

    def test_ls_output_that_is_not_a_regular_file_is_refused(self):
        class Lister(FakeCluster):
            def run(self, argv, **kw):
                return subprocess.CompletedProcess(
                    argv, 0, f"lrwxrwxrwx 1 x x 11 Nov 8 12:00 {argv[-1]} -> /etc/shadow\n", ""
                )

        got = sa.get_slurm_job_output_size("5", "j-1", profile_name="generic", runner=Lister())
        assert got["error_code"] == "output_unavailable"


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
