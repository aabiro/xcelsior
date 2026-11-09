#!/usr/bin/env python3
"""Xcelsior Slurm adapter validation suite — local two-node cluster.

Drives a real Slurm cluster *through the adapter* (workers/slurm_adapter.py,
never raw sbatch) and records PASS/FAIL per test case with evidence.

Typical run, from the ASUS with the adapter's restricted key going to the
tower's gate:

    python3 hpc/slurm-local/tests/validate_adapter.py

Outputs results/<run-id>/{results.json, checklist.md, pod-listing.json,
outputs/*.txt}. Exit status is 0 only if every executed case passed.

Cases (see CHECKLIST.md for the acceptance criteria of each):
  ENV-1   cluster shape seen through the adapter
  GPU-1   GPU job via adapter runs nvidia-smi; GRES honoured
  GPU-2   CPU-only job on the GPU node cannot see the GPU
  CPU-1   CPU job pinned to the CPU node
  LIFE-1..5  pending / running / completed / failed / cancelled (pending
          and running) / timeout, squeue and sacct paths
  FAIL-1..9  unsatisfiable GPU, bad partition, unsatisfiable memory,
          walltime (client + Slurm side), bad node, hostile metadata,
          dead transport, absent GPU type, unknown id on cancel — each a
          clean error code, none a hang
  METER-1..3 accounting timestamps in billing's UsageMeter shape
  POD-1..2   pod listing spec built from the live cluster, and an overclaim
          rejected
  SEC-1..2   a job can't widen the adapter's key; the gate refuses
          scripts with directives outside the adapter's set
  SYNC-1..2  (optional, --with-db) reconcile loop against real jobs; a
          pass costs one squeue and one sacct however many jobs are open
  REG-1      (optional, --with-db) the registry transport: pinned host key,
          config facts, one ControlMaster, a wrong pin refused
  PROG-1     (optional, --with-db) a job's own progress file, fetched once
  VERDICT-1  (optional, --with-db) success needs DerivedExitCode 0:0
  SIZE-1     (optional, --with-db) with a size check, an unchanged file isn't read
  SCEN-1..6  (optional, --scenarios) requeue, spot preemption, OOM, hold,
          suspend/resume and node failure against the reconciler
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

KIT = pathlib.Path(__file__).resolve().parents[1]
REPO = KIT.parents[1]
sys.path.insert(0, str(REPO / "workers"))

import slurm_adapter as sa  # noqa: E402
import slurm_pod as sp  # noqa: E402

PASS, FAIL, SKIP, ERROR = "PASS", "FAIL", "SKIP", "ERROR"


class Ctx:
    def __init__(self, args):
        self.args = args
        self.profile = args.profile
        self.run_id = time.strftime("r%m%d%H%M%S")
        self.out_dir = pathlib.Path(args.results_dir) / self.run_id
        (self.out_dir / "outputs").mkdir(parents=True, exist_ok=True)
        self.runner = sa.get_runner(sa._get_profile(self.profile))
        self.results: list[dict] = []
        self.jobs: dict[str, dict] = {}  # case-local job registry
        self.gpu_node = args.gpu_node
        self.cpu_node = args.cpu_node
        self.started = ""
        # Results one case leaves for a later case (None until set).
        self.inventory: Any = None
        self.gpu1: Any = None
        self.running_acct: Any = None
        self.life: dict = {}
        self.life_failed: Any = None
        self.life_cancelled_running: Any = None
        self.life_timeout: Any = None
        self.pod: Any = None

    # ── helpers ────────────────────────────────────────────────────
    def job(self, tag: str, **fields) -> dict:
        job = {
            "job_id": f"{self.run_id}-{tag}",
            "name": f"validate-{tag}",
            "vram_needed_gb": 0,
            "priority": 0,
            "tier": "validation",
            "num_gpus": 0,
        }
        job.update(fields)
        return job

    def submit(self, tag: str, **fields) -> dict:
        job = self.job(tag, **fields)
        t0 = time.monotonic()
        res = sa.submit_to_slurm(job, profile_name=self.profile, runner=self.runner)
        res["_elapsed_s"] = round(time.monotonic() - t0, 2)
        res["_job"] = job
        if "slurm_job_id" in res:
            self.jobs[tag] = res
        return res

    def status(self, slurm_id: str) -> dict:
        return sa.get_slurm_job_status(slurm_id, profile_name=self.profile, runner=self.runner)

    def wait(self, slurm_id: str, until: set[str], timeout: float) -> tuple[dict, list]:
        """Poll status until slurm_state ∈ until; return (last, trace)."""
        trace = []
        deadline = time.monotonic() + timeout
        last = {}
        while time.monotonic() < deadline:
            last = self.status(slurm_id)
            st = last.get("slurm_state")
            if not trace or trace[-1][1] != st:
                trace.append(
                    (
                        round(time.monotonic() - deadline + timeout, 1),
                        st,
                        last.get("xcelsior_state"),
                        last.get("source"),
                    )
                )
            if st in until:
                return last, trace
            time.sleep(1)
        return last, trace

    def accounting(self, slurm_id: str, timeout: float = 30, want_terminal: bool = True) -> dict:
        deadline = time.monotonic() + timeout
        acct = {}
        while time.monotonic() < deadline:
            acct = sa.get_slurm_job_accounting(
                slurm_id, profile_name=self.profile, runner=self.runner
            )
            if "error" not in acct and (acct.get("terminal") or not want_terminal):
                return acct
            time.sleep(2)
        return acct

    def output(self, slurm_id: str, xjob: str) -> tuple[str | None, str]:
        res = sa.get_slurm_job_output(slurm_id, xjob, profile_name=self.profile, runner=self.runner)
        if "content" in res:
            return res["content"], "submit-host"
        # No shared filesystem in the local cluster: a job on the node this
        # suite runs on writes its output here.
        if self.args.local_output_dir:
            p = pathlib.Path(self.args.local_output_dir) / f"xcl-{xjob}-{slurm_id}.out"
            for _ in range(10):
                if p.exists():
                    return p.read_text(), "local-node"
                time.sleep(1)
        return None, res.get("error", "unavailable")

    def save_output(self, name: str, text: str | None):
        if text is not None:
            (self.out_dir / "outputs" / f"{name}.txt").write_text(text)

    def cleanup_jobs(self):
        for tag, res in self.jobs.items():
            sid = res["slurm_job_id"]
            st = self.status(sid)
            if st.get("xcelsior_state") in ("queued", "running"):
                sa.cancel_slurm_job(sid, profile_name=self.profile, runner=self.runner)


CASES: list[tuple[str, str, str, object]] = []


def case(cid: str, title: str, requirement: str):
    def deco(fn):
        CASES.append((cid, title, requirement, fn))
        return fn

    return deco


def ok(cond: bool, evidence: dict, note: str = "") -> tuple[str, dict, str]:
    return (PASS if cond else FAIL), evidence, note


def terminal_states():
    return {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }


# ── ENV ─────────────────────────────────────────────────────────────


@case(
    "ENV-1",
    "Cluster shape seen through the adapter",
    "One partition spans both nodes; GPU GRES on the GPU node only",
)
def env_1(ctx: Ctx):
    inv = sa.get_cluster_inventory(ctx.profile, runner=ctx.runner)
    if "error" in inv:
        return FAIL, inv, "inventory call failed"
    nodes = {n["name"]: n for n in inv["nodes"]}
    gpu = nodes.get(ctx.gpu_node, {})
    cpu = nodes.get(ctx.cpu_node, {})
    gpu_gres = sp.parse_gres(gpu.get("gres", ""))
    cpu_gres = sp.parse_gres(cpu.get("gres", ""))
    states_ok = all(
        n["state"].rstrip("*~#").lower() in ("idle", "mixed", "allocated") for n in nodes.values()
    )
    ev = {"partition": inv["partition"], "nodes": inv["nodes"], "gpu_gres": gpu_gres}
    ctx.inventory = inv  # reused by POD-1
    return ok(
        set(nodes) == {ctx.gpu_node, ctx.cpu_node}
        and sum(g["count"] for g in gpu_gres) == 1
        and not cpu_gres
        and states_ok,
        ev,
    )


# ── GPU / CPU placement ─────────────────────────────────────────────

GPU1_CMD = (
    'echo "XCL_START=$(date +%s)"; echo "HOST=$(hostname -s)"; '
    'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; '
    'echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"; '
    "nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
    'sleep 5; echo "XCL_END=$(date +%s)"'
)


@case(
    "GPU-1",
    "GPU job through the adapter runs nvidia-smi; GRES honoured",
    "Not pinned: Slurm must place it on the GPU node by GRES alone",
)
def gpu_1(ctx: Ctx):
    res = ctx.submit("gpu1", num_gpus=1, command=GPU1_CMD)
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 180)
    acct = ctx.accounting(sid)
    out, where = ctx.output(sid, res["_job"]["job_id"])
    ctx.save_output("GPU-1", out)
    lines = (out or "").splitlines()
    gpu_lines = [line for line in lines if line.startswith("GPU ")]
    ev = {
        "slurm_job_id": sid,
        "metadata": res.get("metadata"),
        "state_trace": trace,
        "accounting": acct,
        "output_source": where,
        "nvidia_smi_L": gpu_lines,
    }
    ctx.gpu1 = {"sid": sid, "acct": acct, "out": out}
    return ok(
        last.get("slurm_state") == "COMPLETED"
        and acct.get("gpu_count") == 1
        and acct.get("node_list") == ctx.gpu_node
        and len(gpu_lines) == 1
        and ctx.args.expect_gpu in gpu_lines[0]
        and "CUDA_VISIBLE_DEVICES=unset" not in (out or ""),
        ev,
    )


@case(
    "GPU-2",
    "CPU-only job on the GPU node cannot see the GPU",
    "ConstrainDevices: GRES is enforced, not advisory",
)
def gpu_2(ctx: Ctx):
    cmd = "nvidia-smi -L 2>&1 || true; echo DONE"
    res = ctx.submit("gpu2", num_gpus=0, nodelist=ctx.gpu_node, command=cmd)
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 120)
    out, where = ctx.output(sid, res["_job"]["job_id"])
    ctx.save_output("GPU-2", out)
    ev = {"slurm_job_id": sid, "state_trace": trace, "output": out, "output_source": where}
    return ok(
        last.get("slurm_state") == "COMPLETED"
        and out is not None
        and "DONE" in out
        and not any(l.startswith("GPU ") for l in out.splitlines()),
        ev,
    )


@case("CPU-1", "CPU job targeted at the CPU node", "nodelist honoured; no GPU allocated")
def cpu_1(ctx: Ctx):
    cmd = 'echo "HOST=$(hostname -s)"; echo "NPROC=$(nproc)"; echo "NODELIST=$SLURM_JOB_NODELIST"'
    res = ctx.submit("cpu1", num_gpus=0, nodelist=ctx.cpu_node, command=cmd)
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 120)
    acct = ctx.accounting(sid)
    out, where = ctx.output(sid, res["_job"]["job_id"])
    ctx.save_output("CPU-1", out)
    ev = {
        "slurm_job_id": sid,
        "state_trace": trace,
        "accounting": acct,
        "output": out,
        "output_source": where,
    }
    note = (
        ""
        if out
        else "output not visible from the submit host (no shared filesystem); placement verified from accounting"
    )
    return ok(
        last.get("slurm_state") == "COMPLETED"
        and acct.get("node_list") == ctx.cpu_node
        and acct.get("gpu_count") == 0
        and acct.get("exit_code") == "0:0",
        ev,
        note,
    )


# ── Lifecycle ───────────────────────────────────────────────────────


@case(
    "LIFE-1",
    "Lifecycle: pending / running / completed / cancelled-while-pending",
    "Adapter states match squeue/sacct at each step",
)
def life_1(ctx: Ctx):
    a = ctx.submit("lifeA", num_gpus=1, command="sleep 25")
    if "error" in a:
        return FAIL, a, "submit A failed"
    sa_id = a["slurm_job_id"]
    a_run, a_trace = ctx.wait(sa_id, {"RUNNING"}, 90)
    b = ctx.submit("lifeB", num_gpus=1, command="sleep 5")
    if "error" in b:
        return FAIL, {"A": a_run, "B": b}, "submit B failed"
    sb_id = b["slurm_job_id"]
    time.sleep(3)
    b_pend = ctx.status(sb_id)
    # METER-3 needs an accounting read of a job that is still running.
    ctx.running_acct = ctx.accounting(sa_id, timeout=20, want_terminal=False)
    cancel_b = sa.cancel_slurm_job(sb_id, profile_name=ctx.profile, runner=ctx.runner)
    b_end, b_trace = ctx.wait(sb_id, terminal_states(), 30)
    a_end, a_trace2 = ctx.wait(sa_id, terminal_states(), 90)
    rows = [
        {
            "step": "pending",
            "job": sb_id,
            "slurm": b_pend.get("slurm_state"),
            "adapter": b_pend.get("xcelsior_state"),
            "expect": ("PENDING", "queued"),
            "reason": b_pend.get("reason"),
        },
        {
            "step": "running",
            "job": sa_id,
            "slurm": a_run.get("slurm_state"),
            "adapter": a_run.get("xcelsior_state"),
            "expect": ("RUNNING", "running"),
        },
        {
            "step": "cancelled (pending)",
            "job": sb_id,
            "slurm": b_end.get("slurm_state"),
            "adapter": b_end.get("xcelsior_state"),
            "expect": ("CANCELLED", "cancelled"),
        },
        {
            "step": "completed",
            "job": sa_id,
            "slurm": a_end.get("slurm_state"),
            "adapter": a_end.get("xcelsior_state"),
            "expect": ("COMPLETED", "completed"),
        },
    ]
    for r in rows:
        r["ok"] = (r["slurm"], r["adapter"]) == r["expect"]
    ctx.life = {"A": sa_id, "B": sb_id}
    return ok(
        all(r["ok"] for r in rows) and cancel_b.get("cancelled") is True,
        {"mapping": rows, "cancel_B": cancel_b, "traces": {"A": a_trace + a_trace2, "B": b_trace}},
    )


@case("LIFE-2", "Lifecycle: failed (non-zero exit)", "exit 3 → FAILED → failed, exit code 3:0")
def life_2(ctx: Ctx):
    res = ctx.submit("lifeF", num_gpus=0, command="echo about to fail; exit 3")
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 90)
    acct = ctx.accounting(sid)
    ctx.life_failed = sid
    return ok(
        last.get("slurm_state") == "FAILED"
        and last.get("xcelsior_state") == "failed"
        and acct.get("exit_code") == "3:0",
        {
            "slurm_job_id": sid,
            "state_trace": trace,
            "status": last,
            "exit_code": acct.get("exit_code"),
        },
    )


@case(
    "LIFE-3",
    "Lifecycle: cancelled while running",
    "scancel through the adapter → CANCELLED → cancelled",
)
def life_3(ctx: Ctx):
    res = ctx.submit("lifeC", num_gpus=0, command="sleep 300")
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    ctx.wait(sid, {"RUNNING"}, 60)
    time.sleep(4)
    c = sa.cancel_slurm_job(sid, profile_name=ctx.profile, runner=ctx.runner)
    last, trace = ctx.wait(sid, terminal_states(), 60)
    again = sa.cancel_slurm_job(sid, profile_name=ctx.profile, runner=ctx.runner)
    ctx.life_cancelled_running = sid
    return ok(
        c.get("cancelled") is True
        and last.get("slurm_state") == "CANCELLED"
        and last.get("xcelsior_state") == "cancelled"
        and again.get("error_code") == "already_terminal",
        {"slurm_job_id": sid, "cancel": c, "state_trace": trace, "second_cancel": again},
        f"second cancel → {again.get('error_code', 'cancelled')}",
    )


@case("LIFE-4", "Lifecycle: walltime exceeded", "time limit 1 min, sleep 300 → TIMEOUT → failed")
def life_4(ctx: Ctx):
    res = ctx.submit("lifeT", num_gpus=0, command="sleep 300", walltime="1")
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 200)
    ctx.life_timeout = sid
    return ok(
        last.get("slurm_state") == "TIMEOUT" and last.get("xcelsior_state") == "failed",
        {"slurm_job_id": sid, "state_trace": trace, "status": last},
    )


@case(
    "LIFE-5",
    "sacct fallback after the controller forgets the job",
    "After MinJobAge, status comes from sacct with the same mapping; 'CANCELLED by <uid>' normalised",
)
def life_5(ctx: Ctx):
    ids = {
        "completed": getattr(ctx, "life", {}).get("A"),
        "cancelled": getattr(ctx, "life", {}).get("B"),
        "failed": getattr(ctx, "life_failed", None),
    }
    if not all(ids.values()):
        return SKIP, ids, "needs LIFE-1 and LIFE-2"
    deadline = time.monotonic() + ctx.args.minjobage_wait
    rows = {}
    while time.monotonic() < deadline:
        rows = {k: ctx.status(str(v)) for k, v in ids.items()}
        if all(r.get("source") == "sacct" for r in rows.values()):
            break
        time.sleep(10)
    raw_cancel = sa.get_slurm_job_accounting(
        ids["cancelled"], profile_name=ctx.profile, runner=ctx.runner
    ).get("raw_state")
    good = all(r.get("source") == "sacct" for r in rows.values()) and all(
        rows[k].get("xcelsior_state") == k for k in rows
    )
    return ok(
        good,
        {"statuses": rows, "sacct_raw_state_for_cancelled": raw_cancel},
        "" if good else "jobs still in squeue — raise --minjobage-wait or check MinJobAge",
    )


# ── Failure paths ───────────────────────────────────────────────────


def _expect_error(ctx: Ctx, tag: str, codes: set[str], **fields):
    res = ctx.submit(tag, **fields)
    ev = {k: v for k, v in res.items() if k != "script"}
    good = "slurm_job_id" not in res and res.get("error_code") in codes and res["_elapsed_s"] < 30
    return ok(good, ev, f"error_code={res.get('error_code')} in {res['_elapsed_s']}s")


@case(
    "FAIL-1", "Unsatisfiable GPU count", "2 GPUs on a 1-GPU cluster → clean error at submit, no job"
)
def fail_1(ctx: Ctx):
    return _expect_error(
        ctx, "fail1", {"unsatisfiable_request", "invalid_gres"}, num_gpus=2, command="true"
    )


@case("FAIL-2", "Bad partition", "partition 'nosuchpart' → invalid_partition")
def fail_2(ctx: Ctx):
    return _expect_error(
        ctx, "fail2", {"invalid_partition"}, partition="nosuchpart", command="true"
    )


@case("FAIL-3", "Unsatisfiable memory", "64 GB on nodes offering 8/4 GB → unsatisfiable_request")
def fail_3(ctx: Ctx):
    return _expect_error(ctx, "fail3", {"unsatisfiable_request"}, mem_gb=64, command="true")


@case(
    "FAIL-4",
    "Walltime over the limit",
    "client-side (profile max) and Slurm-side (partition MaxTime, EnforcePartLimits=ALL)",
)
def fail_4(ctx: Ctx):
    client = ctx.submit("fail4a", walltime="02:00:00", command="true")
    # Same request through a looser profile, so it reaches sbatch.
    base = json.loads(pathlib.Path(os.environ[sa.PROFILES_FILE_ENV]).read_text())
    loose = dict(base["profiles"][ctx.profile], max_walltime="1-00:00:00")
    base["profiles"]["xcl-local-loose"] = loose
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(base, fh)
    old = os.environ[sa.PROFILES_FILE_ENV]
    os.environ[sa.PROFILES_FILE_ENV] = fh.name
    try:
        t0 = time.monotonic()
        server = sa.submit_to_slurm(
            ctx.job("fail4b", walltime="02:00:00", command="true"),
            profile_name="xcl-local-loose",
            runner=ctx.runner,
        )
        server["_elapsed_s"] = round(time.monotonic() - t0, 2)
    finally:
        os.environ[sa.PROFILES_FILE_ENV] = old
        os.unlink(fh.name)
    ev = {"client_side": client, "slurm_side": {k: v for k, v in server.items() if k != "script"}}
    return ok(
        client.get("error_code") == "walltime_exceeds_limit"
        and "slurm_job_id" not in server
        and server.get("error_code") == "walltime_exceeds_limit"
        and server["_elapsed_s"] < 30,
        ev,
    )


@case("FAIL-5", "Unknown node", "nodelist 'nosuchnode' → invalid_nodelist")
def fail_5(ctx: Ctx):
    return _expect_error(ctx, "fail5", {"invalid_nodelist"}, nodelist="nosuchnode", command="true")


HOSTILE = 'x"; touch /tmp/xcl-pwned-MARK; echo "$(id)`id`'


@case(
    "FAIL-6",
    "Hostile job metadata is inert",
    "A job name with quotes, $(), backticks reaches the job as a literal string",
)
def fail_6(ctx: Ctx):
    name = HOSTILE.replace("MARK", ctx.run_id)
    cmd = f'printf "NAME=%s\\n" "$XCELSIOR_JOB_NAME"; test -e /tmp/xcl-pwned-{ctx.run_id} && echo PWNED || echo CLEAN'
    res = ctx.submit("fail6", name=name, nodelist=ctx.gpu_node, command=cmd)
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, _ = ctx.wait(sid, terminal_states(), 90)
    out, where = ctx.output(sid, res["_job"]["job_id"])
    ctx.save_output("FAIL-6", out)
    return ok(
        last.get("slurm_state") == "COMPLETED"
        and out is not None
        and f"NAME={name}" in out
        and "CLEAN" in out,
        {"slurm_job_id": sid, "output": out, "expected_name": name},
    )


@case(
    "FAIL-7",
    "Dead transport fails fast",
    "SSH to an unreachable submit host → transport_error in seconds",
)
def fail_7(ctx: Ctx):
    if not ctx.runner.remote:
        return SKIP, {}, "suite is running on the submit host (local transport)"
    dead = sa.SlurmRunner(
        ssh_host="aaryn@192.0.2.1", ssh_options=ctx.runner.ssh_options, connect_timeout=4
    )
    t0 = time.monotonic()
    res = sa.submit_to_slurm(
        ctx.job("fail7", command="true"), profile_name=ctx.profile, runner=dead
    )
    el = round(time.monotonic() - t0, 2)
    ev = {k: v for k, v in res.items() if k != "script"}
    ev["_elapsed_s"] = el
    return ok(res.get("error_code") == "transport_error" and el < 20, ev)


def _with_profile_variant(ctx: Ctx, key: str, **overrides):
    """Write a temp profiles file with a variant of the run's profile."""
    base = json.loads(pathlib.Path(os.environ[sa.PROFILES_FILE_ENV]).read_text())
    base["profiles"][key] = dict(base["profiles"][ctx.profile], **overrides)
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(base, fh)
    fh.close()
    return fh.name


@case(
    "FAIL-8",
    "GPU type the cluster does not have",
    "Profile asks for gpu:h100 on an RTX cluster → clean GRES error at submit (the old Nibi-profile failure mode)",
)
def fail_8(ctx: Ctx):
    path = _with_profile_variant(ctx, "xcl-local-h100", gpu_type="h100")
    old = os.environ[sa.PROFILES_FILE_ENV]
    os.environ[sa.PROFILES_FILE_ENV] = path
    try:
        t0 = time.monotonic()
        res = sa.submit_to_slurm(
            ctx.job("fail8", num_gpus=1, command="true"),
            profile_name="xcl-local-h100",
            runner=ctx.runner,
        )
        res["_elapsed_s"] = round(time.monotonic() - t0, 2)
    finally:
        os.environ[sa.PROFILES_FILE_ENV] = old
        os.unlink(path)
    ev = {k: v for k, v in res.items() if k != "script"}
    return ok(
        "slurm_job_id" not in res
        and res.get("error_code") in {"invalid_gres", "unsatisfiable_request"}
        and res["_elapsed_s"] < 30,
        ev,
        f"error_code={res.get('error_code')}",
    )


@case(
    "FAIL-9",
    "Cancel an id Slurm has never seen",
    "scancel exits 0 silently for unknown ids; the adapter must still say not_found",
)
def fail_9(ctx: Ctx):
    t0 = time.monotonic()
    res = sa.cancel_slurm_job("987654321", profile_name=ctx.profile, runner=ctx.runner)
    el = round(time.monotonic() - t0, 2)
    return ok(
        res.get("error_code") == "not_found" and not res.get("cancelled") and el < 30,
        {**res, "_elapsed_s": el},
    )


# ── Security ────────────────────────────────────────────────────────


@case(
    "SEC-1",
    "A job can't widen the adapter's key",
    "Jobs run as the service account, can't write its ~/.ssh or ~/.bashrc, and the key still runs only the gate",
)
def sec_1(ctx: Ctx):
    want_user = sa._get_profile(ctx.profile).get("remote_user") or ""
    cmd = (
        'echo "USER=$(id -un)"; '
        'mkdir -p "$HOME/.ssh" 2>&1 && echo key >>"$HOME/.ssh/authorized_keys" 2>&1 && echo WROTE_KEYS; '
        'echo id >>"$HOME/.bashrc" 2>&1 && echo WROTE_BASHRC; true'
    )
    # On the gate's own node: that is where sshd would read those files.
    res = ctx.submit("sec1", num_gpus=0, nodelist=ctx.gpu_node, command=cmd)
    if "error" in res:
        return FAIL, res, "submit failed"
    sid = res["slurm_job_id"]
    last, trace = ctx.wait(sid, terminal_states(), 120)
    out, where = ctx.output(sid, res["_job"]["job_id"])
    ctx.save_output("SEC-1", out)
    probe: dict = {}
    if ctx.runner.remote:
        r = ctx.runner.run(["id"], timeout=15)
        probe = {"returncode": r.returncode, "stderr": r.stderr.strip()[:200]}
    out = out or ""
    ev = {
        "slurm_job_id": sid,
        "state_trace": trace,
        "output": out,
        "output_source": where,
        "id_probe": probe,
    }
    return ok(
        last.get("slurm_state") == "COMPLETED"
        and f"USER={want_user}" in out
        and "WROTE_KEYS" not in out
        and "WROTE_BASHRC" not in out
        and (not ctx.runner.remote or "xcl-slurm-gate: refused" in probe.get("stderr", "")),
        ev,
    )


@case(
    "SEC-2",
    "The gate checks batch scripts",
    "Scripts with --output aimed at ~/.ssh, or with --chdir, are refused before sbatch runs",
)
def sec_2(ctx: Ctx):
    if not ctx.runner.remote:
        return SKIP, {}, "suite is running on the submit host (local transport, no gate)"
    script, _ = sa.xcelsior_job_to_sbatch(
        ctx.job("sec2", command="true"), ctx.profile, runner=ctx.runner
    )
    head, _, body = script.partition("\n")
    variants = {
        "chdir": f"{head}\n#SBATCH --chdir=/tmp\n{body}",
        "output_to_ssh": "\n".join(
            "#SBATCH --output=/var/lib/xcl-slurm-test/svc-home/.ssh/authorized_keys"
            if line.startswith("#SBATCH --output=")
            else line
            for line in script.splitlines()
        ),
    }
    ev = {}
    for name, text in variants.items():
        r = ctx.runner.run(["sbatch", "--parsable"], input_text=text, timeout=20)
        ev[name] = {
            "returncode": r.returncode,
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip()[:200],
        }
    return ok(
        all(
            v["returncode"] == 126 and not v["stdout"] and "refused" in v["stderr"]
            for v in ev.values()
        ),
        ev,
    )


# ── Metering ────────────────────────────────────────────────────────


@case(
    "METER-1",
    "Accounting timestamps in billing's shape",
    "Epoch start/end, duration = end-start = ElapsedRaw, gpu_seconds = duration × GPUs, job's own clock inside [start, end]",
)
def meter_1(ctx: Ctx):
    g = getattr(ctx, "gpu1", None)
    if not g or "error" in g["acct"]:
        return SKIP, {"gpu1": g}, "needs GPU-1"
    acct, out = g["acct"], g["out"] or ""
    meter = sa.accounting_to_meter_fields(acct)
    vals: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith(("XCL_START=", "XCL_END=")):
            key, val = line.split("=", 1)
            vals[key] = val
    in_start, in_end = int(vals.get("XCL_START", "0")), int(vals.get("XCL_END", "0"))
    s, e = acct["started_at"], acct["completed_at"]
    checks = {
        "epoch_floats": all(isinstance(meter[k], float) for k in meter),
        "start_lt_end": bool(s and e and s < e),
        "duration_eq_end_minus_start": bool(s and e and abs((e - s) - acct["duration_sec"]) <= 1),
        "gpu_seconds_eq_duration_x_gpus": acct["gpu_seconds"]
        == acct["duration_sec"] * acct["gpu_count"],
        "job_clock_inside_window": bool(s and e and s - 2 <= in_start <= in_end <= e + 2),
        "submitted_le_started": bool(acct["submitted_at"] and s and acct["submitted_at"] <= s),
    }
    example_rate = 0.50  # CAD/hr, illustrative
    ev = {
        "accounting": acct,
        "meter_fields": meter,
        "in_job_clock": {"start": in_start, "end": in_end},
        "checks": checks,
        "example_charge": {
            "rate_per_hour": example_rate,
            "formula": "duration_sec / 3600 * rate (billing.UsageMeter path)",
            "amount": round(meter["duration_sec"] / 3600 * example_rate, 6),
        },
    }
    return ok(all(checks.values()), ev)


@case(
    "METER-2",
    "Non-billable and edge cases",
    "cancelled-while-pending → not billable; timeout ≈ walltime; cancelled-running billed to cancel time",
)
def meter_2(ctx: Ctx):
    b = getattr(ctx, "life", {}).get("B")
    t = getattr(ctx, "life_timeout", None)
    c = getattr(ctx, "life_cancelled_running", None)
    if not (b and t and c):
        return SKIP, {}, "needs LIFE-1, LIFE-3, LIFE-4"
    ab, at, ac = ctx.accounting(b), ctx.accounting(t), ctx.accounting(c)
    checks = {
        "pending_cancel_not_billable": ab.get("billable") is False and ab.get("duration_sec") == 0,
        "timeout_billed_about_walltime": at.get("billable") is True
        and 55 <= at.get("duration_sec", 0) <= 150,
        "running_cancel_billed": ac.get("billable") is True and 0 < ac.get("duration_sec", 0) < 120,
    }
    return ok(
        all(checks.values()),
        {"cancelled_pending": ab, "timeout": at, "cancelled_running": ac, "checks": checks},
    )


@case(
    "METER-3",
    "Active metering of a running job",
    "started_at set, completed_at unset; meter completed_at = 0.0",
)
def meter_3(ctx: Ctx):
    acct = getattr(ctx, "running_acct", None)
    if not acct:
        return SKIP, {}, "needs LIFE-1"
    meter = sa.accounting_to_meter_fields(acct) if "error" not in acct else {}
    return ok(
        "error" not in acct
        and acct.get("started_at") is not None
        and acct.get("completed_at") is None
        and meter.get("completed_at") == 0.0,
        {"accounting": acct, "meter_fields": meter},
    )


# ── Pod data model ──────────────────────────────────────────────────


def _probe(ctx: Ctx, tag: str, node: str, gpus: int, command: str):
    res = ctx.submit(tag, num_gpus=gpus, nodelist=node, command=command)
    if "error" in res:
        return None, None
    sid = res["slurm_job_id"]
    ctx.wait(sid, terminal_states(), 120)
    out, _ = ctx.output(sid, res["_job"]["job_id"])
    return sid, out


@case(
    "POD-1",
    "Pod listing backed by the real cluster",
    "Spec built from sinfo + probe jobs + declarations; validates; verification passes",
)
def pod_1(ctx: Ctx):
    inv = getattr(ctx, "inventory", None) or sa.get_cluster_inventory(
        ctx.profile, runner=ctx.runner
    )
    if "error" in inv:
        return FAIL, inv, "inventory failed"
    gpu_probes, nic_probes = {}, {}
    for n in inv["nodes"]:
        count = sum(g["count"] for g in sp.parse_gres(n["gres"]))
        if count:
            sid, out = _probe(ctx, f"podg-{n['name']}", n["name"], count, sp.GPU_PROBE_COMMAND)
            ctx.save_output(f"POD-1-gpu-{n['name']}", out)
            gpu_probes[n["name"]] = {"probe": sp.parse_gpu_probe(out or ""), "slurm_job_id": sid}
        sid, out = _probe(ctx, f"podn-{n['name']}", n["name"], 0, sp.NIC_PROBE_COMMAND)
        ctx.save_output(f"POD-1-nic-{n['name']}", out)
        nic_probes[n["name"]] = {"probe": sp.parse_nic_probe(out or ""), "slurm_job_id": sid}
    declared = json.loads((KIT / "conf" / "pod-declared.json").read_text())
    ver = ctx.runner.run(["sinfo", "--version"], timeout=15).stdout.strip()
    spec = sp.build_pod_spec(
        cluster_id=sa._get_profile(ctx.profile)["cluster_id"],
        partition=inv["partition"],
        inventory=inv,
        gpu_probes=gpu_probes,
        nic_probes=nic_probes,
        declared=declared,
        scheduler_version=ver,
    )
    (ctx.out_dir / "pod-listing.json").write_text(json.dumps(spec, indent=2))
    errors = sp.validate_pod_spec(spec)
    ctx.pod = {"inv": inv, "gpu": gpu_probes, "nic": nic_probes, "declared": declared}
    summary = {
        "node_count": spec["node_count"]["value"],
        "gpu_count": spec["gpu"]["count"]["value"],
        "gpu_model": spec["gpu"]["model"]["value"],
        "gpu_memory_gb": spec["gpu"]["memory_gb"]["value"],
        "nvlink_domain_size": spec["gpu"]["nvlink_domain_size"]["value"],
        "nic_speed_gbps_observed": spec["interconnect"]["nic_speed_gbps"]["value"],
        "interconnect_type": spec["interconnect"]["type"]["value"],
        "verification": spec["verification"]["status"],
        "failed_checks": [c for c in spec["verification"]["checks"] if not c["ok"]],
    }
    return ok(
        not errors and spec["verification"]["status"] == "verified",
        {"summary": summary, "schema_errors": errors, "file": "pod-listing.json"},
    )


@case(
    "POD-2",
    "An overclaimed listing is rejected",
    "Same cluster, facility declares 800 Gb/s NICs → verification fails",
)
def pod_2(ctx: Ctx):
    p = getattr(ctx, "pod", None)
    if not p:
        return SKIP, {}, "needs POD-1"
    declared = json.loads(json.dumps(p["declared"]))
    declared["interconnect"]["nic_speed_gbps"] = 800
    spec = sp.build_pod_spec(
        cluster_id="xcl-local",
        partition=p["inv"]["partition"],
        inventory=p["inv"],
        gpu_probes=p["gpu"],
        nic_probes=p["nic"],
        declared=declared,
    )
    failed = [c for c in spec["verification"]["checks"] if not c["ok"]]
    return ok(
        spec["verification"]["status"] == "failed"
        and any(c["check"] == "nic_speed_declared_not_above_observed" for c in failed),
        {"failed_checks": failed},
    )


# ── Reconcile loop (optional) ───────────────────────────────────────


def _test_db_guard(ctx: Ctx):
    """None when a dev/test database is safely configured, else a SKIP result."""
    if not ctx.args.with_db:
        return SKIP, {}, "run with --with-db against a dev/test database (never prod)"
    # Never touch a real database: .env must not load, and the DSN's
    # database name must say test or dev. Use .env.test (xcelsior_pytest).
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN", "")
    dbname = dsn.rsplit("/", 1)[-1].split("?")[0]
    if os.environ.get("XCELSIOR_SKIP_DOTENV") != "1" or not any(
        t in dbname for t in ("test", "dev")
    ):
        return (
            SKIP,
            {"database": dbname or None},
            (
                "refusing: export the .env.test variables and XCELSIOR_SKIP_DOTENV=1 "
                "(database name must contain 'test' or 'dev')"
            ),
        )
    sys.path[:0] = [str(REPO)] + [
        str(REPO / p) for p in ("core", "api", "auth", "billing", "hosts", "privacy")
    ]
    atexit.register(_close_db_pool)
    return None


def _close_db_pool():
    """Close the pool the DB cases opened, else psycopg_pool complains at exit."""
    db = sys.modules.get("db")
    pool = getattr(db, "_pg_pool", None)
    if pool is not None:
        pool.close()


@case(
    "SYNC-1",
    "Reconcile loop against real jobs (optional)",
    "Pending job → observed 'pending' (no CheckViolation); completed → terminal_at stamped",
)
def sync_1(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    from db import pg_connection, pg_transaction  # type: ignore

    a = ctx.submit("syncA", num_gpus=1, command="sleep 20")
    ctx.wait(a["slurm_job_id"], {"RUNNING"}, 60)
    b = ctx.submit("syncB", num_gpus=1, command="true")
    ids = {"syncA": a["slurm_job_id"], "syncB": b["slurm_job_id"]}
    # Registered under the id the adapter named the job by (xcl-<id>), as
    # production registers it; the reconciler matches jobs by that name.
    xids = [a["xcelsior_job_id"], b["xcelsior_job_id"]]
    try:
        for (k, sid), xid in zip(ids.items(), xids):
            sa.register_slurm_job(xid, sid, cluster_id="xcl-local")
        first = sa.sync_slurm_statuses()
        with pg_connection() as conn:
            rows1 = conn.execute(
                "SELECT xcelsior_job_id, observed_state FROM slurm_job_mappings WHERE xcelsior_job_id = ANY(%s)",
                (xids,),
            ).fetchall()
        ctx.wait(ids["syncB"], terminal_states(), 120)
        second = sa.sync_slurm_statuses()
        rows2 = _mapping_rows(pg_connection, xids)
        # A finished job settles only once it has left the controller
        # (MinJobAge) and accounting agrees; until then it is open.
        _wait_gone_from_controller(ctx, [ids["syncB"]])
        third = sa.sync_slurm_statuses()
        rows3 = _mapping_rows(pg_connection, xids)
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = ANY(%s)", (xids,))
    obs1 = {r[0]: r[1] for r in rows1}
    obs2 = {r[0]: (r[1], r[2]) for r in rows2}
    obs3 = {r[0]: (r[1], r[2]) for r in rows3}
    return ok(
        obs1.get(xids[0]) == "running"
        and obs1.get(xids[1]) == "pending"
        and obs2.get(xids[1], ("",))[0] == "completed"
        and obs3.get(xids[1]) == ("completed", True),
        {
            "first_pass": first,
            "after_first": obs1,
            "second_pass": second,
            "after_second": obs2,
            "third_pass": third,
            "after_third": obs3,
        },
        ""
        if obs2.get(xids[1], ("", True))[1] is False
        else "B had already left the controller at the second pass",
    )


def _wait_gone_from_controller(ctx: Ctx, slurm_ids: list[str]) -> None:
    """Wait until squeue -t all no longer lists these jobs (MinJobAge).

    Not ctx.status(): its `squeue -j <id>` hides a finished job at once,
    while the controller still holds it, which is what -t all shows.
    """
    deadline = time.monotonic() + ctx.args.minjobage_wait
    while time.monotonic() < deadline:
        held = sa.poll_cluster_jobs(profile_name=ctx.profile, runner=ctx.runner).get("jobs", {})
        if not any(sid in held for sid in slurm_ids):
            return
        time.sleep(5)


def _mapping_rows(pg_connection, xids):
    with pg_connection() as conn:
        return conn.execute(
            "SELECT xcelsior_job_id, observed_state, terminal_at IS NOT NULL "
            "FROM slurm_job_mappings WHERE xcelsior_job_id = ANY(%s)",
            (xids,),
        ).fetchall()


def _site_admin_ssh() -> str:
    values = {}
    site = KIT / "site.env"
    if site.exists():
        for line in site.read_text().splitlines():
            key, _, val = line.partition("=")
            values[key.strip()] = val.split("#", 1)[0].strip()
    if values.get("OPERATOR_USER") and values.get("TOWER_ADDR"):
        return f"{values['OPERATOR_USER']}@{values['TOWER_ADDR']}"
    return ""


class _CountingRunner:
    """Wraps the suite's runner to count the commands a reconcile pass sends."""

    def __init__(self, inner):
        self.inner = inner
        self.remote = inner.remote
        self.ssh_host = inner.ssh_host
        self.sent: list[str] = []

    def run(self, argv, **kw):
        self.sent.append(argv[0])
        return self.inner.run(argv, **kw)


@case(
    "SYNC-2",
    "A reconcile pass is batched (optional)",
    "Five open jobs, one running and four finished and gone: one squeue and one sacct for the whole pass, every finished job settled",
)
def sync_2(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    from db import pg_connection, pg_transaction  # type: ignore
    import slurm_reconciler as rc  # noqa: E402

    run = ctx.submit("rec-run", num_gpus=0, command="sleep 600")  # cancelled at the end
    done = [ctx.submit(f"rec-done{i}", num_gpus=0, command="true") for i in range(4)]
    jobs = [run, *done]
    if any("error" in j for j in jobs):
        return FAIL, {"submits": jobs}, "submit failed"
    cid = f"{ctx.run_id}-rec"
    xids = [j["xcelsior_job_id"] for j in jobs]  # the ids the jobs are named by
    try:
        for xid, j in zip(xids, jobs):
            sa.register_slurm_job(xid, j["slurm_job_id"], cluster_id=cid)
        ctx.wait(run["slurm_job_id"], {"RUNNING"}, 60)
        _wait_gone_from_controller(ctx, [j["slurm_job_id"] for j in done])
        counting = _CountingRunner(ctx.runner)
        result = rc.reconcile_cluster(cid, ctx.profile, runner=counting)
        rows = {r[0]: (r[1], r[2]) for r in _mapping_rows(pg_connection, xids)}
        sa.cancel_slurm_job(run["slurm_job_id"], profile_name=ctx.profile, runner=ctx.runner)
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
    ev = {"commands_sent": counting.sent, "pass": result.as_dict(), "rows": rows}
    return ok(
        counting.sent == ["squeue", "sacct"]
        and rows[xids[0]] == ("running", False)
        and all(rows[x] == ("completed", True) for x in xids[1:]),
        ev,
    )


def _adapter_ssh(ctx: Ctx) -> dict:
    """HostName, User and IdentityFile from the profile's own -F config."""
    conf = None
    opts = list(ctx.runner.ssh_options)
    if "-F" in opts:
        conf = pathlib.Path(opts[opts.index("-F") + 1])
    found = {}
    for line in (conf.read_text().splitlines() if conf and conf.exists() else []):
        key, _, val = line.strip().partition(" ")
        if key in ("HostName", "User", "IdentityFile"):
            found[key] = val.strip()
    return found


def _known_host_line(host: str) -> str | None:
    """The host's key as this account already trusts it (~/.ssh/known_hosts)."""
    out = subprocess.run(["ssh-keygen", "-F", host], capture_output=True, text=True).stdout
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    for preferred in ("ssh-ed25519", ""):
        for ln in lines:
            if preferred in ln:
                return ln
    return None


@case(
    "REG-1",
    "Reconciling through the cluster registry (optional)",
    "A registered cluster with a pinned host key: the connection check stores every config fact, passes share one ControlMaster, and a wrong pin is refused as host_key_mismatch",
)
def reg_1(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    import base64
    from db import pg_transaction  # type: ignore
    import slurm_reconciler as rc  # noqa: E402
    import slurm_transport as st  # noqa: E402

    ssh = _adapter_ssh(ctx)
    pin = _known_host_line(ssh.get("HostName", ""))
    if not ssh.get("IdentityFile") or not pin:
        return FAIL, {"adapter_ssh": sorted(ssh)}, "no adapter ssh config or no known_hosts entry"
    cid, bad = f"{ctx.run_id}-reg", f"{ctx.run_id}-badpin"
    wrong = "h ssh-ed25519 " + base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + os.urandom(32)).decode()
    common = {"ssh_host": ssh["HostName"], "remote_user": ssh.get("User")}
    ev: dict = {}
    job = ctx.submit("reg-a", num_gpus=0, command="sleep 20")
    if "error" in job:
        return FAIL, {"submit": job}, "submit failed"
    try:
        rc.register_cluster(
            ctx.profile, cluster_id=cid, known_hosts_line=pin,
            credential_ref=f"file:{ssh['IdentityFile']}", **common,
        )
        ev["check"] = rc.check_cluster_connection(cid)
        sa.register_slurm_job(job["xcelsior_job_id"], job["slurm_job_id"], cluster_id=cid)
        ctx.wait(job["slurm_job_id"], {"RUNNING"}, 60)
        row = rc._cluster_row(cid)
        if row is None:
            return FAIL, ev, "the registered cluster row is missing"
        with st.cluster_runner(row) as runner:
            counting = _CountingRunner(runner)
            first = rc.reconcile_cluster(cid, ctx.profile, runner=counting)
            ev["master_after_first_pass"] = st.master_running(runner)
        with st.cluster_runner(row) as runner:
            # A later use finds the master already up, before sending anything.
            ev["master_before_second_pass"] = st.master_running(runner)
            second = rc.reconcile_cluster(cid, ctx.profile, runner=runner)
        ev["passes"] = [first.as_dict(), second.as_dict()]
        ev["commands_first_pass"] = counting.sent
        rc.register_cluster(
            ctx.profile, cluster_id=bad, known_hosts_line=wrong,
            credential_ref=f"file:{ssh['IdentityFile']}", **common,
        )
        ev["wrong_pin"] = rc.check_cluster_connection(bad)
        with st.cluster_runner(row) as runner:
            st.close_master(runner)
    finally:
        sa.cancel_slurm_job(job["slurm_job_id"], profile_name=ctx.profile, runner=ctx.runner)
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
            conn.execute("DELETE FROM slurm_clusters WHERE cluster_id = ANY(%s)", ([cid, bad],))
    check = ev["check"]
    ev["check"] = {k: check.get(k) for k in ("ok", "missing", "error_code")} | {
        "facts": sorted(check.get("facts", {}))
    }
    return ok(
        check.get("ok") is True
        and check.get("missing") == []
        and ev["master_after_first_pass"]
        and ev["master_before_second_pass"]
        and first.ok and second.ok
        and ev["passes"][0]["in_controller"] == 1
        and ev["wrong_pin"].get("error_code") == "host_key_mismatch",
        ev,
    )


@case(
    "PROG-1",
    "A job reports its own progress (optional)",
    "The job writes XCELSIOR_PROGRESS_FILE per the contract; one reconcile pass fetches it through the gate and the read shows it, reported by the job",
)
def prog_1(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    from db import pg_transaction  # type: ignore
    import slurm_reconciler as rc  # noqa: E402

    # Written as the contract says: a temporary file, then a rename.
    command = (
        'printf \'{"fraction": 0.5, "message": "half way", "updated_at": %s}\' "$(date +%s)" '
        '> "$XCELSIOR_PROGRESS_FILE.tmp" && mv "$XCELSIOR_PROGRESS_FILE.tmp" '
        '"$XCELSIOR_PROGRESS_FILE" && sleep 40'
    )
    # On the controller host: the gate reads files there (no shared filesystem).
    job = ctx.submit("prog", num_gpus=0, nodelist=ctx.gpu_node, command=command)
    if "error" in job:
        return FAIL, {"submit": job}, "submit failed"
    cid, xid = f"{ctx.run_id}-prog", job["xcelsior_job_id"]
    try:
        sa.register_slurm_job(xid, job["slurm_job_id"], cluster_id=cid)
        ctx.wait(job["slurm_job_id"], {"RUNNING"}, 60)
        time.sleep(5)  # let the job write it
        before = sa.read_reported_progress(xid)  # files the request
        result = rc.reconcile_cluster(cid, ctx.profile, runner=ctx.runner)
        after = sa.read_reported_progress(xid)
    finally:
        sa.cancel_slurm_job(job["slurm_job_id"], profile_name=ctx.profile, runner=ctx.runner)
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
    ev = {"before": before, "fetched": result.outputs_fetched, "errors": result.output_errors, "after": after}
    return ok(
        before is None
        and (xid, "progress") in [tuple(f) for f in result.outputs_fetched]
        and bool(after)
        and after.get("valid") is True
        and after.get("fraction") == 0.5
        and after.get("message") == "half way"
        and after.get("source") == "job"
        and after.get("stale") is False,
        ev,
    )


@case(
    "VERDICT-1",
    "The success verdict reads DerivedExitCode (optional)",
    "Two jobs exit 0; in one a step failed first. Settled: the first succeeded, the second failed on DerivedExitCode 1:0",
)
def verdict_1(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    from db import pg_connection, pg_transaction  # type: ignore
    import slurm_reconciler as rc  # noqa: E402

    good = ctx.submit("v-ok", num_gpus=0, command="srun true")
    hidden = ctx.submit("v-step", num_gpus=0, command="srun false; exit 0")
    if "error" in good or "error" in hidden:
        return FAIL, {"submits": [good, hidden]}, "submit failed"
    cid = f"{ctx.run_id}-verdict"
    xids = [good["xcelsior_job_id"], hidden["xcelsior_job_id"]]
    try:
        for j in (good, hidden):
            sa.register_slurm_job(j["xcelsior_job_id"], j["slurm_job_id"], cluster_id=cid)
        _wait_gone_from_controller(ctx, [good["slurm_job_id"], hidden["slurm_job_id"]])
        rc.reconcile_cluster(cid, ctx.profile, runner=ctx.runner)
        with pg_connection() as conn:
            rows = {
                r[0]: r[1:]
                for r in conn.execute(
                    "SELECT xcelsior_job_id, observed_state, exit_code, derived_exit_code, "
                    "terminal_at IS NOT NULL FROM slurm_job_mappings WHERE xcelsior_job_id = ANY(%s)",
                    (xids,),
                ).fetchall()
            }
    finally:
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
    verdicts = {x: sa.success_verdict(*rows[x]) for x in xids if x in rows}
    ev = {"rows": rows, "verdicts": verdicts}
    return ok(
        verdicts.get(xids[0]) == "succeeded"
        and verdicts.get(xids[1]) == "failed"
        and rows[xids[1]][1] == "0:0"
        and rows[xids[1]][2] not in (None, "0:0"),
        ev,
    )


@case(
    "SIZE-1",
    "Output is read only when it changed (optional)",
    "With output_size_check on, a pass asks ls -l first: the first read sends ls and tail, a second pass over an unchanged file sends ls alone",
)
def size_1(ctx: Ctx):
    refused = _test_db_guard(ctx)
    if refused:
        return refused
    from db import pg_transaction  # type: ignore
    import slurm_reconciler as rc  # noqa: E402

    # As on Alliance, where the wrapper allows ls and cat: a size check
    # before every read. This gate reads with tail, so tail does the read.
    base = json.loads(pathlib.Path(os.environ[sa.PROFILES_FILE_ENV]).read_text())
    base["profiles"]["xcl-local-sizecheck"] = dict(
        base["profiles"][ctx.profile], output_size_check=True
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(base, fh)
    old = os.environ[sa.PROFILES_FILE_ENV]
    os.environ[sa.PROFILES_FILE_ENV] = fh.name
    job = ctx.submit("size", num_gpus=0, nodelist=ctx.gpu_node, command="echo first; sleep 60")
    if "error" in job:
        os.environ[sa.PROFILES_FILE_ENV] = old
        os.unlink(fh.name)
        return FAIL, {"submit": job}, "submit failed"
    cid, xid = f"{ctx.run_id}-size", job["xcelsior_job_id"]
    try:
        sa.register_slurm_job(xid, job["slurm_job_id"], cluster_id=cid)
        ctx.wait(job["slurm_job_id"], {"RUNNING"}, 60)
        time.sleep(3)
        sa.read_cached_output(xid)
        first = _CountingRunner(ctx.runner)
        one = rc.reconcile_cluster(cid, "xcl-local-sizecheck", runner=first)
        with pg_transaction() as conn:  # a poll interval passes
            conn.execute(
                "UPDATE slurm_job_outputs SET fetched_at = fetched_at - interval '5 minutes', "
                "attempted_at = attempted_at - interval '5 minutes' WHERE xcelsior_job_id = %s",
                (xid,),
            )
        sa.read_cached_output(xid)
        second = _CountingRunner(ctx.runner)
        two = rc.reconcile_cluster(cid, "xcl-local-sizecheck", runner=second)
        cached = sa.read_cached_output(xid)
    finally:
        os.environ[sa.PROFILES_FILE_ENV] = old
        os.unlink(fh.name)
        sa.cancel_slurm_job(job["slurm_job_id"], profile_name=ctx.profile, runner=ctx.runner)
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (cid,))
    reads = lambda sent: [c for c in sent if c in ("ls", "tail", "cat")]  # noqa: E731
    ev = {
        "first_pass": reads(first.sent),
        "second_pass": reads(second.sent),
        "fetched": one.outputs_fetched,
        "unchanged": two.outputs_unchanged,
        "errors": one.output_errors + two.output_errors,
        "content": cached.get("content"),
    }
    return ok(
        ev["first_pass"] == ["ls", "tail"]
        and ev["second_pass"] == ["ls"]
        and [tuple(u) for u in two.outputs_unchanged] == [(xid, "out")]
        and cached.get("content") == "first\n",
        ev,
    )


# ── Scenarios: real Slurm events against the reconciler ───────────
#
# Opt-in (--scenarios, with --with-db): each case submits through the adapter
# (the job belongs to the service account), makes a Slurm event happen with
# an admin command on the controller host (`sudo -n scontrol` over your own
# SSH login, --admin-ssh), and checks what the reconciler records at each
# step. The rule they all test: a mapping settles only when its job has left
# the controller and its latest accounting run is finished. Job-id reuse is
# staged with a scripted cluster instead (tests/test_slurm_reconciler.py,
# TestJobIdReuse): renumbering needs a controller state wipe.


class _Scenario:
    """One scenario's job, mapping and reconciler, cleaned up on exit."""

    def __init__(self, ctx: Ctx, tag: str):
        self.ctx = ctx
        self.cluster = f"{ctx.run_id}-{tag}"
        self.trace: list = []
        self.jobs: list[str] = []

    def __enter__(self):
        import slurm_reconciler  # noqa: F401  (path set up by _test_db_guard)

        return self

    def __exit__(self, *exc):
        from db import pg_transaction  # type: ignore

        for sid in self.jobs:
            sa.cancel_slurm_job(sid, profile_name=self.ctx.profile, runner=self.ctx.runner)
        with pg_transaction() as conn:
            conn.execute("DELETE FROM slurm_job_mappings WHERE cluster_id = %s", (self.cluster,))
        return False

    def submit(self, tag: str, **fields) -> tuple[str, str]:
        res = self.ctx.submit(tag, **fields)
        if "slurm_job_id" not in res:
            raise RuntimeError(f"submit {tag} failed: {res}")
        sid = res["slurm_job_id"]
        xid = res["xcelsior_job_id"]  # the id the adapter named the job by
        sa.register_slurm_job(xid, sid, cluster_id=self.cluster)
        self.jobs.append(sid)
        return sid, xid

    def admin(self, *args: str) -> str:
        import subprocess

        r = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                self.ctx.args.admin_ssh,
                "sudo",
                "-n",
                "scontrol",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.trace.append(("admin", " ".join(args), r.returncode, r.stderr.strip()[:200]))
        if r.returncode != 0:
            raise RuntimeError(f"scontrol {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout

    def reconcile(self, label: str) -> dict:
        import slurm_reconciler as rc

        result = rc.reconcile_cluster(
            self.cluster, self.ctx.profile, runner=self.ctx.runner, cancel=False
        )
        rows = self.rows()
        self.trace.append((label, result.ok, rows))
        return rows

    def rows(self) -> dict:
        from db import pg_connection  # type: ignore

        with pg_connection() as conn:
            return {
                r[0].rsplit("-", 1)[-1]: (r[1], r[2], r[3])
                for r in conn.execute(
                    "SELECT xcelsior_job_id, observed_state, slurm_state, terminal_at IS NOT NULL "
                    "FROM slurm_job_mappings WHERE cluster_id = %s",
                    (self.cluster,),
                ).fetchall()
            }

    def wait_state(self, sid: str, states: set[str], timeout: float) -> str:
        """Wait until squeue -t all shows one of `states` (or the job is gone, ``GONE``)."""
        deadline = time.monotonic() + timeout
        st = ""
        while time.monotonic() < deadline:
            jobs = sa.poll_cluster_jobs(profile_name=self.ctx.profile, runner=self.ctx.runner).get(
                "jobs", {}
            )
            st = jobs[sid]["state"] if sid in jobs else "GONE"
            if st in states:
                return st
            time.sleep(3)
        return st

    def runs(self, sid: str) -> list[str]:
        acct = sa.accounting_window(
            [sid], since_minutes=120, profile_name=self.ctx.profile, runner=self.ctx.runner
        )
        return [rec["state"] for rec in acct.get("records", {}).get(sid, [])]

    def settle(self, sid: str, label: str = "settled") -> dict:
        self.wait_state(sid, {"GONE"}, self.ctx.args.minjobage_wait)
        return self.reconcile(label)


def _scenario_guard(ctx: Ctx):
    if not ctx.args.scenarios:
        return (
            SKIP,
            {},
            "run with --scenarios (and --with-db): needs admin scontrol on the controller host",
        )
    return _test_db_guard(ctx)


def _scenario(ctx: Ctx, tag: str, body):
    refused = _scenario_guard(ctx)
    if refused:
        return refused
    with _Scenario(ctx, tag) as sc:
        try:
            good, note = body(sc)
        except Exception as e:
            return FAIL, {"trace": sc.trace, "error": repr(e)}, repr(e)
        return ok(good, {"trace": sc.trace}, note)


@case(
    "SCEN-1",
    "Requeue keeps the mapping open",
    "scontrol requeue on a running job: pending again, not settled; it runs again and settles completed with two runs in accounting",
)
def scen_1(ctx: Ctx):
    def body(sc):
        sid, _ = sc.submit("req", num_gpus=0, nodelist=ctx.gpu_node, command="sleep 20")
        sc.wait_state(sid, {"RUNNING"}, 90)
        running = sc.reconcile("running")["req"]
        sc.admin("requeue", sid)
        sc.wait_state(sid, {"PENDING", "REQUEUED"}, 30)
        requeued = sc.reconcile("requeued")["req"]
        sc.wait_state(sid, {"COMPLETED"}, 400)
        settled = sc.settle(sid)["req"]
        runs = sc.runs(sid)
        sc.trace.append(("accounting runs", runs))
        good = (
            running[0] == "running"
            and requeued[0] == "pending"
            and not requeued[2]
            and settled == ("completed", "COMPLETED", True)
            and len(runs) >= 2
            and runs[-1] == "COMPLETED"
        )
        return good, f"accounting runs: {runs}"

    return _scenario(ctx, "scen1", body)


@case(
    "SCEN-2",
    "Spot preemption requeues, never settles early",
    "a spot (QoS) GPU job preempted by a normal one goes back to pending, not settled; it reruns and settles with a PREEMPTED run before the final one",
)
def scen_2(ctx: Ctx):
    def body(sc):
        hog, _ = sc.submit("hog", num_gpus=1, command="sleep 600")
        sc.wait_state(hog, {"RUNNING"}, 90)
        spot, _ = sc.submit("spot", num_gpus=1, command="sleep 30")
        sc.wait_state(spot, {"PENDING"}, 30)
        sc.admin("update", f"jobid={spot}", "qos=spot")
        sa.cancel_slurm_job(hog, profile_name=ctx.profile, runner=ctx.runner)
        sc.wait_state(spot, {"RUNNING"}, 120)
        before = sc.reconcile("spot running")["spot"]
        normal, _ = sc.submit("normal", num_gpus=1, command="sleep 5")
        sc.wait_state(spot, {"PENDING", "REQUEUED", "PREEMPTED"}, 180)
        preempted = sc.reconcile("spot preempted")["spot"]
        sc.wait_state(spot, {"COMPLETED"}, 400)
        settled = sc.settle(spot)["spot"]
        runs = sc.runs(spot)
        sc.trace.append(("accounting runs", runs))
        good = (
            before[0] == "running"
            and preempted[0] in ("pending", "preempted")
            and not preempted[2]
            and settled == ("completed", "COMPLETED", True)
            and "PREEMPTED" in runs[:-1]
            and runs[-1] == "COMPLETED"
        )
        return good, f"accounting runs: {runs}"

    return _scenario(ctx, "scen2", body)


@case(
    "SCEN-3",
    "Out of memory settles as failed",
    "a job allocating past its --mem is killed by the cgroup: settled failed, OUT_OF_MEMORY",
)
def scen_3(ctx: Ctx):
    def body(sc):
        hog = 'python3 -c "b = bytearray(3 * 1024 ** 3)"'
        sid, _ = sc.submit("oom", num_gpus=0, nodelist=ctx.gpu_node, mem_gb=1, command=hog)
        sc.wait_state(sid, {"OUT_OF_MEMORY", "FAILED", "COMPLETED"}, 120)
        settled = sc.settle(sid)["oom"]
        return settled == ("failed", "OUT_OF_MEMORY", True), f"settled as {settled}"

    return _scenario(ctx, "scen3", body)


@case(
    "SCEN-4",
    "A hold is held, not failed",
    "requeuehold State=SpecialExit: held and open; released, it runs again",
)
def scen_4(ctx: Ctx):
    def body(sc):
        sid, _ = sc.submit("hold", num_gpus=0, nodelist=ctx.gpu_node, command="sleep 300")
        sc.wait_state(sid, {"RUNNING"}, 90)
        sc.admin("requeuehold", "State=SpecialExit", sid)
        sc.wait_state(sid, {"SPECIAL_EXIT"}, 60)
        held = sc.reconcile("held")["hold"]
        sc.admin("release", sid)
        sc.wait_state(sid, {"RUNNING"}, 400)
        again = sc.reconcile("released")["hold"]
        sa.cancel_slurm_job(sid, profile_name=ctx.profile, runner=ctx.runner)
        settled = sc.settle(sid)["hold"]
        good = (
            held == ("held", "SPECIAL_EXIT", False)
            and again[0] == "running"
            and settled[0] == "cancelled"
            and settled[2]
        )
        return good, f"held {held}, released {again[0]}, settled {settled}"

    return _scenario(ctx, "scen4", body)


@case(
    "SCEN-5",
    "Suspension is suspended, not pending",
    "scontrol suspend: suspended and open; resumed, it is running again",
)
def scen_5(ctx: Ctx):
    def body(sc):
        sid, _ = sc.submit("susp", num_gpus=0, nodelist=ctx.gpu_node, command="sleep 300")
        sc.wait_state(sid, {"RUNNING"}, 90)
        sc.admin("suspend", sid)
        sc.wait_state(sid, {"SUSPENDED"}, 30)
        suspended = sc.reconcile("suspended")["susp"]
        sc.admin("resume", sid)
        sc.wait_state(sid, {"RUNNING"}, 30)
        resumed = sc.reconcile("resumed")["susp"]
        sa.cancel_slurm_job(sid, profile_name=ctx.profile, runner=ctx.runner)
        settled = sc.settle(sid)["susp"]
        good = (
            suspended == ("suspended", "SUSPENDED", False)
            and resumed[0] == "running"
            and settled[2]
        )
        return good, f"suspended {suspended}, resumed {resumed[0]}, settled {settled}"

    return _scenario(ctx, "scen5", body)


@case(
    "SCEN-6",
    "A node failure never settles the job early",
    "slurmd on the CPU node killed under a running job: while the node is down the mapping stays open; accounting records NODE_FAIL",
)
def scen_6(ctx: Ctx):
    import subprocess

    def body(sc):
        sid, _ = sc.submit("nf", num_gpus=0, nodelist=ctx.cpu_node, command="sleep 900")
        sc.wait_state(sid, {"RUNNING"}, 90)
        sc.reconcile("running")
        kill = subprocess.run(
            ["sudo", "-n", "systemctl", "kill", "-s", "KILL", "slurmd"],
            capture_output=True,
            text=True,
        )
        sc.trace.append(("kill slurmd on this node", kill.returncode, kill.stderr.strip()))
        try:
            # SlurmdTimeout is 120 s: the controller declares the node DOWN
            # and the job NODE_FAIL, then requeues it (pending, node absent).
            st = sc.wait_state(sid, {"PENDING", "REQUEUED", "NODE_FAIL"}, 300)
            during = sc.reconcile("node down")["nf"]
        finally:
            start = subprocess.run(
                ["sudo", "-n", "systemctl", "start", "slurmd"], capture_output=True, text=True
            )
            sc.trace.append(("start slurmd", start.returncode, start.stderr.strip()))
        runs = sc.runs(sid)
        sc.trace.append(("accounting runs", runs))
        good = st in ("PENDING", "REQUEUED", "NODE_FAIL") and not during[2] and "NODE_FAIL" in runs
        return good, f"state {st}, mapping {during}, runs {runs}"

    return _scenario(ctx, "scen6", body)


# ── Runner ──────────────────────────────────────────────────────────


def write_checklist(ctx: Ctx):
    lines = [
        f"# Adapter validation run {ctx.run_id}",
        "",
        f"- Profile: `{ctx.profile}`  ·  transport: {'ssh ' + str(ctx.runner.ssh_host) if ctx.runner.remote else 'local'}",
        f"- Adapter: {sa.ADAPTER_VERSION}  ·  started {ctx.started}  ·  finished {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Case | Title | Result | Note |",
        "|---|---|---|---|",
    ]
    for r in ctx.results:
        lines.append(
            f"| {r['id']} | {r['title']} | **{r['status']}** | {r['note'].replace('|', '/')} |"
        )
    counts = {s: sum(1 for r in ctx.results if r["status"] == s) for s in (PASS, FAIL, SKIP, ERROR)}
    lines += ["", f"Totals: {counts}", "", "Evidence for every case: results.json."]
    (ctx.out_dir / "checklist.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--profile", default="xcl-local")
    ap.add_argument("--profiles-file", default=str(KIT / "conf" / "xcelsior-profiles.json"))
    ap.add_argument("--gpu-node", default="tower")
    ap.add_argument("--cpu-node", default="asus")
    ap.add_argument(
        "--expect-gpu", default="RTX 3060", help="substring of nvidia-smi -L on the GPU node"
    )
    ap.add_argument(
        "--local-output-dir",
        default="/var/lib/xcl-slurm-test/jobs",
        help="where jobs on THIS node write output",
    )
    ap.add_argument("--results-dir", default=str(KIT / "results"))
    ap.add_argument("--minjobage-wait", type=float, default=240)
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument(
        "--with-db", action="store_true", help="also run SYNC-1 (needs a dev/test Xcelsior DB)"
    )
    ap.add_argument(
        "--scenarios",
        action="store_true",
        help="also run SCEN-* (needs --with-db and admin scontrol on the controller host)",
    )
    ap.add_argument(
        "--admin-ssh",
        default=_site_admin_ssh(),
        help="your own SSH login on the controller host (default: OPERATOR_USER@TOWER_ADDR from site.env)",
    )
    args = ap.parse_args()

    os.environ.setdefault(sa.PROFILES_FILE_ENV, args.profiles_file)
    ctx = Ctx(args)
    ctx.started = time.strftime("%Y-%m-%d %H:%M:%S")
    only = {c.strip() for c in args.only.split(",") if c.strip()}
    print(
        f"run {ctx.run_id} · profile {ctx.profile} · transport {'ssh ' + str(ctx.runner.ssh_host) if ctx.runner.remote else 'local'}"
    )
    try:
        for cid, title, req, fn in CASES:
            if only and cid not in only:
                continue
            t0 = time.monotonic()
            try:
                status, ev, note = fn(ctx)  # type: ignore[operator]
            except Exception as e:  # a crash is a result, not a stop
                status, ev, note = ERROR, {"traceback": traceback.format_exc()}, repr(e)
            dur = round(time.monotonic() - t0, 1)
            ctx.results.append(
                {
                    "id": cid,
                    "title": title,
                    "requirement": req,
                    "status": status,
                    "note": note,
                    "duration_s": dur,
                    "evidence": ev,
                }
            )
            print(
                f"  {status:5} {cid:8} {title}  ({dur}s){'  — ' + note if note else ''}", flush=True
            )
    finally:
        ctx.cleanup_jobs()
        (ctx.out_dir / "results.json").write_text(
            json.dumps(
                {
                    "run_id": ctx.run_id,
                    "profile": ctx.profile,
                    "adapter_version": sa.ADAPTER_VERSION,
                    "results": ctx.results,
                },
                indent=2,
                default=str,
            )
        )
        write_checklist(ctx)
    bad = [r for r in ctx.results if r["status"] in (FAIL, ERROR)]
    print(f"\n{len(ctx.results) - len(bad)} ok / {len(bad)} failed · {ctx.out_dir}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
