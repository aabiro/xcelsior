#!/usr/bin/env python3
"""Drive the Slurm shadow week on staging (hpc/IMPLEMENTATION_PLAN.md, 2.3 exit).

The 2.3 exit needs a week of shadow billing on xcl-local with GPU jobs placed
through the scheduler, at least one requeued and one suspended job, and daily
drift under 1 %. Staging's bg-worker places (``tick_slurm_only``), submits,
polls and records drift; this script only supplies the work and the two
operator events:

- every ``tick`` queues one GPU job (``nvidia-smi -L; sleep``) through
  ``scheduler.submit_job`` with ``hpc_cluster_id=xcl-local``, so it is admitted,
  claimed and placed exactly as a tenant job would be. No owner: no quote, no
  wallet hold, and nothing is ever charged;
- once a day it queues one job, waits for it to run, and has the operator
  ``scontrol requeue`` it; and one more that the operator suspends for 90 s and
  resumes. Slurm lets only an operator suspend; the service account cannot.

The jobs hold the tower's GPU GRES while they sleep and compute nothing, so
the shared RTX 3060 keeps serving everything else on it.

Two halves. On the host (stdlib only, run by cron):

    python3 hpc/slurm-local/shadow_week.py tick      # queue work, run due events
    python3 hpc/slurm-local/shadow_week.py drift     # one line per day

``tick`` calls the container half through ``docker exec`` on the staging
bg-worker, whose environment is staging's:

    python /app/hpc/slurm-local/shadow_week.py inside-submit --kind plain
    python /app/hpc/slurm-local/shadow_week.py inside-state <job_id>
    python /app/hpc/slurm-local/shadow_week.py inside-drift

State and a log live in ~/.xcelsior/shadow-week/. The operator login is
``OPERATOR_USER@TOWER_ADDR`` from the git-ignored site.env beside this file.
Never touches the tower's Docker containers or runners: the only remote
commands are ``squeue`` and ``sudo -n scontrol requeue|suspend|resume <id>``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time

CLUSTER = "xcl-local"
GPU_MODEL = "RTX 3060"
CONTAINER = os.environ.get("XCELSIOR_SHADOW_CONTAINER", "xcelsior-staging-bg-worker-1")
IN_IMAGE = "/app/hpc/slurm-local/shadow_week.py"
STATE_DIR = pathlib.Path(os.environ.get("XCELSIOR_SHADOW_DIR", "~/.xcelsior/shadow-week")).expanduser()
KIT = pathlib.Path(__file__).resolve().parent

PLAIN_SECONDS = 240
EVENT_SECONDS = 300
SUSPEND_SECONDS = 90
RUN_BEFORE_EVENT = 60  # let the run accrue time before requeue/suspend
WAIT_RUNNING = 30 * 60  # one GPU: an event job can queue behind a plain one


# ── Container half (staging environment, platform code on the path) ──


def inside_submit(kind: str) -> None:
    import scheduler

    seconds = PLAIN_SECONDS if kind == "plain" else EVENT_SECONDS
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%m%dT%H%M")
    job = scheduler.submit_job(
        name=f"shadow-{kind}-{stamp}",
        vram_needed_gb=0,
        num_gpus=1,
        gpu_model=GPU_MODEL,
        command=f"nvidia-smi -L; sleep {seconds}",
        hpc_cluster_id=CLUSTER,
    )
    print(json.dumps({"job_id": job["job_id"], "name": job["name"], "kind": kind}))


def inside_state(job_id: str) -> None:
    from db import pg_connection

    with pg_connection() as conn:
        sub = conn.execute(
            "SELECT state, slurm_job_id, error_code FROM slurm_submissions "
            "WHERE xcelsior_job_id = %s ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        mapping = conn.execute(
            "SELECT slurm_job_id, observed_state, slurm_state, terminal_at IS NOT NULL "
            "FROM slurm_job_mappings WHERE xcelsior_job_id = %s ORDER BY created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        reason = conn.execute(
            "SELECT status, reason_code FROM jobs WHERE job_id = %s", (job_id,)
        ).fetchone()
    print(
        json.dumps(
            {
                "job_status": reason[0] if reason else None,
                "reason_code": reason[1] if reason else None,
                "submission": sub[0] if sub else None,
                "submission_error": sub[2] if sub else None,
                "slurm_job_id": (mapping[0] if mapping else None) or (sub[1] if sub else None),
                "observed_state": mapping[1] if mapping else None,
                "slurm_state": mapping[2] if mapping else None,
                "settled": bool(mapping[3]) if mapping else False,
            }
        )
    )


def inside_drift() -> None:
    from db import pg_connection

    with pg_connection() as conn:
        rows = conn.execute(
            """
            SELECT day, drift_pct, sacct_gpu_seconds, ledger_gpu_seconds, sacct_records,
                   COALESCE(jsonb_array_length(unmatched->'unattributed'), 0),
                   COALESCE(jsonb_array_length(unmatched->'only_in_sacct'), 0),
                   COALESCE(jsonb_array_length(unmatched->'differing'), 0)
              FROM slurm_billing_drift WHERE cluster_id = %s ORDER BY day
            """,
            (CLUSTER,),
        ).fetchall()
        # Drift days are UTC: the xcl-local profile names no timezone (day_window).
        runs = dict(
            conn.execute(
                """
                SELECT (ended_at AT TIME ZONE 'UTC')::date,
                       count(*) FILTER (
                           WHERE slurm_state LIKE 'REQUEUE%%' OR billing_rule LIKE '%%requeued%%'
                       )::text || ' requeued, '
                       || count(*) FILTER (WHERE suspended_sec > 0)::text || ' suspended'
                  FROM slurm_billing_segments WHERE cluster_id = %s GROUP BY 1
                """,
                (CLUSTER,),
            ).fetchall()
        )
    for day, pct, sacct, ledger, n, kit, missing, differing in rows:
        flag = "OK " if float(pct) < 1.0 else "OVER"
        print(
            f"{day}  {flag} drift {float(pct):.2f}%  sacct {sacct} GPU-s  ledger {ledger} GPU-s  "
            f"platform runs {n} ({runs.get(day, '0 requeued, 0 suspended')})  "
            f"missing {missing}  differing {differing}  kit runs {kit}"
        )


# ── Host half ─────────────────────────────────────────────────────────


def _log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (STATE_DIR / "log").open("a") as f:
        f.write(line + "\n")


def _state() -> dict:
    path = STATE_DIR / "state.json"
    return json.loads(path.read_text()) if path.is_file() else {"days": {}}


def _save(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_DIR / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(STATE_DIR / "state.json")


def _inside(*args: str) -> dict | str:
    r = subprocess.run(
        ["docker", "exec", CONTAINER, "python", IN_IMAGE, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = [line for line in r.stdout.splitlines() if line and not line.startswith('{"event"')]
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: rc={r.returncode} {r.stderr.strip()[-400:]}")
    if args[0] == "inside-drift":
        return "\n".join(out)
    return json.loads(out[-1])


def _operator() -> str:
    values = {}
    site = KIT / "site.env"
    if site.is_file():
        for line in site.read_text().splitlines():
            key, _, val = line.partition("=")
            values[key.strip()] = val.split("#", 1)[0].strip()
    if not (values.get("OPERATOR_USER") and values.get("TOWER_ADDR")):
        raise RuntimeError(f"{site} needs OPERATOR_USER and TOWER_ADDR")
    return f"{values['OPERATOR_USER']}@{values['TOWER_ADDR']}"


def _tower(*args: str) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", _operator(), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def _slurm_state(slurm_job_id: str) -> str:
    # squeue -j <one id> is fine for a single id (the multi-id form is what
    # loads the whole table). Empty once the job has left the controller.
    return _tower("squeue", "-h", "-j", slurm_job_id, "-o", "%T")


def _wait_running(job_id: str) -> str:
    """The Slurm id once the job has run RUN_BEFORE_EVENT seconds."""
    deadline = time.monotonic() + WAIT_RUNNING
    while time.monotonic() < deadline:
        s = _inside("inside-state", job_id)
        assert isinstance(s, dict)
        if s.get("submission") == "rejected" or s.get("reason_code") in (
            "slurm_placement_disabled",
            "slurm_cluster_disabled",
        ):
            raise RuntimeError(f"{job_id} will not run: {s}")
        sid = s.get("slurm_job_id")
        if sid and _slurm_state(str(sid)) == "RUNNING":
            time.sleep(RUN_BEFORE_EVENT)
            if _slurm_state(str(sid)) == "RUNNING":
                return str(sid)
        time.sleep(15)
    raise RuntimeError(f"{job_id} did not reach RUNNING within {WAIT_RUNNING}s")


def _event(kind: str) -> list:
    job = _inside("inside-submit", "--kind", kind)
    assert isinstance(job, dict)
    _log(f"{kind}: queued {job['job_id']} ({job['name']})")
    sid = _wait_running(job["job_id"])
    if kind == "requeue":
        _tower("sudo", "-n", "scontrol", "requeue", sid)
        _log(f"requeue: operator requeued Slurm job {sid} ({job['job_id']})")
    else:
        _tower("sudo", "-n", "scontrol", "suspend", sid)
        _log(f"suspend: operator suspended Slurm job {sid} ({job['job_id']})")
        time.sleep(SUSPEND_SECONDS)
        _tower("sudo", "-n", "scontrol", "resume", sid)
        _log(f"suspend: operator resumed Slurm job {sid} after {SUSPEND_SECONDS}s")
    return [job["job_id"], sid]


def tick() -> int:
    running = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if running != "true":
        _log(f"skip: {CONTAINER} is not running; the shadow loop is down")
        return 1
    state = _state()
    state.setdefault("started", dt.date.today().isoformat())
    today = state["days"].setdefault(dt.date.today().isoformat(), {})
    failures = 0
    for kind in ("requeue", "suspend"):
        if kind in today:
            continue
        try:
            today[kind] = _event(kind)
        except Exception as e:  # one failed event must not stop the week
            failures += 1
            _log(f"{kind}: FAILED {e}")
        _save(state)
    try:
        job = _inside("inside-submit", "--kind", "plain")
        assert isinstance(job, dict)
        today.setdefault("plain", []).append(job["job_id"])
        _log(f"plain: queued {job['job_id']} ({job['name']})")
    except Exception as e:
        failures += 1
        _log(f"plain: FAILED {e}")
    _save(state)
    return 1 if failures else 0


def drift() -> int:
    text = _inside("inside-drift")
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick")
    sub.add_parser("drift")
    s = sub.add_parser("inside-submit")
    s.add_argument("--kind", choices=("plain", "requeue", "suspend"), default="plain")
    st = sub.add_parser("inside-state")
    st.add_argument("job_id")
    sub.add_parser("inside-drift")
    args = p.parse_args()
    if args.cmd == "tick":
        return tick()
    if args.cmd == "drift":
        return drift()
    if args.cmd == "inside-submit":
        inside_submit(args.kind)
    elif args.cmd == "inside-state":
        inside_state(args.job_id)
    elif args.cmd == "inside-drift":
        inside_drift()
    return 0


if __name__ == "__main__":
    sys.exit(main())
