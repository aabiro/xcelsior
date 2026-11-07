# Pass/fail checklist: local Slurm validation

Run date: ____________  Run id: ____________  Adapter version: 3.0.0  Slurm: 25.11.2

Mark each row PASS or FAIL. Automated rows come from `results/<run-id>/checklist.md` (suite) and `30-verify-cluster.sh` (V-rows). Evidence for every suite case is in `results.json` under the same id.

## A. Install and safety

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| I-1 | Tower preflight | `10-install-tower.sh preflight` prints no FAIL lines | ☐ PASS ☐ FAIL | |
| I-2 | ASUS preflight | `20-install-asus.sh preflight` prints no FAIL lines, including reaching 100.64.0.4:6817 | ☐ PASS ☐ FAIL | |
| I-3 | Tower install | `10-install-tower.sh all` ends with "tower services up; boot autostart: off" | ☐ PASS ☐ FAIL | |
| I-4 | ASUS install | `20-install-asus.sh all` ends with "asus services up" | ☐ PASS ☐ FAIL | |
| I-5 | Same IDs on both nodes | `id slurm` → 970 and `id munge` → 971 on both | ☐ PASS ☐ FAIL | |
| I-6 | Manifest complete | `/var/lib/xcl-slurm-test/manifest` on each node lists every package, file, dir, user, ufw rule and key line the install added | ☐ PASS ☐ FAIL | |
| I-7 | Adapter key restricted | `ssh xcl-tower sinfo --version` works; `ssh xcl-tower id` is refused by the gate | ☐ PASS ☐ FAIL | |

## B. Cluster verification (raw Slurm, `30-verify-cluster.sh`)

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| V-1 | One partition spans both nodes | `xcl` lists tower and asus | ☐ PASS ☐ FAIL | |
| V-2 | Nodes schedulable | both idle/mixed/allocated; none down or drained | ☐ PASS ☐ FAIL | |
| V-3 | GPU GRES configured | `scontrol show node tower` → `Gres=gpu:rtx_3060:1` | ☐ PASS ☐ FAIL | |
| V-4 | GPU is a trackable resource | tower `CfgTRES` includes `gres/gpu=1` | ☐ PASS ☐ FAIL | |
| V-5 | ASUS has no GPU | `Gres=(null)` on asus | ☐ PASS ☐ FAIL | |
| V-6 | NVML autodetect | `slurmd -G` shows `Name=gpu Type=rtx_3060 Count=1 … File=/dev/nvidia0` | ☐ PASS ☐ FAIL | |
| V-7 | GPU step sees exactly 1 RTX 3060 | `srun --gres=gpu:1 nvidia-smi -L` → one line, RTX 3060 | ☐ PASS ☐ FAIL | |
| V-8 | GRES enforced | `srun -w tower nvidia-smi -L` without GRES lists no GPU | ☐ PASS ☐ FAIL | |
| V-9 | ASUS runs jobs | `srun -w asus hostname` | ☐ PASS ☐ FAIL | |
| V-10 | Cluster in accounting | `sacctmgr show cluster` lists `xcl-local` | ☐ PASS ☐ FAIL | |
| V-11 | Epoch accounting | `SLURM_TIME_FORMAT=%s sacct` rows carry 10-digit Start/End | ☐ PASS ☐ FAIL | |
| V-12 | Docker and runners untouched | snapshot after install = snapshot at preflight | ☐ PASS ☐ FAIL | |

## C. Adapter validation suite (`tests/validate_adapter.py`)

Everything below goes through `workers/slurm_adapter.py` over the restricted SSH key. No case uses raw `sbatch`.

| ID | Requirement | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| ENV-1 | Adapter sees the cluster | inventory: exactly tower + asus; 1 GPU on tower, none on asus; all schedulable | ☐ PASS ☐ FAIL | |
| GPU-1 | Single-node GPU job via adapter runs `nvidia-smi` and returns output; GRES honoured | job **not pinned**, so placement comes from GRES alone. COMPLETED; accounting `gres/gpu=1`, NodeList=tower; output read back through the adapter shows one GPU line containing "RTX 3060"; `CUDA_VISIBLE_DEVICES` set | ☐ PASS ☐ FAIL | |
| GPU-2 | GRES is enforced, not advisory | CPU-only job pinned to tower: `nvidia-smi -L` lists no GPU | ☐ PASS ☐ FAIL | |
| CPU-1 | CPU job targeted at the ASUS via adapter | COMPLETED; NodeList=asus; 0 GPUs allocated; exit 0:0 | ☐ PASS ☐ FAIL | |
| LIFE-1 | pending / running / completed / cancelled mapping | second GPU job waits: PENDING→queued (Reason Resources); first: RUNNING→running, then COMPLETED→completed; cancelled while pending: CANCELLED→cancelled | ☐ PASS ☐ FAIL | |
| LIFE-2 | failed mapping | `exit 3` → FAILED→failed, ExitCode 3:0 | ☐ PASS ☐ FAIL | |
| LIFE-3 | cancelled while running | adapter cancel → CANCELLED→cancelled; a second cancel → `already_terminal` | ☐ PASS ☐ FAIL | |
| LIFE-4 | timeout mapping | 1-min limit, `sleep 300` → TIMEOUT→failed | ☐ PASS ☐ FAIL | |
| LIFE-5 | sacct path after the controller forgets the job | after MinJobAge: status source = sacct; completed/cancelled/failed still map correctly; raw sacct "CANCELLED by <uid>" → cancelled | ☐ PASS ☐ FAIL | |
| FAIL-1 | Unsatisfiable resource → clean error, not a hang | 2 GPUs: `unsatisfiable_request`, no job id, < 30 s | ☐ PASS ☐ FAIL | |
| FAIL-2 | Bad partition → clean error | `invalid_partition`, no job id, < 30 s | ☐ PASS ☐ FAIL | |
| FAIL-3 | Unsatisfiable memory | 64 GB: `unsatisfiable_request`, no job id | ☐ PASS ☐ FAIL | |
| FAIL-4 | Walltime over limit | refused by the adapter (profile max) **and** by Slurm (partition MaxTime, `EnforcePartLimits=ALL`): `walltime_exceeds_limit` both times | ☐ PASS ☐ FAIL | |
| FAIL-5 | Unknown node | `invalid_nodelist`, no job id | ☐ PASS ☐ FAIL | |
| FAIL-6 | Hostile metadata is inert | a job name with quotes, `$()` and backticks arrives in the job as a literal string; the marker file is never created | ☐ PASS ☐ FAIL | |
| FAIL-7 | Dead transport fails fast | SSH to an unreachable host → `transport_error` in < 20 s | ☐ PASS ☐ FAIL | |
| FAIL-8 | GPU type the cluster lacks | profile asks for `gpu:h100` → GRES error at submit, no job | ☐ PASS ☐ FAIL | |
| FAIL-9 | Cancel an unknown id | `not_found`, although plain `scancel` exits 0 | ☐ PASS ☐ FAIL | |
| METER-1 | Start/end timestamps in the form per-second billing needs | epoch floats; start < end; duration = end − start = ElapsedRaw (±1 s); gpu_seconds = duration × GPUs; the job's own clock lies inside [start, end]; fields match `billing.UsageMeter` | ☐ PASS ☐ FAIL | |
| METER-2 | Edge cases | cancelled while pending: not billable, 0 s; timeout: billed ≈ walltime; cancelled while running: billed to cancel time | ☐ PASS ☐ FAIL | |
| METER-3 | Active metering | running job: started_at set, completed_at unset; meter completed_at = 0.0 | ☐ PASS ☐ FAIL | |
| POD-1 | Cluster as an Xcelsior pod listing, backed by the real cluster | spec built from sinfo + probe jobs (NVML, NIC link speed) + declarations; schema-valid; verification `verified`; node_count 2, GPU 1 × RTX 3060, NVLink domain 1, NIC 1 Gb/s observed | ☐ PASS ☐ FAIL | |
| POD-2 | Overclaims are rejected | same cluster declared at 800 Gb/s → verification `failed` on `nic_speed_declared_not_above_observed` | ☐ PASS ☐ FAIL | |
| SYNC-1 | (optional) Reconcile loop on real jobs | pending job → observed `pending` with no CheckViolation; completed → `completed` with `terminal_at` set; rows removed afterwards | ☐ PASS ☐ FAIL ☐ SKIPPED | |

## D. Rollback

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| R-1 | Dry run | `90-rollback.sh --dry-run` lists only manifest items | ☐ PASS ☐ FAIL | |
| R-2 | Packages | after rollback, `dpkg -l` matches the set before install | ☐ PASS ☐ FAIL | |
| R-3 | Users, files, dirs | no `slurm`, `munge` or `mysql` user; no `/etc/slurm`, `/var/lib/mariadb` or `/var/lib/xcl-slurm-test` | ☐ PASS ☐ FAIL | |
| R-4 | Access | the `xcl-slurm-adapter` line is gone from authorized_keys; the ufw rule is gone (ASUS) | ☐ PASS ☐ FAIL | |
| R-5 | Untouchables | Docker containers and runners are still as in the preflight snapshot | ☐ PASS ☐ FAIL | |

Sign-off: all of A–C PASS (SYNC-1 may be skipped) = local validation complete.
