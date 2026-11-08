# Pass/fail checklist: local Slurm validation

Run date: 2026-11-07  Run id: r1107050537  Adapter version: 3.0.0  Slurm: 25.11.2  Machines: tower + ASUS (real hardware)

Mark each row PASS or FAIL. Automated rows come from `results/<run-id>/checklist.md` (suite) and `30-verify-cluster.sh` (V-rows). Evidence for every suite case is in `results.json` under the same id.

## A. Install and safety

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| I-1 | Tower preflight | `10-install-tower.sh preflight` prints no FAIL lines | ☒ PASS | all 12 checks ok (after fixing a SIGPIPE false failure on the libnvidia-ml check) |
| I-2 | ASUS preflight | `20-install-asus.sh preflight` prints no FAIL lines, including reaching slurmctld on the tower's port 6817 | ☒ PASS | all 10 checks ok |
| I-3 | Tower install | `10-install-tower.sh all` ends with "tower services up; boot autostart: off" | ☒ PASS | 23 packages, users 970/971, services up, boot autostart off |
| I-4 | ASUS install | `20-install-asus.sh all` ends with "asus services up" | ☒ PASS | ufw rule added on tailscale0; services up, boot autostart off |
| I-5 | Same IDs on both nodes | `id slurm` → 970 and `id munge` → 971 on both | ☒ PASS | slurm 970 and munge 971 on both |
| I-6 | Manifest complete | `/var/lib/xcl-slurm-test/manifest` on each node lists every package, file, dir, user, ufw rule and key line the install added | ☒ PASS | tower and ASUS manifests list every package, file, dir, user, rule and key |
| I-7 | Adapter key restricted | `ssh xcl-tower sinfo --version` works; `ssh xcl-tower id` is refused by the gate | ☒ PASS | sinfo allowed; `id` refused by the gate (after moving the alias out of ~/.ssh/config: its `Host *` key bypassed the restriction) |

## B. Cluster verification (raw Slurm, `30-verify-cluster.sh`)

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| V-1 | One partition spans both nodes | `xcl` lists tower and asus | ☒ PASS | asus, tower |
| V-2 | Nodes schedulable | both idle/mixed/allocated; none down or drained | ☒ PASS | both idle |
| V-3 | GPU GRES configured | `scontrol show node tower` → `Gres=gpu:rtx_3060:1` | ☒ PASS | Gres=gpu:rtx_3060:1(S:0) |
| V-4 | GPU is a trackable resource | tower `CfgTRES` includes `gres/gpu=1` | ☒ PASS | CfgTRES=cpu=32,mem=8G,billing=32,gres/gpu=1 |
| V-5 | ASUS has no GPU | `Gres=(null)` on asus | ☒ PASS | Gres=(null) |
| V-6 | NVML autodetect | `slurmd -G` shows `Name=gpu Type=rtx_3060 Count=1 … File=/dev/nvidia0` | ☒ PASS | Type=rtx_3060 Count=1 File=/dev/nvidia0, Flags=HAS_TYPE,ENV_NVML |
| V-7 | GPU step sees exactly 1 RTX 3060 | `srun --gres=gpu:1 nvidia-smi -L` → one line, RTX 3060 | ☒ PASS | GPU 0: NVIDIA GeForce RTX 3060 |
| V-8 | GRES enforced | `srun -w tower nvidia-smi -L` without GRES lists no GPU | ☒ PASS | no GPU line |
| V-9 | ASUS runs jobs | `srun -w asus hostname` | ☒ PASS | asus hostname returned |
| V-10 | Cluster in accounting | `sacctmgr show cluster` lists `xcl-local` | ☒ PASS | xcl-local |
| V-11 | Epoch accounting | `SLURM_TIME_FORMAT=%s sacct` rows carry 10-digit Start/End | ☒ PASS | 10-digit Start/End |
| V-12 | Docker and runners untouched | snapshot after install = snapshot at preflight | ☒ PASS | unchanged after install and again after the full suite (fixed an empty-snapshot bug first) |

## C. Adapter validation suite (`tests/validate_adapter.py`)

Everything below goes through `workers/slurm_adapter.py` over the restricted SSH key. No case uses raw `sbatch`.

| ID | Requirement | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| ENV-1 | Adapter sees the cluster | inventory: exactly tower + asus; 1 GPU on tower, none on asus; all schedulable | ☒ PASS | tower + asus, 1 GPU on tower |
| GPU-1 | Single-node GPU job via adapter runs `nvidia-smi` and returns output; GRES honoured | job **not pinned**, so placement comes from GRES alone. COMPLETED; accounting `gres/gpu=1`, NodeList=tower; output read back through the adapter shows one GPU line containing "RTX 3060"; `CUDA_VISIBLE_DEVICES` set | ☒ PASS | job placed on tower by GRES; output "GPU 0: NVIDIA GeForce RTX 3060", 12288 MiB; CUDA_VISIBLE_DEVICES=0 |
| GPU-2 | GRES is enforced, not advisory | CPU-only job pinned to tower: `nvidia-smi -L` lists no GPU | ☒ PASS | "No devices found." |
| CPU-1 | CPU job targeted at the ASUS via adapter | COMPLETED; NodeList=asus; 0 GPUs allocated; exit 0:0 | ☒ PASS | NodeList=asus, 0 GPUs, exit 0:0; output read on the ASUS |
| LIFE-1 | pending / running / completed / cancelled mapping | second GPU job waits: PENDING→queued (Reason Resources); first: RUNNING→running, then COMPLETED→completed; cancelled while pending: CANCELLED→cancelled | ☒ PASS | PENDING→queued (Resources), RUNNING→running, CANCELLED→cancelled, COMPLETED→completed |
| LIFE-2 | failed mapping | `exit 3` → FAILED→failed, ExitCode 3:0 | ☒ PASS | FAILED→failed, 3:0 |
| LIFE-3 | cancelled while running | adapter cancel → CANCELLED→cancelled; a second cancel → `already_terminal` | ☒ PASS | CANCELLED→cancelled; second cancel → already_terminal |
| LIFE-4 | timeout mapping | 1-min limit, `sleep 300` → TIMEOUT→failed | ☒ PASS | RUNNING→COMPLETING→TIMEOUT→failed |
| LIFE-5 | sacct path after the controller forgets the job | after MinJobAge: status source = sacct; completed/cancelled/failed still map correctly; raw sacct "CANCELLED by <uid>" → cancelled | ☒ PASS | all three from sacct; raw "CANCELLED by 1000" → cancelled |
| FAIL-1 | Unsatisfiable resource → clean error, not a hang | 2 GPUs: `unsatisfiable_request`, no job id, < 30 s | ☒ PASS | unsatisfiable_request, 0.60 s |
| FAIL-2 | Bad partition → clean error | `invalid_partition`, no job id, < 30 s | ☒ PASS | invalid_partition, 0.65 s |
| FAIL-3 | Unsatisfiable memory | 64 GB: `unsatisfiable_request`, no job id | ☒ PASS | unsatisfiable_request, 0.35 s |
| FAIL-4 | Walltime over limit | refused by the adapter (profile max) **and** by Slurm (partition MaxTime, `EnforcePartLimits=ALL`): `walltime_exceeds_limit` both times | ☒ PASS | walltime_exceeds_limit client-side and from Slurm (0.70 s) |
| FAIL-5 | Unknown node | `invalid_nodelist`, no job id | ☒ PASS | invalid_nodelist, 0.38 s |
| FAIL-6 | Hostile metadata is inert | a job name with quotes, `$()` and backticks arrives in the job as a literal string; the marker file is never created | ☒ PASS | literal name echoed; CLEAN |
| FAIL-7 | Dead transport fails fast | SSH to an unreachable host → `transport_error` in < 20 s | ☒ PASS | transport_error, 4.2 s |
| FAIL-8 | GPU type the cluster lacks | profile asks for `gpu:h100` → GRES error at submit, no job | ☒ PASS | unsatisfiable_request, 0.60 s |
| FAIL-9 | Cancel an unknown id | `not_found`, although plain `scancel` exits 0 | ☒ PASS | not_found, 0.69 s |
| METER-1 | Start/end timestamps in the form per-second billing needs | epoch floats; start < end; duration = end − start = ElapsedRaw (±1 s); gpu_seconds = duration × GPUs; the job's own clock lies inside [start, end]; fields match `billing.UsageMeter` | ☒ PASS | start 1794045938, end 1794045944, 6 s, 1 GPU, 6 GPU-s; job clock 1794045939–1794045944 |
| METER-2 | Edge cases | cancelled while pending: not billable, 0 s; timeout: billed ≈ walltime; cancelled while running: billed to cancel time | ☒ PASS | pending-cancel 0 s not billable; timeout 78 s billable; running-cancel 5 s billable |
| METER-3 | Active metering | running job: started_at set, completed_at unset; meter completed_at = 0.0 | ☒ PASS | started_at set, completed_at unset → meter 0.0 |
| POD-1 | Cluster as an Xcelsior pod listing, backed by the real cluster | spec built from sinfo + probe jobs (NVML, NIC link speed) + declarations; schema-valid; verification `verified`; node_count 2, GPU 1 × RTX 3060, NVLink domain 1, NIC 1 Gb/s observed | ☒ PASS | verified: 2 nodes, 1 × RTX 3060 12 GB, NVLink 1, NIC 1 Gb/s on enp38s0 and enp2s0 |
| POD-2 | Overclaims are rejected | same cluster declared at 800 Gb/s → verification `failed` on `nic_speed_declared_not_above_observed` | ☒ PASS | rejected on nic_speed_declared_not_above_observed |
| SYNC-1 | (optional) Reconcile loop on real jobs | pending job → observed `pending` with no CheckViolation; completed → `completed` with `terminal_at` set; rows removed afterwards | ☒ PASS | pending→pending (no CheckViolation); completed + terminal_at; rows removed |

## D. Rollback

| ID | Check | Pass criteria | Result | Evidence / notes |
|---|---|---|---|---|
| R-1 | Dry run | `90-rollback.sh --dry-run` lists only manifest items | not run | cluster kept running; rollback verified in the container rehearsal (package set identical) |
| R-2 | Packages | after rollback, `dpkg -l` matches the set before install | not run | cluster kept running; rollback verified in the container rehearsal (package set identical) |
| R-3 | Users, files, dirs | no `slurm`, `munge` or `mysql` user; no `/etc/slurm`, `/var/lib/mariadb` or `/var/lib/xcl-slurm-test` | not run | cluster kept running; rollback verified in the container rehearsal (package set identical) |
| R-4 | Access | the `xcl-slurm-adapter` line is gone from authorized_keys; the ufw rule is gone (ASUS) | not run | cluster kept running; rollback verified in the container rehearsal (package set identical) |
| R-5 | Untouchables | Docker containers and runners are still as in the preflight snapshot | not run | cluster kept running; rollback verified in the container rehearsal (package set identical) |

Sign-off: all of A–C PASS, SYNC-1 included = **local validation complete (2026-11-07)**. Section D not run: the cluster stays up for further work; run `90-rollback.sh` on each node to remove it.
