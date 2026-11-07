# Slurm adapter and pod tier: production implementation plan

Status as of 2026-11-07. Owner: Aaryn Biro. Audience: whoever implements it, human or agent.

**The claim this plan makes true:** "Slurm schedules GPUs inside a cluster. Xcelsior schedules across clusters." Xcelsior is the meta-scheduler: one control plane that submits into Slurm-run clusters, Kubernetes facilities and independent hosts through one marketplace. Public wording stays "Slurm-native, integration-ready" until a partner route lands.

The local two-node validation kit is in [slurm-local/](slurm-local/README.md). Phase 1 below is that kit.

---

## 1. Where things stand (verified)

### 1.1 What existed before this change (adapter v2.2)

`workers/slurm_adapter.py` turned a job dict into an sbatch script and shelled out to `sbatch`, `squeue`, `sacct` and `scancel` **on whatever machine ran it**. It was reachable from an admin-only API route (`/api/slurm/*` in `api/routes/health.py`), the CLI (`xcelsior slurm-submit`) and the admin HPC dashboard page. It had never run against a real Slurm controller. Reading it and then running it against one found 25 defects; the full list is in the appendix. The ones that matter most:

- It could not express a CPU job, a payload command or a node target. It could not return output or report start/end times. Four of the six local validation cases were impossible to write against it.
- **Reconcile crash.** Any PENDING job raised `CheckViolation` and aborted the whole reconcile pass. The adapter wrote its job state `queued` into `slurm_job_mappings.observed_state`, whose constraint (migration 067) does not allow it. A regression test fails on the old code with exactly that error and passes on the new code.
- **Command injection.** Job name, tier, extra env and image were interpolated raw into the generated script. Through the admin API, that was command injection on the target cluster.
- **Wrong final state.** `sacct` reports `CANCELLED by <uid>`, which mapped to `failed`. Every cancelled job read as failed once it left controller memory.
- **First job always failed.** Output went to a directory the script created *after* Slurm had already opened the output file. The first job on every node failed.
- **Bad profiles.** The `generic` profile requested `gpu:gpu:N`, which no cluster matches. The `nibi` profile described 4 × A100 per node; Nibi is an H100 cluster (288 H100, 8 per node). Unknown profile names silently fell back to `generic`.
- **Lying cancel.** Slurm 25.11's plain `scancel` exits 0 and prints nothing for unknown or aged-out ids, so cancel "succeeded" on any id. The container rehearsal found this.

### 1.2 What this change delivers (adapter v3.0, phase 0)

Done, unit-tested (111 adapter tests, 14 pod tests, 2 new database regression tests) and run end to end against a real Slurm 25.11 cluster in the container rehearsal. All 24 suite cases passed, including the optional SYNC-1 against the pytest database. The pyright gate is at 0 under Python 3.12.

- **Transport.** `SlurmRunner` runs commands locally or over SSH, with the script on stdin and one quoted argv per connection. A forced-command gate, `xcl-slurm-gate`, allowlists exactly those shapes. This is the access model of Alliance automation ("robot") nodes: restricted key, `from=`, `command=`.
- **Profiles.** Operators add profiles through `XCELSIOR_SLURM_PROFILES_FILE`, and unknown names are errors. Profiles now carry:
  - typed or untyped GRES;
  - a CPU partition;
  - a `qos` switch;
  - memory scaling;
  - default and maximum walltime;
  - an output directory;
  - an SSH host;
  - a `cluster_id`, now stored on every mapping.
- **Job spec.** New fields: `command`, `num_gpus: 0` for CPU jobs, `nodelist`, `partition`, `cpus`, `mem_gb` and `walltime`. Every directive value is validated, every shell value is quoted, and `--comment xcelsior:<id>` makes jobs findable after a crash.
- **Errors.** Every failure carries a machine-readable `error_code`: `unsatisfiable_request`, `invalid_partition`, `invalid_nodelist`, `invalid_gres`, `walltime_exceeds_limit`, `transport_error`, `not_found`, `already_terminal`, `invalid_job_spec`, `unknown_profile` and others.
- **States.** The full Slurm state list is mapped. `CANCELLED by <uid>` is normalised, and anything unknown maps to `unknown`, never a guess. A projection onto the database's observed-state vocabulary ends the CheckViolation. The reconcile routes each mapping through its cluster's profile, and one failing row no longer stops the pass.
- **Metering.** `get_slurm_job_accounting()` returns epoch `submitted_at`, `started_at` and `completed_at`, `duration_sec` (ElapsedRaw), `gpu_count`, `gpu_seconds` and `billable`. `accounting_to_meter_fields()` projects that onto `billing.UsageMeter`.
- **Output.** `get_slurm_job_output()` reads the tail of stdout or stderr through the transport.
- **Pod data model.** `workers/slurm_pod.py` builds a pod spec (`xcelsior.pod/v0`) from scheduler inventory, probe jobs and facility declarations. Each field carries provenance: observed or declared, the method, and the Slurm job id as evidence. `verify_pod_spec()` rejects overclaims.
- **Wiring.** The API route and CLI file mappings under the profile's `cluster_id`. The CLI accepts file profiles. The module imports without the database stack.

### 1.3 Known gaps left open on purpose

These need a decision or a larger piece of work. Each is placed in a phase below.

1. **The admin HPC dashboard page is broken.** These are pre-existing defects in `api/frontend/src/app/(dashboard)/dashboard/hpc/page.tsx`:
   - Submit sends `priority: "normal"` (the API wants an int) and omits the required `vram_needed_gb`, so every submission gets a 422.
   - Cancel and refresh pass the Xcelsior `job_id` where the endpoints take a Slurm id.
   - The client type expects `state`, but the API returns `slurm_state` and `xcelsior_state`.

   This is the agent/human parity defect class: TypeScript enforced the wrong shape, so nothing ever failed loudly. → P2.6
2. **`register_slurm_job` still re-points an open mapping.** `ON CONFLICT (xcelsior_job_id) DO UPDATE SET slurm_job_id = …` re-points a non-terminal mapping at a new Slurm id, which contradicts its own docstring. → P2.2
3. **Polling cost.** Reconcile polls one `squeue` per job, which means N SSH round trips per tick. → P2.3
4. **No `preempted` observed state.** The constraint lacks it, so a preempted job projects to `cancelled` (terminal). → decision D2
5. **QoS on built-in profiles.** Built-in profiles still emit `--qos` for priority ≥ 10. Most clusters reject QoS names they don't define. → decision D6
6. **Alliance profiles unverified.** Partitions, modules and QoS in the Alliance profiles have not been checked against docs.alliancecan.ca, which blocks automated reads. Alliance clusters assign partitions themselves. → P3.1
7. **Untested container paths.** The Docker runtime path escapes the job cgroup, and the Apptainer path has never run. → P2.7
8. **Unrelated failing tests.** Two Prometheus `/metrics` tests fail at HEAD with 503. They are unrelated to Slurm.

---

## 2. Target architecture

```mermaid
flowchart LR
  subgraph CP[Xcelsior control plane]
    API[API and MCP tools] --> SUB[Slurm submitter]
    SCHED[Scheduler / placement] --> SUB
    REG[(slurm_clusters registry)] --> SUB
    SUB --> MAP[(slurm_job_mappings)]
    REC[Reconciler task] --> MAP
    ACCT[Accounting ingester] --> MET[(usage_meters / attempts)]
    PODV[Pod verifier] --> POD[(pod_listings + evidence)]
  end
  subgraph CONN[Cluster connector]
    RUN[SlurmRunner over SSH] --> GATE[forced-command gate on the login or robot node]
  end
  SUB --> RUN
  REC --> RUN
  ACCT --> RUN
  PODV --> RUN
  GATE --> SLURM[slurmctld / slurmdbd]
  SLURM --> NODES[compute nodes]
```

**Principles:**

- **The cluster is authoritative for its own state; Xcelsior is authoritative for intent.** Xcelsior records `desired_state`, observes `observed_state` and reconciles the two. It never infers a job's fate from silence.
- **Every remote action is one allowlisted command over one restricted key.** No interactive shells, no agent forwarding, no ad-hoc scripts on the partner's login node.
- **Billing time comes from the cluster's accounting (slurmdbd), not from Xcelsior's polling clock.** Polling lag must never change a bill.
- **The partner keeps its scheduler, its hardware and its policy.** The adapter adapts to the site, never the reverse.
- **Every spec on a pod listing is either observed by a scheduler-run job, with the job id as evidence, or labelled as declared by the facility.**

---

## 3. Phases

Sizes: S ≈ 1–2 days, M ≈ 3–5 days, L ≈ 1–2 weeks of focused work. Each phase ends with an exit gate.

### P0: adapter correctness (done in this change)

See §1.2. Exit gate: done.
- Unit tests green.
- DB regression proven on old code.
- Rehearsal: 24/24 cases, SYNC-1 included.
- Pyright gate at 0.

### P1: local two-node validation (S, ready to run)

Run `slurm-local/` on the tower and the ASUS: Steps 1–6 of its README. Nothing here needs new code.

Exit gate:
- `CHECKLIST.md` sections A–C all PASS (SYNC-1 optional);
- rollback verified on both machines (section D), or the cluster kept running by choice;
- `RESULTS_TEMPLATE.md` filled in from the run's `results.json`.

Watch-outs:
- The ASUS's hardware crashes can take a node DOWN mid-run. Re-run the affected case with `--only` before calling it an adapter failure.
- The tower is production-adjacent. The kit caps Slurm at 8 GB and 8 CPUs per job, starts nothing at boot and proves Docker and the runners unchanged (V-12).

### P2: control-plane integration (the production path)

This turns the adapter from an admin tool into a placement target for real jobs. Everything here runs in the control plane on prod and reaches clusters only through the connector.

#### P2.1 Cluster registry (M, needs a migration)

- Add a table `slurm_clusters` with these columns:
  - `cluster_id` (PK), `display_name`, `profile` (JSONB, the same keys as today's profile dicts), `ssh_host` and `ssh_port`;
  - `known_hosts_line` (the pinned host key, never trust-on-first-use);
  - `credential_ref` (a Vaultwarden / secret-store reference, never the key itself);
  - `remote_user`, `status` (`active | paused | retired`), `owner_org`, `created_at` and `updated_at`.
- `available_profiles()` reads the registry. Built-ins and the JSON file become seed data and a dev override.
- `get_runner()` builds SSH options from the registry: `-o UserKnownHostsFile=<tmp pinned file> -o StrictHostKeyChecking=yes -i <materialised key>`. The key is materialised to a 0600 tmpfs file for the call's lifetime.
- Admin API for CRUD, plus a "test connection" action that runs `sinfo --version` and `scontrol show config` through the gate.
- Bookkeeping per `CONTRIBUTING`/memory rules: a domain claim in `control_plane/db_roles.py`, `EXPECTED_HEAD` bumps, and the migration applied to both the dev and pytest databases.
- Acceptance: the adapter never reads a profile from code in production, and `cluster_id` on every mapping is a foreign key to `slurm_clusters`.

#### P2.2 Submission inside the job lifecycle (L)

- A Slurm cluster becomes a placement target alongside hosts. The scheduler creates a `job_attempts` row (migration 055) whose target is `slurm:<cluster_id>`, then calls the submitter.
- **Idempotent submit:**
  1. The submit key is `attempt_id`.
  2. Before any `sbatch`, look for an existing job carrying `--comment xcelsior:<job_id>`: `squeue --me -h -o "%i %k"`, then `sacct -X -n -o JobID,Comment` (needs `AccountingStoreFlags=job_comment` on the site, or fall back to `JobName=xcl-<job_id>`).
  3. If one exists, register it instead of resubmitting.
  4. This closes the crash window between `sbatch` and the mapping insert.
- Fix `register_slurm_job` so that a conflict on an open mapping with a *different* Slurm id raises, rather than re-pointing. The same id is a no-op. Always pass `tenant_id` and `attempt_id`, so the existing unique indexes do their job.
- Every remote failure maps to an attempt outcome:
  - `error_code` in {`unsatisfiable_request`, `invalid_*`, `walltime_exceeds_limit`} → attempt `rejected` (terminal, not retried);
  - `transport_error`, `timeout`, `controller_unreachable` → retry with backoff, then `lost`.
- Acceptance tests:
  - a crash injected between `sbatch` and register leaves exactly one Slurm job and one mapping after recovery;
  - a retried submit never creates a second job;
  - rejected requests never retry.

#### P2.3 Reconciler as a durable scheduled task (M)

- Register it through `control_plane.scheduled_tasks.register_task(..., service=...)` under its own database identity. It runs every 15–30 s per active cluster.
- **Batch per cluster:** one `squeue -h -j <id,id,…> -o "%i %T %R"` plus one `sacct -X -n --parsable2 -j <ids> -o JobID,State,…` per tick, instead of one call per job. Chunk at about 200 ids per call.
- Persist `observed_state` and `last_observed_at` as today. Alert when `last_observed_at` is older than 3× the interval for any open mapping.
- `not_found` handling: if the job is in neither squeue nor sacct for N consecutive ticks, mark the mapping `unknown` and raise an operator event. Never assume it completed.
- Desired → cluster: `desired_state = cancelled` sends `cancel_slurm_job`. Treat `already_terminal` as success.
- Acceptance tests:
  - a load test with 2,000 open mappings on the rehearsal cluster finishes a tick in under 10 s;
  - a controller restart during a tick loses no transition.

#### P2.4 Metering and billing from accounting (L)

- On a terminal transition, the accounting ingester calls `get_slurm_job_accounting()`. It writes or settles the attempt's meter through the existing attempt-billing path (`control_plane/attempt_billing.py`, `billing.settle_attempt_meter`):
  - `started_at` and `completed_at` come from slurmdbd;
  - `duration_sec` is ElapsedRaw;
  - `gpu_seconds` = ElapsedRaw × AllocTRES `gres/gpu`.
- While running, the active meter uses `started_at` and leaves `completed_at = 0.0`, the shape METER-3 proves.
- Rules, each with a test:
  - a job cancelled while pending is never billed (`billable = false`);
  - a timeout bills to its end;
  - a requeued job bills each run segment, which needs `sacct --duplicates`;
  - a preempted job bills to the preemption time.
- Daily reconciliation: sum the day's `sacct` `ElapsedRaw × GPUs` per cluster and compare it with meter rows. Alert on any drift over 1%.
- Pricing comes from the cluster's rate card in the registry. A partner facility has its own rate, distinct from host offers.
- Acceptance: billing a set of rehearsal jobs reproduces `sacct` to the second, and the reconciliation report is clean.

#### P2.5 Output and artifacts (M)

- Shared-filesystem clusters: the adapter reads output through the gate (`tail -c`) and caps it at 64 KiB per fetch. A streaming endpoint pages with an offset (add `tail -c +<offset>` to the gate allowlist with bounds).
- Large artifacts: an `rsync` pull through the site's transfer wrapper (Alliance ships one), into Xcelsior's artifact store with tenant scoping. Retention follows the artifact policy.
- Acceptance: a 100 MB result file arrives intact with its checksum verified, and output from another tenant's job is unreachable.

#### P2.6 API, MCP and UI parity (M)

- New tenant-facing endpoints:
  - `POST /api/v1/hpc/jobs` (command, image, num_gpus, walltime, cluster or pod);
  - `GET /api/v1/hpc/jobs/{id}` (Xcelsior id, never a raw Slurm id);
  - `DELETE` the same path;
  - `GET /api/v1/hpc/jobs/{id}/output`;
  - `GET /api/v1/hpc/clusters`.
- These are scoped to the tenant, not admin-only. Map `error_code` onto HTTP status: 400 `invalid_*`, 409 `already_terminal`, 422 `unsatisfiable_request` or `walltime_exceeds_limit`, 503 `transport_error` or `controller_unreachable`.
- MCP tools `submit_hpc_job`, `get_hpc_job`, `cancel_hpc_job`, `get_hpc_job_output` and `list_hpc_clusters`. They return `error_code` verbatim so agents can branch. Regenerate `mcp/openapi-v1.json` and run `npm run generate:api`.
- Rebuild the HPC dashboard page on the v1 endpoints; this fixes the three defects in §1.3. Add a contract test that the frontend type and the response schema share fields.
- Keep the old `/api/slurm/*` admin routes until the UI moves, then remove them.
- Acceptance: an agent books, polls, cancels and fetches output through MCP alone, and the UI shows the same states as the API.

#### P2.7 Security (M, gates any external cluster)

- **Payload trust (decision D1).** Who may run arbitrary `command`s on a partner cluster, and under whose account?
- **Gate per cluster.** The gate's allowlist is versioned in the repo. Every allow and deny is shipped to the audit log with tenant, job and cluster ids.
- **Keys.** One key per cluster, rotated quarterly. `from=` is pinned to the control plane's egress IPs; the prod Hetzner address must be stable or proxied. Rotation runbook and tested revocation.
- **Containers.** Apptainer only on shared clusters, with `--containall` and no host mounts beyond the job's work dir. Remove the Docker runtime path from non-owned profiles, because it escapes Slurm's cgroups.
- **Secrets** never appear in generated scripts. Pass them through the site's mechanism or not at all.
- Run a `/security-review` and a red-team pass on the script generator (fuzz every job field through `bash -n` and execution in a sandbox).
- Acceptance: the fuzzer finds no field that reaches shell evaluation, and a pen-test of the gate finds no escape.

#### P2.8 Observability (S)

- Metrics: submissions by `error_code`, submit latency, reconcile lag, transport failures per cluster, accounting drift, and the gate deny rate. Wire them into the existing Prometheus and Grafana setup, with alerts on reconcile staleness, transport failures over 5 minutes and drift over 1%.

#### P2.9 Rollout (S)

1. Feature flag `slurm_placement`.
2. **Shadow mode:** the reconciler and accounting ingester run against the local cluster while billing is computed but not charged, and compared for a week.
3. Admin-only.
4. Allowlisted tenants.
5. General availability.

The rollback switch is the flag; mappings stay for audit.

**P2 exit gate:**
- the local cluster is registered as a real cluster in staging;
- jobs placed through the scheduler run, reconcile and bill to the second;
- the MCP round trip works;
- the security review is closed;
- the dashboards are live.

### P3: academic scale validation (research; contingent on a PI and Mitacs)

Nothing here starts until a sponsoring PI exists. Research use only; no commercial workloads on the cluster.

- **P3.1 Profile truth (S).** By hand, read the Nibi pages on docs.alliancecan.ca for these and correct the `nibi` profile:
  - partitions (Alliance assigns them automatically, so the profile should probably *omit* `--partition`);
  - GPU request syntax (`--gpus-per-node=h100:N`) and MIG names;
  - modules and QoS;
  - output locations (`/scratch/$USER`);
  - account flags (`--account=def-<pi>`).

  Add a profile option to omit the partition, and `gpus_per_node` syntax support. Retire `graham`.
- **P3.2 Automation access (S, needs the PI).** Request robot-node access with the exact command list (the gate's allowlist). Alliance asks for the commands and tooling up front. Use the site's `command=` wrapper or ours, as they prefer.
- **P3.3 Validation runs (M).** Run the same suite with a `nibi` profile at small scale: CPU, 1 GPU and 1 node × 8 GPUs. Add cases for multi-node jobs, requeue/preemption accounting and the NCCL all-reduce probe (P4.2).
- **P3.4 Write-up (S).** The results report is the research output and the proposal's evidence.

Exit gate: the suite passes on Nibi at small scale, and the profile is corrected from observed behaviour.

### P4: pod tier productization

Build on `slurm_pod.py`'s spec. The facility partner brings the hardware and capex; Xcelsior brings orchestration, metering and the agent booking layer.

- **P4.1 Data model (M, migration).** Tables:
  - `pod_listings`: `pod_id`, `cluster_id` FK, `partition` or `reservation`, `status`, `spec` JSONB in `xcelsior.pod/v0`, `listing_title` and `rate_card`;
  - `pod_verifications`: append-only, with `pod_id`, `spec_snapshot`, `checks`, `status`, `probe_job_ids` and `verified_at`.

  A listing is bookable only while its latest verification is `verified` and younger than its re-verification interval.
- **P4.2 Verification pipeline (L).** Probe jobs at onboarding and on a schedule:
  - per-node GPU (NVML);
  - NVLink topology (`nvidia-smi topo -m`, replacing v0's link-count approximation);
  - NIC link speed and RDMA devices (`ibv_devinfo`, `rdma link`);
  - fabric bandwidth with `nccl-tests` all-reduce across all nodes, which verifies "800G per GPU" as achieved bus bandwidth against the declared figure with a tolerance.

  Drift fails verification and suspends the listing with an operator event. The spec version moves to `v1` with the fabric fields.
- **P4.3 Booking model (L, decision D5).** Whole-pod or block bookings map to a Slurm reservation (`scontrol create reservation`, which needs operator rights the partner grants) or a dedicated partition. Jobs submitted under a booking carry `--reservation`. Billing is per pod-hour for the booked window, not per job. There is a cancellation and no-show policy.
- **P4.4 Marketplace surface (M).** Listing pages with each spec's provenance shown ("verified by job 4123 on …" vs "declared by facility"). Book-a-pod in the API, MCP (`list_pods`, `book_pod`, `get_pod_booking`) and UI.
- **P4.5 Partner onboarding (S, contract).** The partner provides:
  - a Slurm account and a robot/automation key path;
  - accounting read access (`sacct` for our jobs);
  - a reservation or partition for bookings;
  - a declared spec sheet;
  - an SLA.

  Xcelsior provides orchestration, metering, payment and the agent booking layer.

Exit gate: one partner pod listed with `verified` status from P4.2 probes, booked end to end through MCP, and billed per pod-hour.

---

## 4. Testing strategy

| Layer | What | Where |
|---|---|---|
| Unit | job translation, quoting, error codes, state maps, accounting parsing, pod spec; Slurm output shapes captured from a real cluster | `tests/test_slurm_adapter.py`, `tests/test_slurm_pod.py` |
| Database contract | mapping constraints, reconcile projection, idempotent register, terminal retention | `tests/test_shared_state_contract.py` |
| Real Slurm, no hardware | the container rehearsal: two nodes, real slurmctld/slurmdbd/MariaDB, GPU via NVML, SSH through the gate; also install and rollback | `hpc/slurm-local/rehearsal/run-rehearsal.sh` |
| Real hardware | the same suite on tower + ASUS | `hpc/slurm-local/tests/validate_adapter.py` |
| Live gates | P2 acceptance cases (crash recovery, batching load, billing to the second) added to `tests/live/`, which skips silently without a cluster; check it before calling a gate unasserted | `tests/live/` |
| Security | field fuzzer through `bash -n` + sandboxed execution; gate allowlist tests | P2.7 |

CI note: GitHub Actions has been billing-locked since about 2026-07-21. Until it unlocks, run lint, black, pyright (Python 3.12 from `uv.lock`) and the Slurm tests locally, and treat a green push as unverified. When CI returns, add the rehearsal as a nightly job; it needs a runner with Docker, and a GPU only for the GPU cases.

---

## 5. Decisions that are yours

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Whose account runs tenant code on a partner cluster | (a) one Xcelsior service account per cluster; (b) per-tenant cluster accounts | (a) for partner pods, with strict payload policy and per-tenant Unix groups for output separation. (b) is how academic sites work and is what P3 uses |
| D2 | `preempted` as an observed state | add it to the constraint (migration) vs keep projecting to `cancelled` | Add it in P2.1's migration: spot and preemptible capacity will want it |
| D3 | Billing source of truth for Slurm jobs | slurmdbd accounting vs Xcelsior's observation clock | slurmdbd (§2 principle); polling lag must not move money |
| D4 | Tenant `command` payloads | open, allowlisted images only, or templates only | Apptainer images from an allowlist plus free-form command inside the container |
| D5 | Pod booking mechanism | Slurm reservation vs dedicated partition | Reservation: time-boxed, native to Slurm, and leaves the partner's partitions alone |
| D6 | Built-in profiles' `--qos` | keep, or default `qos: false` | Default false; enable per cluster from P3.1's observed behaviour |

---

## 6. Definition of production-ready

The adapter is production-ready when all of these hold:

1. No profile, key or host is read from code in production; the registry and secret store hold them, and host keys are pinned.
2. Submits are idempotent across crashes, and rejected requests never retry.
3. Reconcile is batched, durable and alerting, and never infers completion from silence.
4. Billing for Slurm jobs reproduces slurmdbd to the second, with a daily drift report.
5. Tenant-facing API, MCP and UI share one schema; the dashboard works.
6. A security review and a fuzzer pass on script generation and the gate are closed.
7. The suite passes on the local cluster and, if P3 happens, on an academic cluster.

---

## Appendix A: defects found in adapter v2.2 and their status

| # | Defect | Status | Proof |
|---|---|---|---|
| 1 | CPU jobs impossible: `partition_cpu` unused, `--gres` always emitted | fixed | CPU-1, unit tests |
| 2 | No payload command | fixed | GPU-1 |
| 3 | No node targeting | fixed | CPU-1 |
| 4 | No output retrieval | fixed | GPU-1, FAIL-6 |
| 5 | No start/end timestamps for billing | fixed | METER-1..3 |
| 6 | `CANCELLED by <uid>` from sacct mapped to `failed` | fixed | LIFE-5, unit test |
| 7 | PENDING written as `queued` into `observed_state` → CheckViolation aborts reconcile | fixed | regression test fails on old code; SYNC-1 |
| 8 | Unknown states guessed (`running` from squeue, `failed` from sacct) | fixed → `unknown` | unit test |
| 9 | Shell injection via name, tier, extra env, image | fixed | FAIL-6, bash-execution unit test |
| 10 | `#SBATCH` directive injection via newlines in fields | fixed (validation) | unit tests |
| 11 | Output dir created inside the script, after Slurm opened the file | fixed | every rehearsal job |
| 12 | Temp sbatch scripts leaked on the submit host | fixed (stdin) | unit test |
| 13 | Unknown profile silently became `generic` | fixed → `unknown_profile` | unit test |
| 14 | `generic` requested `gpu:gpu:N` | fixed → untyped | unit test |
| 15 | `nibi` described 4 × A100; Nibi is 8 × H100 | corrected; partitions/modules still unverified | P3.1 |
| 16 | `--qos` always emitted for priority ≥ 10 | profile switch added; built-ins unchanged | D6 |
| 17 | Fixed memory formula (12 GB for one GPU) | profile-scaled | unit test |
| 18 | Walltime never checked against the profile maximum | fixed (client and Slurm side) | FAIL-4 |
| 19 | Every mapping filed under cluster `unknown` → cross-cluster id collisions | fixed in route and CLI | unit test |
| 20 | Module imported the DB stack at load; unusable on a submit host | fixed (lazy) | unit test |
| 21 | Plain `scancel` reports success for unknown ids | fixed (`-v` + accounting) | FAIL-9, LIFE-3 |
| 22 | No machine-readable errors | added `error_code` | FAIL-* |
| 23 | No remote transport; adapter assumed it ran on the login node | SSH runner + gate | every rehearsal case |
| 24 | Docker multi-GPU flag never expanded (single quotes); Docker escapes cgroups | flag fixed (`SLURM_JOB_GPUS`); escape documented | P2.7 |
| 25 | One failing row stopped the reconcile pass | per-row isolation | DB tests |

## Appendix B: sources for the Alliance facts above

- Nibi: 288 H100 GPUs, successor to Graham. Alliance's Nibi service page, https://www.alliancecan.ca/en/services/compute/nibi
- 8 H100 per node and 400 Gb/s Ethernet on GPU nodes: SHARCNET's 2025 migration webinar, https://helpwiki.sharcnet.ca/wiki/images/a/ac/Migration_webinar_2025.pdf. Read through a search summary; the PDF is bot-protected. Verify by hand.
- Automation nodes: restricted keys (`restrict,from=…,command=…`), approval through Alliance support, and hostnames like `robot.<cluster>.alliancecan.ca`. https://docs.alliancecan.ca/wiki/AiiDA, which points to the "Automation in the context of multifactor authentication" page.
