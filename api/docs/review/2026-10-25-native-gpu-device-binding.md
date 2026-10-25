# Executed physical GPU binding and startup authority handoff

Production API, scheduler, background worker and SSH gateway run
**7a1aa0ee**, at schema **141**, with no migrations in this handoff. Seven
runtime database-role contexts passed. All four containers and the unchanged MCP
runtime are healthy with zero restarts. All eight repair policies remain
`report_only`; preemption and deadline force remain disabled.

Both idle native GPU workers were handed off from the prior signed bundle to
this release. Their attempt journals and environment were retained, twelve
signatures and the pinned public key were verified on each host, and both
confirmed the native execution protocol with zero automatic restarts. Tower's
other workloads were preserved.

Lease claim now returns only the exact physical GPU UUIDs in that attempt's
reserved allocation. Invalid counts, duplicate or synthetic IDs, wrong-host
allocations and retired devices abort the claim transaction. The worker requires
`leases.gpu_device_binding` and `leases.start_authority` during negotiation.
Docker receives those UUIDs and an enforced `NVIDIA_VISIBLE_DEVICES` value;
there is no tenant execution fallback to `all` or device indices. Immediately
before container startup, a fresh renewal rechecks current running intent and
job/attempt/host/fence/spec/generation authority. Stop/restart intent or lost
authority during preparation prevents startup. Authority lost during the Docker
call requests a stop of that exact attempt-labeled container.

The frozen source passed **4,121 tests**, with zero failures, errors or skips,
**1,138 unchanged source hashes**, clean source at the start and finish, and
successful PostgreSQL/Redis teardown. The actual non-root image passed twelve
checks covering startup, schema, metrics, signed distributions, retired HTTP
transports and the new GPU/start contract. Black and Ruff passed for fourteen
files; pyright passed for four runtime files. MCP schema/types were regenerated
and the generated client typecheck passed. Public OpenAPI and endpoint inventory
remain unchanged. Named gates include `tests/test_native_gpu_binding.py`,
`tests/test_worker_execution_boundary.py`, `tests/test_control_plane_leases.py`,
`tests/test_agent_v2.py` and `tests/test_security.py`. Test execution used the
isolated `api/scripts/run_security_gates.py` wrapper. Earlier interrupted or
failed diagnostic runs are retained and are superseded by this full qualification.

Actual TUF job **181a77c4**, attempt
**eac7c8e8-c786-45d2-add3-24be4e9460d8**, fence **12**, succeeded. While it was
running, Docker's device request contained exactly the single physical UUID
reserved in PostgreSQL. The submitted `NVIDIA_VISIBLE_DEVICES=all` was replaced
with that UUID. Durable HTTP logs independently report the matching UUID,
`XCL_NATIVE_GPU_COUNT=1` and `XCL_NATIVE_CUDA_RESULT=32768`. The allocation was
released and the attempt/lease settled. Database tests cover distinct and
multiple GPU reservations, and Docker configuration parsing accepted multiple
UUIDs; an actual multi-GPU hardware journey is not claimed.

The normal scheduled billing cycle charged **400 micro-CAD** for this job.
Its spot spec, captured attempt quote, meter, cycle and wallet debit agree.
Across seven real CUDA/lifecycle jobs, eight meters and nine charged cycles and
wallet debits agree at **57,900 micro-CAD**, with zero wallet-ledger drift.
No active attempt pointers, leases or GPU allocations remain. Eight repeated
HTTP log reads, log download, read-only invoice preview and CSV match the usage.
No external provider payment was tested.

The first post-import handoff was refused by the existing disk headroom guard,
before runtime/configuration changes. The imported daemon's image size exceeded
the local daemon's estimate. Only the unused, never-deployed 66f31bc5 candidate
image was removed, after proving no container references and exact retained
local archive custody; no forced deletion or running image/backup removal was
used. That reclaimed 1,447,612,416 bytes. The guard passed before the successful
handoff. Future private staging uses the greater of the existing eight-GiB budget
floor and a conservative size estimate; this check does not reserve filesystem
space or guarantee other writers cannot consume it.

Three recovery timers remain enabled/active, and their last executions succeeded
on **1a0ae249**. They were not rerun or relabeled as 7a1aa0ee proof. Fifteen prior
backups and the dated full 155-table hash-verified restore are retained. Schema
141, retired projection triggers/columns and absence of restore clones were
checked again. The prior read-only population scan covers 177 physical tables
and 2,077 columns with zero scan errors. The 201,301 qualified empty invoices
retain their exact record-set digest and have not been archived or deleted.

The [sanitized receipt](2026-10-25-native-gpu-device-binding-receipt.json) records
the source, runtime, worker, CUDA and billing evidence. It supersedes the runtime
and worker state in the [previous handoff](2026-10-25-native-release-production-handoff.md).
The [shared-filesystem recovery](2026-10-25-shared-filesystem-recovery.md) remains
the dated incident evidence.

Remaining work is explicit:

- Canonical physical GPU/model/VRAM/freshness storage and reservation revalidation
  still have JSONB and synthetic-inventory compatibility paths. Total VRAM is not
  proof of measured free VRAM; busy-host inventory refresh also needs work.
- Observation-session retirement still depends on retained observation history.
  Durable session retirement must survive retention without allowing old reports
  to become current again.
- Docker inspect/kill failures need durable stop retry; the current stop request
  is not an unconditional physical-stop guarantee. Lost claim responses still
  rely on lease expiry rather than idempotent same-session grant replay.
- Two attempts from July 18 and October 17 lack captured quotes/meters, so
  `XcelsiorUsageEvidenceMissing` remains firing at 2. Historical invoice
  disposition, remaining unused helpers and the REST unified-launch adapter
  residual are not described as complete.
- `TowerDiskHigh` remains firing at about **96.8%** as of the recorded check;
  unrelated models and workloads were preserved.
- Independent offsite/WAL/PITR/object recovery and failover acceptance remain
  unverified. PostgreSQL `archive_mode` and `data_checksums` remain off.
  Production repair/preemption enforcement remains disabled.

No complete control-plane or data-layer cutover claim is made.
