# Legacy observation gates and recovery sources — 2026-10-20

B16 remains partial. This increment measures two compatibility handlers still
used during the v2 transition and extends the stopped-job recovery audit. No
production job, volume, container, service or repair policy was changed.

## Status and command observations

`xcelsior_legacy_execution_total{path,outcome}` now includes `status_patch` and
`command_drain`, with all series initialized for scraping. Measurement starts
after authentication, before ownership/fencing and database operations.

- `requested` includes empty command polls and later ownership/fencing refusals.
- Status `used` counts completed lifecycle mutator calls, including idempotent
  calls and fenced compatibility projections. It does not count unique jobs.
- Drain `used` counts rows removed in the committed delivery transaction. It
  proves destructive delivery, not worker execution or ACK. Fenced commands
  remain excluded by the existing drain predicate.
- `blocked` counts client refusals before a known effect. `failed` counts server
  errors and errors after a known effect. An exception with uncertain commit
  state is never described as a side-effect-free refusal.

`xcelsior_legacy_path_retired{path}` supplies an expectation per replica and
path. Scheduler active mode retains the existing expectation for six execution
paths. It does **not** declare status PATCH or command draining retired: v2
workers still use them. The optional `XCELSIOR_RETIRED_LEGACY_PATHS` setting
declares that expectation only after their separate gates close. It changes
alerts, not handler availability, and rejects unknown path names on startup.

The alert matches `instance`, `job` and `path`; another replica or a retired
work-poll path cannot make an in-use command path alert. The operations
dashboard shows usage and per-replica policy. Deploy the API series before its
new alert rule and verify every replica before starting a zero-use window.

## Recovery sources checked

Read-only production metadata still shows the same twelve legacy jobs stopped,
with no registered user images whose `source_job_id` matches those jobs.
Identity-verified Docker image censuses completed on both hosts: 46 images on
the workstation and 18 on tower, with no matching Xcelsior job labels.
Unlabeled or externally exported images are not ruled out by this census.

The retained production-host copies dated **2026-10-09 through 2026-10-19**
(eleven snapshots, each recording rsync exit 0) and the 2026-10-07 old-server
cutover copy all contain `/exports/volumes`, but none contains any of:

```
exports/volumes/vol-36592ad0362e
exports/volumes/vol-36592ad0362e.img
exports/volumes/_snapshots/vol-36592ad0362e
```

These are API/storage-host copies, not backups of either GPU worker's writable
Docker layers. Logical PostgreSQL dumps preserve metadata, not those files.
No suitable recovery source was found in the checked locations, so no restore
was attempted. This does not exhaust earlier backups or external exports and
does not prove irrecoverable loss. No empty replacement volume was created.

Production API, background worker and scheduler worker all report `report_only`
for all eight enforceable reconciliation findings. Scheduler mode remains the
same scoped `canary`; the new observation policy is unset.

## Executed gates

- **294 selected tests passed**, with zero failures, errors or skips, against a
  fresh local PostgreSQL database migrated to head. The separate opt-in Docker
  Compose smoke class was explicitly deselected.
- The real worker/HTTP/PostgreSQL acceptance passed five fenced CUDA journeys
  and the additional legacy writable-layer marker preservation case.
- Prometheus rule tests passed all five cases, covering active/canary policy,
  replica isolation, path isolation and explicitly retired status calls.
- All 12 worker distribution signatures verified; worker source is unchanged
  in this increment. Ruff, targeted Pyright, dashboard JSON and whitespace
  checks passed.

The [machine-readable evidence](2026-10-20-legacy-observation-evidence.json)
records source hashes, gate results, curated recovery inventories and runtime
policy. Full local logs remain in `artifacts/legacy-observation-evidence`.

## Candidate image and live monitoring readiness

Implementation commit `97849ee1a9e14f92f1c7800b1874d109f210c9c6` was built
from its clean worktree as `xcelsior-api:control-plane-97849ee1`, image
`sha256:c1ef3e15ac3de512c2a6a32109c77fcf1342fd1fe3d0ca715939cd02b447fe81`.
A network-disabled, read-only container verified the selected source hashes,
all twelve signatures and the absence of the private environment/signing key.
This proves the packaged candidate, not production deployment.

The live native Prometheus on `127.0.0.1:9091` has eight scrape targets but no
`xcelsior-api` target, and its loaded rules do not include
`XcelsiorRetiredExecutionPathUsed`. It has one configured Alertmanager;
notification delivery was not tested. An inactive systemd unit named
`alertmanager` alone does not establish that its configured receiver is down.
Preserve this shared host's existing scrape jobs and receivers when adding
Xcelsior monitoring; the repository's Compose configuration is not the running
native configuration.

There is also a producer-side gate: the current Prometheus client counters are
process-local, and production is configured for two Gunicorn workers without
multiprocess metrics enabled. A local fork probe incremented the actual legacy
request counter in a child that exited successfully; the scraping parent saw a
delta of zero. The endpoint instrumentation therefore does **not** yet prove
all-worker usage. Implement and test shared worker aggregation, including a
worker exit before the next scrape, before using these counters for retirement.
No zero-use window has started.

## Remaining gates

1. Complete shared Gunicorn worker counter aggregation and its failure/restart
   acceptance. Add and validate the API scrape target and retirement rules in
   the existing native monitoring configuration, preserving other targets and
   receiver routing. Verify notification delivery.
2. Deploy the accumulated API/worker safeguards and observability changes in
   API-first order, keeping scheduler scope and repairs unchanged. Verify the
   exact deployed bytes, signatures and worker journeys.
3. Obtain a verified recovery source or an explicit disposition for each
   stopped row and the missing volume. A new fenced attempt is reprovisioning,
   not recovery of the old writable layer.
4. Instrument the other B16 paths, including internal direct status writers,
   direct scheduler SSH, process-memory preemption, old launch/confirmation
   surfaces and broad scope issuance.
5. Establish the agreed continuous zero-use window before contract migration,
   code removal and the dependent fairness/preemption work. No point-in-time
   inventory or passing synthetic gate closes that window.
