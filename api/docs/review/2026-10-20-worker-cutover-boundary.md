# Worker execution boundary and legacy retirement — 2026-10-20

The prior cutovers established scoped scheduler ownership, fenced commands and
private agent ingress. They did not remove every legacy execution path. Track A
P4.4b described code-readiness while Track B B16 contraction/removal remained
open. Its heading now makes that distinction explicit.

## Current production evidence

Read-only API/scheduler database and configuration checks on October 20 found:

- Scheduler mode **canary**, with claims enabled for RTX 2060/3060 models and
  hosts `aaryn-tuf-rtx2060` / `tower-rtx3060`.
- API v2 enrollment is `*`. Enrollment does not mean all jobs have fenced
  attempts, and a `/agent/v2/work` URL still calls the compatibility handler.
- No queued, assigned, starting or running jobs. There are **12 stopped jobs**;
  none has any `job_attempts` history. The other jobs are terminal.
- All five maintenance repair actions remain **report_only**.

The production records are embedded in the
[evidence JSON](2026-10-20-worker-cutover-evidence.json). No production service,
worker, database row or policy was changed by this increment.

## Defects closed

The compatibility work handler trusted assigned/cached payloads without an
authoritative fenced-history filter. The compatibility lease writer could insert
a legacy lease before a later fenced status mutation failed. The worker runner
changed containers before acquiring that lease and continued when its claim
failed. Startup adoption registered monitoring without a successful reclaim.
Negotiation errors silently selected v1 execution.

The API now checks current host/status and full attempt history before delivering
compatibility work, including through the v2 alias. Failed ownership reads return
503 and retain queued hints. Legacy grant/renew/release lock the placement job
row and refuse any fenced history before changing a lease. A job between fenced
attempts cannot revert to the compatibility protocol. Direct event-store calls
receive the same protection as HTTP routes.

The worker requires a matching local attempt/fence/container tuple for a v2
start. Compatibility execution must receive a successful lease before container
cleanup or startup; adoption must reclaim before tracking or monitoring. Valid
fenced restart survivors now populate active-job tracking using their real job
ID, keeping heartbeat, image warmers and benchmarks aware of their workload.

Negotiation explicitly returns `legacy_execution_allowed`; the worker accepts
only boolean `true`. Scheduler active mode disables compatibility work and new
legacy lease claims. Unconfirmed negotiation waits and retries instead of
falling back to v1. Deploy this API contract before upgrading mixed-mode workers:
an older API that omits the flag cannot authorize legacy execution for the new
worker. Existing workers need a rollout/restart to pick up a changed policy.

## Measurable use and validation

`xcelsior_legacy_execution_total{path,outcome}` tracks authenticated work polling
and lease claim/renew/release. Outcomes distinguish requests, actual delivery or
lease changes, and blocked requests. Even an empty poll or a blocked request
proves the caller still uses the path. The gauge
`xcelsior_legacy_execution_enabled` reflects each API replica's scheduler policy.
The operations dashboard plots each path/outcome. The new alert joins policy
and requests on the same replica, so an active replica does not classify another
replica's permitted canary calls as retired use.

- The first boundary run produced **17 failures** before implementation
  (`artifacts/worker-cutover-evidence/red`).
- **457 tests passed** across 18 suites. The separately opt-in Docker Compose
  smoke class is excluded from this selected regression gate. Earlier broad
  collection exposed that test module clearing the worker token during import;
  preserving the synthetic conftest token fixed those unrelated 401 failures
  without changing production authentication.
- The real CUDA/HTTP/PostgreSQL acceptance passed all five journeys: normal
  success/failure, pre-commit failure, lost committed response and a worker
  process restart. Container/journal retention, single meter settlement,
  lease/allocation release and exact invoice deltas remain verified.
- All 12 distribution signatures verify. Targeted Ruff, Pyright and whitespace
  checks pass.
- Prometheus 3.5.0 `promtool check rules` validates 57 rules. Three executable
  alert scenarios pass: active use alerts, canary use does not, and a different
  replica's active policy cannot turn permitted canary traffic into an alert.

Full manifests, JUnit/logs and CUDA journey details are under
`artifacts/worker-cutover-evidence/{red,final,cuda}`; selected source hashes and
full-manifest checksums are retained in the linked evidence JSON. The final
runner requires zero skips among selected tests. Its explicit exclusion is
`--deselect=tests/test_instance_flow.py::TestDockerComposeSmoke` after the
runner's `--` argument separator.

## Remaining retirement gates

1. Roll out the API contract/metrics and signed worker candidate while keeping
   repairs report-only; verify the deployed bytes and worker journeys.
2. Inventory and choose a state-preserving migration/disposition for each of the
   twelve stopped legacy jobs. Their container data must not be discarded by
   deleting resume handlers.
3. Qualify widening from canary to active, drain old execution and measure a
   continuous agreed zero-use window. An idle snapshot, a missing scrape or
   undeployed counters is not that evidence.
4. Instrument and retire the other B16 paths: legacy status/lifecycle writes,
   direct SSH execution, destructive administrative drains/preemption, launch
   compatibility, broad scope issuance and old confirmation flows. This increment
   covers only the four execution endpoint families listed above.
5. Perform B16.2 schema contraction and B16.3 code/table removal after their
   migration, retention, rollback and deployment gates pass.

Legacy cleanup is therefore still partial. This commit closes execution bypasses
and provides measured-use signals; it does not claim fleet cutover or delete the
remaining compatibility handlers.
