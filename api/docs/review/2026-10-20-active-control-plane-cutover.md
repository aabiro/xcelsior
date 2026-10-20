# Active control-plane cutover — 2026-10-20

Production now runs the v2 scheduler in **active** mode across the API,
maintenance worker and scheduler worker. Both signed GPU workers negotiated
v2 with legacy execution disabled and acknowledged real GPU-free health
commands. The operator requested this cutover while no customer jobs were
running, without extending the former retirement-observation project.

The native dashboard is now
[Xcelsior Control Plane](https://grafana.xcelsior.ca/d/xcelsior-control-plane/xcelsior-control-plane).
The former retirement dashboard, its dedicated provider and native alert file
and reference were removed. This record supersedes the observation-only
[native dashboard record](2026-10-20-native-legacy-dashboard.md).

## Runtime behavior

Host-management commands use durable v2 claim/ACK delivery. Workers journal
successful results before ACK, replay the result after a lost ACK, and NACK
failed execution or journal writes. Existing attempt-only clients retain their
command partition. A read-only `health_probe` exercises real worker claim,
dispatch and ACK without creating a job or touching a container.

Runtime connection metadata is validated and written within the fenced status
transaction under the job lock. A stale fence or out-of-order status cannot
change connection details. Repeating the current status can refresh metadata.
Fenced attempts no longer send the compatibility status PATCH.

Active mode returns HTTP 410 for the old work poll, destructive command drain,
lease claim/renew/release, instance status PATCH, process-memory preemption poll
and admin preemption enqueue. Admin rejection occurs before any queue write or
notification. Durable fair-share/preemption remains the separately tracked
B16.4 feature; the obsolete endpoint cannot claim it scheduled work.

Implementation is in `742c63b1`; the final obsolete-preemption guard is in
`e388434f`. The final image is `xcelsior-api:active-cutover-e388434f`. Its
local/imported image IDs differ, so Config, RootFS layers, Created, Architecture
and Os were compared independently and all matched. The
[machine-readable evidence](2026-10-20-active-control-plane-cutover-evidence.json)
pins both IDs, all changed runtime files and the preserved live files.

## Operational metrics

The dashboard has 20 panels and 32 expressions using the existing native
Prometheus datasource. It shows API reachability, durable metric availability,
scheduler mode, ready hosts, queue depth/age, active attempts and GPU allocations,
pending/claimed/dead-letter commands, service heartbeat age and fresh replicas,
reconciliation backlog/errors/delay, lease and observation problems, billing
invariants, event/projection delivery, background failures and metric freshness.
Idle work is a valid zero; missing data remains unknown.

Six new database-backed gauges cover command states, active attempts, active
GPU allocations and ready hosts. Pending commands exclude expired entries;
active attempts include preparation and reservations before container start.
Ready hosts require admission, readiness and a fresh v2 observation.

All 32 expressions returned nonempty results through Grafana's authenticated
datasource proxy. An authenticated browser visited all 20 panels: all query
responses succeeded and supplied data frames, with no browser errors or
missing-data display. Tower's dashboard, provider and datasource hashes remain
unchanged. Native Grafana remained PID 182240 throughout activation.

## Executed verification and preservation

- The final selected regression gate passed **420 tests**, with zero failures,
  errors or skips, against a fresh local Unix-socket-only PostgreSQL cluster.
  The optional Docker Compose smoke class was explicitly deselected. No live
  credentials, Tower database or GPU were used by that gate.
- Both production workers durably acknowledged health commands, carrying
  protocol 2 and their actual worker-session IDs. Six original compatibility
  routes were probed over authenticated live HTTP; the final preemption poll
  was also checked over live HTTP. The admin preemption guard was checked in
  an isolated process using the deployed handler and an injected admin identity,
  with assertions that neither enqueue nor notification could run.
- All three application containers are healthy, active, and match twelve
  runtime source fingerprints each: eight changed files and four preserved
  production files. All eight enforceable repair policies remain `report_only`.
- Twelve worker package signatures were verified locally and against public
  distribution. Both installed workers match source SHA-256
  `791751ace206dad948d813e491cbf8e52de6ccfa6a88df28b8111064ea575ab1`.
- Worker handoffs explicitly deferred startup GPU benchmarking and restored
  normal unit configuration afterward. The first workstation verification raced
  a later startup log message and automatically restored the prior worker;
  waiting for both negotiation and benchmark-deferral messages fixed the
  verifier. The retry and Tower handoff passed. No benchmark or customer
  workload was launched.
- Tower VTT PID 1515078 and serving container PID 1514666 retained their
  identities. Workstation VTT identity also matched before/after. Existing
  serving containers were not restarted by the worker handoffs.
- Schema remains 125. All twelve protected stopped-job row hashes match the
  baseline. The protected volume metadata, zero attachments and zero snapshots
  remain unchanged. Unrelated local work and four live production source files
  were preserved. Environment files are ignored and no credentials are recorded.
- Public HTTPS checks using curl returned 200 for health, readiness, pricing
  and OAuth providers; metrics remained blocked with 404. A separate urllib
  probe with its default Python user agent received 403 and is recorded
  separately; it was not counted as a passing smoke check.

Initial gate failures remain in private local evidence. They included old
compatibility expectations, an invalid test command-claim fixture, and the
new preemption test expecting FastAPI's default error envelope instead of the
application envelope. The final gate above passed after those fixture fixes.

## Rollback and remaining scope

The production release directories
`/opt/xcelsior-releases/active-cutover-20261020` and
`/opt/xcelsior-releases/active-cutover-final-20261020` retain previous images,
configuration and source backups. Each worker retains its prior signed files.
To roll back the protocol, first restore compatibility policy on the current
API by changing scheduler mode to the prior canary scope before starting older
workers. Then restore the worker package and prior application images using
readiness-checked handoffs. Roll back monitoring through only its owned files;
preserve Tower providers, dashboards and datasource settings. A complete
production rollback drill was not performed by this change.

The runtime cutover is complete. Physical schema contraction, historical data
recovery/disposition, fair-share/preemption, repair enforcement qualification
and alert receiver delivery are separate work. No long zero-use observation
wait is imposed on this completed runtime cutover, and no restoration or data
deletion is claimed for the historical records.
