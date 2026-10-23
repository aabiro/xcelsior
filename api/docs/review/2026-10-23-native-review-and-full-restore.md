# Native review follow-up and restore qualification

Runtime `231589345eb7f6bdfa9bce2c73e5ccc3283685c1` supersedes the operator
candidate `821d1a07`. Four review findings are addressed:

- Host eviction consumes the locked host version atomically with commands and
  rejects another executing eviction. Distinct approved plans cannot issue
  duplicate stop sets from the same snapshot. Contended job locks return a
  retryable conflict; no plan or host version is consumed on rollback.
- Preemption planning evaluates net team usage after selecting and validating
  exact victims. Only the nominated team's matching attempts receive quota
  credit. Reservation receives no credit and still waits for confirmed
  physical release. An oversized replacement issues no stop commands.
- A fenced failed terminal report for a preempted compute attempt can requeue
  its job at a new generation. The failed attempt remains recorded. Explicit
  user intent and mining policy still take precedence; a failure alone never
  creates preemption authority.
- An incomplete native burn estimate cannot clear wallet warnings. The
  response explicitly reports the unavailable estimate; zero balance remains
  independent of whether every running attempt has a captured rate.

## Passing source and image gates

| Gate | Evidence and scope |
|---|---|
| `native-review-final` | 154 passed, zero failures/errors/skips. Covers concurrent eviction plans, same-team replacement quotas, failed terminal settlement, native wallet estimates, fairness, worker protocol, billing snapshots/controllers, serverless lifecycle and agent/scheduler behavior. All eight changed source/test hashes match the committed runtime. Owned PostgreSQL stopped successfully. |
| `review-images-pg16` | Three actual images booted with Gunicorn on isolated PostgreSQL 16: transition at schema/floor 126, intermediate at schema 136/floor 134, and final at schema/floor 137. Each verified twelve worker signatures and exported metrics availability equal to one. UID 10001, read-only root, no network/capabilities; all containers and owned PostgreSQL cleaned up. |
| Static checks | Black/Ruff pass on the eight changed Python files; targeted runtime Pyright reports zero errors/warnings. The nine protected files in the user's root checkout retain their saved hashes. |

The 154-test run started with an uncommitted tree based on `342bf7e6`; its
recorded file hashes match `23158934`. The image gate used that clean commit.
The earlier clean 3,336-test selection predates both operator commits; it was
not repeated here. Its three explicit baseline exclusions and the existing
MCP unit skip remain open. Test selections overlap.

## Frozen handoff artifacts

The final image is `xcelsior-api:cutover-review-23158934`, ID
`sha256:788f6b8084caa223fd4a3a8d4b2e4f4444822041e6048ac8f2948f187ce71ec3`.
Its build context contains 3,407 tracked files, no runtime environment files
and no private signing key. Context archive SHA-256:
`c9a5d6844c521fe4640551f18dc2b7ea96c3257181c0f2261d97ca9d6fff743e`.

`review-handoff-images.tar.gz` contains this image and the required
`1af572b6` transition and `7757bdc3` intermediate images. Its 966,587,391
bytes hash to
`5ed3be54e4f272a390b75dca4084c28adad16d8249861958de7d6925fefb5a4b`.
Independent archive inspection verifies each image's configuration digest,
configuration contents, filesystem layer identities and platform against the
frozen local images. The archive has not been transferred to production.

The [staged handoff](2026-10-23-staged-control-plane-handoff.md) names the
three immutable images and their ordered writer/schema boundaries. The older
operator image/archive receipts remain historical evidence.

## Complete production backup and isolated restore

`full-production-restore-accepted` passed on PostgreSQL 16. A read-only
exported snapshot from the configured production database on port **5434**
was shared with `pg_dump` at 2026-10-23 19:20:22 UTC. The private 52,421,073-byte
archive has SHA-256
`0da31acb43195c568e4c224fe9c50d253512cd07f748b9eb679eeb6dc7dcde1d`.
Archive and snapshot metadata remain mode 0600 outside Git.

Restoring into a fresh local database matched counts and full-row aggregate
hashes for all **138** source tables. The cluster matched production's major
version and `C.UTF-8` database collation/ctype. Each migration target, 126
through 137, succeeded independently and preserved all rows and aggregate
hashes in **30 financial/marketplace tables**, plus `job_attempts`,
`agent_commands`, `job_logs` and `volumes`: **34 protected tables** at every
stage. All 364 jobs remained retained, with four exact recorded specs and 360
explicitly retained as unavailable. The final historical qualification audit
passed. No missing execution specification was invented.

The migration source used in the drill is frozen at `342bf7e6`; all 140
migration Python files match the final `23158934` runtime byte for byte.
The gate also checked that the runner's separate test database retained its
original schema. Fixture teardown dropped the restored database; runner
PostgreSQL shutdown passed. No application services contacted customers or
payment providers. The gate took 276.12 seconds and reported one passed test,
zero failures/errors/skips. Exact test arguments and script/manifest hashes
are in the adjacent JSON. This closes the complete-backup/isolated-restore
qualification gap recorded in the earlier scoped historical receipt.

## Production boundary and residual work

The fresh read-only probe at 2026-10-23 20:04 UTC still found production at
schema 125, no active attempt pointers/leases/allocations, three fresh service
heartbeat identities and dedicated roles without login. All eight repair
policies remain `report_only`. The serving API/scheduler/maintenance image is
still `cutover-audit-cb71106b`.

This qualification does not deploy the candidate. Repairs remain
`report_only`; default scheduling policy enables no preemption. Dedicated
roles and signed workers require the staged handoff, followed by a deployed
native launch/stop/restart, terminal-report recovery and financial/provider
settlement journey. Local image startup and an isolated restore cannot prove
those production outcomes.

Remaining observation adapters, planned scheduler/worker module decomposition,
the disabled old spot-cycle execution surface, baseline exclusions and MCP
skip remain explicit. B16 is not complete. Exact arguments, hashes, source
identities and cleanup receipts are retained in the adjacent JSON; raw private
backup data and diagnostic logs remain outside Git.
