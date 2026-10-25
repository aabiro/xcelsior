# Executed native queue, transport and reservation handoff

**Superseded runtime/worker state:** the
[physical GPU binding handoff](2026-10-25-native-gpu-device-binding.md) records
7a1aa0ee and newly activated signed agents. This receipt retains the earlier
1a0ae249 execution, billing and recovery proof.

Production API, scheduler, background worker and SSH gateway now run
**1a0ae249**, at schema **141**. This handoff ran no migrations. Seven dedicated
database-role contexts passed. All eight repairs remain `report_only`, and
preemption/deadline force remain disabled. Both existing signed GPU agents were
verified byte-identical to this source and reused with zero restarts. The MCP
runtime image is unchanged; its readiness and generated client typecheck pass.

The release permanently retires v1 work polling, destructive command draining,
raw instance status PATCH and inline placement endpoints behind authenticated
410 responses. Their execution bodies and the old queue walkers are physically
removed. Admission persists intent; the dedicated native scheduler owns placement.
Lease claim and queue backoff now commit their job projection and outbox event
atomically, with rollback and replay checks.

Placement reads canonical geography/capabilities. Reservation rechecks locked
readiness, region, egress isolation and the approved price ceiling. Unknown or
invalid host prices cannot pass a positive ceiling. Native spot admission freezes
an explicit active-catalog quote before the immutable spec is persisted; replay
does not reprice. Spot policy and unreleased GPU reservations are revalidated
under host/device locks, including simultaneous claims for one spot slot.

The final frozen regression passed **4,064 tests**, with zero failures, errors or
skips and 1,137 unchanged source hashes. PostgreSQL and Redis teardown passed.
The actual non-root image passed startup, schema, metrics, twelve signatures and
five retired HTTP endpoint checks. Generated inventory/OpenAPI checks, MCP client
typecheck and scoped static checks pass. Earlier failed diagnostics remain
retained; they are superseded by this complete qualification.

Actual spot job **8061e0b0** succeeded on TUF, attempt
**72fd947b-cf3d-49d0-ad05-c5e38be467da**, fence **11**. Durable HTTP logs contain
`XCL_NATIVE_CUDA_RESULT=32768`. Its admitted spot rate, captured attempt quote and
meter rate agree; its wallet debit is **100 micro-CAD**. Across six actual
CUDA/lifecycle jobs, seven meters, eight charged cycles and eight wallet debits
agree at **57,500 micro-CAD**. All attempt pointers, leases and allocations are
released, with zero wallet-ledger mismatches. Repeated HTTP log reads/download,
read-only invoice preview and CSV also pass. No external provider payment is
claimed.

Backup, direct-migrator restore drill and partition maintenance executed
successfully on this release. All three timers remain enabled/active, with
fifteen retained prior backups. The prior full 155-table hash-verified restore
remains a dated proof; this receipt does not relabel it as a new full snapshot.
The post-incident read-only data audit covered 177 physical tables and 2,077
columns, with zero scan errors. The 201,301 qualified empty invoices retain their
exact prior record-set digest and have not been archived or deleted.

The [sanitized receipt](2026-10-25-native-release-production-receipt.json) records
source, image and runtime evidence. The
[shared-filesystem incident](2026-10-25-shared-filesystem-recovery.md) records the
PixelEnhance temporary-file/Redis failures, recovery and new release-space gate.
That gate was used before this image transfer and runtime handoff.

Remaining work is explicit:

- Native physical GPU/model/VRAM and freshness reads still have JSONB and
  synthetic-inventory compatibility paths; their reservation revalidation is
  not claimed complete.
- Two older attempts, from July 18 and October 17, lack captured billing quotes
  and meters. The usage-evidence alert remains active. Historical invoice
  disposition and other unknown historical facts still require qualified work.
- Permanent observation-session retirement and remaining unused diagnostic/raw
  lifecycle helpers are not described as removed.
- Tower root usage was about 94%, with its disk alert active. Its unrelated
  workloads and data were preserved.
- Independent offsite/WAL/PITR/object recovery and failover acceptance remain
  unverified. Production repair/preemption enforcement remains disabled.

No complete control-plane or data-layer cutover claim is made.
