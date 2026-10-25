# Canonical geography and capability facts for placement

**Accepted follow-up:** deployed as `1a0ae249`/schema 141 after 4,064 passing checks and real spot CUDA/billing acceptance. The [executed receipt](2026-10-25-native-release-production-handoff.md) supersedes the source-candidate status and diagnostics below. Repairs remain `report_only`.

The scheduler fleet snapshot now overlays canonical region, country, province
and capabilities exactly as host API reads already do. Payload aliases cannot
change geography or grant tenant egress isolation, and null/empty canonical
facts never fall back to a payload claim. This closes four remaining JSONB
placement reads without converting unknown historical facts into authority.

Three native database/filter tests cover conflicting and missing canonical facts.
The expanded placement, preference, shadow and native worker selection passed
126 checks with no failures/errors/skips. Positive raw-SQL fixtures explicitly
write their approved capability into the native column; the first diagnostic
run exposed 37 tests that had only seeded payload capability, and is retained.
No runtime fallback was added for those fixtures.

Runnable gates include tests/test_native_fleet_projection.py,
tests/test_native_host_column_contract.py, tests/test_agent_v2.py,
tests/test_control_plane_shadow.py, tests/test_native_placement_preference.py
and tests/test_control_plane_reservation.py through the isolated security harness.
The broader diagnostic was deliberately interrupted after 526 checks exposed
15 stale fixture failures and one setup error. Native fixture authority was
corrected, then all 76 impacted migration/placement/reconciliation/outbox checks
passed in a fresh database. PostgreSQL and Redis teardown passed on the
interrupted run; it is not a passing qualification. The full unified candidate
regression remains required. This source follow-up is not
pushed or deployed; production remains b935d694/schema 141.

Synthetic inventory creation, payload GPU/model/VRAM and freshness reads, full
reservation hard-filter revalidation, permanent observation-session retirement,
historical invoice disposition and independent offsite recovery remain open.
