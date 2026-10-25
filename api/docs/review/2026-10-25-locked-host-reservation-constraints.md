# Host constraints revalidated at reservation

**Accepted follow-up:** deployed as `1a0ae249`/schema 141 after 4,064 passing checks and real spot CUDA/billing acceptance. The [executed receipt](2026-10-25-native-release-production-handoff.md) supersedes the source-candidate status and diagnostics below. Repairs remain `report_only`.

The reservation transaction now checks the locked host's native status, region
and capabilities against the admitted workload before allocating any device.
It also checks the current host price against the customer's approved ceiling.
A score from an earlier transaction cannot bypass a region change, revoked
egress isolation, an inactive host or a price increase. Canonical null/empty
facts do not fall back to payload aliases.

Filter policy `filters/v5` rejects unknown, malformed, Boolean or non-finite host
prices when a positive approved ceiling applies. Invalid explicit ceilings
cannot silently become unbounded. An absent or zero ceiling retains the existing
unbounded admission contract; this change does not invent a price for billing.

The 21 new checks in `tests/test_reservation_host_constraints.py` cover a host
that passed scoring, subsequent changes without an inventory-generation bump,
complete reservation rollback and conflicting payload aliases. Together with
reservation atomicity, concurrent spot capacity and native fleet filters, the
focused gate passed **53 checks**, with no failures, errors or skips. Scoped
Black, Ruff and Pyright passed.

Run these tests through `api/scripts/run_security_gates.py` using a fresh local
Unix-socket PostgreSQL instance. The complete frozen regression and actual-image
qualification remain required before release. Production is still
`b935d694`/schema 141; all eight repairs remain `report_only`.

This closes the listed host-row races only. Physical GPU/model/VRAM and freshness
still have remaining JSONB reads and synthetic-inventory compatibility code.
Historical invoice disposition, permanent observation-session retirement and
independent offsite recovery remain open. Production fairness/preemption
enforcement stays disabled.
