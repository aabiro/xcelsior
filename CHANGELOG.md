# Changelog

All notable changes to Xcelsior are documented here.

## [Unreleased]

### Breaking changes

- **Spot bidding removed.** The `max_bid` field is no longer accepted on `POST /instance` or any spot API. Launch interruptible workloads with `pricing_mode: "spot"` and accept the published spot rate (CAD/hr) shown at launch. Preemption is capacity-driven (on-demand contention), not bid-based.

### Added

- **Spot instances (production):** Unified spot pricing, `pricing_mode=spot` launch path, capacity-based preemption, provider spot controls (`spot_enabled`, `spot_gpu_slots`, `spot_min_cents`), and observability (`spot.*` logs, `xcelsior_spot_*` Prometheus metrics).
- **Feature flag:** `XCELSIOR_SPOT_ENABLED` — global kill switch for new spot launches (503 when disabled).
- **Ops:** `SPOT_RUNBOOK.md`, `SPOT_ROLLOUT.md`, `scripts/spot_staging_smoke.py`, `POST /spot/preemption-cycle?dry_run=true`, `GET /api/pricing/spot-enabled`.
- **Phase 10:** `tests/test_spot_e2e_staging.py` maps staging checklist to CI; `billing_prod_smoke.py` includes spot endpoint checks.

### Changed

- Spot rate is locked at allocation (`spot_rate_cad`); billing meters at that rate for the job lifetime.
- Provider earnings distinguish `spot_earned_cad` vs `on_demand_earned_cad`.

### Removed

- `max_bid`, `submit_spot_job`, bid-gated queue logic, and bid-based preemption.