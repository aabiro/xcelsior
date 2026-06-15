# Changelog

All notable changes to Xcelsior are documented here.

## [1.0.0] — 2026-06-15

First stable production release for [xcelsior.ca](https://xcelsior.ca).

### Added

- **Interactive instances:** Web terminal (xterm.js + WebGL), SSH via `connect.xcelsior.ca`, root password on dashboard, ASCII MOTD banner.
- **Blue-green deploy:** Zero-downtime API and SSH gateway handoff; deploy colour state persisted outside the synced tree.
- **SSH connect panel:** Command + password surfaced above the web terminal for running interactive instances.
- **GPU picker:** Admission-aware slot display; queue messaging when GPUs are busy.
- **Container UX:** Default `cd /workspace`, host tools bind-mount (rsync, etc.), profile/MOTD injection on launch.

### Fixed

- **Worker monitor:** Interactive jobs no longer marked `cancelled` on worker restart, requeue, or deploy — fixes false “cancelled during execution” in the UI.
- **Requeue / reaper:** `submitted_at` reset on requeue; reaper skips `gpu_busy` hosts.
- **Terminal MOTD:** Box borders aligned for xterm (fixed inner width).
- **Instance enrichment:** `host_ip` and computed public SSH port always surfaced for interactive instances.
- **Deploy:** Conditional service restarts; no in-place API kill on standby failure; SSH gateway drain capped at 300s.

### Changed

- Timeline copy for ended instances: “Instance ended while running” instead of implying a user cancel.
- Web terminal renderer restored to WebGL with canvas fallback.

[Release commit](https://github.com/aabiro/xcelsior/commit/2e9ba59)

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