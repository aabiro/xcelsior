# Spot Instances — Operations Runbook

*Last updated: 2026-06-09*

On-call guide for interruptible (spot) GPU instances alongside on-demand workloads.

---

## Quick reference

| Signal | Likely cause | First action |
|--------|--------------|--------------|
| Spike in `xcelsior_spot_preemptions_total` | On-demand contention or host drain | Check queue depth + host GPU utilization |
| Spot rate ≈ on-demand (`xcelsior_spot_rate_cad`) | Supply exhaustion / surge cap | Review `spot.price_updated` logs; add capacity |
| Provider payout dispute | Wrong allocation price | Verify `gpu_allocations.price_cents_per_hour` |
| Customers cannot launch spot (503) | `XCELSIOR_SPOT_ENABLED=false` | Confirm intentional kill switch |

---

## Spike in preemptions

**Symptoms:** Many `spot.preempted` / `spot.requeued` log lines; renters report frequent interruptions.

**Diagnosis**

1. Check metrics: `GET /metrics` → `spot.jobs_running`, `spot.preemptions_total`
2. Prometheus: `rate(xcelsior_spot_preemptions_total[5m])`
3. List queued on-demand jobs: high `queue_depth` with low `active_hosts` suggests capacity crunch
4. Review recent host drains: `host_update` events with `status=draining`

**Mitigation**

- Add GPU capacity (register hosts or undrain after maintenance)
- Temporarily reduce spot pool on congested hosts (`spot_gpu_slots`)
- If abuse/noise: disable new spot launches (`XCELSIOR_SPOT_ENABLED=false`) — running jobs continue until preempted or stopped

**Dry-run preemption (ops)**

```bash
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$API/spot/preemption-cycle?dry_run=true"
```

---

## Spot price stuck at on-demand cap

**Symptoms:** `savings_pct` near 0; spot launch modal shows little discount.

**Diagnosis**

1. `GET /spot-prices` — compare `rate_cad` vs `on_demand_cad` per GPU model
2. Logs: `spot.price_updated` with `supply` / `demand` fields
3. Unified engine caps spot at on-demand; zero supply triggers max surge

**Mitigation**

- Register additional hosts for the affected GPU model
- Confirm providers have `spot_enabled=true` and reasonable `spot_min_cents` floors
- Trigger manual refresh: `POST /spot-prices/update` (admin)

---

## Provider complains about spot payouts

**Symptoms:** Earnings mismatch; spot job ran but payout lower than expected.

**Verification**

1. Instance detail: `pricing_mode=spot`, `spot_rate_cad` locked at allocation
2. Marketplace path: `gpu_allocations.price_cents_per_hour` for `allocation_type=spot`
3. Provider dashboard: earnings split `spot_earned_cad` vs `on_demand_earned_cad`
4. Billing: meter closed at `preempted_at` — payout is **actual runtime only**

**Reminder for providers:** Spot jobs are interruptible; preemption is capacity-driven, not bid-based.

---

## Kill switch

Set `XCELSIOR_SPOT_ENABLED=false` and restart API.

- New spot launches return **503** with a clear message
- UI shows a maintenance banner on the launch modal
- Running spot instances are **not** mass-killed; they drain via normal preemption or renter stop

Re-enable after fix: `XCELSIOR_SPOT_ENABLED=true`

---

## Related endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/pricing/spot-enabled` | Feature flag status for UI |
| `GET /spot-prices` | Live unified spot quotes |
| `POST /spot/preemption-cycle` | Capacity preemption (or `?dry_run=true`) |
| `GET /metrics/prometheus` | `xcelsior_spot_*` series |