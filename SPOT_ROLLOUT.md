# Spot Instances — Staging & Production Rollout

*Last updated: 2026-06-09*

Operator checklist for Phase 10 of the spot integration plan. On-call incidents: [SPOT_RUNBOOK.md](SPOT_RUNBOOK.md).

---

## Prerequisites

- Migrations through **040** (`040_retire_spot_bidding.py`) merged and tested in CI
- API, worker agent, and frontend deployed from the same release tag
- `XCELSIOR_SPOT_ENABLED` documented in environment (default **true**)

---

## 10.1 Staging checklist

Run automated coverage locally:

```bash
pytest tests/test_spot_e2e_staging.py -q
```

Run live staging smoke (public endpoints; no login required):

```bash
python scripts/spot_staging_smoke.py --base-url https://staging.xcelsior.ca
```

Optional full launch (needs audit credentials in `.env.audit`):

```bash
python scripts/spot_staging_smoke.py --base-url https://staging.xcelsior.ca --launch
```

| Step | Verification |
|------|----------------|
| Migration 040 on staging DB | `alembic current` shows `040`; `jobs.pricing_mode` and `jobs.spot_rate_cad` exist |
| Spot launch → meter at spot rate | Launch RTX 4090 with `pricing_mode: "spot"`; wallet debits at locked `spot_rate_cad`, not host on-demand |
| On-demand preempts spot | With one GPU occupied by spot, launch on-demand on same host; spot → `queued`, worker stops container |
| Spot requeues | After preemption, `preemption_count` increments; job returns to queue with `pricing_mode=spot` |
| `spot_enabled=false` offer skipped | Provider disables spot; spot launch allocates elsewhere or queues |
| Provider floor | `spot_min_cents` never undercut in allocation / quote preview |
| No bid fields | Browser network tab: launch payload has `pricing_mode` only; no `max_bid` |

---

## 10.2 Production rollout

1. **Maintenance window** — apply migration 040 (brief API restart).
2. **Lockstep deploy** — API + worker agent + frontend from one release.
3. **Smoke tests**

   ```bash
   python scripts/billing_prod_smoke.py
   python scripts/spot_staging_smoke.py
   ```

   Manual: one spot launch on production; confirm published rate and preemption acknowledgement in UI.

4. **Monitor 24h** — `xcelsior_spot_preemptions_total`, `spot.preempted` / `spot.requeued` logs, support tickets.
5. **Feature flag** — if smoke used `XCELSIOR_SPOT_ENABLED=false` during cutover, set `true` after smoke pass and restart API.

---

## 10.3 Rollback

| Action | Effect |
|--------|--------|
| `XCELSIOR_SPOT_ENABLED=false` + API restart | New spot launches return **503**; UI shows kill-switch banner |
| Running spot jobs | **Drain** (recommended): let jobs finish or preempt normally; do not mass-kill |
| Migration 040 | **Not reversible** — `max_bid` stripped from payloads; acceptable per product decision |

Re-enable after fix: `XCELSIOR_SPOT_ENABLED=true`.

---

## Phase 10 exit criteria

- Production spot and on-demand coexist on shared hosts with correct priority
- Bidding fully absent (`pytest tests/test_no_max_bid_references.py`)
- Wallet debits match locked spot rate ± tax; provider payout matches allocation price
- Staging contention test demonstrated (on-demand preempts spot)

---

## Related scripts & tests

| Artifact | Purpose |
|----------|---------|
| `scripts/spot_staging_smoke.py` | Staging / prod infra smoke |
| `scripts/billing_prod_smoke.py` | Billing path + spot feature endpoints |
| `tests/test_spot_e2e_staging.py` | CI mapping to §10.1 checklist |
| `SPOT_RUNBOOK.md` | On-call playbooks |