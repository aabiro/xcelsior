# Mobile perf — phase 3 changes (2026-06-08)

Commit: `8a7dd88` + follow-ups on `main`.

## Goal

Cut mobile slow-4G TBT on `/` and `/pricing` by ≥50% vs 2026-06-05 baseline (was ~28–31s).

## Changes landed

| Area | Change |
|------|--------|
| Homepage | Hero-only in `page.tsx`; below-fold in `home-below-fold.tsx` via `dynamic()` |
| Pricing | `SavingsCalculator` dynamically imported |
| Fonts | `Geist_Mono` scoped to dashboard layout only |
| Auth | `AuthProvider` mounted on dashboard + auth routes only (not marketing root) |
| Shell | Footer deferred via `dynamic()`; footer logos no longer `priority` |
| Cache | `/_next/static/*` immutable 1y headers |
| i18n | `en-public` / `fr-public` static; dashboard dictionaries lazy on `/dashboard` |
| Motion | `MarketingMotion` uses `LazyMotion` + `domAnimation` |
| PWA / chat | `RootClientWidgets` + `MarketingChatWidget` idle-deferred |

## Measurement

### Bundle analyzer (local)

```bash
cd frontend && npm run analyze
```

Opens treemap for client chunks on `/` and `/pricing` after build.

### Perf MCP (prod)

```bash
bash scripts/redo_when_prod_up.sh          # full — includes perf MCP
bash scripts/redo_when_prod_up.sh --quick  # skip perf
```

Artifacts: `/tmp/xcelsior-audit/raw/perf-all.json`

### Targets

| Route | Metric | Baseline (2026-06-05) | Target |
|-------|--------|-------------------------|--------|
| `/` | Mobile TBT | ~28–31s | < 8s (stretch < 5s) |
| `/` | Desktop TBT | ~7s | < 2s |
| `/pricing` | Mobile TBT | ~28–31s | < 8s |

## Next items

- Re-run perf MCP post-deploy and paste rows into this file
- Lazy-load remaining heavy marketing routes (`/gpu-availability` is already fetch-only)
- Verify GTM `lazyOnload` has no regression in Network waterfall