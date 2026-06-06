# Xcelsior — Post-Recovery Site Audit Report

Date: **2026-06-06** (verification run)  
Target: https://xcelsior.ca  
Local `main`: `05dbb0b`  
Production (est.): `05dbb0b`

This is the **third verification pass** — synthesizing automated script output after VPS recovery (2026-06-05). It supersedes the stale generator output in `site-audit-report-2026-06-04-reaudit.md` and consolidates:

| Prior report | Role |
|--------------|------|
| `docs/site-audit-report-2026-06-04.md` | Baseline MCP audit (pre-remediation) |
| `docs/site-audit-report-2026-06-04-reaudit-mcp.md` | Second MCP crawl (pre-deploy prod) |
| **This report** | Post-deploy + post-recovery script verification |

Raw artifacts: `/tmp/xcelsior-audit/raw/` (perf, dashboard, desktop), `/tmp/post-deploy-check.json`, `/tmp/cli-coverage.json`.

---

## Executive summary

1. **Production is healthy** — Public `healthz` **200**, SSH to origin OK, all automated verification scripts pass.
2. **P1 findings cleared on prod** — F-001, F-002, F-004, F-005 verified via **51/51** post-deploy checks and **5/5** hydration routes.
3. **F-003 improved, not done** — Marketing JS **~374 KB** on `/` (was **~812 KB**); desktop TBT **~7.2 s** (was **~9.2 s**). Mobile slow-4G TBT still **25–31 s** on key routes.
4. **Dashboard coverage complete** — Playwright authenticated crawl **10/10** routes (hardened audit script, `ef6ff30`).
5. **Deploy status** — `05dbb0b` deployed to production (2026-06-06).

### Verification matrix (2026-06-06)

| Suite | Result | Artifact |
|-------|--------|----------|
| Post-deploy checks | **51/51** passed at 2026-06-06T03:33:41.832Z | `/tmp/post-deploy-check.json` |
| CLI coverage | **51/51** passed at 2026-06-06T03:12:21.001Z | `/tmp/cli-coverage.json` |
| Hydration repro | **5/5** routes clean | `frontend/scripts/hydration-repro.mjs` |
| Dashboard Playwright | **10/10** passed at 2026-06-06T03:13:16.854Z (1 nav warnings) | `/tmp/xcelsior-audit/raw/dashboard-all.json` |
| Perf MCP | captured 2026-06-05 (30 rows) | `/tmp/xcelsior-audit/raw/perf-all.json` |

### API / SEO signals (post-deploy)

| Check | Result |
|-------|--------|
| `GET /api/auth/me` (anon) | 200 — user null ✓ |
| `GET /api/auth/me` (invalid bearer) | 401 |
| `GET /api/v2/gpu/available` | 200 |
| GPU canonical | https://xcelsior.ca/gpu-availability |
| Sitemap /gpu-availability | yes |
| Sitemap /download | yes |

---

## Recovery timeline (2026-06-05)

| Event | Action |
|-------|--------|
| Cloudflare **522** + SSH timeout | Origin audits via Tailscale (`AUDIT_ORIGIN_IP=100.64.0.1`) |
| Vultr hard restart | Postgres/Docker volumes intact |
| Headscale ACL fix | SSH for `linuxuser@tag:xcelsior` |
| Redeploy F-003 | `dfeff69`, `5212499` on prod |
| Full redo | `bash scripts/redo_when_prod_up.sh` → 51/51 + hydration + dashboard |

---

## Remediation status (F-001 – F-022)

| ID | Sev | Commit(s) | Fix | Prod |
|----|-----|-----------|-----|------|
| F-001 | P1 | 5fc95d3 | 200 + user:null; defer /api/auth/me on marketing | prod ✓ |
| F-002 | P1 | 5fc95d3 | Public GPU endpoint; honest degraded UI | prod ✓ |
| F-003 | P1 | dfeff69,5212499,ef6ff30 | i18n split, lazy framer-motion, idle PWA/chat/toaster | prod ✓ |
| F-004 | P1 | eb338b2 | CF email obfuscation hydration; ObfuscationSafeMailto | prod ✓ |
| F-005 | P1 | 5fc95d3 | Per-route canonicals; sitemap download + gpu-availability | prod ✓ |
| F-006 | P1 | 5fc95d3 | Deduped security headers at app layer | prod ✓ |
| F-007 | P1 | 5fc95d3 | no-store for auth/dashboard/offline | prod ✓ |
| F-008 | P2 | 5fc95d3 | Header overlap / BETA placement | prod ✓ |
| F-009 | P2 | 5fc95d3 | 44px touch targets | prod ✓ |
| F-010 | P2 | 5fc95d3 | 16px inputs on mobile | prod ✓ |
| F-011 | P2 | 5fc95d3 | Mobile pricing affordance (partial) | prod ✓ |
| F-012 | P2 | 5fc95d3 | SW update flow improvements | prod ✓ |
| F-013 | P2 | 5fc95d3 | aria-expanded, body scroll lock | prod ✓ |
| F-014 | P2 | 181ad74 | Contrast: Beta, sovereignty badge, support CTA | prod ✓ |
| F-015 | P2 | 181ad74 | H1 spacing (hero_line1 trailing space) | prod ✓ |
| F-016 | P2 | 5fc95d3 | Terms required + a11y errors | prod ✓ |
| F-017 | P2 | 181ad74 | Privacy third-party disclosure (GTM, CF) | prod ✓ |
| F-018 | P3 | 181ad74 | Organization @id; blog publisher reference | prod ✓ |
| F-019 | P3 | b87c8f1 | GTM lazyOnload, no preload | prod ✓ |
| F-020 | P3 | 181ad74 | Install banner route allowlist | prod ✓ |
| F-021 | P3 | 181ad74 | Stable UTC dates in sitemap, RSS, OG | prod ✓ |
| F-022 | P3 | — | SVG logos OK (accepted) | accepted |

---

## F-003 performance (MCP `perf-all.json`, 2026-06-05)

| Route | Baseline desktop TBT | Post-F-003 TBT | Δ TBT | Post JS (KB) | Mobile slow-4G TBT |
|-------|---------------------:|---------------:|------:|-------------:|-------------------:|
| / | 9166 | 7168 | -1998 | 374 | 30747 |
| /pricing | 52596 | 6043 | -46553 | 374 | 30940 |
| /blog | 13662 | 4398 | -9264 | 353 | 28058 |
| /download | 9721 | 4911 | -4810 | 359 | 28469 |
| /privacy | 9239 | 5720 | -3519 | 341 | — |
| /terms | 9209 | 5610 | -3599 | 337 | — |
| /gpu-availability | — | 4968 | — | 359 | 28093 |

Desktop `/`: JS **374 KB** vs baseline **812 KB** (−438 KB, −54%).

**F-003 phase 2** (`ef6ff30`, deployed): idle-defer PWA widgets, lazy `DeferredClientToaster`, hardened dashboard audit script.

---

## F-004 hydration (Cloudflare email obfuscation)

React #418 on `/privacy` and `/terms` was caused by Cloudflare Email Obfuscation rewriting mailto text after SSR. Fixed with `ObfuscationSafeMailto` (`eb338b2`). Verified **5/5** on prod via `hydration-repro.mjs`.

Optional operator action: Cloudflare Scrape Shield → disable Email Obfuscation if plaintext mailto in View Source is required.

---

## Dashboard audit (authenticated)

Account: `site-audit@xcelsior.ca` (provisioned via `scripts/provision_audit_user.sh`).

```bash
bash scripts/run_audit_dashboard.sh   # Playwright UI crawl
```

Routes probed: `/dashboard`, `/dashboard/instances`, `/dashboard/hosts`, `/dashboard/billing`, `/dashboard/settings`, `/dashboard/notifications`, `/dashboard/marketplace`, `/dashboard/volumes`, `/dashboard/analytics`, `/dashboard/compliance`.

---

## Security sweep (`ef6ff30`, repo)

| Change | Detail |
|--------|--------|
| `/v1/chat/completions` | Requires auth + `inference:write` |
| `/api/v2/inference/complete/{id}` | Worker auth required |
| Events / inference v1 | Ownership guards + IDOR regression tests |
| `/api/audit/instance/{job_id}` | Added `events:read` scope |

---

## Regenerate this report

```bash
bash scripts/redo_when_prod_up.sh          # full suite incl. perf MCP
bash scripts/redo_when_prod_up.sh --quick  # skip perf (~7 min)

# Or step-by-step:
node scripts/post_deploy_audit_check.mjs > /tmp/post-deploy-check.json
node scripts/audit_cli_coverage.mjs > /tmp/cli-coverage.json
BASE_URL=https://xcelsior.ca node frontend/scripts/hydration-repro.mjs
bash scripts/run_audit_dashboard.sh
REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs
```

---

## Recommended next steps

1. **Re-run perf MCP** — compare post-`ef6ff30` TBT vs `perf-all.json` baseline (2026-06-05)
2. **Monitor** — `bash scripts/redo_when_prod_up.sh --quick` after material changes
3. **F-003 target** — sub-2 s desktop TBT / acceptable mobile INP (optional further splitting)
4. **Full CI** — `CI=true XCELSIOR_ENV=test bash run-tests.sh` (2863 passed, 6 skipped as of 2026-06-05)
5. **Cloudflare optional** — disable Email Obfuscation if legal pages need plaintext mailto in HTML source

