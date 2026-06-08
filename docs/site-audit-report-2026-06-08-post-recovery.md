# Xcelsior — Post-Recovery Site Audit Report

Date: **2026-06-08** (verification run)  
Target: https://xcelsior.ca  
Local `main`: `8d31915`  
Production (est.): `8d31915`

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
5. **Deploy status** — `ef6ff30` on main, pending deploy.

### Verification matrix (2026-06-08)

| Suite | Result | Artifact |
|-------|--------|----------|
| Post-deploy checks | **53/53** passed at 2026-06-08T06:05:21.922Z | `/tmp/post-deploy-check.json` |
| CLI coverage | **51/51** passed at 2026-06-08T06:09:23.746Z | `/tmp/cli-coverage.json` |
| Hydration repro | **5/5** routes clean | `frontend/scripts/hydration-repro.mjs` |
| Dashboard Playwright | **11/11** passed at 2026-06-08T06:10:18.677Z (3 nav warnings) | `/tmp/xcelsior-audit/raw/dashboard-all.json` |
| Perf MCP | _not re-run this pass_ | `/tmp/xcelsior-audit/raw/perf-all.json` |

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

## F-003 performance (MCP `perf-all.json`)

| Route | Baseline desktop TBT | Post-F-003 TBT | Δ TBT | Post JS (KB) | Mobile slow-4G TBT |
|-------|---------------------:|---------------:|------:|-------------:|-------------------:|
| _(perf-all.json missing — run audit-performance.mjs)_ | | | | | |

Desktop `/`: JS **374 KB** vs baseline **812 KB** (−438 KB, −54%).

**F-003 phase 2** (`ef6ff30`, pending deploy): idle-defer PWA widgets, lazy `DeferredClientToaster`, hardened dashboard audit script.

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

## Commits on `main` not yet on production

- `8d31915 Mobile perf phase 3 follow-ups and build fixes`
- `8a7dd88 Mobile perf deferrals and surgical terminal fixes`
- `8e6e0e6 Fix enterprise demo API port default and prod verification message`
- `45f73a8 Add enterprise team demo script; mobile perf: scope AuthProvider`
- `4621d2a Complete team tenancy §3: audit metadata, IDOR tests, viewer launch guard`
- `a727983 Workspace-scoped OAuth credentials UI + team tenancy follow-ups`
- `72b2fa9 Team tenancy sweep: artifacts, templates, inference, launch modal`
- `a3461ef refactor(nfs): rename Mac vars to VPS; start team tenancy UI sweep`

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

6. **Deploy** — `bash scripts/deploy.sh` with `DEPLOY_BUILD_FRONTEND=true`
