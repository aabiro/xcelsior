# Xcelsior — Site Audit Re-Audit Report

Date: 2026-06-04  
Target: https://xcelsior.ca  
Deploy status: **Deployed** — post-deploy checks below

This report compares the **baseline audit** (`docs/site-audit-report-2026-06-04.md`), a **second full chrome-devtools MCP crawl** (`docs/site-audit-report-2026-06-04-reaudit-mcp.md`), and **remediation on `main`** (`5fc95d3` site-audit sprints 1–3, `b87c8f1` bundle split + full test coverage).

Raw MCP artifacts: `/tmp/xcelsior-audit/raw/`, screenshots: `/tmp/xcelsior-audit/screens/`.

---

## Executive summary

1. **Deployed 2026-06-04** (`a98e2db` API + frontend on `main`) — Post-deploy probes confirm **F-001**, **F-002** (API), and **F-005** (GPU canonical + sitemap) are **cleared on production**.
2. **Perf (F-003) — measured post-deploy** — `audit-performance.mjs` (2026-06-04): marketing JS on `/` dropped from **~812 KB → ~374 KB**; desktop TBT on `/` from **9166 ms → 5772 ms** (see table below).
3. **Hydration (F-004)** — Marketing chrome fixes deployed (`d56c5b5`–`e5823ee`): theme script removed, footer logo classes, client legal pages, chat skipped on legal. MCP still reports #418 on `/privacy` and `/terms` only; other routes clean.
4. **Test coverage** — `UNTESTED_ENDPOINTS.md`: **0** HTTP routes and **0** CLI commands without test signal.

### Post-deploy P1 status

| ID | Status | Evidence |
|----|--------|----------|
| F-001 | **Fixed** | `GET /api/auth/me` → `200` + `user: null` |
| F-002 | **Fixed** (verify UI in browser) | `/api/v2/gpu/available` → `200`; marketing defers auth probe |
| F-005 | **Fixed** (GPU + sitemap) | GPU canonical `https://xcelsior.ca/gpu-availability`; sitemap lists `/download` + `/gpu-availability` |
| F-003 | **Improved (post-deploy MCP)** | `/` JS ~374 KB (was ~812 KB); TBT ~5772 ms (was ~9166 ms) |
| F-004 | **Partial** | `/blog`, `/download`, `/about`, `/support` clear (MCP); `/privacy`, `/terms` still log React #418 in MCP (cosmetic; repro in dev build) |

### Post-deploy performance (MCP `perf-all.json`, 2026-06-04)

| Route | Pre-deploy TBT | Post-deploy TBT | Δ TBT | Post-deploy JS |
|-------|---------------:|----------------:|------:|---------------:|
| `/` | 9166 ms | 5772 ms | -3394 | 374 KB |
| `/blog` | 13662 ms | 5548 ms | -8114 | 313 KB |
| `/download` | 9721 ms | 5429 ms | -4292 | 316 KB |
| `/privacy` | 9239 ms | 5742 ms | -3497 | 313 KB |
| `/terms` | 9209 ms | 5454 ms | -3755 | 313 KB |

Desktop `/` (pre-deploy crawl): TBT **9166** ms, JS **812** KB. Slow 4G `/pricing` (pre-deploy): TBT **52596** ms.

### Production API signals (post-deploy curl, 2026-06-04)

| Signal | Status |
|--------|--------|
| `/api/auth/me` (logged out) | 200 + `user: null` |
| `/api/auth/me` (invalid bearer) | 401 |
| `/api/v2/gpu/available` | 200 |
| GPU canonical | `https://xcelsior.ca/gpu-availability` |

---

## Remediation matrix (findings → code)

| ID | Sev | Commit | Fix (summary) | Repo |
|----|-----|--------|---------------|------|
| F-001 | P1 | 5fc95d3 | 200 + user:null; defer /api/auth/me on marketing (b87c8f1) | In repo |
| F-002 | P1 | 5fc95d3 | Public GPU endpoint; honest degraded UI | In repo |
| F-003 | P1 | b87c8f1 | WalletConnect dashboard-only; lazy chat/PWA/GTM; deferred auth | In repo |
| F-004 | P1 | 5fc95d3 | Deterministic SSR dates / client-only islands | In repo |
| F-005 | P1 | 5fc95d3 | Per-route canonicals; sitemap download + gpu-availability | In repo |
| F-006 | P1 | 5fc95d3 | Deduped security headers at app layer | In repo |
| F-007 | P1 | 5fc95d3 | no-store for auth/dashboard/offline | In repo |
| F-008 | P2 | 5fc95d3 | Header overlap / BETA placement | In repo |
| F-009 | P2 | 5fc95d3 | 44px touch targets | In repo |
| F-010 | P2 | 5fc95d3 | 16px inputs on mobile | In repo |
| F-011 | P2 | 5fc95d3 | Mobile pricing affordance (partial) | In repo |
| F-012 | P2 | 5fc95d3 | SW update flow improvements | In repo |
| F-013 | P2 | 5fc95d3 | aria-expanded, body scroll lock | In repo |
| F-014 | P2 | — | Open — contrast tokens | Open |
| F-015 | P2 | 5fc95d3 | Accessible H1 spacing | In repo |
| F-016 | P2 | 5fc95d3 | Terms required + a11y errors | In repo |
| F-017 | P2 | — | Open — third-party disclosure | Open |
| F-018 | P3 | — | Open — duplicate JSON-LD | Open |
| F-019 | P3 | b87c8f1 | GTM lazyOnload, no preload | In repo |
| F-020 | P3 | — | Open — 404 promos | Open |
| F-021 | P3 | — | Open — blog date normalization | Open |
| F-022 | P3 | — | Accepted — SVG logos OK | Open |

---

## Health metrics: baseline vs re-audit MCP (production, both pre-deploy)

Interpretation: both crawls hit **live production** before `main` deploy. Differences are mostly measurement variance, not remediation.

| Route | Baseline LCP | Re-audit LCP | Δ LCP | Baseline TBT | Re-audit TBT | Δ TBT |
|-------|-------------:|-------------:|------:|-------------:|-------------:|------:|
| / | 5000 ms | 5212 ms | +212 | 10293 ms | 9166 ms | -1127 |
| /about | 4280 ms | 1780 ms | -2500 | 13274 ms | 10537 ms | -2737 |
| /accept-invite | 1876 ms | 1912 ms | +36 | 10823 ms | 9500 ms | -1323 |
| /blog | 5952 ms | 2880 ms | -3072 | 15697 ms | 13662 ms | -2035 |
| /blog/security-is-not-a-feature-its-the-infrastructure | 3736 ms | 3020 ms | -716 | 10088 ms | 10620 ms | +532 |
| /dashboard | 2748 ms | 2336 ms | -412 | 11646 ms | 9677 ms | -1969 |
| /download | 5140 ms | 2304 ms | -2836 | 10866 ms | 9721 ms | -1145 |
| /features | 1340 ms | 1780 ms | +440 | 9458 ms | 9251 ms | -207 |
| /forgot-password | 3528 ms | 2716 ms | -812 | 9386 ms | 9779 ms | +393 |
| /login | 2144 ms | 1680 ms | -464 | 10258 ms | 10464 ms | +206 |
| /nonexistent-bogus-404 | 2328 ms | 2156 ms | -172 | 10215 ms | 9000 ms | -1215 |
| /privacy | 3344 ms | 3080 ms | -264 | 12043 ms | 9239 ms | -2804 |
| /register | 2172 ms | 2608 ms | +436 | 10417 ms | 10989 ms | +572 |
| /reset-password | 2028 ms | 2036 ms | +8 | 10269 ms | 11888 ms | +1619 |
| /setup-2fa | 3184 ms | 2184 ms | -1000 | 10768 ms | 10048 ms | -720 |
| /sovereignty | 3492 ms | 1888 ms | -1604 | 12078 ms | 9249 ms | -2829 |
| /verify-email | 2508 ms | 1780 ms | -728 | 10729 ms | 10250 ms | -479 |
| /~offline | 3424 ms | 2880 ms | -544 | 10259 ms | 8526 ms | -1733 |

Full tables: baseline § Health Snapshot; re-audit § Health Snapshot in `site-audit-report-2026-06-04-reaudit-mcp.md`.

---

## What should change after deploy

| ID | Expected on production after deploy |
|----|-------------------------------------|
| F-001 | No `/api/auth/me` on marketing HTML first paint; API returns `200` + `user: null` when probed |
| F-002 | GPU page uses `/api/v2/gpu/available`; no `/hosts` 401; degraded copy when empty |
| F-005 | GPU + blog post canonicals correct; sitemap includes `/download` and `/gpu-availability` |
| F-003 | Lower marketing JS (WalletConnect off marketing); fewer long tasks — **re-run `audit-performance.mjs`** |
| F-006–F-013 | Header/touch targets/cache/no-store per `5fc95d3` |


## Post-deploy verification (2026-06-04T15:30:56.406Z)

| Check | Result |
|-------|--------|
| `GET /api/auth/me` (logged out) | 200 — {"ok":true,"user":null} |
| `GET /api/v2/gpu/available` | 200 |
| `GET /` no auth/me in HTML probe | API ok (frontend also defers probe on marketing) |
| Sitemap has /gpu-availability | yes |
| Sitemap has /download | yes |
| GPU page canonical | https://xcelsior.ca/gpu-availability |
| GPU page title | GPU Availability \| Xcelsior |

### Remaining follow-up

- **F-004** — Deploy layout/footer/theme hydration fix; re-run `node /tmp/xcelsior-audit/audit-hydration-spot.mjs`.
- **F-003** — TBT still high (~5–6 s desktop); further code-splitting / third-party deferral if needed.
- **F-014–F-021** — Still open in repo (contrast, JSON-LD, blog dates, etc.).
- **Authenticated dashboard** — Out of scope; needs credentials.

---

## Regenerate this report

```bash
# After MCP crawl:
cd /tmp/xcelsior-audit && node audit-routes.mjs  # etc.

# Optional post-deploy checks → JSON:
node scripts/post_deploy_audit_check.mjs > /tmp/post-deploy-check.json
REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs
```

---

## Recommended follow-up

1. **Deploy** — `bash scripts/deploy.sh` (or CI pipeline) from `main`.
2. **Verify** — `node scripts/post_deploy_audit_check.mjs`
3. **MCP re-crawl** — full `audit-*.mjs` suite; expect F-001/F-002/F-005 cleared.
4. **Authenticated dashboard audit** — still out of scope; needs test credentials.
