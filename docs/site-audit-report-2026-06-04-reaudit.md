# Xcelsior — Site Audit Re-Audit Report

Date: 2026-06-05 (updated)  
Target: https://xcelsior.ca  
Deploy status: **Deployed** — `eb338b2` on production (frontend hydration + F-014–F-021 backlog)

This report compares the **baseline audit** (`docs/site-audit-report-2026-06-04.md`), a **second full chrome-devtools MCP crawl** (`docs/site-audit-report-2026-06-04-reaudit-mcp.md`), and **remediation on `main`** (`5fc95d3` sprints 1–3, `b87c8f1` bundle split, `181ad74`–`eb338b2` hydration + a11y/SEO backlog).

Raw MCP artifacts: `/tmp/xcelsior-audit/raw/`, screenshots: `/tmp/xcelsior-audit/screens/`.

---

## Executive summary

1. **Deployed 2026-06-05** (`eb338b2` frontend) — Post-deploy probes confirm **F-001**, **F-002**, **F-005**, and **F-004** (legal hydration) are **cleared on production**.
2. **Perf (F-003) — measured 2026-06-04** — Marketing JS on `/` **~374 KB** (was ~812 KB); desktop TBT **~5.7 s** (was ~9.2 s). Still above target; further splitting optional.
3. **Hydration (F-004) — fixed 2026-06-05** — React #418 on `/privacy` and `/terms` was **Cloudflare Email Obfuscation** rewriting mailto text after SSR. Fixed with post-mount mailto rendering and i18n string splits (`ObfuscationSafeMailto`, commits `8357133`, `eb338b2`). Repro: `frontend/scripts/hydration-repro.mjs`, `hydration-diff.mjs`.
4. **F-014–F-021 backlog** — Shipped in `181ad74` (contrast, JSON-LD `@id`, blog dates, install-banner allowlist, privacy third-party disclosure).

### Post-deploy P1 status

| ID | Status | Evidence |
|----|--------|----------|
| F-001 | **Fixed** | `GET /api/auth/me` → `200` + `user: null`; invalid bearer → `401` |
| F-002 | **Fixed** | `/api/v2/gpu/available` → `200` |
| F-005 | **Fixed** | GPU canonical `https://xcelsior.ca/gpu-availability`; sitemap lists `/download` + `/gpu-availability` |
| F-003 | **Improved** | `/` JS ~374 KB; TBT ~5.7 s desktop (re-run MCP for fresh numbers) |
| F-004 | **Fixed** | `hydration-repro.mjs` on prod: `/privacy`, `/terms`, `/blog`, `/about`, `/support` OK |

### Post-deploy performance (MCP `perf-all.json`, 2026-06-04)

| Route | Pre-deploy TBT | Post-deploy TBT | Δ TBT | Post-deploy JS |
|-------|---------------:|----------------:|------:|---------------:|
| `/` | 9166 ms | 5772 ms | -3394 | 374 KB |
| `/blog` | 13662 ms | 5548 ms | -8114 | 313 KB |
| `/download` | 9721 ms | 5429 ms | -4292 | 316 KB |
| `/privacy` | 9239 ms | 5742 ms | -3497 | 313 KB |
| `/terms` | 9209 ms | 5454 ms | -3755 | 313 KB |

Desktop `/` (pre-deploy crawl): TBT **9166** ms, JS **812** KB. Slow 4G `/pricing` (pre-deploy): TBT **52596** ms.

### Production API signals (post-deploy curl, 2026-06-05)

| Signal | Status |
|--------|--------|
| `/api/auth/me` (logged out) | 200 + `user: null` |
| `/api/auth/me` (invalid bearer) | 401 |
| `/api/v2/gpu/available` | 200 |
| GPU canonical | `https://xcelsior.ca/gpu-availability` |

---

## Remediation matrix (findings → code)

| ID | Sev | Commit | Fix (summary) | Repo / prod |
|----|-----|--------|---------------|-------------|
| F-001 | P1 | 5fc95d3, a98e2db | 200 + user:null; invalid bearer 401 | In repo + prod |
| F-002 | P1 | 5fc95d3 | Public GPU endpoint; honest degraded UI | In repo + prod |
| F-003 | P1 | b87c8f1 | WalletConnect dashboard-only; lazy chat/PWA/GTM | In repo + prod |
| F-004 | P1 | eb338b2 | CF email obfuscation hydration; mailto after mount | In repo + prod |
| F-005 | P1 | 5fc95d3 | Per-route canonicals; sitemap download + gpu-availability | In repo + prod |
| F-006 | P1 | 5fc95d3 | Deduped security headers at app layer | In repo |
| F-007 | P1 | 5fc95d3 | no-store for auth/dashboard/offline | In repo |
| F-008 | P2 | 5fc95d3 | Header overlap / BETA placement | In repo |
| F-009 | P2 | 5fc95d3 | 44px touch targets | In repo |
| F-010 | P2 | 5fc95d3 | 16px inputs on mobile | In repo |
| F-011 | P2 | 5fc95d3 | Mobile pricing affordance (partial) | In repo |
| F-012 | P2 | 5fc95d3 | SW update flow improvements | In repo |
| F-013 | P2 | 5fc95d3 | aria-expanded, body scroll lock | In repo |
| F-014 | P2 | 181ad74 | Contrast: Beta, sovereignty badge, support CTA | In repo + prod |
| F-015 | P2 | 181ad74 | H1 spacing (`home.hero_line1` trailing space) | In repo + prod |
| F-016 | P2 | 5fc95d3 | Terms required + a11y errors | In repo |
| F-017 | P2 | 181ad74 | Privacy §9/§11 third-party disclosure (GTM, CF) | In repo + prod |
| F-018 | P3 | 181ad74 | Organization `@id`; blog publisher reference | In repo + prod |
| F-019 | P3 | b87c8f1 | GTM lazyOnload, no preload | In repo |
| F-020 | P3 | 181ad74 | Install banner route allowlist | In repo + prod |
| F-021 | P3 | 181ad74 | Blog dates: sitemap, RSS, OG use stable UTC dates | In repo + prod |
| F-022 | P3 | — | Accepted — SVG logos OK | Open (accepted) |

---

## F-004 root cause (for operators)

**Symptom:** Minified React error #418 (`args[]=text`) on `/privacy` and `/terms` only in production.

**Cause:** Cloudflare **Email Address Obfuscation** replaces `privacy@xcelsior.ca` (and similar) in the HTML edge response with `[email protected]` while React hydrates the original strings.

**Mitigation in app:** `ObfuscationSafeMailto` + no inline emails in legal i18n strings.

**Optional Cloudflare:** Scrape Shield → disable Email Obfuscation if you want plaintext emails in View Source.

**Verify:**

```bash
BASE_URL=https://xcelsior.ca node frontend/scripts/hydration-repro.mjs
node frontend/scripts/hydration-diff.mjs   # SSR vs client text diff
```

---

## Post-deploy verification (2026-06-05)

| Check | Result |
|-------|--------|
| `GET /api/auth/me` (logged out) | 200 — `{"ok":true,"user":null}` |
| `GET /api/auth/me` (invalid bearer) | 401 |
| `GET /api/v2/gpu/available` | 200 |
| Sitemap has /gpu-availability | yes |
| Sitemap has /download | yes |
| GPU page canonical | https://xcelsior.ca/gpu-availability |
| Hydration `/privacy`, `/terms` | OK (`hydration-repro.mjs`) |
| SSR legal pages omit raw emails | OK (no `privacy@` in HTML before hydration) |

### Remaining follow-up

- **F-003** — TBT still ~5–6 s desktop; re-run `/tmp/xcelsior-audit/audit-performance.mjs` after major JS changes.
- **Authenticated dashboard** — MCP crawl needs test credentials.
- **Deploy script** — `health_check()` now reads `/opt/xcelsior/.deploy_colour` (was typo `.deploy-colour`).

---

## Regenerate this report

```bash
node scripts/post_deploy_audit_check.mjs > /tmp/post-deploy-check.json
REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs  # if extending generator

BASE_URL=https://xcelsior.ca node frontend/scripts/hydration-repro.mjs
```