# Xcelsior — Site-Wide Audit Prompt (chrome-devtools MCP)

A reusable, agent-agnostic prompt for a full discovery audit of the Xcelsior site using the
**chrome-devtools MCP** server, then a prioritized report + fix plan. Tailored to this project's
stack and goals.

## Project context (so the audit is on-target)
- **Product:** Xcelsior — "Sovereign GPU Compute for Canada". B2B/developer SaaS; routes AI workloads
  to admission-gated GPU hosts over a private mesh. PIPEDA / data-sovereignty positioning, CAD-native billing.
- **Frontend:** Next.js 16 (App Router + React Server Components), React 19, Tailwind v4, **Serwist PWA**
  (service worker + `~offline` page). Lives in `frontend/`. Dev: `npm run dev` → http://localhost:3000.
- **Backend:** Python (FastAPI/gunicorn, alembic, Postgres, Redis). API at xcelsior.ca / docs at docs.xcelsior.ca (Fern).
- **This site MUST be indexable** — SEO, SSR/SSG correctness, and structured data are first-class (unlike a noindex staging site). robots.txt should ALLOW crawling of public pages.

## How to run
1. Ensure the agent has the `chrome-devtools` MCP server enabled (user scope). Approve on first use.
2. Run from inside the repo so findings map back to `frontend/src/app/**`.
3. Pick a Target below. For local: `cd frontend && npm run dev` then use http://localhost:3000.
4. Sanity-check ONE route first, then the full list. Discovery only — do NOT change code.

You can also tell any agent: *"Follow docs/site-audit-prompt.md and audit https://xcelsior.ca."*

---

## The Prompt

You have the chrome-devtools MCP server available, which drives a real Chrome browser. Use it as your ONLY source of truth — navigate, measure, resize, throttle, and observe the live site. Do not guess or rely on source code for findings; every finding must be backed by something you actually saw or measured. This is a DISCOVERY + PLANNING pass: do NOT change any code.

### Target
- Base URL: https://xcelsior.ca        # <-- or http://localhost:3000 (run `npm run dev` in frontend/)
- Public routes (visit each):
  - /                  (landing)
  - /features
  - /pricing           (CAD pricing — verify currency + plan clarity)
  - /sovereignty       (compliance/positioning)
  - /about
  - /blog              (+ open one post)
  - /support
  - /download
  - /privacy , /terms  (PIPEDA / legal)
  - /gpu-availability  (real-time data page)
- Auth flow routes (render + validation, logged-out): /login, /register, /forgot-password, /reset-password, /setup-2fa, /verify-email, /accept-invite
- Gated: /dashboard (note redirect when logged-out; audit authed separately — see below)

**Authenticated dashboard (optional):** provision creds with `bash scripts/provision_audit_user.sh`
(uses `site-audit@xcelsior.ca` → `.env.audit`, gitignored). Then:
`bash scripts/run_audit_dashboard.sh` or `node frontend/scripts/audit-dashboard.mjs`.
- Special: /feed.xml (RSS), /~offline (PWA offline), /sitemap.xml, /robots.txt
- Conditions: each page (a) unthrottled and (b) under "Slow 4G" + 4× CPU. Device matrix: 360×800, 375×667, 390×844, 430×932, 768×1024, 1280×800; spot-check one mobile LANDSCAPE.

### 1. Console
All errors/warnings (full text): uncaught exceptions, **React hydration mismatches**, CSP violations, failed assertions, deprecation warnings, 404s, unhandled rejections. Note load vs interaction, and which route group ((marketing) vs (dashboard)).

### 2. Network
List requests; flag/capture (method, URL, status, size, timing): failures/4xx/5xx, >1s, >250KB, render-blocking, missing/weak cache headers, no compression (br/gzip), HTTP/1.1 where h2/h3 expected, mixed content, CORS errors, redirect chains, duplicates, third-party calls, and any sensitive data in URLs. Note Next.js RSC payloads (`?_rsc=`), `/_next/static` caching, and `/_next/image` usage.

### 3. Performance (Core Web Vitals — these are REAL here, SSR)
Run a load trace per key route. Extract LCP, CLS, INP/TBT, FCP, TTFB and attribute each to a cause. Specifically check: **First Load JS / bundle size**, hydration cost, RSC streaming, render-blocking, long tasks >50ms, and whether `next/image` + `next/font` are used (no layout shift from images/fonts).

### 4. Next.js specifics
- **SSR/SSG correctness**: does HTML arrive server-rendered (real content in initial HTML, not just a JS shell)? Any hydration errors or content flash?
- **next/image**: images optimized/sized/lazy with correct `sizes`; no raw oversized `<img>`.
- **Prefetch/navigation**: link prefetch working; client navigations fast and not refetching everything.
- **Caching**: static assets immutable-hashed + long max-age; ISR/route cache headers sane.
- **404 / error**: `not-found` and error boundaries render properly (visit a bogus route).

### 5. SEO & metadata (indexable B2B — make this thorough)
Per route: `<title>`, meta description, canonical, `<html lang>`, robots meta (public pages must be **indexable**), Open Graph + Twitter cards, JSON-LD structured data (Organization, Product/Offer for pricing, BreadcrumbList, Article for blog, FAQ where relevant). Site-level: **robots.txt allows** public pages, **sitemap.xml** present + valid + lists routes, **feed.xml** valid RSS. Flag duplicate/missing titles & descriptions, missing canonicals, and any public page accidentally `noindex`.

### 6. PWA / Serwist
- `manifest.webmanifest`: name, icons (incl. maskable), theme/background color, display, start_url — installability.
- Service worker registers; **offline**: go offline and confirm `~offline` page (and cached routes) work; broken/blank offline = finding.
- SW caching strategy sane (no stale-forever HTML); update flow doesn't trap users on old assets.

### 7. Accessibility (B2B credibility)
Color contrast vs WCAG AA; images missing alt; buttons/links/icons without accessible names; keyboard nav + visible focus + logical order (esp. nav, pricing toggles, auth forms); heading hierarchy; form fields labeled and errors announced; respects reduced-motion; zoom NOT disabled.

### 8. Security & compliance (infra product — high bar)
- Response headers: **CSP**, **HSTS**, X-Content-Type-Options, X-Frame-Options/frame-ancestors, Referrer-Policy, Permissions-Policy. Flag missing/weak.
- Cookies (esp. auth/session): Secure + HttpOnly + SameSite.
- No secrets/tokens/API keys exposed in JS bundles, RSC payloads, network responses, or URLs.
- Auth pages over HTTPS; OAuth callback handles errors; password/2FA flows don't leak.
- PIPEDA/sovereignty: privacy + terms reachable and linked; any third-party that ships data offshore is worth flagging given the Canadian-data positioning.

### 9. Conversion / marketing UX
Pricing clarity (CAD shown, plans/limits legible, CTA obvious), signup funnel (/register) friction, hero/value-prop above the fold, working CTAs, no dead links, consistent nav/footer.

### 10. Mobile UI / responsive (every route, each mobile viewport)
Screenshot each (portrait + one landscape). Check: horizontal overflow (scrollWidth vs innerWidth, name the culprit), clipping/cutoff, touch targets ≥44×44px, body text ≥16px and inputs ≥16px (iOS zoom), images scale correctly, sticky header/nav + mobile menu usable, modals fit/scroll/dismiss, pricing tables reflow, auth forms usable with the on-screen keyboard, safe-area/notch respected, viewport meta correct.

### 11. Resilience / error states
Bogus route → proper 404; offline → `~offline`; /gpu-availability with no/slow data → loading + empty + error states (not a blank/spinner-forever); logged-out /dashboard → clean redirect to /login; Slow-4G behavior.

### Output — produce ONE markdown report
- **Executive summary** (5–8 bullets): biggest issues + overall health; mobile vs desktop; call out SEO and security explicitly (they matter most here).
- **Health snapshot** table per route: LCP / CLS / INP / First-Load-JS / # console errors / # failed requests / indexable? (Y/N).
- **Findings table**: ID | Area (Console / Network / Perf / Next.js / SEO / PWA / A11y / Security / UX / Mobile / Resilience) | Severity (P0→P3) | Page(s) + Viewport | Evidence (exact value/status/console text/px) | Likely root cause | Suggested fix | Effort (S/M/L).
- **Grouped detail** by severity, evidence quoted.
- **Quick wins**, then an ordered **remediation plan** (sprint-style, with dependencies), then **Needs manual follow-up** (authed dashboard, full viewport matrix, etc.).

### Rules
- Be specific and quantitative — real numbers, URLs, status codes, console messages, px values. Never "seems slow".
- Measure via the real browser only; do not infer from source.
- Do NOT change code in this pass — discovery + planning only.
- Public pages must be indexable: treat an accidental `noindex`/robots block on a public route as a finding (this site WANTS to rank).
- Every finding names the exact page and (for UI) the exact viewport.
