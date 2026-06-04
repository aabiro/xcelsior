# Xcelsior - Site Audit Report

Date: 2026-06-04
Target: https://xcelsior.ca
Method: Real Chrome via chrome-devtools-mcp, driven from /home/aaryn/storage/projects/xcelsior. No application code was changed.

Raw artifacts:

- Desktop route JSON and screenshots: /tmp/xcelsior-audit/raw/desktop-all.json, /tmp/xcelsior-audit/screens/desktop/
- Responsive matrix JSON and screenshots: /tmp/xcelsior-audit/raw/responsive-all.json, /tmp/xcelsior-audit/screens/responsive/
- Web Vitals/resource metrics: /tmp/xcelsior-audit/raw/perf-all.json
- DevTools traces: /tmp/xcelsior-audit/raw/trace-all.json, /tmp/xcelsior-audit/traces/
- Site-level checks: /tmp/xcelsior-audit/raw/site-all.json, /tmp/xcelsior-audit/raw/robots-full.json

## Executive Summary

- Overall health is mixed: the site is server-rendered and indexable, internal links work, robots/sitemap/feed/manifest parse, no bundle secret-pattern matches were found, and no horizontal page overflow was detected across 132 responsive route/viewport loads.
- The largest production bug is global logged-out auth noise: every public route hits /api/auth/me with 401, and /gpu-availability also hits /hosts?active_only=true with 401 while presenting itself as live real-time availability.
- Performance needs focused work. Cold desktop loads transfer roughly 695-813 KB of JavaScript and show 9.3-15.7 seconds of measured TBT in the observer pass. DevTools traces attribute LCP mainly to render delay, not TTFB.
- SEO has important gaps for an indexable B2B site: /gpu-availability uses home metadata/canonical, the selected blog post canonical points to the home page, /download and /gpu-availability are missing from sitemap.xml, and auth/offline/special pages reuse home metadata.
- Security headers are present but messy: strong HSTS and nosniff exist, but several headers are duplicated or conflicting, CSP still allows unsafe-inline, and most HTML is cached for one year at the shared cache layer.
- Mobile layout is mostly overflow-safe, but the header is visibly broken at narrow and landscape widths: BETA overlaps adjacent controls/nav, the logo is clipped, Sign In wraps, and touch targets are below 44px across every tested viewport.
- PWA installability assets are mostly present and offline cached routes loaded, but the service worker remained stuck in installing state with no controller and serviceWorker.ready timing out.
- No P0 data-loss/security-critical issue was observed in this logged-out public audit, but there are multiple P1 issues that should be fixed before relying on the site for search, conversion, or compliance messaging.

## Scope

Routes visited:

- Public: /, /features, /pricing, /sovereignty, /about, /blog, /blog/security-is-not-a-feature-its-the-infrastructure, /support, /download, /privacy, /terms, /gpu-availability
- Auth logged out: /login, /register, /forgot-password, /reset-password, /setup-2fa, /verify-email, /accept-invite
- Gated/special: /dashboard, /~offline, /nonexistent-bogus-404, /robots.txt, /sitemap.xml, /feed.xml, /manifest.webmanifest

Viewports:

- Desktop/tablet/mobile: 1280x800, 768x1024, 430x932, 390x844, 375x667, 360x800
- Mobile landscape spot-check: 844x390
- Throttled key routes: Slow 4G + 4x CPU at 390x844

## Health Snapshot

INP was not meaningfully measurable in a passive logged-out crawl, so the table reports TBT from the performance observer as the interaction-risk proxy requested by the prompt.

| Route | Final URL | Status | LCP | CLS | INP/TBT proxy | First Load JS | Console | Failed requests | Indexable |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| / | / | 200 | 5000 ms | 0.0443 | 10293 ms | 812 KB | 1 err / 1 warn | 401 /api/auth/me | Y |
| /features | /features | 200 | 1340 ms | 0.0001 | 9458 ms | 777 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /pricing | /pricing | 200 | 2536 ms | 0.0083 | 11776 ms | 778 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /sovereignty | /sovereignty | 200 | 3492 ms | 0.0098 | 12078 ms | 810 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /about | /about | 200 | 4280 ms | 0.0001 | 13274 ms | 777 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /blog | /blog | 200 | 5952 ms | 0.0048 | 15697 ms | 775 KB | 2 err / 0 warn | 401 /api/auth/me | Y |
| /blog/security-is-not-a-feature-its-the-infrastructure | /blog/security-is-not-a-feature-its-the-infrastructure | 200 | 3736 ms | 0.0002 | 10088 ms | 774 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /support | /support | 200 | 1888 ms | 0.0101 | 11286 ms | 776 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /download | /download | 200 | 5140 ms | 0.0058 | 10866 ms | 778 KB | 2 err / 0 warn | 401 /api/auth/me | Y |
| /privacy | /privacy | 200 | 3344 ms | 0.0001 | 12043 ms | 775 KB | 2 err / 0 warn | 401 /api/auth/me | Y |
| /terms | /terms | 200 | 3460 ms | 0.0102 | 9780 ms | 775 KB | 2 err / 0 warn | 401 /api/auth/me | Y |
| /gpu-availability | /gpu-availability | 200 | 2912 ms | 0.0009 | 10774 ms | 761 KB | 1 err / 0 warn | 401 /hosts?active_only=true<br>401 /api/auth/me | Y |
| /login | /login | 200 | 2144 ms | 0.008 | 10258 ms | 813 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /register | /register | 200 | 2172 ms | 0.0092 | 10417 ms | 812 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /forgot-password | /forgot-password | 200 | 3528 ms | 0.0042 | 9386 ms | 777 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /reset-password | /reset-password | 200 | 2028 ms | 0.0042 | 10269 ms | 778 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /setup-2fa | /login | 200 | 3184 ms | 0.0001 | 10768 ms | 788 KB | 0 err / 0 warn | 401 /api/auth/me | Y |
| /verify-email | /verify-email | 200 | 2508 ms | 0.0001 | 10729 ms | 776 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /accept-invite | /accept-invite | 200 | 1876 ms | 0.0092 | 10823 ms | 777 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /dashboard | /login?redirect=%2Fdashboard | 200 final login redirect | 2748 ms | 0.0001 | 11646 ms | 813 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /~offline | /~offline | 200 | 3424 ms | 0.0006 | 10259 ms | 697 KB | 1 err / 0 warn | 401 /api/auth/me | Y |
| /nonexistent-bogus-404 | /nonexistent-bogus-404 | 404 | 2328 ms | 0.0021 | 10215 ms | 695 KB | 2 err / 0 warn | 404 /nonexistent-bogus-404<br>401 /api/auth/me<br>404 /nonexistent-bogus-404 | N |

## Slow 4G + 4x CPU Key Route Snapshot

Note: slow-home transfer was served mostly from cache/service-worker state (31 KB), so slow-home resource transfer is not comparable to other slow runs.

| Route | LCP | FCP | TBT proxy | Max long task | JS transfer | Total transfer | Slowest resources |
|---|---:|---:|---:|---:|---:|---:|---|
| / | 8168 ms | 8168 ms | 10463 ms | 3573 ms | 4 KB | 31 KB | 4705 ms 3 KB /_next/static/chunks/0o4.3~2e0sq8_.js<br>1544 ms 24 KB /_next/static/chunks/1235czw0ipicq.css<br>1084 ms 1 KB /xcelsior-logo-wordmark-iconbg-light.svg |
| /pricing | 6528 ms | 6528 ms | 54501 ms | 21245 ms | 778 KB | 866 KB | 7356 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7077 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>6797 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /blog | 23328 ms | 6620 ms | 57885 ms | 33345 ms | 775 KB | 833 KB | 7317 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7014 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>6803 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /blog/security-is-not-a-feature-its-the-infrastructure | 22840 ms | 7332 ms | 49471 ms | 20614 ms | 774 KB | 832 KB | 7576 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7229 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>7125 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /download | 6632 ms | 6632 ms | 54192 ms | 21977 ms | 778 KB | 836 KB | 7692 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7368 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>7077 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /gpu-availability | 19636 ms | 7104 ms | 48692 ms | 22374 ms | 761 KB | 818 KB | 7930 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7624 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>7314 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /login | 7096 ms | 7096 ms | 49355 ms | 21988 ms | 813 KB | 871 KB | 7878 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7588 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>7386 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |
| /register | 6068 ms | 6068 ms | 46602 ms | 22101 ms | 812 KB | 870 KB | 7747 ms 148 KB /_next/static/chunks/12mrwcs3j1~op.js<br>7426 ms 100 KB /_next/static/chunks/0h-r7qi8u4mrj.js<br>7164 ms 69 KB /_next/static/chunks/07c4zh_ug2.bf.js |

## DevTools Trace Snapshot

Manual DevTools traces were captured for five key routes. Trace file names in the MCP output include .json.json.gz because the tool appends .json.gz to the requested path.

| Route | Trace LCP | TTFB | Render delay | Interpretation |
|---|---:|---:|---:|---|
| / | 6504 ms | 435 ms | 5671 ms | Render delay dominates LCP. |
| /pricing | 5398 ms | 166 ms | 4964 ms | Render delay dominates LCP. |
| /blog | 2838 ms | 171 ms | 2667 ms | Render delay dominates LCP. |
| /gpu-availability | 6533 ms | 299 ms | 6233 ms | Render delay dominates LCP and live-data request fails. |
| /login | 2383 ms | 142 ms | 2016 ms | Render delay dominates LCP. |

## Findings Table

| ID | Area | Severity | Page(s) + Viewport | Evidence | Likely root cause | Suggested fix | Effort |
|---|---|---|---|---|---|---|---|
| F-001 | Console / Network | P1 | All routes, desktop and mobile | Every audited desktop route issued `GET /api/auth/me [401]`; 21 of 22 desktop routes logged a console error. Example `/`: `Failed to load resource: the server responded with a status of 401 ()`. | Global logged-out auth probe runs on public pages and lets the browser treat expected anonymous state as a failed request. | Do not call `/api/auth/me` on fully public routes, or return/cache a non-error anonymous session shape. Suppress expected anonymous 401s before they hit the console. | M |
| F-002 | Resilience / UX / Network | P1 | /gpu-availability | Route body says `Live Availability`, `Real-time GPU availability`; network shows `GET /hosts?active_only=true [401]` plus `/api/auth/me [401]`. | The public GPU page likely calls an authenticated hosts endpoint and then renders fallback/static data. | Expose a public read-only availability endpoint or gate the page honestly. Add loading, empty, and error copy when live data cannot load. | M |
| F-003 | Performance | P1 | All routes; Slow 4G key routes | Cold desktop First Load JS was 695-813 KB per route; desktop TBT ranged 9386-15697 ms. Slow 4G + 4x CPU TBT was 46602-57885 ms on key routes. DevTools traces show render delay dominates LCP: home 5671 ms, pricing 4964 ms, GPU availability 6233 ms. | Large client bundle and hydration/main-thread work dominate render delay. Third-party GTM and Cloudflare Insights also run on public pages. | Audit client components, remove unused shared code, dynamically import non-critical widgets/chat/auth/PWA code, defer analytics, and split dashboard/auth code away from marketing routes. | L |
| F-004 | Next.js / Console | P1 | /blog, /download, /privacy, /terms | Console logged `Uncaught Error: Minified React error #418 ... args[]=text` on `/blog`, `/download`, `/privacy`, `/terms`. | Server-rendered text differs from hydrated client text. Likely date/locale, randomized, time-dependent, or client-only text rendered during SSR. | Reproduce in non-minified build, find the mismatched text node, and make server/client rendering deterministic or isolate client-only content behind a mounted state. | M |
| F-005 | SEO | P1 | /gpu-availability, /download, blog post, auth/special pages | `/gpu-availability` title/description/canonical are the home values; selected blog post canonical is `https://xcelsior.ca`; `/download` and `/gpu-availability` are absent from sitemap. Auth/offline/special pages reuse home title/description/canonical. | Metadata generation is missing for several routes and canonical defaults fall back to root. | Add route-specific metadata and canonical URLs. Add `/download` and `/gpu-availability` to sitemap. Set appropriate `noindex` for auth/offline/error pages. | M |
| F-006 | Security / Headers | P1 | All 200 HTML routes | Response headers include `X-Frame-Options: DENY, SAMEORIGIN`, duplicate `X-Content-Type-Options: nosniff, nosniff`, duplicate `Referrer-Policy`, duplicate `X-XSS-Protection`, and CSP allows `script-src ... unsafe-inline` plus `style-src ... unsafe-inline`. | Headers are emitted by more than one layer and CSP is loosened for inline scripts/styles. | Deduplicate at the edge/app boundary, prefer CSP `frame-ancestors` over conflicting XFO, remove obsolete X-XSS-Protection, and move toward nonce/hash CSP for inline code. | M |
| F-007 | Caching / Next.js | P1 | Most HTML routes including auth/legal/offline | Most HTML responses returned `cache-control: s-maxage=31536000`; pricing returned `s-maxage=3600, stale-while-revalidate=31532400`. | HTML route cache policy is too broad for pages that can change, show auth state, display legal/compliance text, or recover from offline/PWA updates. | Set route-specific cache policies. Keep immutable caching for hashed static assets, but use shorter ISR/no-store policies for auth, legal, live data, offline, and error pages. | M |
| F-008 | Mobile / UX | P2 | 360x800 and 844x390 landscape header | At 360px, logo link is 129x44 with scrollWidth 160; BETA is x=145-184 and language button starts x=153. At 844x390, logo link is 49x44 with scrollWidth 88; BETA x=73-112 and nav starts x=73. `Sign In` is 33x40 and wraps to two lines. | Header breakpoint and fixed gap/layout rules do not fit narrow or landscape viewports; BETA badge is inside the logo row with negative margin. | Move BETA into a non-overlapping badge slot, hide/reduce wordmark sooner, adjust breakpoints, and enforce min widths for auth/nav controls. | S |
| F-009 | Mobile / A11y | P2 | All routes, all tested mobile/tablet viewports | Small target totals: 242 targets at each 360/375/390/430 viewport and 246 at 768/landscape. Examples: theme and menu buttons 40x40; nav links 44x20; support buttons 42px high. | Interactive controls use visual text/link dimensions without minimum hit area. | Set `min-height`/`min-width` >= 44px for header controls, nav links, footer links, chat/dismiss controls, pricing CTAs, and support actions. | S |
| F-010 | Mobile / Forms / A11y | P2 | /pricing, /login, /register, /forgot-password, /dashboard, /setup-2fa | Responsive matrix counted sub-16px inputs: pricing 3, login 2, register 5, forgot 1, dashboard 2, setup-2fa 2. Focused pricing probe showed controls at `fontSize: 14px`, 262x40, and no associated labels for select/number controls. | Form/control typography is optimized for desktop density and not iOS/mobile input behavior. | Use >=16px font on all mobile inputs/selects/textarea controls. Add labels or aria-labels for pricing calculator controls. | S |
| F-011 | Mobile / Pricing UX | P2 | /pricing at 360x800 | Pricing table container is 310px client width with 550px scrollWidth. Screenshot shows only GPU, VRAM, On-Demand, and partial Spot columns; reserved columns are off-screen. | Wide tabular pricing is placed in horizontal scroll without strong affordance or alternate mobile layout. | Use stacked plan cards or sticky first column plus scroll shadows/labels. Ensure all critical CAD price columns are discoverable on 360px. | M |
| F-012 | PWA / Serwist | P2 | Home/PWA | After a longer browser check, `navigator.serviceWorker.controller` was false, `navigator.serviceWorker.ready` timed out, and registration remained `installing` with `sw.js`; cache `serwist-precache-v2-https://xcelsior.ca/` existed. Offline `/`, `/pricing`, and `/~offline` loaded from cache. | Service worker install/activation lifecycle is not completing promptly in this Chrome session, even though precache exists. | Investigate SW install errors, asset cache failures, skipWaiting/clientsClaim/update flow, and whether stale cached SW versions are trapped. | M |
| F-013 | A11y / Mobile Menu | P2 | Mobile header | Mobile menu opens, but trigger has `aria-expanded: null` before and after. After open, header exposes duplicated EN/FR and theme controls, and body overflow remains `visible`. | Menu state is visually controlled without full ARIA state/scroll management. | Set `aria-expanded`, `aria-controls`, focus management, escape/outside-click close, and body scroll handling. Avoid duplicate controls in the accessibility tree. | S |
| F-014 | A11y / Contrast | P2 | Global header, support CTA, small red labels | Automated contrast found `BETA` at 4.1:1 vs 4.5:1, `Data Sovereignty Explained` red label at 4.1:1, and support `View Pricing` white-on-cyan at 1.77:1. | Accent colors are used for small text and button backgrounds without contrast tokens. | Define AA-safe accent text and button color pairs. Keep gradient text decorative or provide solid fallback text color with sufficient contrast. | S |
| F-015 | SEO / A11y | P2 | Home | Home H1 accessible text captured as `Sovereign GPUCompute for Canada` with no space between GPU and Compute. | H1 is split across inline elements without an actual whitespace text node. | Insert a literal space or restructure the H1 so accessible text reads `Sovereign GPU Compute for Canada`. | S |
| F-016 | Auth UX / A11y | P2 | /login, /register, /forgot-password | Empty submit uses native browser validation only; invalid fields had no `aria-invalid`, no `aria-describedby`, and no `[role=alert]`/`aria-live`. Register `agree-terms` checkbox was not required. | Validation relies on native constraints and does not announce app-level errors. Terms consent is visually present but not required by DOM. | Add accessible field errors, `aria-invalid`, descriptions, live region summaries, and require the terms/privacy agreement before account creation. | M |
| F-017 | Compliance / Third Party | P2 | All public routes | Cold loads contacted `www.googletagmanager.com` and `static.cloudflareinsights.com`; CSP allows Google Analytics/GTM, Cloudflare Insights, Stripe, WalletConnect endpoints. | Marketing analytics and third-party scripts may conflict with Canada-first/data-sovereignty expectations if not disclosed and controlled. | Document data flows in privacy copy, gate analytics with consent where required, and minimize third-party loading on sovereignty/compliance-focused pages. | M |
| F-018 | SEO / Structured Data | P3 | /blog, /download, /privacy, /terms | JSON-LD types showed duplicate `Organization` entries on `/blog`, `/download`, `/privacy`, `/terms`. Blog post has `BlogPosting`, pricing has `Product` and `FAQPage`. | Shared Organization schema is emitted more than once on some layouts/pages. | Emit one site-level Organization entity and connect page schemas via `publisher`/`isPartOf` references. | S |
| F-019 | Network / Perf | P3 | Home | Console warning: GTM preload `https://www.googletagmanager.com/gtag/js?id=G-EPD8EJ9R5D` was preloaded but not used within a few seconds from load. | Analytics preload is not aligned with actual use timing. | Remove the preload or load analytics later without pretending it is critical path. | S |
| F-020 | UX / Error State | P3 | /nonexistent-bogus-404 | 404 rendered correctly with status 404 and `robots noindex`, but also showed the desktop app promo and chat affordance in the compact 404 body. | Global promotional widgets render on error pages. | Consider suppressing promos/chat on 404/offline/error routes or lowering their prominence. | S |
| F-021 | SEO / Content | P3 | Blog index vs selected post | Blog index lists the selected post as April 10, 2026; selected post body shows April 11, 2026; sitemap lastmod is April 11, 2026. | Post date source differs between card/list and article/detail. | Normalize article publish/update dates across blog cards, article pages, RSS, sitemap, and JSON-LD. | S |
| F-022 | Next.js / Images | P3 | Global | Core content is server-rendered and images have alt text, but route measurements show `next/image` usage count effectively zero and raw SVG/logo assets on critical path. | Brand assets and page visuals are mostly raw/static assets rather than Next image-optimized responsive media. | This is acceptable for small SVG/logo files, but use `next/image` for any raster or hero/media additions and keep explicit dimensions. | S |

## Grouped Detail

### P1 - Fix First

#### F-001 Global auth 401 noise

Every public route triggers a logged-out auth request that fails with 401. The exact browser console text is generic but user-visible in DevTools:

```
Failed to load resource: the server responded with a status of 401 ()
```

The network evidence ties the console noise to `GET https://xcelsior.ca/api/auth/me [401]`. This pollutes monitoring, hides real failures, and costs a request on every marketing page load.

#### F-002 GPU availability is not actually public-live

`/gpu-availability` rendered useful-looking prices and "Updated" text, but the browser network also recorded:

```
GET https://xcelsior.ca/hosts?active_only=true [401]
GET https://xcelsior.ca/api/auth/me [401]
```

Because the page says "Live Availability" and "Real-time GPU availability", an unauthenticated live-data failure is more than a cosmetic problem. It makes the page look current while its source request is unauthorized.

#### F-003 Performance and hydration cost

The custom observer pass measured very high main-thread work on every route. Worst desktop TBT values:

- /blog: 15697 ms TBT, 5952 ms LCP, 775 KB JS
- /about: 13274 ms TBT, 4280 ms LCP, 777 KB JS
- /sovereignty: 12078 ms TBT, 3492 ms LCP, 810 KB JS
- /pricing: 11776 ms TBT, 2536 ms LCP, 778 KB JS

The DevTools trace summaries explain the LCP issue as render delay, not server response time. Example: /gpu-availability LCP was 6533 ms with 6233 ms of render delay.

#### F-004 React hydration mismatch

Production console logged minified React error #418, a text hydration mismatch, on:

- /blog
- /download
- /privacy
- /terms

This is especially important because the prompt treats SSR/SSG correctness as first-class. The pages do serve real HTML, but some text differs during hydration.

#### F-005 SEO metadata and sitemap gaps

Important observations:

- /gpu-availability title: `Xcelsior — Canada-First GPU Compute for Teams Worldwide`
- /gpu-availability canonical: `https://xcelsior.ca`
- selected blog post canonical: `https://xcelsior.ca`
- sitemap.xml URL count: 21
- missing from sitemap: /download, /gpu-availability
- blog post meta description length: 201

Public pages are indexable, which is good, but canonical errors can consolidate valuable pages into the home URL and undermine search performance.

#### F-006 Headers and CSP

Main document response headers include good basics like HSTS and Permissions-Policy, but they are not clean:

```
X-Frame-Options: DENY, SAMEORIGIN
X-Content-Type-Options: nosniff, nosniff
Referrer-Policy: strict-origin-when-cross-origin, strict-origin-when-cross-origin
X-XSS-Protection: 1; mode=block, 1; mode=block
Content-Security-Policy: script-src ... 'unsafe-inline'; style-src ... 'unsafe-inline'
```

The conflict between DENY and SAMEORIGIN should be resolved, duplicate headers should be removed, and CSP should move toward nonce/hash based inline execution.

#### F-007 HTML caching

Most 200 HTML routes returned:

```
cache-control: s-maxage=31536000
```

This is risky for legal pages, auth pages, offline page, live data pages, and any page where metadata or content may need correction quickly.

### P2 - Next Sprint

#### Mobile header

At 360x800:

- logo link: x=24-153, 129x44, scrollWidth 160
- logo image: x=24-145, 121x44
- BETA badge: x=145-184, 39x19
- language button: x=153-232, 79x40

That means the BETA badge starts exactly where the logo image ends and overlaps into the language button area. At 844x390 landscape:

- logo link: x=24-73, 49x44, scrollWidth 88
- BETA badge: x=73-112
- nav starts at x=73
- Sign In is 33x40 and wraps to two lines

The responsive matrix did not report body-level horizontal overflow, but screenshots show the header is visually broken.

#### Mobile touch targets

Across the responsive matrix, every viewport had hundreds of sub-44px controls. Repeated examples:

- Header theme button: 40x40
- Header menu button: 40x40
- Header language toggle: 79x40
- Nav text links: often 20px high
- Support actions: 42px high

This affects repeated navigation and accessibility, even though the site avoids horizontal page scroll.

#### Pricing mobile table

At 360x800, the pricing table scroll container was:

- client width: 310 px
- scroll width: 550 px
- visual screenshot: right-side columns are cropped off-screen

The page does technically contain a horizontal scroll container, but there is no obvious affordance and the critical reserved-price columns are not visible by default.

#### PWA service worker

The manifest is strong: name, short_name, display standalone, maskable 192/512 icons, shortcuts, screenshots, start_url and scope were present. Offline `/`, `/pricing`, and `/~offline` loaded under emulated Offline.

But the service worker registration remained unhealthy after a longer wait:

```json
{
  "controller": false,
  "ready": "timeout",
  "regs": [
    {
      "scope": "https://xcelsior.ca/",
      "active": null,
      "waiting": null,
      "installing": {
        "url": "https://xcelsior.ca/sw.js",
        "state": "installing"
      }
    }
  ],
  "caches": ["serwist-precache-v2-https://xcelsior.ca/"]
}
```

### P3 - Cleanup

- Duplicate Organization JSON-LD appears on some pages.
- Home GTM preload is unused within a few seconds of load.
- The 404 route works and is noindex, but global promos/chat make the error page noisier than needed.
- Blog post date differs between blog index and article/sitemap.
- Raw SVG logos are fine, but any future raster/hero media should use optimized image handling.

## SEO And Site-Level Checks

- robots.txt loaded with 200 and includes `User-agent: * Allow: /`.
- robots.txt advertises `Sitemap: https://xcelsior.ca/sitemap.xml`.
- sitemap.xml loaded with 200, parsed as `urlset`, and listed 21 URLs.
- feed.xml loaded with 200, parsed as RSS, and listed 10 items.
- manifest.webmanifest loaded with 200 and parsed as JSON.
- Internal same-origin link check found no broken public links. The only redirect noted was `/dashboard` to `/login?redirect=%2Fdashboard`, which is expected for logged-out users.
- Bundle secret-pattern scan found no matches for Stripe secret keys, AWS keys, private keys, bearer literals, JWT-like values, or Google API keys.

Full robots.txt crawler directives captured by browser:

```
# As a condition of accessing this website, you agree to abide by the following
# content signals:

# (a)  If a Content-Signal = yes, you may collect content for the corresponding
#      use.
# (b)  If a Content-Signal = no, you may not collect content for the
#      corresponding use.
# (c)  If the website operator does not include a Content-Signal for a
#      corresponding use, the website operator neither grants nor restricts
#      permission via Content-Signal with respect to the corresponding use.

# The content signals and their meanings are:

# search:   building a search index and providing search results (e.g., returning
#           hyperlinks and short excerpts from your website's contents). Search does not
#           include providing AI-generated search summaries.
# ai-input: inputting content into one or more AI models (e.g., retrieval
#           augmented generation, grounding, or other real-time taking of content for
#           generative AI search answers).
# ai-train: training or fine-tuning AI models.

# ANY RESTRICTIONS EXPRESSED VIA CONTENT SIGNALS ARE EXPRESS RESERVATIONS OF
# RIGHTS UNDER ARTICLE 4 OF THE EUROPEAN UNION DIRECTIVE 2019/790 ON COPYRIGHT
# AND RELATED RIGHTS IN THE DIGITAL SINGLE MARKET.

# BEGIN Cloudflare Managed content

User-agent: *
Content-Signal: search=yes,ai-train=no
Allow: /

User-agent: Amazonbot
Disallow: /

User-agent: Applebot-Extended
Disallow: /

User-agent: Bytespider
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: ClaudeBot
Disallow: /

User-agent: CloudflareBrowserRenderingCrawler
Disallow: /

User-agent: Google-Extended
Disallow: /

User-agent: GPTBot
Disallow: /

User-agent: meta-externalagent
Disallow: /

# END Cloudflare Managed Content

User-agent: *
Allow: /

Sitemap: https://xcelsior.ca/sitemap.xml
```

## Quick Wins

1. Stop the global public-page `/api/auth/me` 401 console error.
2. Fix `/gpu-availability` to use a public live-data endpoint or show a clear unavailable/error state.
3. Add route-specific metadata/canonicals for `/gpu-availability`, auth/special pages, and the selected blog post.
4. Add `/download` and `/gpu-availability` to sitemap.xml.
5. Fix the H1 text spacing on home.
6. Move or restyle the BETA badge so it cannot overlap header controls.
7. Raise header/menu/theme/lang controls to at least 44x44.
8. Remove unused GTM preload or load analytics later.
9. Deduplicate security headers at one layer.
10. Make pricing calculator controls labeled and >=16px on mobile.

## Remediation Plan

### Sprint 1 - Production Correctness

1. Fix the public auth probe and `/gpu-availability` 401s.
2. Fix React #418 hydration mismatches on /blog, /download, /privacy, and /terms.
3. Correct SEO metadata/canonicals and sitemap membership for public routes.
4. Deduplicate/conflict-resolve headers and adjust route cache policies.

Dependencies: route metadata/cache fixes should be reviewed together because metadata errors can persist behind long shared-cache TTLs.

### Sprint 2 - Performance And Mobile

1. Analyze bundle composition and remove shared client code from marketing routes.
2. Dynamically import chat/download/PWA/auth-only widgets where possible.
3. Defer analytics and non-critical third-party code.
4. Repair mobile header layout and touch targets.
5. Replace mobile pricing table with a stacked or visibly scrollable layout.

Dependencies: performance work should happen after hydration fixes so traces are cleaner.

### Sprint 3 - Accessibility, PWA, And Polish

1. Add accessible form error handling and require terms consent on register.
2. Fix contrast tokens for small accent text and cyan buttons.
3. Add ARIA/focus/scroll handling for the mobile menu.
4. Debug service worker install/activation and update lifecycle.
5. Clean duplicate JSON-LD, date mismatches, 404 promo behavior, and noindex policy for auth/offline pages.

## Needs Manual Follow-Up

- Authenticated dashboard audit with real credentials: dashboard routes, billing, marketplace, notifications, host flows, cookies, session attributes, and destructive action guards were out of scope.
- External link and download verification: browser same-origin link checks passed; external docs/download hosts should be checked separately.
- Real INP: passive crawl did not produce meaningful INP events. Use field data or scripted interactions once auth/test accounts are available.
- Payment/OAuth flows: GitHub/Google/Hugging Face login and Stripe/payment surfaces need explicit flow testing.
- Cross-browser manual review: Safari iOS and Firefox should be used for header, input zoom, PWA, and sticky/fixed element behavior.
- Legal/privacy review: third-party analytics and Cloudflare/GTM data flow should be reviewed against the Canada-first/PIPEDA positioning.
