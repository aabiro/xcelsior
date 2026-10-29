# Xcelsior Desktop PWA Plan

Status: Draft
Date: 2026-04-08
Scope: Desktop installable PWA built from the existing Next.js frontend and FastAPI backend

## Executive Summary

Xcelsior should ship one installable PWA for web, mobile web, and desktop from the existing frontend. The desktop track must not create a separate desktop UI and must not change the current web visual language. Desktop gets installability, a standalone window, offline shell behavior, resume behavior, and push notifications. The pages, routes, cards, nav, and dashboard structure stay the same.

The implementation path for this repo is:

- one shared Next.js App Router frontend
- one shared FastAPI backend
- Serwist configurator-style integration with `@serwist/next`
- no `@serwist/turbopack`
- no Electron or Tauri in v1

Because Serwist's official `@serwist/next` guide is the webpack-oriented path, the PWA build should use explicit webpack-backed Next.js scripts until this repo proves a supported alternative. That is a build-system choice, not a UI fork.

## Product Rules

- Keep every existing route and layout intact.
- Do not introduce a desktop-only redesign.
- Do not split web, mobile, and desktop into separate frontends.
- Desktop behavior must be additive only: install, notifications, offline, resume, and app metadata.
- Production stability beats build-tool novelty.
- Chromium desktop is the primary installed-app target. Other browsers must degrade gracefully.

## What The Desktop PWA Is

The v1 desktop app is an installed web app, not a native shell. That means:

- same frontend code as the website
- same backend APIs
- opens in a standalone app window
- supports app shortcuts, install flows, offline shell, and web push
- updates with web deployment, not with an app store release

## What The Desktop PWA Is Not

The v1 desktop app is not:

- a separate Electron app
- a separate Tauri app
- a tray daemon
- a menu-bar app
- a native file-system utility

If Xcelsior later needs real tray integration, auto-launch at OS login, deep protocol registration everywhere, or long-running background behavior outside normal browser/PWA constraints, the correct follow-on is a Tauri 2 wrapper around the same built frontend. That is explicitly out of scope for this first desktop release.

## Current Repo Assessment

The repo already has a partial PWA baseline in the frontend:

- `frontend/src/app/manifest.ts`
- `frontend/src/app/sw.ts`
- `frontend/src/components/InstallBanner.tsx`
- `frontend/src/components/ServiceWorkerRegistrar.tsx`

The repo also already has user notifications on both sides:

- frontend reads notifications from `frontend/src/lib/api.ts`
- notification UI already exists in `frontend/src/components/NotificationBell.tsx`
- FastAPI notification routes already exist in `routes/notifications.py`
- notification persistence already exists in `db.py` via `NotificationStore`
- existing event delivery already exists in `routes/_deps.py`

Current issues that must be cleaned up before shipping desktop PWA:

- the current frontend worktree is moving toward `@serwist/turbopack`, but this plan does not use that path
- `frontend/package.json` currently references `concurrently` in `dev`, but `concurrently` is not installed
- `frontend/src/app/sw.ts` currently imports from `serwist/worker`; this plan standardizes on `@serwist/next/worker`
- the repo references an offline fallback URL (`/~offline`) but does not currently provide a dedicated offline page
- `frontend/src/middleware.ts` is deprecated in Next.js 16 and should become `frontend/src/proxy.ts`
- `frontend/tsconfig.json` does not yet include the Serwist typings and worker lib additions recommended by the official guide

## Desktop UX Rules

Desktop must feel like the website in a window, not like a different product.

Allowed UI additions:

- install entry point in existing navigation or settings surfaces
- notification permission control inside existing settings
- subtle standalone/offline status indicator
- app shortcuts from the installed icon

Not allowed:

- new desktop-only sidebar model
- new desktop-only page hierarchy
- new native-looking chrome that changes the core app identity
- separate marketing and desktop dashboards

## Recommended Technical Direction

### Frontend

- Keep Next.js 16 App Router.
- Use Serwist configurator integration with `@serwist/next`.
- Use `serwist` for service worker generation.
- Use the current manifest route and extend it rather than replacing it.
- Keep the dashboard shell and marketing surfaces visually unchanged.

### Build Tooling

This plan intentionally avoids `@serwist/turbopack`.

Instead:

- add `@serwist/next`
- remove `@serwist/turbopack`
- restore `withSerwistInit(...)` wrapping in `frontend/next.config.ts`
- use explicit webpack-backed Next.js scripts for the PWA build path

Practical script direction:

- `dev`: `next dev --webpack`
- `build`: `next build --webpack`

or, if the team wants to preserve a non-PWA turbo workflow for day-to-day UI work:

- `dev`: current preferred local dev path
- `dev:pwa`: `next dev --webpack`
- `build`: `next build --webpack`

For production, the release build should follow the supported PWA path, not a split path.

### Service Worker Strategy

Use one Serwist service worker with:

- precache for app shell, icons, manifest, and offline route
- runtime caching for safe GET requests only
- network-first for HTML documents with offline fallback
- stale-while-revalidate or network-first for selected read-only API responses
- no interception for auth mutations, billing mutations, launches, deletes, or SSE/WebSocket flows

### Backend

FastAPI remains the source of truth.

Add web push on top of the existing notification model instead of creating a second notification system:

- keep `NotificationStore` as the canonical in-app notification source
- add a web-push subscription store keyed to authenticated users
- emit push messages from the same event paths that already create in-app notifications

For Python, use `pywebpush` with VAPID keys rather than trying to bolt on a Node-only push sender to the FastAPI stack.

## File-Level Plan

### Frontend Foundation

Files to change:

- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/next.config.ts`
- `frontend/src/app/sw.ts`
- `frontend/tsconfig.json`
- `/home/aaryn/storage/projects/xcelsior/.gitignore`

Work:

- add `@serwist/next`
- remove `@serwist/turbopack`
- keep `serwist`
- remove `frontend/serwist.config.js`
- switch `frontend/src/app/sw.ts` to `@serwist/next/worker`
- wrap `next.config.ts` with `withSerwistInit(...)`
- add `additionalPrecacheEntries` for `"/~offline"`
- add worker typing support in `tsconfig.json`
- exclude generated `public/sw.js` from TypeScript and Git noise
- explicitly serve the service worker with no-cache headers

### Next.js Cleanup

Files to change:

- `frontend/src/middleware.ts`
- optionally `frontend/src/proxy.ts` as the renamed replacement

Work:

- rename `middleware.ts` to `proxy.ts`
- rename the exported handler accordingly
- keep existing auth and CSP logic unchanged

This is not a design change. It is maintenance required by Next.js 16.

### Manifest And Metadata

Files to change:

- `frontend/src/app/manifest.ts`
- `frontend/src/app/layout.tsx`

Manifest additions for desktop:

- add explicit `id`
- add explicit `scope`
- keep `display: "standalone"`
- keep the existing icon set and validate sizes
- keep existing screenshots and add true desktop-oriented screenshots if needed
- keep shortcuts, but verify they map to real high-value desktop tasks
- keep experimental members out of the v1 critical path unless they are purely additive

Metadata additions:

- set `applicationName`
- add explicit `appleWebApp` metadata
- keep viewport theme handling
- keep current brand colors and OG metadata

### Offline Route

Files to add:

- `frontend/src/app/~offline/page.tsx`

Rules:

- use the current Xcelsior visual language
- no new layout system
- clearly explain that the user is offline
- provide links back to cached routes when possible
- show retry and reconnect guidance

### Install Experience

Files to change:

- `frontend/src/components/InstallBanner.tsx`
- `frontend/src/components/ServiceWorkerRegistrar.tsx`
- `frontend/src/app/(dashboard)/dashboard-shell.tsx`
- `frontend/src/app/(marketing)/layout.tsx` or equivalent nav component if there is a shared header entry point
- `frontend/src/app/(dashboard)/dashboard/settings/page.tsx`

Plan:

- keep the existing mobile install banner behavior for mobile only
- add a desktop install CTA in an existing surface, not a new floating desktop UI
- recommended placement:
  - dashboard gear/help surface
  - settings page under notifications/app preferences
  - optional marketing-nav install action when eligible
- hide install affordances when already installed or unsupported
- use `beforeinstallprompt` progressively, never as the only path

### Standalone App Behavior

Files to change:

- `frontend/src/app/layout.tsx`
- `frontend/src/app/(dashboard)/dashboard-shell.tsx`
- small shared utility/hooks for display-mode detection and route restore

Desktop-specific behavior to add:

- detect installed/standalone mode
- restore the last meaningful in-app route on launch when possible
- keep regular browser sessions unchanged
- optionally suppress install CTA after successful install
- keep browser and app sessions visually identical

### Cache Strategy

Files to change:

- `frontend/src/app/sw.ts`
- any route handlers that need cache headers

Runtime caching policy:

- static assets: cache-first
- fonts and brand images: cache-first
- HTML navigations: network-first with `"/~offline"` fallback
- marketplace and notification GET endpoints: network-first with short timeout, then stale cache if available
- authenticated dashboard shell assets: stale-while-revalidate or network-first depending on risk
- POST/PUT/PATCH/DELETE: never cache
- SSE and websocket traffic: never cache

Important safety rule:

- billing, launches, destructive actions, and settings writes must never appear to succeed offline

### Push Notifications

Files to change:

- `frontend/src/app/(dashboard)/dashboard/settings/page.tsx`
- `frontend/src/components/NotificationBell.tsx`
- `frontend/src/lib/api.ts`
- `routes/notifications.py`
- `db.py`
- possibly `routes/_deps.py`
- a new backend module for push delivery if we want to keep route files small

Backend additions:

- add a `web_push_subscriptions` table
- store:
  - `id`
  - `user_email` or `user_id`
  - `endpoint`
  - `p256dh`
  - `auth`
  - `user_agent`
  - `created_at`
  - `last_used_at`
  - `revoked_at`
- add subscribe and unsubscribe endpoints
- send push from the same notification event pipeline that already creates in-app notifications
- remove stale subscriptions on push-provider 404 or 410 responses

Frontend additions:

- add a browser-push permission toggle in the existing settings page
- register the push subscription through the service worker
- keep the in-app bell as-is
- allow desktop users to opt in without changing the app layout

Recommended backend dependency:

- `pywebpush`

Recommended config:

- VAPID public key exposed to frontend
- VAPID private key kept server-side only

### Optional Desktop-Only Nice-To-Haves After V1

Only after the core desktop PWA ships cleanly:

- unread badge support when browser support is present
- protocol handlers
- file handlers
- launch-handler tuning

These are not v1 requirements because the manifest features above are still experimental or uneven across browsers.

## Detailed Execution Phases

### Phase 0: Normalize The Foundation

Goal: make the existing frontend PWA path coherent.

Exit criteria:

- `@serwist/next` is the only Serwist integration package in use
- `next.config.ts` is wrapped with Serwist config
- `sw.ts` uses the correct worker helper
- `proxy.ts` replaces deprecated middleware naming
- `npm run build` produces a working service worker and standalone output

### Phase 1: Ship Desktop Installability Without UI Drift

Goal: install Xcelsior as a desktop app with no redesign.

Exit criteria:

- app installs from supported desktop browsers
- manifest is valid
- shortcuts are visible and useful
- installed window launches correctly
- desktop install CTA appears only when eligible

### Phase 2: Ship Offline Shell And Safe Resume

Goal: make the desktop app resilient to weak or lost connectivity.

Exit criteria:

- cached app shell loads
- document navigation falls back to `"/~offline"` cleanly
- stale read-only data is clearly marked
- writes never fake success offline

### Phase 3: Ship Real Desktop Push Notifications

Goal: notify users outside the app window using the existing notification model.

Exit criteria:

- desktop user can opt in from settings
- subscription is persisted server-side
- a test notification can be sent end to end
- invalid subscriptions are cleaned up automatically

### Phase 4: Hardening And Release

Goal: make the desktop PWA safe to advertise publicly.

Exit criteria:

- CI covers build, manifest, offline route, and service worker generation
- desktop install smoke test passes in Chromium
- push notification smoke test passes
- service worker caching behavior is verified for auth flows

## Testing Plan

### Existing

Current frontend CI already has:

- lint
- unit tests
- typecheck
- build

### Add

Add Playwright-based browser checks for Chromium:

- installability check
- standalone launch check
- offline fallback check
- push-subscription permission flow
- notification click-through routing

Add build-time assertions for:

- `public/sw.js` exists
- manifest route resolves
- service worker headers are correct
- offline route resolves

## Release Criteria

The desktop PWA is ready to announce only when all of the below are true:

- the app installs on Chrome and Edge desktop
- the installed experience uses the same UI as the web app
- offline shell works without breaking auth or writes
- notifications work end to end
- no service worker caching bug can expose stale billing or control-plane actions
- CI exercises the PWA path, not just the plain web path

## Explicit Non-Goals For This First Pass

- Electron packaging
- Tauri packaging
- tray icons
- auto-start background process
- offline job launches
- offline billing actions
- desktop-only dashboard redesign

## References

Xcelsior Desktop Infrastructure Plan
Summary

Expand the current desktop strategy from “browser-installed PWA only” to a dual-track desktop product: keep the existing PWA for web/mobile/browser install, and add a Tauri 2 native desktop app as the primary infrastructure client.
Use a hybrid desktop architecture: a bundled native Control Center window plus tray/orchestrator, and a separate main app window that loads the shared hosted Next app. This keeps the web UI familiar while allowing native tray, autostart, deep links, updater, and desktop-only infrastructure surfaces.
Keep shared routes visually consistent with the website. Desktop-specific UI changes apply only when running inside the native shell or standalone PWA. New desktop-only routes live in the native Control Center, not in the shared web dashboard.
Public Interfaces And Config

Add the already-used env vars everywhere they belong: XCELSIOR_WEB_PUSH_STALE_AFTER_DAYS, XCELSIOR_NOTIFICATION_RETENTION_DAYS, XCELSIOR_WEB_PUSH_REVOKED_RETENTION_DAYS, XCELSIOR_PWA_SMOKE_BASE_URL, XCELSIOR_PWA_SMOKE_EMAIL, XCELSIOR_PWA_SMOKE_PASSWORD, XCELSIOR_PWA_SMOKE_REQUIRE_CLICK, and XCELSIOR_PWA_SMOKE_CLICK_TIMEOUT_MS.
Keep desktop build/runtime location config in Tauri config, not scattered envs. The native app config must define: production app origin, local dev server URL, updater endpoint, bundle ID, deep-link scheme xcelsior://, and per-platform bundle targets.
Add one shared desktop capability contract exposed to the frontend. It must report: isNativeDesktop, isStandalonePwa, canInstall, isInstalled, isOnline, notificationsEnabled, trayConnected, autostartEnabled, updateAvailable, currentDesktopRoute, and lastRemoteRoute.
Define one native event contract between the hosted app and Tauri: unread count updates, critical alert count, current route changes, notification click-through targets, auth-required state, and quick-action requests from tray or deep links.
Desktop-only route hierarchy is fixed to: /desktop, /desktop/activity, /desktop/launch, /desktop/settings, and /desktop/links. These belong to the bundled native Control Center UI and are not exposed as general web routes.
Implementation Changes

Create a dedicated Tauri 2 desktop workspace in the repo with a Rust src-tauri backend and a bundled Control Center frontend. Do not use Electron.
The Tauri app must run as a single-instance desktop process. Closing the main app window hides it to tray by default instead of quitting; quitting happens only through tray/app-exit actions.
The Control Center window is the persistent native shell. It owns tray state, autostart, updater state, deep-link intake, startup/offline recovery, and desktop-only infrastructure tools.
The main app window loads the shared hosted Next app in production and the local Next dev server in development. Shared routes remain the source of truth for dashboard, billing, instances, marketplace, admin, and other normal app flows.
Keep the current PWA implementation intact for browser/mobile and as a browser-installed fallback. Tauri does not replace the web app; it wraps and coordinates it for infrastructure users.
Add Tauri-native features using official Tauri 2 capabilities: tray/menu bar, autostart, deep-link handling, updater, single-instance, persistent local store, notifications, and secure shell/open handling where needed.
Use the native Control Center as the desktop-only route layer:
/desktop is the operator overview with connectivity, tray state, updater state, notification health, and quick actions.
/desktop/activity is the native event and alert center, backed by the same backend notification model and current push health data.
/desktop/launch is the launch surface for recent work, dashboard shortcuts, deep-link landings, and reopen targets.
/desktop/settings owns native preferences such as launch-on-login, hide-to-tray, notification behavior, updater channel, and desktop route defaults.
/desktop/links is the explicit landing and debug surface for xcelsior:// deep links and tray-driven route handoff.
Add a desktop runtime provider to the hosted frontend so the shared app can detect native-shell mode and adjust presentation without redesigning routes.
Apply desktop-only polish to the shared app when running inside Tauri or standalone PWA: titlebar-safe spacing, denser content gutters, improved table/card overflow, persistent sidebar and AI-panel state, keyboard shortcuts, route restore, stronger large-screen empty/loading states, and a compact topbar status strip for connection, notifications, and updates.
Run a route audit and polish pass over the most important shared pages: overview, instances, marketplace, telemetry, billing, notifications, admin, and AI. The goal is not a redesign; it is to make the installed/native desktop experience feel deliberate and stable.
Keep backend notifications canonical. Do not build a second desktop notification system. Browser/PWA keeps web push; the native desktop shell consumes the same notification data and surfaces it through tray state, native notifications, and Control Center views.
Extend deployment and release infrastructure for native desktop delivery: signed macOS, Windows, and Linux builds, updater artifacts, version manifests, and staged release channels.
Test Plan

Frontend unit tests must cover the desktop capability provider, native event bridge adapters, shell-state mapping, desktop-only status strip behavior, and manifest shortcuts.
Rust unit tests must cover tray state transitions, deep-link parsing, autostart preference handling, updater state, single-instance behavior, and route handoff logic.
Web/PWA checks stay required: npx tsc --noEmit, vitest PWA tests, npm run build, npm run verify:pwa, and npm run test:pwa:smoke.
Add native desktop integration smoke coverage for: launch to tray, hide/reopen window, autostart bootstrap, deep-link open, single-instance re-focus, offline bootstrap fallback, updater prompt, and tray quick actions.
Keep the authenticated staging push smoke and extend it so a native desktop session can subscribe, receive a real notification, click through, and cleanly unsubscribe.
Manual acceptance must pass on macOS, Windows, and Linux for: first launch, login, hide-to-tray, reopen from tray, Control Center routes, deep-link handling, update flow, notification click-through, and polished shared dashboard routes.
Assumptions And Defaults

This plan supersedes the older parts of PWA_PLAN.md that explicitly kept Tauri, tray integration, and desktop-only routes out of scope.
Tauri 2 is the only native desktop target. Electron is not part of the implementation plan.
The native app ships as direct-download installers with auto-update, not app-store distribution.
macOS, Windows, and Linux are first-class in the initial rollout.
Shared web routes keep the current UI language. New desktop-only functionality lives in the native Control Center route layer, and shared-route desktop polish is additive only.

OAuth 2.0 Migration: Next 3 Phases
Summary
Replace the current mixed session/API-key model with a real internal OAuth 2.0 authorization server.
Use a dedicated Redis-compatible cache layer for ephemeral auth state; do not reuse the generic shared state table.
Keep api_old.py out of scope for these phases.
Target grant coverage: Authorization Code + PKCE for browser, Device Authorization Grant for CLI/device, Client Credentials for service-to-service.
Token model: 15-minute access tokens, 30-day refresh tokens with rotation enabled. Use hybrid token format:
Browser/device/user grants: opaque access + opaque rotating refresh.
Client credentials: signed JWT access tokens, no refresh token.
Public API / Interface Changes
Add OAuth-standard endpoints:
GET /oauth/authorize
POST /oauth/token
POST /oauth/device/authorize
GET /.well-known/oauth-authorization-server
Add OAuth client management endpoints for machine clients and first-party app registration.
Keep /api/auth/me as the canonical user profile endpoint, but make it read from the new OAuth principal resolver.
Keep /api/auth/login only as the first-party credential entrypoint that establishes the browser auth session used by /oauth/authorize; do not add password grant support.
Mark /api/auth/refresh, /api/auth/device, /api/auth/token, /api/auth/verify, and /api/keys/* as deprecated compatibility surfaces during migration.
Add deprecation headers and structured telemetry on every API-key-authenticated request during the transition window.
Phase 1 — OAuth Foundation
Build a dedicated auth/OAuth service layer and move device-flow logic out of routes/health.py into the auth surface.
Add durable auth schema for:
OAuth clients
Client secrets / secret metadata
Refresh token families with rotation state and reuse-detection fields
Optional consent records if browser consent is modeled explicitly
Add a dedicated Redis-compatible cache abstraction for:
authorization codes
PKCE verifier/challenge state
device codes and user codes
opaque access tokens
refresh rotation locks / replay guards
short-lived login/approval state
Implement a unified principal resolver in routes/_deps.py that can validate:
opaque user access tokens from cache
JWT client-credentials tokens by signature, issuer, audience, expiry, and scopes
legacy session/API-key credentials during migration
Fail closed if Redis is unavailable for opaque token issuance or introspection.
Add metrics and logs for token issuance, token refresh, token reuse detection, device-code polling, and deprecated API-key usage.
Phase 2 — Browser and Device Flow Migration
Browser flow:
Use Authorization Code + PKCE end-to-end.
Keep the current login UX, but after credential/MFA verification, redirect into /oauth/authorize instead of minting long-lived bearer tokens directly.
Exchange the code server-side and store OAuth-backed browser auth in secure httpOnly cookies.
Device flow:
Replace the current ad hoc device endpoints with OAuth Device Authorization Grant semantics.
Keep the signup bridge: an unauthenticated browser user may sign up or log in during device approval.
Do not issue anonymous broad-access sessions; tokens are issued only after browser auth is completed and the device code is approved.
Refresh behavior:
Rotate refresh tokens on every successful refresh.
Detect refresh-token reuse and revoke the entire token family.
Return a stable error code for reused or revoked refresh tokens.
Compatibility:
Make the legacy /api/auth/device, /api/auth/token, and /api/auth/verify endpoints thin adapters to the new OAuth device flow for one migration window, with deprecation headers.
Make /api/auth/refresh a compatibility wrapper over refresh-token grant behavior during the same window.
Phase 3 — Machine-Client Migration and API-Key Deprecation
Add confidential OAuth clients and Client Credentials Grant for service-to-service callers.
Issue JWT access tokens for client credentials with 15-minute TTL, explicit scopes, issuer, audience, and key rotation support.
Introduce preferred client-management APIs/UI for OAuth clients; keep /api/keys/* only as a deprecated migration surface during the transition window.
Restrict user/self-service endpoints to user-grant tokens or browser session cookies; machine credentials must not call refresh, logout, sessions, password, MFA, profile mutation, preferences mutation, or account-deletion routes.
Preserve existing API-key auth only for the migration window on non-user programmatic APIs, with:
deprecation headers
request telemetry
sunset date in docs and responses
Default migration window: 90 days from OAuth client-credentials launch.
Test Plan
Browser auth:
authorization code + PKCE success path
code replay rejection
PKCE mismatch rejection
MFA + OAuth handoff
cookie-backed browser session renewal through refresh rotation
Device auth:
device authorization issuance
poll pending / slow-down / expiry behavior
login-or-signup bridge during approval
single-use approval
code replay rejection
Refresh rotation:
refresh success rotates token
old refresh token reuse revokes family
revoked family cannot mint new access tokens
Client credentials:
JWT validation success
bad signature / audience / expiry rejection
scope enforcement on protected routes
Migration compatibility:
deprecated /api/auth/device|token|verify|refresh wrappers map cleanly to the new engine
deprecated API keys still work only where allowed during the transition window
deprecation headers and telemetry are emitted
Failure modes:
Redis unavailable -> opaque-token issuance/introspection fails closed
DB unavailable -> client/refresh issuance fails closed
JWT validation remains local and does not depend on Redis
Assumptions
api_old.py retirement is explicitly excluded from this plan.
No OpenID Connect ID-token work is included; /api/auth/me remains the user-info source unless a downstream consumer requires OIDC later.
Browser/device user tokens are opaque and Redis-backed; service-client tokens are JWTs.
Existing API keys are deprecated, not removed immediately.
Redis-compatible means Redis or Valkey semantics are acceptable, with one dedicated auth cache abstraction rather than generic state-table reuse.