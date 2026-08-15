# Xcelsior MCP — surface changelog

Every change to the published tool surface at `https://mcp.xcelsior.ca/mcp`,
newest first. Breaking changes are listed first within a release and carry a
removal date. The contract this changelog serves is
[docs/mcp-tool-versioning.md](../docs/mcp-tool-versioning.md).

The machine-readable form of the current surface is
[`tool-surface.json`](tool-surface.json); a breaking change that is not
reflected here and version-bumped fails the build.

---

## Unreleased

### Breaking

- **Seven tools no longer claim that repeating a call is free.** `run_training_job`,
  `schedule_under_budget`, `create_volume`, `snapshot_volume`, `run_pipeline` and
  `open_instance_access` move to `idempotency: "none"` and
  `idempotentHint: false` (**2.0.0 → 2.1.0**); `schedule_under_budget` also gains
  `openWorldHint: true` (**→ 2.2.0**).

  `idempotency` defaulted to `"keyed"` for anything not read-only, so 25 tools
  advertised *"calling this again has no additional effect"* when only four sent
  an idempotency key. `run_training_job` and `schedule_under_budget` both POST
  `/instance` with no key, so a client that trusted the hint and retried a call
  that appeared to time out would launch **a second instance and be billed for
  both**; `open_instance_access` mints a fresh single-use ticket per call rather
  than returning the previous one.

  **No notice period, and that is deliberate.** §3's notice exists so a client
  is not surprised by a deliberate design change. This is the retraction of a
  promise that was never true, and every day it stands is a day a client can
  lose money by believing it. Tools whose repeat really is harmless are
  unaffected — `terminate_instance`, `register_ssh_key`, which 409s on a
  duplicate fingerprint, and `promote_artifact_to_volume`, whose endpoint
  carries its own idempotency key.

  `schedule_under_budget` reads `/api/v2/gpu/available` and
  `/api/v2/marketplace/spot-prices` — the two live third-party feeds behind
  `list_available_gpus` and `get_spot_prices`, both already flagged — and then
  spends against the answer, so a cached reading must not be assumed to hold.

  **Action:** if you retry these tools on timeout, check state first —
  `list_instances`, `list_volumes`, or the volume's snapshots — rather than
  calling again. Each tool's description now says so.

### Added

- **PostHog MCP analytics.** The pinned `@posthog/mcp` beta captures standard
  MCP lifecycle/tool metadata for HTTP and STDIO, groups stateless calls by the
  authenticated principal, preserves the reviewed tool schemas, strips request
  and response content before send, and flushes on graceful shutdown.
- **Connector OAuth front door.** `WWW-Authenticate` now accompanies every 401,
  naming the protected-resource metadata, so a connector can discover how to
  authenticate. Client identification by CIMD or RFC 7591 dynamic registration,
  a consent screen, and port-agnostic loopback redirects.
- **Canonical resource identifier** is now `https://mcp.xcelsior.ca/mcp` (the
  exact URL a user pastes). Tokens bound to the previous origin
  `https://mcp.xcelsior.ca` remain valid until **2026-11-30**.
- **`search` and `fetch`** (ChatGPT company knowledge) over the documentation
  site, `llms.txt`, pricing, and marketplace listings. Off by default; enable
  with `XCELSIOR_MCP_COMPANY_KNOWLEDGE=1`.

### Changed

- **Trust-surface split.** The public connector serves the customer profile
  only. `drain_host`, `undrain_host`, `evict_host_workloads`,
  `retry_agent_command`, `get_scheduler_health`, `get_host_capacity`, and
  `list_reconciliation_findings` moved to the unlisted operator profile. They
  were never usable without operator scopes, so no credential loses access it
  could previously exercise.
- **`openWorldHint` corrected to `true`** on `list_available_gpus`,
  `get_spot_prices`, and `search_marketplace` — they read a live third-party
  marketplace. Annotation accuracy is now enforced against the contract at
  registration time.
- **Every tool description rewritten** to state when to use it and what it
  costs or changes. Descriptions are prose, not contract; no version changed.
- **Connector access tokens now live ~1 hour** (previously 15 minutes).
  Refresh tokens are unchanged at 30 days.
