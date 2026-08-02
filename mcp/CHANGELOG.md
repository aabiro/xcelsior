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

_None._

### Added

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
