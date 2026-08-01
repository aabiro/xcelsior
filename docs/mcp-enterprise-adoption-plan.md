# Xcelsior MCP — Enterprise Adoption & Connector Distribution Plan

> Get `mcp.xcelsior.ca` from "a well-built hosted MCP server" to "listed, discoverable, and
> installable from inside every major LLM product," without loosening a single one of the
> security properties it already has.

**Status:** plan of record. Written 2026-07-29 and revised 2026-08-01 against a live audit of
`mcp/`, `routes/auth.py`, `oauth_service.py`, `nginx/mcp-xcelsior.conf`, the production endpoint,
and the current first-party connector documentation for Anthropic, OpenAI, xAI, Microsoft, GitHub,
Google Cloud, and Cloudflare.
**Implementation status (2026-08-01):** X0, X1, X2, X6 and X7 are **built and merged**; X5's
one-click installs are built. What remains is what only a human with a provider account can do —
see [§9 Delivery status](#9-delivery-status-2026-08-01) for the exact line between the two,
and [mcp-submission-readiness.md](mcp-submission-readiness.md) for the pre-submission checklist.
**Companion docs:** [mcp-server-plan.md](mcp-server-plan.md) (v1/v2 build plan, delivered),
[mcp-submission-readiness.md](mcp-submission-readiness.md) (X2 checklist),
[mcp-tool-versioning.md](mcp-tool-versioning.md) (X6.28 published contract),
[mcp-quickstarts.md](mcp-quickstarts.md) (X7.35),
[runbooks/mcp-edge.md](../runbooks/mcp-edge.md) (X6.32),
[xcelsior-production-control-plane-mcp-blueprint.md](xcelsior-production-control-plane-mcp-blueprint.md).
**Scope boundary:** this is xcelsior product work. It is independent of `agent-forge`; forge is
simply one more client of `mcp.xcelsior.ca` once this lands.

---

## 1. The goal, stated plainly

Three outcomes, in priority order:

1. **A user of Claude / ChatGPT / Grok / Gemini / Copilot can add Xcelsior in under 60 seconds** —
   paste a URL or click an entry in a directory, log in with their Xcelsior account, approve, done.
2. **We appear in the places people browse for connectors**, not just the places they can paste a
   URL into. Directory presence is the difference between "our customers can use it" and
   "people who have never heard of us discover it."
3. **Nothing about (1) or (2) weakens the control plane.** Scoped tokens, action-plan approval on
   destructive operations, per-tool rate limits, and the audit trail are the product's
   differentiators — they are not overhead to be traded away for adoption.

### 1a. Decision record: what “remote” means

**There is no general “must be hosted by somebody else” rule.** Provider-hosted connector clients
need a stable MCP endpoint they can reach. “Remote,” “internet-hosted,” and “publicly reachable”
describe network reachability; they do not require AWS, Azure, GCP, or any other third party to own
the machine running Xcelsior.

The current rules differ by surface:

- **Claude remote connectors:** Anthropic initiates the connection from its cloud. The server and
  authorization server must be reachable from Anthropic's published egress ranges. Claude local
  MCP/Desktop extensions are a separate distribution mode.
- **OpenAI custom/private use:** a private or on-premises MCP can use OpenAI Secure MCP Tunnel in
  supported products. **Public plugin submission is stricter:** it requires a stable, publicly
  reachable HTTPS Streamable HTTP endpoint. Secure MCP Tunnel alone, a temporary tunnel, or a local
  URL does not qualify. A public HTTPS proxy in front of a private origin does qualify.
- **Grok custom connectors:** xAI's cloud must reach a public URL. xAI documents ngrok and
  Cloudflare Tunnel for local testing; a stable endpoint is the production choice.
- **Microsoft Copilot Studio:** connects to an existing Streamable HTTP endpoint through its MCP
  onboarding wizard or a Power Platform connector. Public certification additionally requires a
  verified publisher, endpoint ownership/control, approved authentication, packaging, and review.
- **GitHub Copilot / VS Code / Copilot CLI:** supports local stdio, localhost HTTP, and remote HTTP.
  A public endpoint is not required when the client runs locally; a cloud-hosted Copilot client
  still needs network reachability from its execution environment.

**Decision:** keep `https://mcp.xcelsior.ca/mcp` as the canonical public connector. Do not move it
or provision a VM merely to satisfy a perceived third-party-hosting rule. Hosting may move later
for uptime, scaling, or operations, but that is an SLO decision, not a connector-compliance gate.

Current source references, re-check at execution time because provider rules change:

- [Claude remote connector network requirements](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)
- [Claude connector building and authentication](https://claude.com/docs/connectors/building/authentication)
- [Claude directory submission](https://claude.com/docs/connectors/building/submission)
- [OpenAI MCP deployment requirements](https://developers.openai.com/plugins/build/mcp-server)
- [OpenAI MCP authentication](https://developers.openai.com/plugins/build/auth)
- [OpenAI plugin submission](https://developers.openai.com/plugins/deploy/submission)
- [OpenAI private MCP tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
- [Grok custom connectors](https://docs.x.ai/grok/connectors)
- [Microsoft Copilot Studio MCP onboarding](https://learn.microsoft.com/en-us/microsoft-copilot-studio/mcp-add-existing-server-to-agent)
- [Microsoft MCP certification](https://learn.microsoft.com/en-us/microsoft-copilot-studio/mcp-server-certification)
- [GitHub Copilot CLI MCP reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference)
- [Google Cloud Run MCP hosting](https://docs.cloud.google.com/run/docs/host-mcp-servers)
- [Cloudflare remote MCP hosting](https://developers.cloudflare.com/agents/model-context-protocol/guides/remote-mcp-server/)

---

## 2. What already exists (audited, with references)

This is a mature server. Recording the verified baseline so no phase below re-does delivered work.

| Capability | State | Evidence |
|---|---|---|
| Hosted Streamable HTTP endpoint | ✅ `https://mcp.xcelsior.ca/mcp`, stateless mode (no session round-trip), blue/green on :8770/:8771 | [mcp/src/index.ts:92](../mcp/src/index.ts#L92), [nginx/mcp-xcelsior.conf](../nginx/mcp-xcelsior.conf) |
| Protected Resource Metadata (RFC 9728) | ✅ *(was ⚠️ — fixed in X0.2)* served at both discovery paths; `resource` is now the exact `/mcp` URL | [mcp/src/index.ts](../mcp/src/index.ts), [mcp/src/config.ts](../mcp/src/config.ts) |
| Authorization Server Metadata (RFC 8414) | ✅ `authorization_code`, `refresh_token`, `client_credentials`, device code; S256 only; `resource_indicators_supported: true` | [routes/auth.py:478](../routes/auth.py#L478) |
| Audience binding (RFC 8707) | ✅ enforced, with a test proving resource substitution is rejected | [tests/test_mcp_oauth_contract.py:143](../tests/test_mcp_oauth_contract.py#L143) |
| Asymmetric signing + JWKS | ✅ RS256, production refuses symmetric machine tokens | [tests/test_mcp_oauth_contract.py:67](../tests/test_mcp_oauth_contract.py#L67) |
| Scope model | ✅ 20+ scopes, per-tool required-scope map, tenant vs operator classes | [mcp/src/auth/scopes.ts](../mcp/src/auth/scopes.ts), [mcp/src/tools/contracts.ts](../mcp/src/tools/contracts.ts) |
| Tool annotations | ✅ **all four hints on every tool**, injected centrally; registration *throws* for any tool lacking a contract | [mcp/src/audit/context.ts:44-50](../mcp/src/audit/context.ts#L44) |
| Structured output schemas | ✅ defaulted for every tool | [mcp/src/audit/context.ts:49](../mcp/src/audit/context.ts#L49) |
| Human-approval gate on destructive ops | ✅ server-bound action plans; `confirm:true` never substitutes for approval; drain never evicts | [mcp/README.md:63](../mcp/README.md#L63) |
| Audit trail | ✅ per-call record: principal, tenant, scopes, redacted arg hash, plan id, idempotency key, upstream route/status, trace id, latency, outcome | [mcp/src/audit/context.ts:84](../mcp/src/audit/context.ts#L84) |
| Secret redaction | ✅ regex-guarded before hashing/logging | [mcp/src/audit/context.ts:13](../mcp/src/audit/context.ts#L13) |
| Rate limiting | ✅ per-principal, per-tool, plus auth-failure abuse tracking and a `watch_instance` slot cap | [mcp/src/rate-limit.ts](../mcp/src/rate-limit.ts) |
| Observability | ✅ OpenTelemetry traces, Prometheus metrics, pino structured logs | [mcp/src/observability/](../mcp/src/observability/) |
| Tool surface | ✅ 35+ single-purpose tools, 2 resources, 3 prompt playbooks | [mcp/README.md:45](../mcp/README.md#L45) |
| stdio distribution | ✅ `npx @xcelsior-gpu/mcp` | [mcp-cli/](../mcp-cli/) |
| CI | ✅ typecheck, OpenAPI drift check, unit, `npm audit`, docker build, 13-step real-stack E2E against real Postgres | [.github/workflows/mcp.yml](../.github/workflows/mcp.yml) |
| Deploy safety | ✅ readiness + authenticated `initialize` + `tools/list` against standby before nginx switch | [mcp/README.md:78](../mcp/README.md#L78) |
| TLS | ✅ Let's Encrypt, TLS 1.2/1.3, HSTS | [nginx/mcp-xcelsior.conf](../nginx/mcp-xcelsior.conf) |
| Legal/support pages | ✅ `/privacy`, `/terms`, `/support` exist | `frontend/src/app/(marketing)/` |

**Read this table as the answer to "are we behind?" — we are not.** The remaining work is a
short list of protocol-contract details, a public-vs-operator surface boundary, and distribution
paperwork—not a second MCP implementation.

---

## 3. Gap analysis

Ranked by what actually blocks a listing.

### BLOCKER 1 — `WWW-Authenticate` is missing from the 401
Unauthenticated requests return a 401 with a JSON body and **no `WWW-Authenticate` header**
([mcp/src/index.ts:46](../mcp/src/index.ts#L46) and [:57](../mcp/src/index.ts#L57)).

The MCP auth spec — and Claude's connector flow explicitly — require that *every* unauthenticated
request, **including the first `initialize`**, returns 401 carrying `WWW-Authenticate` that points
at the protected-resource metadata. That header is the only breadcrumb a client has to discover
where to authenticate. We serve correct metadata at `/.well-known/oauth-protected-resource`; no
client is ever told to look for it.

**Failure mode:** the connector appears broken with nothing in our logs looking wrong. This is
the single highest-value fix in this document and it is roughly three lines.

```
WWW-Authenticate: Bearer realm="xcelsior",
  resource_metadata="https://mcp.xcelsior.ca/.well-known/oauth-protected-resource",
  error="invalid_token", error_description="..."
```
(`error`/`error_description` only on the invalid-token branch; the no-credentials branch carries
`realm` + `resource_metadata` alone.)

The same fix must settle the canonical resource identifier. Production metadata currently returns
`resource: "https://mcp.xcelsior.ca"`, while Claude's connector documentation requires it to match
the URL entered by the user, including the path component: `https://mcp.xcelsior.ca/mcp`. Change
the metadata, OAuth `resource` propagation, token `aud`, validation, configuration, and tests as one
deliberate migration. Do not change only the JSON document and strand existing audience-bound
tokens.

### BLOCKER 2 — the documented front door is `client_credentials`
[mcp/README.md:32](../mcp/README.md#L32) and the dashboard flow both tell users to mint a machine
token. **Anthropic does not support pure machine-to-machine `client_credentials` as the
user-facing OAuth flow**—the directory path is *paste URL → Connect → our login page → approve*.
Anthropic supports administrator-supplied static headers in beta for custom organization
connectors, but authenticated directory submissions require OAuth. OpenAI's public plugin path is
likewise user-consent OAuth for per-user Xcelsior data and actions.

The good news is this is wiring, not building: `authorization_code` + S256 PKCE + `refresh_token`
already work **and are already audience-bound to the MCP resource**
([tests/test_mcp_oauth_contract.py:80](../tests/test_mcp_oauth_contract.py#L80)). What is missing:
- `https://claude.ai/api/mcp/auth_callback` registered as a permitted redirect URI.
- Port-agnostic loopback redirect matching (`http://localhost:*`, `http://127.0.0.1:*`) for Claude
  Code and other native clients, which use a random local port per attempt.
- The token endpoint accepting `application/x-www-form-urlencoded` (a documented silent-failure
  cause when servers accept only JSON) — **verify, don't assume**.
- Access ~1h / refresh ~30d lifetimes, with `refresh_token` grant reachable by connector clients.
- README + dashboard reframed so `client_credentials` is the *automation* path and OAuth consent
  is the *default* path.

### BLOCKER 3 — no scalable connector client-registration path
`/.well-known/oauth-authorization-server` advertises neither CIMD support nor a DCR
`registration_endpoint` ([routes/auth.py:478](../routes/auth.py#L478)); `/api/auth/register` is
human signup, not RFC 7591 client registration.

Claude and OpenAI support Client ID Metadata Documents (CIMD), DCR, or provider-held/predefined
credentials. Microsoft Copilot Studio documents DCR discovery as its simplest OAuth onboarding
path. Current Anthropic guidance recommends CIMD or Anthropic-held credentials over DCR for a
high-traffic directory connector because DCR can create very large client-registration tables.
OpenAI also prioritizes CIMD when it is available.

**Recommendation: implement both, in this order:**

1. **CIMD as the preferred Anthropic/OpenAI path.** Advertise
   `client_id_metadata_document_supported: true` and public-client token exchange with S256 PKCE;
   validate HTTPS metadata-document client ids, exact redirect URIs, and resource indicators.
2. **RFC 7591 DCR as the compatibility path** for Microsoft and other clients: allowlisted
   redirect hosts, exact loopback rules, MCP-audience-only clients, read-biased default scopes,
   rate limits, and expiry for unused registrations.
3. **Provider-held static OAuth credentials only as a contingency**, not the portable default.

### BLOCKER 4 — OpenAI domain-verification route absent
When the submission portal issues a challenge, it requires
`https://mcp.xcelsior.ca/.well-known/openai-apps-challenge` to return **only** that plugin's
generated token—not JSON, not a list. Build a configuration-backed route now, then deploy the
portal-issued token before scanning/submission. Do not invent or permanently hardcode a token.

### OPTIONAL EXPANSION 5 — no `search` / `fetch` tools
OpenAI no longer requires `search` and `fetch` for an ordinary connected MCP server or plugin
listing. Implementing their schemas remains valuable if we deliberately pursue ChatGPT
**company knowledge**. That surface needs absolute user-openable URLs for citable sources and
internal ids in the result `id` field. Good corpora exist: `llms.txt`, pricing reference,
marketplace, and docs. This work must not block the base OpenAI plugin submission.

### GAP 6 — two annotation-accuracy nits
Annotations are wired correctly; two values are arguably wrong, and reviewers check that
annotations **match real behavior**:
- `openWorldHint` is hardcoded `false` for every tool
  ([mcp/src/tools/contracts.ts](../mcp/src/tools/contracts.ts)). Discovery tools
  (`list_available_gpus`, `get_spot_prices`, `search_marketplace`) read a live external
  marketplace whose contents we do not control — that is open-world by definition.
- `DESTRUCTIVE` contains only `cancel_instance`, `terminate_instance`, `evict_host_workloads`.
  `drain_host` is disruptive to running workloads and reads as destructive to an operator.
  Deliberate call required, then documented either way.

### GAP 7 — submission paperwork
Privacy/terms/support pages exist. Still needed: a **reviewer demo account with real sample data,
no MFA, no email/SMS confirmation, no private-network access** (the most common cause of a failed
review); production-ready logo; category; short + long descriptions; verified business identity;
a Claude **Team or Enterprise** org (individual plans cannot submit) with an Owner running it;
and — if we ship any MCP-Apps UI — 3–5 PNG carousel screenshots ≥1000px wide with paired prompts
(no video, no GIF).

### GAP 8 — WAF / egress
A WAF that blocks Anthropic, OpenAI, xAI, Microsoft, or a cloud-hosted GitHub client produces
discovery that works in a browser and fails from the provider. Test from an independent network
and through real provider connection flows. A generic cloud runner proves foreign-network
reachability; only a provider-side probe or actual connection proves that provider's egress path.

---

## 4. Target surfaces

| Surface | Path in | Notes |
|---|---|---|
| **Anthropic Connectors Directory** | Submission portal inside Claude.ai org admin | Needs Team/Enterprise; Team Owners submit, while Enterprise can delegate Directory management. Highest-signal listing for our buyer. |
| **OpenAI Plugin Directory** (ChatGPT + Codex) | Partner portal; app directory migrated to *Plugins* on 2026-07-09 | Domain-verification challenge required. Cannot reference an existing published integration — submit fresh. |
| **ChatGPT company knowledge** | Optionally implement `search`/`fetch` schemas after the base plugin path | Distribution multiplier, but not a base connector or listing requirement. |
| **Grok custom MCP connector** | Team admin adds the canonical URL in Grok Business/Enterprise connector management | xAI requires public reachability. Use the production URL; tunnels are development aids. Do not claim public-catalog distribution until xAI documents a vendor-submission path. |
| **Gemini Enterprise custom MCP data store** | Google Cloud console; "data store" = connector, "tools" = *actions* | StreamableHTTP only (we comply — no SSE). Constraints: ≤100 enabled actions, no PSC, no VPC-SC in preview, off by default behind an org policy, and Gemini Enterprise must be registered as an OAuth client with the customer's IdP. Enterprise-customer-led, not a public directory. |
| **Microsoft Copilot Studio / M365 distribution** | Partner Center offer type "Connectors and Agents for Microsoft Copilot Studio" | Requires verified publisher, enrollment in the M365+Copilot program, and ownership of the endpoint. Cross-tenant publishing is an explicit, deliberate choice. Re-check downstream discovery surfaces at submission time. |
| **GitHub Copilot / VS Code / Copilot CLI** | Remote HTTP URL, local stdio package, and optional enterprise MCP registry | Local clients do not impose a public-hosting rule. Cloud clients must reach the endpoint. Keep both the hosted URL and `npx` path. |
| **Official MCP registry** | `registry.modelcontextprotocol.io` | Cheap, neutral discoverability. Verify current publish flow at execution time. |
| **One-click install** | Cursor / VS Code deeplinks, Claude Code `claude mcp add`, `npx @xcelsior-gpu/mcp` | Owned by us — buttons on `/mcp` marketing page and the dashboard connect card. |
| **Direct enterprise** | URL + docs + scoped credentials | Already works; improves as a by-product of everything above. |

---

## 4a. Architecture decision: one product, separate trust surfaces

Do **not** create a second repository or rewrite the MCP as a separate product. The existing
`mcp/` package is already the correct separately deployable Node application. Keep one codebase
and preferably one container image, with configuration selecting the exposed trust surface.

```text
Claude / ChatGPT / Grok / Gemini / Copilot
                    |
             public HTTPS edge
                    |
        mcp.xcelsior.ca/mcp
        curated customer connector
                    |
       authenticated FastAPI contract
                    |
        Xcelsior control-plane services

ops-mcp.xcelsior.ca/mcp (unlisted, stronger policy)
                    |
         same image, operator profile
                    |
       authenticated FastAPI contract
```

The boundaries are:

- **Public customer connector:** `https://mcp.xcelsior.ca/mcp`, stable tool schemas, customer and
  tenant workflows, provider-directory submission, user-consent OAuth, and no platform-global
  operator surface.
- **Operator connector:** the same implementation deployed with an operator profile at a separate
  unlisted hostname or equivalently strong boundary. Require operator scopes, stronger identity
  policy, and explicit action-plan approval. Never submit this surface to a public directory.
- **Company knowledge:** if pursued, add the read-only `search`/`fetch` tools to the customer
  profile or expose a deliberately read-only profile. Do not make a third deployment by default.
- **Backend:** both profiles remain stateless MCP-to-FastAPI adapters. They never connect directly
  to PostgreSQL, workers, Docker, SSH, or host management planes.

This split preserves the comprehensive internal capability without forcing every public model to
inspect or every reviewer to exercise platform-operator tools. It also prevents a provider's
frozen tool snapshot from becoming an accidental promise that every end user can see operator
actions.

## 4b. Hosting decision and migration trigger

**`mcp.xcelsior.ca` does not move for connector compliance.** It is already deployed—systemd
under `/opt/xcelsior`, nginx + Let's Encrypt, blue/green on :8770/:8771—and is reachable over
public HTTPS. The immediate gaps are protocol, authentication, surface curation, and paperwork.

Hosting options, in recommended order:

1. **Keep the existing deployment now.** This is the lowest-risk path to the first listing.
2. **If SLO, power/network independence, scaling, or operational load justifies a move, use GCP
   Cloud Run.** The MCP is stateless, containerized Node, and Streamable HTTP; Cloud Run is a
   direct fit and is preferable to a raw VM. Preserve the canonical custom domain so clients and
   OAuth metadata do not change. See [Google's MCP hosting guidance](https://docs.cloud.google.com/run/docs/host-mcp-servers).
3. **Azure Container Apps** is the strongest alternative if Entra/Microsoft distribution becomes
   the dominant concern. **Cloudflare** is useful as the TLS/WAF/rate-limit edge or can host a
   stateless MCP with more runtime adaptation. AWS App Runner/ECS, Fly.io, Render, and Railway are
   valid HTTPS container hosts but do not currently justify migration churn.
4. **Temporary tunnels are development tools.** Grok documents ngrok and Cloudflare Tunnel;
   OpenAI Secure MCP Tunnel supports private custom use. Neither a temporary tunnel nor OpenAI's
   tunnel alone satisfies OpenAI public-plugin submission.

No separate cloud project is required by any provider. If Cloud Run becomes the production host,
create dedicated production and staging GCP projects (or equivalently strong project boundaries)
for IAM, secrets, budgets, quotas, audit logs, and blast-radius isolation. Keep the source in this
monorepo; the project boundary is operational, not a fork of the application.

## 4c. External validation without a permanent VM

GX0 and GX2 still require discovery, the 401 challenge, the OAuth round trip, and TLS to be proven
from outside the Xcelsior/headscale network. **The requirement is independent execution, not a
specific VM.** Use an ephemeral scheduled runner:

1. First choice: a scheduled GitHub Actions workflow with protected environment secrets.
2. If provider/network separation or longer execution makes that unsuitable: a GCP Cloud Run Job
   triggered by Cloud Scheduler in a small validation project.
3. Optional supplement: an external synthetic-monitoring provider for TLS, metadata, and
   unauthenticated discovery checks.

The runner executes the full GX0 chain and GX2 TLS/WAF checks on a schedule. It can also run the
forge black-box eval suite. A permanent GCP VM is justified only if a later workload genuinely
needs an always-on process that cannot run as a job; connector compliance does not justify it.

**Honest limitation:** any generic cloud runner proves reachability from a foreign network, which
catches broad WAF and TLS-chain failures. It does not prove a specific provider's egress is
allowed. Only a real provider connection, provider-side probe, or submission review closes that
provider-specific assertion.

## 4d. agent-forge as the continuous conformance client

forge is an MCP **client**. Pointing it at `mcp.xcelsior.ca` turns it into exactly the workload a
directory reviewer simulates — an unfamiliar agent, holding only an OAuth token, reading only our
published tool descriptions. This is the highest-value early use of forge and it directly feeds
three gates:

- **GX0** — forge exercises both supported third-party registration paths: CIMD for the preferred
  public-client flow and DCR for compatibility. If it can identify/register, authorize, call, and
  refresh without hand-holding, BLOCKER 3 is genuinely closed.
- **GX1** — the evaluation set becomes a forge **eval suite** (ADK eval) that runs continuously
  rather than being a one-time submission checkbox. "Every tool invoked, zero unhandled errors"
  becomes a standing regression test.
- **GX1/GX6** — forge's usage ledger is an **independent** count of tool calls. Reconciling it
  against our own `mcp/tool-audit` records catches audit gaps that self-testing structurally
  cannot find.

**The one rule that makes this valid — forge must stay a black box.** It gets no bypass token, no
direct API access behind `/mcp`, no shared types or generated client that could paper over a
schema mismatch, and no privileged scopes. The instant forge is treated as special, it stops
being evidence. It must be indistinguishable from Claude on the wire.

**What it does not replace:** GX1 still requires a *human who did not write the tools* to walk the
surface. forge catches the mechanical failures — which is most of them, early and cheaply — not
the ambiguous-description failures that need fresh human judgment.

## 5. Gate system

Same discipline as the agent-forge plan: gates are machine-checked, run against **staging over the
public internet** (not localhost — that is precisely what hides WAF and TLS problems), and a gate
that cannot run is recorded **BLOCKED(env)**, never green.

- **No self-attestation.** "OAuth works" means a scripted client completed
  discovery → 401 challenge → metadata fetch → authorize → code+PKCE exchange → authenticated
  `tools/list`, from outside our network.
- **External-vantage requirement.** Every discovery/auth assertion runs at least once from a
  non-xcelsior egress IP. This may be an ephemeral job; no permanent VM is implied.
- **No regressions on delivered properties.** Every gate re-runs the existing OAuth contract
  tests, the 13-step real-stack E2E, and a scope-denial assertion. Adoption work must not
  loosen the control plane.
- **Submission gates are outcome-gated, not effort-gated.** X3/X4 close on *reviewer approval*,
  not on "submitted." Rejections are recorded with reviewer feedback and re-entered at the
  phase that owns the cause.

---

## 6. Phases

### X0 — Protocol contract & the auth front door  (Gate GX0)
The blockers. Nothing else ships first; every other phase depends on a connectable server.

1. `WWW-Authenticate` on both 401 branches, pointing at resource metadata (BLOCKER 1).
2. Migrate the canonical OAuth resource identifier from the origin to the exact connector URL
   `https://mcp.xcelsior.ca/mcp` across metadata, authorization requests, token `aud`, validation,
   configuration, compatibility handling, and tests (BLOCKER 1).
3. Register `https://claude.ai/api/mcp/auth_callback`; add port-agnostic loopback matching
   (BLOCKER 2).
4. Verify the token endpoint accepts `application/x-www-form-urlencoded`; fix if not.
5. Confirm/settle token lifetimes (~1h access, ~30d refresh) and that connector clients can
   refresh.
6. Implement and advertise CIMD as the preferred Anthropic/OpenAI client-identification path
   (BLOCKER 3).
7. Implement RFC 7591 DCR as the compatibility path, hard-gated: allowlisted redirect hosts,
   MCP-audience-only, read-biased default scopes, rate limits, and unused-client expiry
   (BLOCKER 3).
8. Reframe README + dashboard: OAuth consent is the default path; `client_credentials` is the
   automation path.

**GX0 (all from an external vantage point):**
- Unauthenticated `POST /mcp` `initialize` ⇒ **401 with `WWW-Authenticate` naming the resource
  metadata URL**; following that header end-to-end reaches a working authorize endpoint.
- Protected-resource metadata `resource`, authorization `resource`, token `aud`, and validator
  expectation all resolve to the exact canonical `/mcp` identifier; old-origin tokens follow the
  documented compatibility/expiry path rather than failing unpredictably.
- Full scripted `authorization_code` + PKCE run ⇒ token whose `aud` is the MCP resource ⇒
  authenticated `tools/list` returns the stable public customer profile.
- Loopback redirect succeeds on **two different random ports** in the same test run.
- Token endpoint accepts form-encoded; JSON-only would fail here.
- `refresh_token` grant returns a working access token.
- CIMD: identify → authorize → call, then reject an invalid/untrusted metadata-document client id.
- DCR: register → authorize → call, then assert a registration with an off-allowlist redirect is
  **rejected**, and that a DCR-issued client **cannot** obtain operator scopes.
- Resource-substitution rejection and RS256-only assertions still pass (no auth regressions).
- MCP Inspector connects against staging with zero manual header surgery.

### X1 — Tool surface hardening for review  (Gate GX1)

> **X1 has its own detailed plan: [`mcp-tool-surface-plan.md`](./mcp-tool-surface-plan.md).**
> It covers the curated tool list (37 → ~91 actions, under the 100-action Gemini cap), the journey
> completeness test, and the payments architecture. Headline findings: the API exposes **514
> endpoints** and MCP reaches ~7% of them; **storage, compliance, teams and payouts have zero
> tools**; `billing:write`, `events:read` and `mcp_actions:approve` are declared scopes wired to no
> tool at all; and the installed SDK (**1.29.0**) already supports **URL Mode Elicitation**
> (SEP-1036), which is the correct — and spec-mandated — mechanism for deposits, adding payment
> methods, and payout onboarding. Gates GT0–GT4 there are prerequisites to GX1 below.

9. Define and enforce two tool profiles from the same codebase: the stable public customer profile
   and the unlisted operator profile. Public-directory credentials must never enumerate
   platform-global operator tools.
10. Fix `openWorldHint` on genuinely open-world discovery tools; decide and document `drain_host`'s
   `destructiveHint` (GAP 6).
11. **Optional company-knowledge track:** implement `search` + `fetch` with OpenAI's exact schemas
   over llms.txt / pricing / marketplace / docs, returning absolute user-openable URLs and
   internal ids in `id`. Do not block base plugin submission on this item.
12. Description pass: every tool description states *when to use it* and its cost/impact. Reviewers
   call every tool — descriptions must match behavior exactly.
13. Response hygiene audit: no internal identifiers, debug payloads, auth material, or undisclosed
    user fields in any tool output. Check against our own privacy policy, line by line.
14. Evaluation set: direct requests, indirect phrasings, follow-ups reusing earlier ids, write
    actions requiring approval, and requests that should call **no** tool.

**GX1:** annotation-vs-behavior conformance test (a read-only-hinted tool that mutates fails the
build) · public credentials enumerate no operator tools · public schemas remain stable · **every
public tool invoked once against staging with reviewer-grade credentials, zero unhandled errors**
(this is literally what reviewers do) · every operator tool invoked separately with operator-grade
credentials · response-hygiene scanner clean · eval set ≥ target pass rate with no-tool cases
correctly abstaining. If company knowledge is in scope, add `search`/`fetch` schema conformance and
an assertion that every returned URL resolves.

### X2 — Submission readiness  (Gate GX2)
15. Configuration-backed `/.well-known/openai-apps-challenge` route implemented; after the portal
    issues a token, deploy it and verify the route returns that bare token (BLOCKER 4).
16. Reviewer demo account: real sample data, MFA off, no email/SMS step, no private-network need.
17. Listing assets: logo, category, short/long descriptions, docs/support/privacy/terms URLs.
18. Verified business identity with both Anthropic and OpenAI; Claude Team Owner or Enterprise
    member with delegated Directory management able to submit; OpenAI org role with Apps
    Management = Write.
19. External test job plus real Anthropic and OpenAI connection attempts proving the public edge,
    auth host, and redirects are not WAF-blocked (GAP 8).
20. Screenshots only if we ship MCP-Apps UI (3–5 PNG ≥1000px, cropped to app response, paired
    prompts).

**GX2:** challenge endpoint returns exactly the token, nothing else · a **fresh human** completes
the reviewer-account path start to finish on an unfamiliar machine, timed · all listing URLs
return 200 with valid TLS from an external vantage · real connection green from Anthropic and
OpenAI—not merely a generic cloud IP · pre-submission checklists for both providers pass
item-by-item, recorded.

### X3 — Anthropic submission  (Gate GX3)
Submit; track in the submissions dashboard; hold reviewer feedback as the gate's evidence.
**GX3:** listed in the Connectors Directory. A rejection is recorded verbatim, its cause routed
back to the owning phase, and re-submitted — the gate does not close on "submitted."

### X4 — OpenAI submission  (Gate GX4)
Submit through the plugin portal from scratch (no referencing an existing integration); the portal
scans the server and validates tool metadata during review. Pursue company-knowledge eligibility
as a separate optional track after the base plugin path is ready.
**GX4:** published in the Plugins Directory (ChatGPT + Codex). Same rejection-handling rule.

### X5 — Google, Grok, Microsoft, GitHub, registry, one-click  (Gate GX5)
21. **Gemini Enterprise:** custom-MCP-data-store onboarding guide for enterprise customers —
    org-policy override, IdP OAuth client registration, FQDN allowlisting, and a **curated ≤100
    action** subset (we exceed the limit if every tool is enabled; choose the subset deliberately
    rather than letting it truncate).
22. **Grok:** verify the canonical production URL as a custom connector in a real Grok
    Business/Enterprise team. Document team-admin provisioning and end-user auth. Treat public
    xAI catalog distribution as uncommitted until xAI publishes a vendor submission path.
23. **Microsoft:** first verify the existing server through Copilot Studio's MCP onboarding wizard;
    then pursue verified publisher, M365+Copilot program enrollment, Partner Center
    "Connectors and Agents for Microsoft Copilot Studio" submission, MCP package. Cross-tenant
    publishing is a deliberate decision, made and recorded.
24. **GitHub Copilot:** verify hosted HTTP and local `npx` paths in VS Code and Copilot CLI; publish
    enterprise-registry instructions without implying that local clients require public hosting.
25. **Official MCP registry** entry.
26. **One-click installs** on the `/mcp` marketing page and dashboard connect card: Cursor and
    VS Code deeplinks, `claude mcp add`, Copilot CLI, and `npx @xcelsior-gpu/mcp`.

**GX5:** Gemini Enterprise connector verified against a real project (action subset ≤100, actions
explicitly enabled, tool call succeeds) · Grok custom connector verified · Microsoft submission
accepted · GitHub Copilot hosted and local paths verified · registry entry live · **every
one-click button exercised on a clean machine** and lands in a working authenticated session.

### X6 — Enterprise-grade operations  (Gate GX6)
What an enterprise buyer's security review asks for after they've found us:
27. Public status page + uptime history for `mcp.xcelsior.ca`; SLO published. Use measured SLO
    misses—not connector folklore—as the trigger for a Cloud Run migration review.
28. Tool **versioning and deprecation policy** — `_meta["xcelsior/toolVersion"]` already exists;
    give it a contract (notice period, overlap window, changelog).
29. Security posture page: data handling, retention, sub-processors, the action-plan approval
    model, scope reference, audit-log availability to customers.
30. Customer-visible audit export (we already record everything at
    [audit/context.ts:84](../mcp/src/audit/context.ts#L84) — expose it).
31. Load/soak test at the published rate limits; confirm graceful degradation, not collapse.
32. Incident runbook for the MCP edge specifically (blue/green rollback is already there — write
    down when to pull it).

**GX6:** status page live and honest · deprecation policy published and enforced by a CI check
that fails on an unversioned breaking tool change · soak test at 2× published limits with graceful
degradation · security page reviewed against an actual enterprise security questionnaire.

### X7 — Adoption engine  (Gate GX7)
33. Instrument the **activation funnel**: discovery hit → 401 challenge → authorize → first
    successful tool call → first *paid* workload. We have the audit trail; turn it into a funnel.
34. Per-surface attribution (which directory produced which activations).
35. Quickstarts per client (Claude, ChatGPT, Grok, Cursor, VS Code, Claude Code, GitHub Copilot,
    Gemini Enterprise, Microsoft Copilot Studio), each verified end-to-end on a clean machine.
36. Publish the differentiator explicitly: **agent-safe destructive operations via server-bound
    action plans.** Nobody else in this category ships that. It is the reason an enterprise
    lets an agent near real infrastructure, and it should be the headline of the security page,
    not a footnote in the README.

**GX7:** funnel dashboard live with real numbers · every quickstart reproduced from scratch by
someone who did not write it · drop-off between "connected" and "first tool call" measured, with a
named owner for the biggest cliff.

---

## 7. Sequencing

X0 gates every hosted connector—without interoperable discovery and user OAuth, the public URL is
reachable but not connectable. X1 and X2 run in parallel after X0 (engineering vs. paperwork;
different people, no shared blocker). The public/operator profile split closes in X1 before any
directory scan. X3 and X4 are submit-and-wait, so start both as soon as X1 and X2 close and let
review queues run concurrently. X5's custom-connector checks can begin as soon as GX0 closes, but
its public Microsoft certification and enterprise-customer-led Google work may trail. X6 and X7
are continuous once X3/X4 land.

The honest critical path to a first listing is **X0 → X1/X2 → X3**. Everything before X3 is under
our control; X3's clock is not.

---

## 8. Non-goals

- **No loosening of the approval model to please a reviewer.** If a directory's UX expectations
  conflict with server-bound action plans, we document the flow rather than remove the gate.
- **No `client_credentials` removal.** It stays as the automation/CI path; it just stops being the
  documented front door for humans.
- **No second MCP repository or duplicated application.** Public customer and unlisted operator
  profiles share the implementation and build artifact while retaining separate trust boundaries.
- **No VM or cloud migration merely to satisfy “remote.”** Independent validation runs as an
  ephemeral external job. Production hosting changes only for measured reliability, scale,
  residency, or operating-cost reasons.
- **No operator tools in public directory submissions.** Comprehensive internal capability is
  preserved behind the operator profile; public curation is a security and model-quality boundary.
- **No company-knowledge delay to the base OpenAI submission.** `search`/`fetch` is an optional
  distribution track, not a general connected-server requirement.
- **No new tools purely to look bigger.** Reviewers and models both degrade with tool bloat, and
  Gemini Enterprise caps enabled actions at 100. Curation beats count.

---

## 9. Delivery status (2026-08-01)

Recorded here rather than in a commit message because the line between "built"
and "needs a human with a provider account" is the thing anyone picking this up
next needs first. Every ✅ below is code in the repository with a test that
fails if it regresses.

### X0 — protocol contract & auth front door ✅

| Item | Where |
|---|---|
| 1. `WWW-Authenticate` on both 401 branches | [mcp/src/auth/challenge.ts](../mcp/src/auth/challenge.ts), wired in `src/index.ts`; asserted in the hosted E2E and on every deploy by `scripts/deploy.sh` |
| 2. Canonical resource `https://mcp.xcelsior.ca/mcp` | `oauth_service.py` — one `normalize_resource_indicator` replaces six string comparisons; legacy origin accepted until **2026-11-30** and then rejected on its own |
| 3. Claude callback + port-agnostic loopback | `CONNECTOR_REDIRECT_URIS`, `redirect_uri_matches` (RFC 8252 §7.3); widening is confined to loopback |
| 4. Form-encoded token endpoint | Already correct; now pinned by `test_token_endpoint_accepts_json_as_well_as_form_encoding` |
| 5. ~1h connector access, 30d refresh | `MCP_ACCESS_TOKEN_TTL_SEC`, applied by audience; `expires_in` now reports the real TTL |
| 6. CIMD | [oauth_registration.py](../oauth_registration.py) — same-origin redirects, SSRF-guarded fetch, advertised as `client_id_metadata_document_supported` |
| 7. RFC 7591 DCR | `POST /oauth/register`, hard-gated: allowlisted redirect hosts, MCP-audience pin, read-biased scopes, no operator scopes, per-IP and global rate limits, unused-registration expiry |
| 8. README + dashboard reframed | OAuth consent is the documented default; `client_credentials` is labelled the automation path |
| **Consent screen** | Not a numbered item, but §1 says "log in … approve". `POST /oauth/authorize` with a single-use staged request, per-client/per-scope/per-resource grants, revocable |

**GX0:** `scripts/gx0_conformance.py` runs the whole chain from an external
vantage and reports `BLOCKED(env)` rather than green when it cannot.
[.github/workflows/mcp-conformance.yml](../.github/workflows/mcp-conformance.yml)
schedules it daily.

### X1 — tool surface hardening ✅

- **9. Two profiles from one codebase.** Operator tools are not *registered* under the customer profile, so they cannot appear in `tools/list` or be called by name. Asserted on the wire in both E2E suites.
- **10. GAP 6 resolved, both halves.** `openWorldHint` is now true for the three live-marketplace reads. `drain_host` was reviewed and deliberately **left non-destructive**: the versioned endpoint it calls returns *"new placements stopped; running workloads untouched"* — the destructive counterpart is `evict_host_workloads`, which is flagged. Documented in `contracts.ts` and pinned by test.
- **11. Company knowledge** (`search`/`fetch`, OpenAI's exact schemas) over docs, `llms.txt`, pricing, and marketplace. **Off by default** so it cannot delay the base submission. All 17 cited doc URLs verified to resolve.
- **12. Descriptions** rewritten into one reviewed file, injected centrally; a tool with no entry fails registration. Each states what it does, *when to use it*, and its cost/impact — enforced by test, including that destructive tools warn and non-idempotent ones say so.
- **13. Response hygiene.** Auth material, debug payloads, and undisclosed fields are stripped from every tool result — both `structuredContent` and the `content` text, since most clients render the text. Removals increment a metric.
- **14. Evaluation set.** 31 cases across direct, indirect, follow-ups reusing earlier ids, approval-gated writes, and **no-tool abstention**. `scripts/mcp_tool_eval.py` grades against the *live published* surface.

### X2 — submission readiness 🔧

Code complete; three items need a provider to act first. See
[mcp-submission-readiness.md](mcp-submission-readiness.md).

- **15.** `/.well-known/openai-apps-challenge` ships and returns the bare token; 404s until `XCELSIOR_MCP_OPENAI_APPS_CHALLENGE` is set. 🔧
- **16.** `scripts/seed_reviewer_account.py` — non-admin, no MFA, no email step, **no IP gate**, $250 credit, three instances across running/completed/failed. Needs a password and a production run. 🔧
- **17–20.** Listing assets recorded; provider accounts and the real connection tests are 👤.

### X6 — enterprise operations ✅

- **27.** `/status` (live, honest — it does not claim uptime history it cannot compute) and the SLO table, also in the runbook. `/api/status` now probes the MCP connector.
- **28.** [mcp-tool-versioning.md](mcp-tool-versioning.md) + `mcp/tool-surface.json` + a CI check that names the exact field when a breaking change lands without a version bump. Verified by deliberately introducing one.
- **29.** `/security` — leads with server-bound action plans, which is X7.36's requirement too.
- **30.** `GET /api/v1/mcp/tool-audit` — the customer's own trail, keyset-paged, strictly tenant-scoped.
- **31.** `scripts/mcp_soak.py` — passes only when excess load is *refused* with `Retry-After`, p95 holds, and **nothing** returns 5xx.
- **32.** [runbooks/mcp-edge.md](../runbooks/mcp-edge.md), organised by symptom, including the failure where nothing in our logs looks wrong.

### X7 — adoption engine ✅

- **33/34.** `GET /api/v1/mcp/activation-funnel` — authorized → first tool call → first success → first write → first paid, with per-surface attribution recorded **at consent time** (the same connector client serves every surface, so it is unrecoverable later). It states plainly that discovery hits and 401s are Prometheus counters, not rows — persisting one row per unauthenticated request would make an unauthenticated endpoint a write amplifier.
- **35.** [mcp-quickstarts.md](mcp-quickstarts.md) — nine clients, one URL.
- **36.** The differentiator is the headline of `/security`, not a footnote.

### X5 partially ✅

**26.** One-click installs (Cursor, VS Code, Claude Code, Copilot CLI, `npx`) on
the dashboard connect card, with a test that no link or command carries a
credential. Items 21–25 (Gemini Enterprise, Grok verification, Microsoft
certification, GitHub registry, MCP registry) need real accounts on those
platforms.

### What is *not* done, and why

- **X3, X4, and most of X5** are submit-and-wait: they need a Claude Team/Enterprise Owner, an OpenAI org with Apps Management = Write, verified business identity, and real connection attempts from each provider. No code can close them.
- **Real-provider egress** stays unproven until a provider actually connects. A scheduled cloud runner proves foreign-network reachability and nothing more — §4c is explicit about this and the conformance script says so in its own output.
- **`XCELSIOR_MCP_OPENAI_APPS_CHALLENGE`** is empty until the portal issues a token.

### Three pre-existing defects the E2E uncovered, now fixed

Driving `mcp/tests/e2e/real-stack.test.ts` to green surfaced three real bugs.
All three predate this work — verified by stashing every change, restarting the
API from a clean tree, and reproducing each failure identically. Two of them
were production-affecting, and both sat directly on the path a directory
reviewer walks.

1. **Agent-key tenant resolution** (`oauth_service.py`). A machine token minted
   by Quick Connect resolved its workspace from a live user lookup only, and
   dropped `team_id` entirely. `_canonical_owner_id` then fell back to the
   *user id*, so a plan was filed under a tenant the owner's own browser never
   resolves to and approval returned `404 plan_not_found`. Fixed by
   `_agent_key_workspace`, which prefers the live customer/team, falls back to
   the key's stored tenant, and distinguishes a team tenant from a customer
   tenant instead of guessing.

2. **Agent keys rejected as invalid API keys** (`routes/_deps.py`). An agent key
   (`xcel_ai_…`) also starts with the serverless key prefix (`xcel_`), so a
   prefix test alone routed every Quick Connect credential into the
   serverless-key validator, which failed it with `401 Invalid API key`. Fixed
   with an explicit `looks_like_agent_key` exclusion; regression test in
   `tests/test_serverless_security.py::TestAgentKeyPrefixCollision`, which was
   confirmed to fail when the guard is removed.

3. **Fixture host was never admissible** (`scripts/mcp_e2e_fixture.py`).
   Migration 082 made `admission_state` the authority and had the projection
   trigger *overwrite* `payload->>'admitted'`. The fixture set only the payload
   flag, so every placement returned `no_eligible_host`. Test-only, but it had
   been masking the two bugs above.

The suite now passes end to end — thirteen steps, three consecutive clean runs —
including the assertion that the public listing is exactly the customer profile
and that `drain_host` is unreachable on the customer surface while a separately
deployed operator instance serves it.

### A fourth: the published OpenAPI spec regenerated from itself

Re-running the gate surfaced one more, and it matters here because the published
spec is what a directory reviewer, an SDK generator, and an enterprise customer
read. `api.py` reassigns `app.openapi` to serve the previously generated
`public/openapi.json`; `scripts/generate_public_openapi.py` called that same
accessor, so its input was its own last output.

The checked-in spec carried the damage: the "public developer surface only" note
had accumulated **ten copies**; component ordering was reshuffled on every run,
churning ~900 lines per regeneration; and — the part that matters — allowlisted
operations never picked up model changes, so **five schemas had drifted from the
API they document**. `ServerlessEndpointCreate` was missing seven fields the API
accepts: `execution_mode`, `lora_adapters`, `managed_engine`,
`queue_timeout_sec`, `source_ref`, `source_ref_branch`, `source_type`.

The generator now reads `FastAPI.openapi(app)`, which bypasses the instance
override and builds from the live routes; components are emitted sorted; the
note is idempotent. **The curated surface did not change** — 70 paths / 79
operations before and after, because the double allowlist (Fern overrides ∩
`CLIENT_OPERATION_ALLOWLIST`) is what decides publication, and nothing internal
leaked in. The change is purely additive.

`tests/test_public_openapi.py` was the gate that missed this: it compared the
operation set and tag names only, so it could answer "are the right endpoints
published?" but never "is the document still true?". It now asserts
whole-document equality against a fresh generation, that the generator does not
read its own output (confirmed to fail when the loop is put back), that
components are sorted, and that the note appears once.

### Gate state at close (2026-08-01)

Every gate re-run *after* the four fixes, not before:

| Gate | Result |
|---|---|
| `ruff check .` | clean |
| `pyright` | 0 errors (0 in the ratcheted `reportCallIssue` / `reportArgumentType`) |
| `./run-tests.sh` (full pytest) | **4649 passed, 7 skipped** |
| `mcp` vitest, **including both E2E suites** | 17 files, **151 passed, 1 skipped** |
| `mcp/tests/e2e/real-stack.test.ts` | 13/13 steps, 3 consecutive runs, no leaked listeners |
| frontend | `tsc` clean, 226 vitest passed, production build |
| `docker build` MCP image / `docker compose config` | both pass |
| `npm audit --omit=dev` (mcp) | 0 vulnerabilities |

The real-stack E2E counts toward that total now; it was previously excluded
because it did not pass. Two `eslint` errors remain in
`frontend/public/site-assets/reference/support.js`, a vendored marketing asset
from commit `0b235f6` that this work never touched and that is not part of the
app bundle — reported rather than fixed in passing.

Two teardown races in the E2E itself were also fixed, because both read as
product failures when they were not: the restart step rebound the port before
the old listener released it (`EADDRINUSE`, surfacing as "MCP restart failed"),
and `afterAll` signalled the replicas without waiting, so a back-to-back run
died on startup. Teardown now waits for exit and escalates to `SIGKILL` if a
replica ignores `SIGTERM` — which, verified across three runs, it never does.
