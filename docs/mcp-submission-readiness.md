# MCP connector submission readiness (X2 / gate GX2)

> Companion to [mcp-enterprise-adoption-plan.md](mcp-enterprise-adoption-plan.md).
> That document is the plan; this one is the checklist you work down immediately
> before pressing submit, and the record of what was actually done.

**Status legend.** ✅ shipped and verified · 🔧 shipped, needs a value only the
provider can issue · 👤 requires a human with an account we cannot automate ·
⬜ not started.

---

## 1. What is already in the code

| Item | State | Where |
|---|---|---|
| `WWW-Authenticate` on every 401, naming resource metadata | ✅ | [mcp/src/auth/challenge.ts](../mcp/src/auth/challenge.ts) |
| Protected-resource metadata at both RFC 9728 paths | ✅ | [mcp/src/index.ts](../mcp/src/index.ts), [nginx/mcp-xcelsior.conf](../nginx/mcp-xcelsior.conf) |
| Canonical resource identifier `https://mcp.xcelsior.ca/mcp` | ✅ | [oauth_service.py](../oauth_service.py) |
| OAuth consent screen (sign in → approve → callback) | ✅ | `_oauth_consent_page` in [routes/auth.py](../routes/auth.py) |
| CIMD client identification | ✅ | [oauth_registration.py](../oauth_registration.py) |
| RFC 7591 dynamic registration, hard-gated | ✅ | `POST /oauth/register` |
| Provider callbacks registered (Claude, ChatGPT, Grok, Copilot Studio) | ✅ | `CONNECTOR_REDIRECT_URIS` |
| Port-agnostic loopback redirects | ✅ | `redirect_uri_matches` |
| Public/operator tool profiles | ✅ | [mcp/src/tools/profiles.ts](../mcp/src/tools/profiles.ts) |
| Response-hygiene filter on every tool result | ✅ | [mcp/src/lib/hygiene.ts](../mcp/src/lib/hygiene.ts) |
| `/.well-known/openai-apps-challenge` | 🔧 | Route ships; 404s until `XCELSIOR_MCP_OPENAI_APPS_CHALLENGE` is set |
| Reviewer account seeder | 🔧 | [scripts/seed_reviewer_account.py](../scripts/seed_reviewer_account.py); needs a password and a production run |

---

## 2. Reviewer account

The single most common cause of a failed connector review is a reviewer who
signs in and sees nothing, or who is asked for a code sent to an inbox they do
not own.

```bash
# On the production host, with the real database environment:
XCELSIOR_REVIEWER_PASSWORD='<strong, generated, stored in the secret manager>' \
  python3 scripts/seed_reviewer_account.py

# Verify at any time — exits non-zero if anything drifted:
python3 scripts/seed_reviewer_account.py --check
```

What the seeder guarantees, and why each one matters:

| Property | Why |
|---|---|
| **Not an admin** (`role=submitter`, `is_admin=0`) | A reviewer must see the public customer profile. An admin token would enumerate the operator surface we deliberately keep unlisted. |
| Email already verified, no verification token | No inbox round-trip. |
| MFA disabled | No second factor a reviewer cannot receive. |
| No IP allowlist | The existing `demo@xcelsior.ca` button is gated to our own networks — the opposite of what a reviewer connecting from Anthropic's egress needs. This account is reachable from anywhere, so **the password is its only control**: generate it, store it in the secret manager, and deliver it through the submission form, never over email. |
| $250 CAD wallet credit | `should_i_run_this` and `estimate_job_cost` return a real verdict instead of "insufficient funds". |
| Three instances (running / completed / failed) | `list_instances`, `get_instance`, `get_instance_logs`, and `get_instance_timeline` all return something, including an error path a reviewer can inspect. |

**Do not hand over `demo@xcelsior.ca`.** It is a platform admin behind an IP
allowlist — wrong on both counts.

---

## 3. OpenAI domain verification (BLOCKER 4)

The route is configuration-backed and returns **the bare token, nothing else** —
no JSON, no list, no trailing newline (the value is trimmed on load, because a
newline from a heredoc or a secret manager is a one-byte mismatch that reads as
"verification just doesn't work").

1. Start the OpenAI plugin submission; the portal issues a token.
2. Set `XCELSIOR_MCP_OPENAI_APPS_CHALLENGE=<token>` in the production `.env`.
3. Redeploy the MCP service (blue/green; see [mcp/README.md](../mcp/README.md)).
4. Verify from **outside** our network:

```bash
curl -sS https://mcp.xcelsior.ca/.well-known/openai-apps-challenge | xxd | tail -2
# must be exactly the token, with no trailing 0a
```

Until step 2 the route returns 404 with an explanatory body. That is deliberate:
a hardcoded placeholder would either be a lie now or stale later.

---

## 4. Listing assets

| Asset | Value | State |
|---|---|---|
| Connector URL | `https://mcp.xcelsior.ca/mcp` | ✅ |
| Name | Xcelsior | ✅ |
| Category | Developer tools / Cloud infrastructure | ✅ |
| Short description | Rent GPUs by the hour from a marketplace of independent hosts, or run serverless inference — priced and settled in CAD. | ✅ |
| Long description | See §4a below | ✅ |
| Logo | [xcelsior_icon_app_gradient.png](../xcelsior_icon_app_gradient.png) — confirm the portal's minimum dimensions before upload | 👤 |
| Privacy policy | `https://xcelsior.ca/privacy` | ✅ |
| Terms | `https://xcelsior.ca/terms` | ✅ |
| Support | `https://xcelsior.ca/support` | ✅ |
| Documentation | `https://docs.xcelsior.ca` | ✅ |
| Security posture | `https://xcelsior.ca/security` | ✅ (X6.29) |
| Screenshots | Only required if we ship MCP-Apps UI. We do not, so none. | ✅ (n/a) |

### 4a. Long description

> Xcelsior is a distributed GPU marketplace where independent hosts compete on
> price. Connect it to your assistant to discover live GPU availability and spot
> pricing, estimate what a job will cost before you commit, launch and monitor
> training runs, and call serverless inference endpoints billed per token with
> no idle cost. Everything is quoted and settled in Canadian dollars, and
> workloads can be pinned to a jurisdiction when data residency matters.
>
> Destructive operations — cancelling, terminating, evicting — never execute on
> the model's say-so. They produce a server-bound action plan that a human
> approves out of band; the assistant's `confirm` flag expresses intent and
> never substitutes for that approval. Every call is scope-checked, rate-limited
> per tool, and written to an audit trail with the principal, tenant, arguments
> hash, and outcome.

---

## 5. Accounts and roles (👤 — nobody can automate these)

| Provider | Requirement | Owner | State |
|---|---|---|---|
| Anthropic | Claude **Team or Enterprise** org (individual plans cannot submit); an Owner submits, or an Enterprise admin delegates Directory management | | ⬜ |
| Anthropic | Verified business identity | | ⬜ |
| OpenAI | Org role with **Apps Management = Write** | | ⬜ |
| OpenAI | Verified business identity | | ⬜ |

Record who holds each one here as they are obtained — a submission that stalls
because nobody knows whose account it is under is a self-inflicted delay.

---

## 6. GX2 — what closes the gate

Each line is a check somebody runs and records, not a box somebody ticks.

- [ ] **Challenge endpoint returns exactly the token.** `curl … | xxd` shows no
      trailing newline and no JSON. (§3)
- [ ] **A fresh human completes the reviewer path on an unfamiliar machine,
      timed.** Paste the connector URL into Claude and into ChatGPT, sign in as
      the reviewer account, approve, and call three tools. Record the wall-clock
      time; the target in §1 of the plan is under 60 seconds to connected.
      *A person who has not seen this before — reading the walkthrough
      cold — is the test. Someone who built it cannot run it.*
- [ ] **All listing URLs return 200 with valid TLS from an external vantage.**
      The scheduled conformance job covers the connector and its metadata; check
      the marketing URLs in §4 the same way.
- [ ] **A real connection succeeds from Anthropic and from OpenAI** — not merely
      from a generic cloud IP. This is the assertion no script can make for us:
      a cloud runner proves foreign-network reachability, which catches broad WAF
      and TLS failures, and proves nothing about a specific provider's egress.
- [ ] **Pre-submission checklists for both providers pass item by item**, with
      the result recorded here.

### Running the machine-checked half

```bash
# GX0: discovery, challenge, OAuth round trip, TLS — from an external vantage.
python3 scripts/gx0_conformance.py \
  --base https://mcp.xcelsior.ca/mcp \
  --email reviewer@xcelsior.ca --password "$XCELSIOR_REVIEWER_PASSWORD"

# GX1: does a model pick the right tool from our published descriptions?
XCELSIOR_MCP_TOKEN=... python3 scripts/mcp_tool_eval.py --base https://mcp.xcelsior.ca/mcp

# Reviewer account state.
python3 scripts/seed_reviewer_account.py --check
```

The scheduled workflow in
[.github/workflows/mcp-conformance.yml](../.github/workflows/mcp-conformance.yml)
runs the GX0 chain daily from GitHub's egress. **A blocked gate is not a passed
gate** — both scripts report `BLOCKED(env)` rather than green when they cannot
run, and that state has to be resolved before the gate closes.

---

## 7. Rejection handling

A rejection is recorded here **verbatim**, routed to the phase that owns the
cause, and re-submitted. The gate does not close on "submitted".

| Date | Provider | Verbatim reviewer feedback | Owning phase | Fix | Re-submitted |
|---|---|---|---|---|---|
| | | | | | |
