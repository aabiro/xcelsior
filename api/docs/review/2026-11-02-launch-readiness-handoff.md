<!-- residency-guard: documents-removal — dated launch-readiness review -->
# Xcelsior launch readiness — exploration and agent handoff

*2026-11-02. Research only: nothing on production or in the code was changed to write this.*

Xcelsior is a **global** GPU marketplace. It has no data-residency requirements and
is not pitched as a Canadian or sovereign cloud. Read every finding below with that in mind.

## How this was checked

| Source | What it covered | Limits |
| --- | --- | --- |
| Live probes of `xcelsior.ca` (curl) | 22 public pages and endpoints, `/readyz`, `/api/status`, `/api/v2/gpu/available`, `/api/pricing/reference`, response headers, JS weight | Unauthenticated only |
| `api/frontend/e2e-prod` Playwright pass, run today | 42 public-page checks, desktop + mobile, all passed; screenshots viewed | **The signed-in half (48 tests) did not run.** The demo gate refuses this network because the home public IP changed |
| Repo at `faeae010`, rechecked at `a3292583` (residency references removed) | Routes, billing, tax, signup, marketing components, roadmap docs. The CA/ON signup stamp and the CAD pricing badge survive `a3292583` | No prod shell or prod DB access |
| Deep-research workflow (web) | Competitor pricing, payouts, trust tiers, isolation | 19 claims survived a 3-vote check, all about Vast.ai, RunPod, Salad and Akash |
| Targeted follow-up (web, 2026-11-03) | Lambda, TensorDock, Hyperbolic, io.net, DigitalOcean/Paperspace, SLAs, MCP, currency/tax, serverless pricing, referrals, startup credits, HN/Product Hunt launches | Each claim read on the vendor's own page or HN's search API; single reader, no 3-vote check (§5b) |

Anything marked **unverified** was not checked against a live system.

---

## 1. What works (verified today)

* **Production is up and on the repo's schema.** `/readyz`: storage on PostgreSQL, Redis auth cache,
  NFS volumes `mode=full, reachable=true`, agent gateway on, schema `current=149` (= repo head
  `149_drop_tier_multiplier`). `/api/status`: all ten services `operational`.
* **Spot and serverless are switched on in production.** `/api/pricing/spot-enabled` and
  `/api/v2/serverless/enabled` both report `enabled: true, global_enabled: true`.
* **Serverless is substantially built**, not a stub: about 1,950 lines in `api/routes/serverless.py`
  (endpoints CRUD, warm, health, metrics, batches, jobs + SSE stream, keys, `/v1/serverless/{id}/run|runsync`)
  plus a `/dashboard/inference` page. `api/docs/NEXT_PRIORITIES_ROADMAP.md` §6 still lists all of it as
  unchecked, so that roadmap is stale.
* **Every public page renders cleanly** on desktop and mobile: no `NaN`/`undefined`/raw i18n keys, no
  sideways overflow, no broken images, no CSP violations, no same-origin 4xx/5xx, no hydration errors.
  The pricing page is populated from the live rate table, and a wrong password gets a visible message.
* **Edge setup is sound.** Cloudflare in front, HSTS with preload, a strict CSP, HTTP/3 advertised
  (`alt-svc: h3`), and marketing pages answering in about 180 ms. (`/api/status` took about 3 s,
  which is worth a look if a status page will poll it.)
* **Unauthenticated API surface is closed.** Marketplace, pricing-reserved, reputation, transparency
  and inference model lists all return 401 without a session.

## 2. What doesn't work, or isn't true yet

Ordered by how much each would hurt a public launch.

### 2.1 International customers are taxed as Ontario residents — **blocker for "global"**

* Every new account is stamped `country: "CA", province: "ON"` at signup, for email registration
  (`api/routes/auth.py:1938`) and OAuth (`api/routes/auth.py:2404`).
* Deposit tax resolves from the **account** address first and only falls back to IP when the account
  has none (`billing/stripe_tax.py:80-110`, `api/routes/billing.py:162`). Because the account is never
  empty, the IP path never runs.
* The dashboard has a **Canadian province selector** (`src/components/ui/province-selector.tsx`) and no
  country selector, so a US or EU customer cannot correct it.
* Result: if Stripe Tax has an Ontario registration, every customer worldwide is charged 13% HST.
  Which registrations are live in production is **unverified**.

### 2.2 The supply that exists is hidden, and the supply on show does not exist

* Live inventory (`/api/v2/gpu/available`) is **three GPUs**: one RTX 2060 (`ca-central`, $0.08/hr) and two
  RTX 3060s (`ca-central`, `ca-east`, $0.12/hr).
* The `/gpu-availability` "Live fleet" table shows twelve rows (B200, H200, H100, H100 NVL, A100, L40S, L40, L4,
  A40, RTX 5090, RTX 4090, RTX 3090), **every one "on request" or "by reservation", and none of the three
  GPUs that are actually free.** Cause: `curateMarketingGpus` in `api/frontend/src/lib/marketing-gpu.ts:93-110`
  fills its 12 slots from `MARKETING_GPU_PRIORITY` (reference-price catalogue rows with zero stock) before it
  considers in-stock extras.
* The same page's hero card reads **"0 regions"**, and says "B200 and H100-class GPUs are on the fleet,
  ready when you are."
* The home hero shows an "RTX 4090 · Live Utilization 72% · 18.4 GB · 310 W" card. There is no RTX 4090
  on the fleet, so that "live" card is illustrative data presented as live.
* The existing e2e check "the GPU availability page shows real inventory or says there is none" passes,
  so it does not catch this.

### 2.3 Copy contradicts itself and the product

| Claim | Where | Problem |
| --- | --- | --- |
| "Billed by the second" / "Rent by the minute" / "billed by the hour" | home hero / availability hero / home feature card | Three granularities. Confirm the real one in `billing/` and use one phrase everywhere |
| "From $0.30/hr" (RTX 4090 spot) vs "Spot from $0.11" (RTX 3090) | home / availability | Neither GPU exists; the real floor is the 2060 at $0.08 on-demand ($0.032 spot by the reference table) |
| "Get Started · $5 Free Credits" | home CTA | The body says *deposit $5 to claim a $5 bonus*. It is a match, not free credit |
| Comparison table: "AWS Canada" column, "$0.30 CAD" vs "$0.54 CAD" | home §03 | Wrong frame for a global product; competitors publish in USD |
| "All prices in Canadian Dollars" badge | `/pricing` | Fine for honesty, but a global buyer sees no USD figure anywhere |
| "No auction … no bidding you can lose" | home hero | Migration `148_paid_queue_bids` adds *funded queue bids*. Check the copy still holds |
| "Agent-Native (MCP) — 86 tools, native · first, Jun 2026" | home comparison | "first" is a priority claim with no source on the page |
| "Mining detection built in", "7 checks", "6-tier reputation" | home | Unverified here; each should link to the doc that backs it |
| "Reputation-scored hosts, tiered SLAs" | `/gpu-availability` | SLA credits are not paid as designed (§2.6) |
| Canadian provincial tax table (ON 13% HST, QC, BC, Atlantic) | `llms.txt` lines ~314–318 | The only tax guidance an agent reads; nothing for non-Canadian buyers |

### 2.4 Operations a launch would lean on

* **CI does not run.** `gh run list`: today's runs of *Frontend CI*, *MCP connector conformance* and
  *Sync to Org Mirror* are all `skipped`. This is the GitHub Actions billing lock that has been in place
  since about 2026-07-21. A green push proves nothing until it is lifted or replaced.
* **`status.xcelsior.ca` is broken.** DNS points it at the production box (`46.225.20.97`), which has no
  certificate for that name: TLS fails with `unrecognized name`. Nothing in the repo links to it, so it is
  a dangling record. A status page on the box it reports on would also go down with it.
* **Backups do not leave the box, as written.** `api/scripts/backup-db.sh` writes compressed `pg_dump`s to
  `/var/backups/xcelsior` with 14-day rotation and a Telegram alert on failure. Nothing in the script
  copies them anywhere else. Whether another job pulls them off-box is **unverified**.
* **Single box.** API, DB, NFS volumes, Headscale proxy and the agent gateway share one Hetzner host.
  The LUKS volumes need a manual reopen after every reboot (`scripts/volumes_reopen_luks.py`), so an
  unattended reboot is an outage until someone intervenes.
* **Workload identity is still interim.** `/readyz` reports `spiffe_strict: false`. The Envoy + SPIRE
  cutover has open decisions and one README step that would take the fleet down (reusing
  `XCELSIOR_AGENT_GATEWAY_SECRET` is mandatory).
* **The signed-in production check cannot run** from the operator's machine until the new home IP is
  whitelisted (`XCELSIOR_DEMO_IP_WHITELIST_EXTRA`). The dashboard, billing and launch flows were therefore
  **not** re-examined live today.

### 2.6 SLA credits are not really paid

The site advertises "tiered SLAs". In code:

* **Nothing computes monthly SLA records on a schedule.** `sla_monthly` rows are written only by
  `POST /api/sla/enforce` (`api/routes/sla.py:30`), an admin call where a person supplies the host, tier,
  month and **the customer's spend by hand**.
* The hourly task `auto_issue_credits` (`api/api.py:247`, `core/sla.py:448`) pays only those rows. Its own
  docstring says that until a recent fix the query used a dropped column, so "no SLA credit has ever been
  issued by this path."
* **When it does pay, it pays the wrong person.** The recipient is
  `SELECT DISTINCT owner_id FROM jobs WHERE host_id = … AND status = 'completed' ORDER BY owner_id LIMIT 1`:
  the alphabetically first customer who *ever* completed a job on that host, in any month. It isn't the
  customers whose jobs the outage hit, and it ignores everyone else.
* **The tier doesn't change the credit.** `CREDIT_TIERS` starts at < 99.0%, so a Dedicated (99.9%) or
  Secure (99.5%) host at 99.2% uptime breaches its target and owes nothing.
* Whether any `sla_monthly` row exists in production is **unverified** (no prod DB access).

### 2.7 Front-end weight and polish

* The home page loads **18 JS chunks, about 341 KB compressed**. The CSP allows WalletConnect and web3modal
  origins. Check whether wallet code ships on marketing routes. The roadmap's mobile TBT targets
  (§5: < 8 s on slow 4G, stretch < 5 s) are still unchecked.
* On mobile the cookie banner sits over the hero's price card, and the home comparison table clips its
  cell text ("86 tools, nati…").
* `script-src` includes `'unsafe-inline'`. A nonce-based CSP is the normal next step.

---

## 3. UI improvements, in priority order

1. **Make the availability page true.** Show in-stock models first, always. Label catalogue-only rows
   "Not on the fleet yet — request" in a separate group, not inside "Live fleet". Fix the "0 regions"
   counter. Drop "on the fleet, ready when you are" until it is.
2. **Make the hero card real or say it is a sample.** Feed it from the cheapest in-stock GPU, or caption it.
3. **One price story.** The "from" price on every page should come from one computed value: the cheapest
   in-stock spot rate from `/api/v2/gpu/available` joined with `/api/pricing/reference`. Never hard-code it
   in copy (see [`align-duplicated-values`]).
4. **Global buyer basics.** A country field at signup or first deposit; a country selector in settings
   (replacing the province-only picker, with province/state as a dependent field); show a USD estimate
   next to CAD on pricing and in the launch modal; drop "AWS Canada" from the comparison table in favour
   of Vast.ai, RunPod and Lambda, priced in USD.
5. **Honest credits CTA.** "Deposit $5, get $5" instead of "$5 Free Credits".
6. **Provider-side marketing.** With three GPUs, supply is the constraint. A provider landing page with an
   earnings estimate per GPU model, payout terms, countries supported and the one-command install
   (see §6) matters more for launch than another buyer page.
7. **Mobile polish.** Cookie banner should not cover the hero CTA; comparison table should scroll or stack.
8. **Signed-in pass.** Re-run `npx playwright test -c playwright.prod.config.ts` once the IP is
   whitelisted, and look at every screenshot (dashboard, billing, launch, instances, earnings, volumes,
   inference), not just the assertion results.

## 4. Backend processes

| Area | State | Next |
| --- | --- | --- |
| Signup country / tax | CA/ON stamped on every account | §7 step 1 |
| Stripe Tax registrations | Unknown which are live | Operator reads them back (Dashboard → Tax → Registrations) and decides which non-CA jurisdictions to register |
| Currency | Ledger, deposits and quotes are CAD (12 hard-coded `currency="cad"` sites in routes/billing) | Decide: keep CAD and show a USD estimate, or present USD at checkout (Stripe presentment currency). Ledger stays integer micros either way |
| Provider payouts | Stripe Connect; provider `country` is a request field defaulting to `CA` (`api/routes/providers.py:60,166`; `XCELSIOR_CONNECT_COUNTRY`, `api/routes/stripe_connect_v2.py:85`). PayPal payouts code exists, prod env unset (roadmap §2) | Verify which countries a Canadian Stripe platform can pay out to; that list *is* the provider-country list until PayPal is live |
| Serverless | Built and enabled | Decide what it can honestly host on 6–12 GB cards. Open item: `gpu_model="any"` resolves to the cheapest SKU (RTX 2060) |
| Scheduler / admission / spot | Live; spot is published-rate with capacity preemption | Nothing blocking found from outside |
| Agent gateway / identity | Live, `SPIFFE_STRICT=0` | SPIRE cutover after its open decisions |
| Volumes (NFS) | `mode=full`, reachable | Off-box copy of volume data and of DB dumps |
| CI | Not running | Lift the billing lock or move to a self-hosted runner on this box |
| Status page | Dangling DNS | External status page, or remove the record |
| Backups | On-box only (as written) | Off-box copy + a timed restore drill |
| SLA credits | Manual admin enforcement; credit goes to one arbitrary customer; bands ignore tier | §7 step 6c |

## 5. Competitive benchmark

*Filled from the deep-research workflow. Claims here passed a 3-vote adversarial check unless marked.*

Vote counts are refute-checks out of three; a claim needed two refutes to be dropped.

**Vast.ai**
* **Rental types** ([rental-types](https://docs.vast.ai/documentation/reference/faq/rental-types), 3-0):
  on-demand (fixed host price, cannot be interrupted), interruptible (the renter bids, a higher bid can stop
  the instance), and reserved (pre-paid, up to 50% off by commitment length, convertible from a running
  on-demand instance). An interrupted instance is stopped, not destroyed, and resumes when it regains
  priority. Vast says interruptible saves 50–80% against on-demand (2-1), but that is its marketing range:
  an independent price aggregator puts the typical discount at about 40–70%. The reserved "up to 50%" is
  a ceiling; Vast's CLI default curve gives about 20% for one month and 30% for three, the prepaid credit
  is tied to one host, and cancelling early forfeits the discount plus a 5% fee.
* **Host payouts** ([host/payment](https://docs.vast.ai/documentation/host/payment), 3-0): weekly. Invoices
  are raised every Friday for the previous week and paid the following Friday, so 1–2 weeks of lag. $20 USD
  minimum, smaller balances roll forward. Paid only through Wise, PayPal and Stripe: no direct bank, ACH, wire
  or SWIFT, and no published country list.
* **Trust tiers** ([compliance](https://vast.ai/compliance), 3-0): in practice three levels. An unverified
  base; *Verified Hosts* for general AI/HPC work; and *Secure Cloud* for regulated work, limited to vetted
  datacenter partners with at least 5 flagship GPU servers in ISO 27001, Tier 3/4 facilities, signed DPAs,
  and audits of ownership, identity and source of funds. Vast's host self-test (driver/CUDA, network,
  ports, PCIe bandwidth, VRAM, RAM, CPU, a short simulated load) is a prerequisite for verification, not
  a guarantee of it: verification itself is "entirely automated by proprietary algorithms."

**RunPod**
* **Stopped taking new Community Cloud hosts in February 2026** and took down the host program's docs
  (3-0): `docs.runpod.io/hosting` is 404 and the old subpages redirect to a security page with no host,
  payout or country terms.
* Tenant isolation is documented as one container per Pod/worker in a multi-tenant environment, with no
  VM or confidential-computing isolation from the host operator (2-1).

**Salad** — the strongest model for untrusted consumer hosts
* **Isolation** ([security](https://salad.com/security), 3-0): each workload runs in an OCI container
  inside a Linux VM on a hypervisor-partitioned node, kept apart from the host's Windows environment. The
  stated premise is that the host machine cannot be trusted.
* **Intrusion response** (3-0): constant host-intrusion detection (folder access, shell opening). A host
  that touches the Linux environment has it destroyed and the machine blacklisted. Falco runtime
  detection is described as in progress.
* **Placement by trust** (3-0): a proprietary behavioural trust rating per node; only nodes holding a
  solid rating get work, weighed with the customer's hardware and location requirements.
* **Payouts** (3-0): Salad is not a payout marketplace. Contributors earn reward points with no cash
  value and redeem them for storefront rewards. PayPal is one of them, fulfilled by a third-party vendor in
  about 115 listed countries; gift cards, prepaid Mastercards and RENDER are others. That structure
  sidesteps the identity, tax-reporting and cross-border obligations a real payout rail carries.

**Akash** — graduated provider verification ([AEP-86, draft](https://akash.network/roadmap/aep-86/), 3-0)
* Proposes replacing its binary audited-attributes model with five auditor-attested tiers, Level 0
  (permissionless) to Level 4 (Trusted), because human accreditation does not scale.
* Level 2 (Verified): automated resource-delivery and network checks, a minimum uptime (e.g. 30 days) and
  a slashable bond sized to capacity.
* Level 3 (Established): a rolling uptime threshold (e.g. 90 days at 99%+), a lease completion rate
  (e.g. 98%), and automated cross-tenant isolation tests (network namespace, filesystem, process).
* AEP-86 is a draft (created 2026-04-04), off by default, not live.
* **Hardware attestation** ([AEP-29](https://akash.network/roadmap/aep-29/), 2-1): planned TEE remote
  attestation (IETF RATS) of CPU, GPU, firmware and driver measurements rooted in Intel TDX, AMD SEV-SNP or
  NVIDIA NVTrust. Not in production. It needs datacenter confidential-computing GPUs (Hopper, and per
  Akash's later docs Blackwell); whether consumer RTX cards can be attested at all is unanswered.

**Checked and refuted, so not to be repeated:** that Vast's general tier offers only an unprivileged
Docker container (0-3); that RunPod protects data from hosts by contract alone (0-3); that io.net pays
providers only in its IO token (1-2) and bills in USDC (0-3); that GPU attestation is limited to
H100/H200 (refuted 2-1 as out of date).

Most of these sources are vendors describing their own controls: they show what each company says, not
that it works.

### 5b. Gaps filled by a second, targeted pass (2026-11-03)

The workflow left Lambda, TensorDock, Hyperbolic, io.net, SLAs, MCP, currency/tax and launch tactics
uncovered. Each claim below was read on the vendor's own page (or, for history, a dated news report).
Where a secondary source disagreed with the primary page, the primary page wins and the disagreement is noted.

**Lambda** ([pricing](https://lambda.ai/pricing))
* Self-serve instances from **$0.69/GPU-hr** (Quadro RTX 6000, 24 GB). H100 SXM **$3.99–$4.29/GPU-hr**
  depending on configuration. 1-Click Clusters have a 16-GPU minimum.
* Prices carry a bare "$" with no currency code, "plus applicable sales tax/VAT/GST."
* Reserved capacity is contact-sales only ("our lowest prices").
* The pricing page states **no billing granularity and no SLA**. Third-party sites disagree on both
  (per-minute vs rounded to the hour; 99.9% vs 99.95%), and none cites a Lambda document.

**TensorDock** ([host](https://www.tensordock.com/host.html))
* Hosts set their own prices and pay a **25% revenue share**. Payouts through Stripe, withdrawable any
  time; TensorDock covers fees on withdrawals above $10K. (Third-party sites quote 20% and an 80% host
  share; the host page says 25%.)
* **Datacenter-grade hosts only**: 1 Gbps per location, static IP, a facility with power and network
  redundancy, 16 GB/s PCIe per GPU, 16 GB RAM and 8 vCPUs per GPU, 1 TB NVMe per GPU, 24/7 operation, and
  an approval process. Excessive downtime is penalised; maintenance booked two weeks ahead is not.

**Hyperbolic** ([on-demand docs](https://hyperbolic.ai/docs/on-demand/overview.md))
* On-demand GPUs bill **hourly**, funded by card from **$5**, with "no charges for failed instances."
  Reserved is a fixed term paid up front.
* Supplier terms are **not in the public docs**: the old marketplace-pricing page now redirects to a
  landing page with no supplier, payout or fee information. A third-party tracker's 10% supplier fee could
  not be confirmed.

**io.net** ([payments](https://io.net/docs/guides/payment/io-cloud-payments),
[worker rewards](https://io.net/docs/guides/workers/rewards-wallets))
* Compute is paid only in IO Credits. Top-ups by **card carry a 5% fee** (via Stripe); top-ups in **USDC
  or the IO token are free**. (A search summary said USDC carried 2%; the payments page says no fee.)
* "All worker earnings are paid in IO Coin." The rewards page shows a USDC claim option but does not say
  how conversion works or what it costs.

**SLAs — nobody in this set publishes a self-serve SLA with credits**
* RunPod's [terms](https://www.runpod.io/terms-of-service): "does not make any specific uptime warranties"
  for Community Cloud, and state no SLA or service credits for Secure Cloud either. A competitor's blog
  claims a 99.5% Secure Cloud commitment; RunPod's terms don't contain it.
* Lambda's pricing page states none (above). Enterprise SLAs are reportedly negotiated.
* **Xcelsior has one in code, but it doesn't work end to end** (§2.6): `core/sla.py` defines Community 99.0%,
  Secure 99.5% and Dedicated 99.9% availability with support response targets of 4, 2 and 1 hours, and a
  credit table (95–99% uptime → 10%, 90–95% → 25%, < 90% → 100%). No public SLA page promises it.

**MCP / agent-native APIs**
* **RunPod ships an official MCP server** ([docs](https://docs.runpod.io/get-started/mcp-servers)):
  hosted at `mcp.runpod.io` with OAuth sign-in or an API key, or locally as `@runpod/mcp-server`. It manages
  Pods, Serverless endpoints, templates, network volumes and container registries, and is documented for
  Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Codex CLI, Gemini CLI, Cline and ChatGPT. A separate
  unauthenticated docs MCP server sits at `docs.runpod.io/mcp`.
* **Vast.ai has no official MCP server** that could be found: only a Vast tutorial on hosting your own,
  and third-party servers of unclear origin.
* So the home page's "Agent-Native (MCP) … first, Jun 2026" and the comparison row calling RunPod's MCP
  "Official" while implying Xcelsior was first are risky. RunPod's server covers resource management.
  What it does not advertise is Xcelsior's *binding quote before spend* and *ask before spending*.

**Currency and tax**
* Lambda, RunPod and Hyperbolic all show a bare "$" with no currency code on their pricing pages. Xcelsior's
  explicit "All prices in Canadian Dollars" badge is the outlier. A buyer reading "$0.30" next to
  competitors will assume USD, which here *understates* how cheap Xcelsior is (CAD is worth less).
* RunPod's [sales-tax model](https://docs.runpod.io/accounts-billing/sales-tax) is the one to copy: the
  rate comes from **billing address and payment method**, tax is **added on top** of the credit purchase
  ($100 + 18% = $118 charged, $100 credited), shown as its own receipt line, applied to every auto-reload,
  and a **valid business tax ID can exempt** B2B buyers. Lambda's page adds "sales tax/VAT/GST" on top too.

**Referral, credits and launch tactics**
* **RunPod referrals** ([docs](https://docs.runpod.io/accounts-billing/referrals)): 3% of a referee's Pod
  spend and 5% of Serverless spend for six months, as credits. Both sides get a bonus once the referee loads
  $10: a random weighted $5–$500 outside Europe, a fixed $5 in Europe. After 25 paying referrals, an
  affiliate tier pays **10% cash** for six months through PartnerStack.
* **Vast referrals** ([docs](https://docs.vast.ai/guides/reference/referral-program)): **3% of everything a
  new client spends, for the lifetime of their account.** Up to 75% can be cashed out via Stripe Connect,
  PayPal or Wise from a dedicated referral account.
* **How RunPod started** ([TechCrunch, 2026-01-16](https://techcrunch.com/2026/01/16/ai-cloud-startup-Runpod-hits-120m-in-arr-and-it-started-with-a-reddit-post)):
  two first-time founders converted about $50K of crypto-mining rigs in their basements into AI servers and,
  in early 2022, posted in AI subreddits offering **free compute in exchange for feedback**. Beta testers
  became paying customers; they hit $1M revenue within nine months and bootstrapped past $24M before a
  $20M seed in May 2024. Today: $120M ARR, 500,000 developers, 31 regions. In Lu's words, "let's just post
  on Reddit."
* **RunPod closed its community host program** ([post, 2026-02-24](https://www.plushcap.com/content/runpod/blog/runpod-an-update-on-the-community-cloud-hosting-application-process)):
  "stop accepting new hosts for its Community Cloud," citing demand for more robust infrastructure.
  Existing hosts keep running.

**DigitalOcean (Paperspace)** ([GPU Droplets pricing](https://www.digitalocean.com/pricing/gpu-droplets))
* DigitalOcean bought Paperspace (YC W15) for $111M in cash in 2023
  ([HN, 193 pts](https://news.ycombinator.com/item?id=36614924)); GPU compute is now sold as GPU Droplets.
* **Billed per second with a 5-minute minimum.** From $0.76/GPU-hr (RTX 4000 Ada); HGX H100 $4.41/GPU-hr
  on-demand, $3.26 on a 12-month reserved contract (through sales). Bare "$", no SLA on the page.

**Serverless pricing**
* **RunPod** ([serverless pricing](https://docs.runpod.io/serverless/pricing), [pricing](https://www.runpod.io/pricing)):
  billed from worker start to full stop, **rounded up to the second**. Cold start (loading the model into
  GPU memory) is billed; pulling the image and cached models while scaled down is not. Idle timeout
  defaults to 5 s, and a scaled-down worker costs nothing. *Flex* workers scale to zero; *active* workers
  run 24/7 with a sales-negotiated discount. Listed rates: 24 GB class (L4/A5000/3090) **$0.75/hr**,
  A100 80 GB $2.72/hr, H100 $5.39/hr, against $3.99/hr for an H100 SXM Pod. Structured data on the
  page names USD.
* **Vast.ai** ([serverless pricing](https://docs.vast.ai/guides/serverless/pricing)): serverless costs
  **the same as a normal instance, no surcharge**, per second. Workers that are *Ready* or *Loading* are
  billed for GPU; *Creating* and *Inactive* workers pay only storage and bandwidth.
* **DigitalOcean** ([Gradient pricing](https://docs.digitalocean.com/products/gradient-ai-platform/details/pricing/)):
  token-billed, prepaid, USD per 1M tokens: gpt-oss-120b $0.10 in / $0.70 out; Ministral 3 14B $0.20 /
  $0.20; Gemma 4 $0.18 / $0.50; Llama 4 Maverick $0.25 / $0.87; DeepSeek V3.2 $0.50 / $1.60.
* **Hyperbolic**: `hyperbolic.ai/pricing` returns 404 (checked 2026-11-03). A price tracker reports it
  dropped its serverless-inference page in August 2026 (secondary source).
* *For Xcelsior:* the market has two shapes. RunPod and Vast sell per-second GPU time with scale to zero.
  DigitalOcean sells tokens. Xcelsior advertises "token-billed endpoints", so its comparison is
  DigitalOcean's table: tens of cents per million tokens on big models. A 6–12 GB fleet can't serve
  those models, so a token price for small models needs to be a clear step cheaper to be worth choosing.

**Startup credits — RunPod** ([startup program](https://www.runpod.io/startup-program))
* *Starter*: $1,000 in credits for pre-Series A startups with a working product and near-term GPU needs.
* *Growth*: commit $50K up front on a 12-month agreement and get $75K in credits, aimed at Seed–Series B
  startups spending $5K–$25K+/month. Venture backing "strongly preferred". No credit reload; expiry not stated.

**Launch venues — what the record shows**
* Hacker News (via HN's search API, `hn.algolia.com`):

  | Company | Biggest launch-type post | Points | Date |
  | --- | --- | --- | --- |
  | Paperspace | "A full computer you can access from any web browser" (YC launch) | 358 | 2015-03-03 |
  | Vast.ai | "marketplace for renting out your GPU, or renting someone else's" (not a Show HN) | 179 | 2022-03-19 |
  | TensorDock | Show HN: "GPU servers from $0.29/hr" | 147 | 2022-07-28 |
  | Salad | Show HN: "a distributed cloud for AI (like Airbnb for GPUs)" | 15 | 2024-05-03 |
  | Hyperbolic | "Rent H100 SXMs for 0.99" | 10 | 2024-11-20 |
  | RunPod, Lambda, io.net | no launch post of note | — | — |

* Product Hunt has listings for RunPod and TensorDock (TensorDock's "Cloud GPUs from $0.29/hour", about four
  years old). None were found for Vast.ai, Salad or Hyperbolic.
* *For Xcelsior:* Hacker News is where GPU marketplaces have actually launched. The two that landed (Vast,
  TensorDock) led with **the marketplace idea or one concrete price**, not a feature list. RunPod, the
  biggest of them, skipped launch sites entirely and grew from Reddit. Product Hunt has been a minor venue
  for this category.

**What it means for Xcelsior**
* **RunPod closing its community host program is the clearest opening found.** Consumer-GPU owners who
  were turned away in February 2026 are exactly the supply Xcelsior lacks (§2.2). Aim host recruitment
  at them.
* **Published spot rates with no bidding** is a real difference from Vast's interruptible bids. Keep it,
  make sure funded queue bids don't blur the claim, and benchmark the spot discount against Vast's
  realistic 40–70%, citing measured Xcelsior prices rather than a headline percentage. Publish the
  preemption rules side by side with Vast's: what triggers it, stop vs destroy, resume, notice period.
* **Payout terms are the provider's first question.** Vast's bar is weekly, a $20 floor, and three
  third-party rails. Xcelsior should publish its schedule, minimum and supported countries. PayPal
  provider payouts (code exists, prod env unset) would widen country coverage the way it does for Vast
  and Salad.
* **Isolation is where Xcelsior can lead or lag.** Community hosts at RunPod are container-isolated. Salad
  goes further with a VM per workload, intrusion detection that destroys and blacklists, and trust-rated
  placement. Xcelsior already has reputation tiers, host verification and mining detection. Publish
  exactly what they check, and decide whether untrusted hosts get a VM boundary.
* **Tiered verification is converging.** Vast (Secure Cloud vs the rest) and Akash (Levels 0–4, uptime
  and isolation tests) both grade providers. Xcelsior's 6-tier reputation maps onto this; showing the
  tier on every listing would make it a buyer-facing feature.
* **Consumer GPU owners have nowhere to go.** RunPod closed community hosting (February 2026) and
  TensorDock only takes datacenter-grade hosts. Xcelsior accepts a home 2060. That is the supply pitch,
  and RunPod's own origin (basement mining rigs, free compute for feedback on Reddit) is the playbook.
* **Host economics are competitive.** The live site already promises an **"85% revenue share to
  providers"**, which matches the 15% default (`XCELSIOR_PLATFORM_CUT`, `billing/provider_settlement.py:82`).
  TensorDock takes 25%. The production env value itself is unverified; if it differs, the public copy is wrong.
* **A published SLA would be unique in this set**, since neither RunPod nor Lambda publishes self-serve
  uptime credits. But Xcelsior's credit path is not real yet (§2.6). Fix it before promising it.
* **Referral design:** Vast's 3% for life is the simplest to explain; RunPod's random $5–$500 signup bonus
  is the most viral. Xcelsior's "$5 match on a $5 deposit" is closer to RunPod's $10-load trigger.
* **Payment fees shape behaviour.** io.net charges 5% on card top-ups and nothing on crypto. Xcelsior's
  BTC deposit is "Coming soon"; if it ships, a fee difference is a lever.

## 6. Future planning ideas

* **Supply first.** A marketplace with three GPUs (all the operator's own) is a private cloud. Before a
  public buyer launch: recruit 10–20 outside hosts (homelab, small mining-to-AI operators, university
  labs), with a referral or bonus for the first cohort, paid from platform fee, not cash.
* **One-command host install** surfaced on the provider page: the wizard and `worker_agent` exist; make
  "install → verified → earning" a single visible path with a progress page.
* **USD presentment** for buyers, leaving the ledger in CAD micros.
* **External status page** (Better Stack, Instatus or similar) fed by the existing `/api/status`.
* **Second region / warm standby** for the DB (streaming replica off-box), so one Hetzner incident is
  not a full outage.
* **Agent-native as the headline, minus "first".** The MCP server (86 tools) and the in-app assistant are
  the clearest differentiator, but RunPod ships an official hosted MCP server with OAuth (§5b). Lead with
  what RunPod doesn't advertise: binding quotes before spend, and every money-spending action asking first.
  Match RunPod on setup: a hosted endpoint with OAuth sign-in plus a one-command installer per client.
* **A public SLA page** built from `core/sla.py`, once credits are confirmed to land in wallets.
* **Launch sequencing, RunPod-style.** RunPod's start (free compute for feedback in AI subreddits,
  basement hardware, $1M in nine months) is the closest analogue to Xcelsior today. Sequence: (1) recruit
  hosts where consumer GPU owners are, with RunPod's closed community program as the hook; (2) offer free
  compute for feedback to a small buyer cohort; (3) only then a Show HN, once several GPU models are
  visibly in stock. Vast (179 points) and TensorDock (147) landed on HN by leading with the marketplace
  idea or one concrete price ("GPU servers from $0.29/hr"), so lead with the real cheapest in-stock rate.
  Treat Product Hunt as optional: no competitor in this set grew from it.

---

## 7. Handoff — implementation steps

Each step names what, where, how to verify, and what it is blocked on. Do them in this order; 1–4 are
the launch prefix.

### Step 1 — Stop taxing every customer as Ontario
* **What:** don't stamp `CA/ON` at signup. Store the country from Cloudflare's `cf-ipcountry` header as a
  provisional value flagged unconfirmed, and confirm it from the Stripe billing address at the first
  deposit. Resolve tax from a confirmed billing address, else IP, else refuse to estimate rather than
  defaulting to Ontario. Follow RunPod's model (§5b): tax added on top of the credit purchase as its own
  receipt line, the same on every auto-reload, and a business tax ID field that exempts B2B buyers.
* **Where:** `api/routes/auth.py:1938`, `:2404`; `billing/stripe_tax.py:50-110`
  (`normalize_customer_address`, `resolve_tax_customer_details`); `api/routes/billing.py:162`
  (`_account_tax_address`), `:1452` (`_estimate_province`); a country selector replacing
  `src/components/ui/province-selector.tsx` in settings.
* **Verify:** new tests: a registration with `cf-ipcountry: US` stores `US` and no province; a deposit
  tax estimate for a US account is not HST; an account with no confirmed address uses IP. Run the full
  suite locally with `run-tests.sh` (CI won't).
* **Blocked on:** the user's decision on which tax jurisdictions to register in Stripe Tax. The code
  change is safe without it.

### Step 2 — Make the availability page and hero true
* **What:** in-stock models first in `curateMarketingGpus`; catalogue-only SKUs in their own labelled
  group; fix "0 regions"; feed or caption the home "Live utilization" card; remove "on the fleet, ready
  when you are" while no such GPU is online.
* **Where:** `api/frontend/src/lib/marketing-gpu.ts:93-110`;
  `api/frontend/src/app/(marketing)/gpu-availability/content.tsx`; the home hero component; i18n strings
  in `src/lib/i18n/en-public.ts` (and FR).
* **Verify:** tighten `e2e-prod/public-surface.spec.ts` "shows real inventory": every model with
  `count_available > 0` in `/api/v2/gpu/available` must appear in the live-fleet group. Run it against
  prod after deploy and view the screenshot.
* **Blocked on:** nothing.

### Step 3 — One price and one billing-granularity story
* **What:** compute the "from" price once from live data; pick the true billing granularity and use it in
  every string; rename "$5 Free Credits"; replace the "AWS Canada" column; add a USD estimate on pricing;
  drop "first, Jun 2026" from the MCP row (RunPod has an official MCP server, §5b).
* **Where:** home and pricing components, `en-public.ts`/FR strings, `llms.txt`, the wizard copy.
* **Verify:** a vitest that fails if a hard-coded `$0.30`/`$0.11` "from" price appears in marketing
  copy; grep for "by the minute|by the hour|by the second" returns one phrasing.
* **Blocked on:** the user confirming the billing granularity and the comparison set.

### Step 4 — Operator-safety basics
* **What:** (a) restore CI or add a self-hosted runner; (b) remove or serve `status.xcelsior.ca`, ideally
  from an external status service; (c) copy DB dumps off-box nightly and run one timed restore;
  (d) alert when the LUKS volumes are closed after a reboot.
* **Where:** `.github/workflows/`; Cloudflare DNS (zone in memory `cloudflare-dns`);
  `api/scripts/backup-db.sh`; `api/runbooks/prod-hetzner.md`.
* **Verify:** a workflow run with `conclusion: success`; `curl https://status.xcelsior.ca` returns 200 or
  NXDOMAIN; a restore drill log with duration; an alert fires on a staged closed volume.
* **Blocked on:** the user, for the GitHub billing, DNS and off-box storage credentials. These are
  production changes an agent should stage and hand over, not run.

### Step 5 — Re-examine the signed-in product
* **What:** whitelist the new home IP for the demo gate, then run the full prod Playwright pass and fix
  what the screenshots show.
* **Where:** prod `.env` `XCELSIOR_DEMO_IP_WHITELIST_EXTRA=<home-ip>/32`;
  `api/frontend/e2e-prod/`.
* **Verify:** 91/91 tests pass and every signed-in screenshot has been looked at.
* **Blocked on:** the user making the prod `.env` change.

### Step 6 — Provider supply path
* **What:** a provider landing page (earnings by GPU model from the reference table, payout schedule,
  minimum, supported countries), and a visible install → verified → earning progress page.
* **Where:** `src/app/(marketing)/` (new route), `src/app/(dashboard)/dashboard/hosts`, `earnings`.
* **Verify:** an outside tester installs a host from the page alone and appears in `/api/v2/gpu/available`.
* **Also:** publish the payout terms (schedule, minimum, rails, countries per rail) read from the payout
  code and ledger, not written by hand, with a drift test the way `provider_expectations.py` guards its
  copy. Vast publishes cadence and minimum but no country list; publishing all four beats it.
* **Blocked on:** the list of payout countries the Stripe platform supports (operator reads it from
  Stripe), and the user's payout schedule and minimum.

### Step 6b — Write down the trust and isolation position
* **What:** inventory what the worker actually enforces on an untrusted host (container only, VM, or
  gVisor), whether anything detects host tampering and what happens when it does, whether the scheduler
  *excludes* low-reputation hosts or only ranks them lower, and whether a host can list without a burn-in
  benchmark. Map the 6-tier reputation onto Akash-style evidence Xcelsior already collects (heartbeat
  uptime, job completion rate, benchmark). Publish the result as the security page and show the tier as
  a badge and filter in the GPU picker.
* **Where:** `api/scripts/worker_agent.py` container launch config; `reputation.py`,
  `provider_expectations.py`; scheduler admission; the GPU picker in the launch modal.
* **Verify:** inspect a running container's config on the test stack; seed a low-reputation host and
  confirm the scheduler's behaviour matches the published text; badge equals backend tier for a seeded host.
* **Blocked on:** the user's call on whether untrusted hosts need a VM boundary, and whether to offer a
  separate audited datacenter-partner tier like Vast's Secure Cloud.

### Step 6c — Make SLA credits real, or stop advertising them
* **What:** compute `sla_monthly` on a schedule from `sla_downtime` instead of a hand-run admin call.
  Attribute each credit to **the customers whose jobs ran on the host during the downtime**, sized by their
  own spend on that host that month. Make the credit bands depend on the tier: a breach of the tier's own
  target owes something. Until all of that ships, change "tiered SLAs" in the copy to "uptime-scored hosts".
* **Where:** `core/sla.py` (`enforce_monthly`, `CREDIT_TIERS`, `compute_credit_pct`, `auto_issue_credits`);
  the scheduled tasks in `api/api.py:247`; `api/routes/sla.py`; availability-page copy.
* **Verify:** a test with two customers on one host, a seeded outage overlapping one customer's job: only
  that customer is credited, by their own spend, once (idempotent on `sla:{host}:{month}:{customer}`);
  a Dedicated host at 99.2% owes a credit. Then a read-only prod query for `sla_monthly` rows.
* **Blocked on:** the user's credit table per tier (keep Google/Azure bands for Community, tighter for
  Secure and Dedicated?).

### Step 7 — Identity and resilience (after launch prefix)
* SPIRE + Envoy cutover to `SPIFFE_STRICT=1`, following the pre-flight's open decisions.
* Off-box streaming replica of Postgres.
* Nonce-based CSP; check whether wallet code loads on marketing routes; measure mobile TBT.
* Update `api/docs/NEXT_PRIORITIES_ROADMAP.md` so §6 reflects that serverless is built and enabled.

## 8. Questions for the user

1. Which tax jurisdictions should Stripe Tax be registered in, beyond Canada?
2. Keep charging in CAD with a USD estimate shown, or present USD at checkout?
3. What is the true billing granularity: per second, per minute, or per 5-minute cycle?
4. Is a provider-supply push the next priority, ahead of buyer-facing launch work?
5. OK to whitelist the new home IP for the demo gate so the signed-in prod check can run?
6. ~~Is the platform cut 15%, and are SLA credits paid?~~ Answered (§2.6, §5): the site promises 85% to
   providers, matching the 15% default; SLA credits are not really paid. Left open: confirm
   `XCELSIOR_PLATFORM_CUT` in production's `.env` matches the public "85%", and choose the credit
   bands per SLA tier (Step 6c).
