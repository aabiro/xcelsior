<!-- residency-guard: documents-removal — tracks the removal item by item -->
# Xcelsior — Business Strategy Checklist

> Actionable checklist mirroring `docs/BUSINESS_STRATEGY.md` (updated direction:
> compete on **price, reliability, and developer experience** — Canadian is a
> quiet bonus, not the pitch). Check items off as they ship. Exit criteria are
> the gate to call a phase done.

**Thesis:** Xcelsior is a GPU marketplace that wins on price, reliability, and
ease of use. We make money through healthy margins on serverless inference and a
fair take rate on dedicated GPUs. Give developers competitive prices, reliable
access to GPUs, and a smooth experience that just works. We are **not** competing
on sovereignty / data residency as the main story.

**North Star:** Weekly paid GPU-seconds + token revenue.

---

## Phase 0 — Instrumentation & unit-economics truth  ·  ~1 week

**Goal:** Get real numbers on what we actually make or lose before changing anything.

- [x] Build a simple unit-economics dashboard — `/api/admin/unit-economics` + `dashboard/admin/unit-economics`: marketplace gross/payout/margin(%), serverless revenue + GPU-hours, computed from the live DB.
- [~] Track key funnel events. *PostHog events done (`user_registered`, `serverless_endpoint_deployed`, `gpu_instance_launched`, `serverless_inference_run`, `first_run_card_*`); the dashboard also computes the funnel from the DB (signups → deployed → ran → paid). Still TODO: `wallet_credit_granted` + `first_paid` PostHog events.*
- [x] Monitor weekly **paid GPU-seconds**, **token revenue**, **supply liquidity**. *Weekly revenue + GPU-hours; token revenue now persisted (`billing_cycles.token_cost_cad`) and shown; supply-liquidity (% of jobs that got a host) shown.*

**Exit criteria**
- [x] One dashboard clearly shows whether we're making money (margin) and where users drop off (funnel).

---

## Phase 1 — Pricing correctness & token-billing  ·  ~2–3 weeks

**Goal:** Make our pricing competitive on spot GPUs and profitable on serverless.

- [x] Adjust spot GPU pricing in `db.py` so popular cards are at or below Vast / RunPod. *(A100 + A6000 repriced toward market via migration 044; 4090/3090 spot were already below market — left as-is.)*
- [x] Replace flat token pricing with size-based tiers in `serverless/metering.py`: *(live — `acd9309`)*
  - [x] ≤ 9B → **$0.15 input / $0.45 output** per M tokens (CAD)
  - [x] 10–34B → **$0.35 / $1.05**
  - [x] 35–80B → **$0.70 / $2.10**
  - [x] 80B & MoE → **$1.10 / $3.30**
- [x] Implement **blended billing** — charge the higher of (GPU-seconds cost, token cost). *Mechanism + per-endpoint token-accrual (migration 045, accrue on completion / consume on charge) + `XCELSIOR_SERVERLESS_BLENDED_BILLING` flag (default OFF) + dual-write recording. Flag stays off until the dual-write validation below.*
- [x] Simplify to a flat **15% platform fee** (remove reputation-based discounts for now). *`_platform_cut_for_host` now returns the flat `PLATFORM_CUT` for everyone (no Gold/Platinum/Diamond discount); reputation page hidden from the dashboard nav.*
- [x] Remove the **sovereign tier multiplier** (1.43× → 1.0; sovereignty_premium → 0). *db.py + migration 044; tier string retained, no price premium.*

**Exit criteria**
- [x] New token pricing live for all serverless presets.
- [x] Spot GPUs are price-competitive.
- [ ] Billing changes tested with dual-write during a transition period *(blended flag OFF; watch the recorded gpu vs. token amounts, then flip `XCELSIOR_SERVERLESS_BLENDED_BILLING=1`).*
- [x] Clear margin-per-model-size visibility *(needs the Phase 0 unit-economics dashboard).* — added `serverless_by_model_size` breakdown (GPU-seconds vs. token revenue per size band) to `/api/admin/unit-economics` and the admin dashboard page.

---

## Phase 2 — Catalog expansion (demand magnets)  ·  ~2–4 weeks, parallel to Phase 1

**Goal:** Add the models people are actually looking for.

- [x] Add **DeepSeek R1** — `DeepSeek-R1-Distill-Qwen-32B` + `-Llama-8B`. *(The distills are what's actually serveable on marketplace hardware; full R1/V3 (671B MoE) needs 8–16 GPUs — add only once that supply exists.)*
- [x] Replace Llama 3.1 70B with **Llama 3.3 70B**.
- [x] Add **Qwen2.5-Coder** (`Qwen2.5-Coder-32B-Instruct`).
- [x] Ungate and add a proper **embeddings** preset (`BAAI/bge-m3`). *Proxy ungated by model task; vLLM launches `--task embed`.*
- [x] Add a small, fast **reranker** model (`BAAI/bge-reranker-v2-m3`). *`--task score`, `/v1/rerank` route.*

**Positioning:** market as "fast, simple, OpenAI-compatible inference with good prices" — not as a Canadian sovereign option.

**Exit criteria**
- [x] DeepSeek and Llama 3.3 live.
- [x] Embeddings working *(preset + ungated proxy + vLLM embed mode; needs a deploy to exercise on real hardware).*
- [x] Clean copy-paste snippet for each model — the Try-It console now emits task-aware curl + Python (chat / embeddings / rerank).

---

## Phase 3 — Demand conversion (signup → active → paying)  ·  ~3–4 weeks

**Goal:** Make it dead simple for new signups to run their first job.

- [x] Build a clear **"Try it now"** card — `FirstRunCard` on the dashboard shows until the user has launched something: pre-filled Python snippet (Llama 3.3) + "Deploy a model" CTA. The Try-It console emits the working per-endpoint snippet after deploy.
- [x] Make the **free credit highly visible** — the card shows "Your first run is covered by your free credit ($X)" (wallet balance); CreditsButton remains in the topbar.
- [x] Lead with developer experience: **"OpenAI-compatible — just change your `base_url`."** (card headline + snippet comments).
- [ ] Add gentle **activation nudges** (follow-up emails / in-app messages) for users who sign up but don't run anything. *(Deferred — needs the funnel events + a scheduled job; infra exists: NotificationStore + _send_team_email.)*

**Exit criteria**
- [ ] Signup → first-inference rate is visible and improving *(needs the Phase 0 funnel dashboard).*
- [ ] ≥ **25%** of new signups run their first inference in week one *(metric — measure after Phase 0).*

---

## Phase 4 — Supply & liquidity  ·  ~4–8 weeks, ongoing

**Goal:** Make sure users can always get a GPU when they need one.

- [ ] Make the provider **wizard** genuinely one-command easy to start providing GPUs.
- [x] Offer **0% platform fee for the first 90 days** for new providers. *Backend (`PROVIDER_INTRO_FEE_DAYS`, env-configurable) + surfaced provider-side: earnings page "X days left" banner, hosts page acquisition banner.*
- [ ] Keep provider **payouts fast** (biggest provider-retention lever). *(operational)*
- [ ] Maintain a **small rented buffer** (1–2 good cards) purely for basic availability — treat as marketing spend, not a profit center. *(operational/infra)*

**Exit criteria**
- [ ] Most requests get a GPU quickly.
- [ ] **70–80%** of total GPU-hours come from community providers (not rented).

---

## Phase 5 — Polish & growth  ·  ~4–6 weeks

**Goal:** Raise the overall quality so users keep coming back.

- [ ] Significantly improve the **dashboard** experience (instantly see usage, costs, available GPUs).
- [ ] Add quality-of-life features: **usage alerts**, better **logging**, easier **SSH setup** for dedicated instances.
- [ ] Strengthen **docs & developer experience** (more working example code, more reliable OpenAI-compatible endpoint).
- [ ] Begin **light marketing**: directory listings, relevant community mentions, easy organic discovery.

**Exit criteria**
- [ ] Platform feels reliable and trustworthy; one-time users come back; steady organic signups.

---

## Phase 6 — Flywheel & scale  ·  ongoing

**Goal:** Create a self-reinforcing marketplace (more users → more supply → more users).

- [ ] Keep the model catalog **fresh** — regularly add the latest popular open-source models.
- [ ] **Reinvest margin** into aggressive provider acquisition and keeping liquidity high.
- [ ] **Double down** on whatever's working (serverless, spot GPUs, or specific model types).
- [ ] Stay obsessively focused on **price and simplicity**.

---

## Appendix — Page copy to apply (from the strategy doc)

Marketing headlines chosen in the strategy session — apply per page (drop the sovereignty framing):

- [ ] **Homepage** — "GPUs when you need them. No bullshit." / *Spot instances by the minute and fast serverless inference.*
- [ ] **Features** — "Built for developers who just want it to work." / *Powerful GPUs, simple serverless, and the tools you actually need.*
- [ ] **Pricing** — "Competitive prices. Zero surprises." / *Clear spot rates and smart token pricing that scales with your model.*
- [ ] **GPUs / Dedicated** — "Rent real GPUs. Full control." / *Start in seconds. Pay only for the time you use. No queues.*
- [ ] **MCP / Agents** — "Fast GPUs for your AI agents." / *Built for MCP, Claude, and heavy workloads. No more waiting.*
- [ ] **Serverless** — "Serverless inference that's actually fast." / *OpenAI-compatible, token-priced, and always ready.*
- [ ] **Download & Done (Provider)** — "Download. Install. Start earning." / *Turn your idle GPU into income in under a minute.*
- [ ] **Docs** — "Documentation that doesn't suck." / *Clear guides, working code samples, and zero fluff.*
- [ ] **Dashboard** — "Your compute, beautifully simple." / *Track usage, manage costs, and scale without the headache.*
- [ ] **Sovereignty page** — decide: retire it or demote to a quiet footnote (no longer the main story).

**Logo direction (noted in the doc):** bold, geometric **X** mark; clean futuristic sans-serif "Xcelsior"; dark navy / black with an electric cyan or violet accent; Canadian maple-leaf/flag **removed**. Keep old assets archived.
