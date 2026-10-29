# Xcelsior — Business Strategy & Execution Plan

> A grounded, phased plan to turn the platform's strategic thesis into shipped
> work. Every technical step references real code (`db.py`, `serverless/`,
> `reputation.py`, `billing.py`, `frontend/src/features/serverless/`). Revenue
> *targets* are templates to calibrate — flagged **[CALIBRATE]** — everything
> else is grounded in what's in the repo today.

---

## 0. Thesis (the one-paragraph version)

Xcelsior does **not** make money reselling raw GPU hours — that arbitrage is
break-even at best and actively undercuts our own providers and our sovereignty
story. The moat and the margin are in two places: **(1) the serverless / token
markup** we control end-to-end, and **(2) the sovereign-compliance premium**
enterprises will pay for Canadian data residency. Supply grows by **converting
the signups we already get into providers**, not by buying hardware. Everything
below serves that thesis.

**What we are explicitly NOT doing:** buying GPU inventory for resale, chasing
catalog breadth to match Fireworks/Together, or competing on raw $/hr against US
hyperscalers.

---

## 1. Where we are — verified baseline (from the code, not vibes)

| Lever | Reality in repo | File |
|---|---|---|
| GPU on-demand base rates (CAD/hr) | 3060 = 0.13, 3090 = 0.30, 3090 Ti = 0.35, 4090 = 0.55–0.62, A100 80GB SXM = 2.60 | `db.py` `_GPU_PRICING_BASE` |
| Tier multipliers | standard 1.0 · premium 1.30 · **sovereign 1.43** (premium ×1.10) | `db.py` `_TIER_MULT` |
| Sovereignty premium | +0.10 only on sovereign tier | `db.py` `_TIER_SOVEREIGNTY` |
| Spot discount | 0.40× on-demand | `db.py` `"spot": 0.40` |
| Token prices | input 0.50 / output 1.50 CAD/M — **observability-only, "never debited as primary meter"** | `serverless/metering.py` |
| Primary meter | **GPU-seconds**, not tokens | `serverless/metering.py` `estimate_cost_cad` |
| Platform take-rate | **15%**, reduced by reputation (Gold 12% · Platinum 10% · Diamond 8%) | `billing.py` (`platform_fee_pct=0.15`), `reputation.py` |
| Reputation tiers | Bronze · Silver · Gold · Platinum · Diamond (850+) + Sovereign (incorporation) | `reputation.py` |
| Serverless catalog | 5 presets: Llama 3.1 8B/70B, Mistral 7B v0.3, Qwen2.5 7B, Gemma 2 9B | `frontend/src/features/serverless/constants.ts` |
| Embeddings | **gated** in the OpenAI proxy | `serverless/openai_proxy.py` |
| Supply onboarding | one-command wizard (Ink CLI) + mesh | `wizard/` |
| Demand credit | signup wallet credit + `subsidized_starter_cad` (0.4×) | `billing.py` |

**The arbitrage math (why we don't resell):** a 4090 lists at 0.55 CAD/hr
(~$0.40 USD); renting one from RunPod/Vast is $0.20–0.35 USD/hr — after our 15%
take and idle time, margin ≈ 0. Bigger cards are worse: A100 80GB at 2.60 CAD
(~$1.90 USD) is below cloud on-demand rent. Resale also (a) undercuts our own
providers and (b) puts a GPU in a US datacenter, breaking the PIPEDA/CLOUD-Act
pitch that is the entire differentiator (`jurisdiction.py`).

**The single biggest correction vs. prior strategy notes:** token markup is
currently *metadata only*. To make "sell tokens at a markup" a real revenue
lever we must **promote token billing to a primary (or blended) meter** — that's
a Phase 1 engineering task, not a constant change.

---

## 2. The goal — what "pulling it off" looks like (12 months)

A liquid, self-reinforcing Canadian compute marketplace where:

1. **Serverless gross margin is positive and size-aware** — small models priced
   to win volume, large models priced to capture value. **[CALIBRATE]** target
   blended serverless gross margin ≥ 55%.
2. **Supply is mostly converted, not bought** — **[CALIBRATE]** ≥ 80% of active
   GPU-hours come from signed-up providers; ≤ 20% from a tiny rented baseline
   held only for SLA/demo guarantees.
3. **The funnel converts** — signup → first inference → first paid is
   instrumented and improving; **[CALIBRATE]** ≥ 25% of signups run an inference
   in week 1.
4. **The sovereign premium is a product, not a multiplier** — enterprise tier
   with SLA, audit logs, residency attestation, and a real attach rate.
5. **Liquidity is always-on** — "something is always available" is true 24/7,
   funded by the sovereign premium as CAC, never as a profit center.

**North-star metric:** *paid GPU-seconds + token revenue per week*, with supply
liquidity % and signup→active conversion as the two leading indicators.

---

## 3. Phased execution plan

Each phase: **Goal · Why now · Steps (file-grounded) · Effort · Exit criteria.**
Phases 0–2 are pure software (no capex, highest ROI) and run first.

### Phase 0 — Instrumentation & unit-economics truth *(prerequisite, ~1 week)*

**Goal:** know the real numbers before changing any price.
**Why now:** every later decision (token tier bands, premium sizing, CAC ceiling)
needs a baseline. PostHog is already wired (`instrumentation-client.ts`, 11
events).

**Steps**
1. Add a **unit-economics view**: per-job GPU-cost (provider payout) vs. price
   charged vs. platform fee — surface margin per tier and per GPU class from
   `billing.py` ledger + `serverless/metering.py`.
2. Instrument the **funnel** events end-to-end: `signup → wallet_credit_granted
   → first_instance_launched → first_inference → first_paid`. Reuse PostHog.
3. Dashboard: weekly **paid GPU-seconds**, **supply liquidity %** (hours
   available ÷ hours requested), **signup→active**, **sovereign attach**.

**Exit criteria:** one dashboard answers "are we making money on each tier, and
where do signups drop off?" Numbers replace `[CALIBRATE]` placeholders below.

---

### Phase 1 — Pricing correctness & token-billing productization *(~2–3 weeks)*

**Goal:** stop overcharging small models, stop leaking value on large ones, and
make the token markup a *real* meter.
**Why now:** flat 0.50/1.50 CAD/M is the same for an 8B and a 70B — we overcharge
small (US 8B serverless is ~$0.10–0.20/M) and undercharge 70B. And tokens aren't
even billed yet.

**Steps**
1. **Size-tiered token pricing.** Replace the two flat constants in
   `serverless/metering.py` with model-size bands, e.g.:
   - ≤ 9B → input 0.15 / output 0.45 CAD/M
   - 10–34B → input 0.35 / output 1.05 CAD/M
   - 35–80B → input 0.70 / output 2.10 CAD/M
   - MoE / >80B → input 1.10 / output 3.30 CAD/M
   Source the band from each preset's `vram`/param size in
   `constants.ts` (mirror the band table server-side).
2. **Promote token cost from observability to a billable meter.** Decide the
   billing model and implement in `serverless/metering.py`:
   - **Option A (recommended): blended** — bill `max(gpu_seconds_cost,
     token_cost)` or `gpu_seconds_cost + token_markup`, so we capture value on
     high-throughput cheap-GPU workloads without double-charging idle.
   - Option B: per-token only for serverless presets, GPU-seconds for dedicated.
   Keep GPU-seconds as the meter for dedicated instances.
3. **Validate tier multipliers per segment** (no change expected; document the
   intent): hobbyist → spot 3060/3090; pro → on-demand 4090; enterprise →
   sovereign + SLA + serverless. Make each marketing page lead with its own
   differentiator (Phase 5 ties this to UI).
4. **Spot** stays at 0.40× (matches the "40–60% below" claim) — no change.

**Effort:** medium (metering refactor + tests). **Risk:** billing changes are
sensitive — ship behind a flag, dual-write old/new cost for one billing cycle,
reconcile before cutover.

**Exit criteria:** token meter live for serverless; blended margin per model band
visible and ≥ target; no billing regressions in reconciliation.

---

### Phase 2 — Catalog expansion (demand magnets) *(~2–4 weeks, parallel to P1)*

**Goal:** add the models people actually search for. Highest value-per-effort,
pure software.
**Why now:** the 5-preset list is one generation stale and missing the single
most-requested open model.

**Steps** (add to `constants.ts` presets + ensure engine/vLLM support + ungate
where needed):
1. **DeepSeek R1 / V3** — most-requested open model. Top priority.
2. **Llama 3.3 70B** — replaces the stale 3.1 70B as the flagship.
3. **Qwen2.5-Coder** — coding workloads (pairs with the MCP/agent story).
4. **Embeddings preset** — *ungate* embeddings in `serverless/openai_proxy.py`
   and add a `bge`/`nomic`/`Qwen3-Embedding` preset. Embeddings are sticky and
   high-frequency.
5. **Reranker** — small, cheap, high-attach for RAG users.
6. (Stretch) vision/multimodal or one image-gen (SDXL/Flux) once the above land.

**Positioning:** lead with **OpenAI-compatible + per-second billing + Canadian
residency + CAD-native**, not catalog size. Don't try to out-breadth Fireworks.

**Exit criteria:** DeepSeek + Llama 3.3 + embeddings live and discoverable; each
new preset has a copy-paste OpenAI-compatible snippet.

---

### Phase 3 — Demand conversion (signup → active → paying) *(~3–4 weeks)*

**Goal:** convert the signups we already get; make the free path obvious.
**Why now:** we already grant signup credit + `subsidized_starter_cad` (0.4×) but
the path to first inference isn't obvious.

**Steps**
1. **First-run experience:** after signup, a one-screen "run your first
   inference for free" using the existing wallet credit — pre-filled
   OpenAI-compatible curl/SDK snippet against a live preset (DeepSeek/Llama 3.3).
2. **Make credit visible** in the dashboard (`CreditsButton` exists) and on the
   serverless launch flow; show "covered by your free credit."
3. **Drop-in migration copy:** "change one base_url, keep your OpenAI code" —
   landing section + docs; this is the real wedge.
4. **Activation nudges:** funnel-based emails/notifications (infra exists —
   `NotificationStore`, `_send_team_email`) for signups who haven't run a job.

**Exit criteria:** signup→first-inference conversion measured and improving
toward the **[CALIBRATE]** ≥ 25% week-1 target.

---

### Phase 4 — Supply liquidity without capex *(~4–10 weeks, ongoing)*

**Goal:** guarantee "something is always available" while growing provider supply
we don't pay for.
**Why now:** liquidity is the chicken-and-egg of any marketplace; we solve it
with incentives + a tiny rented floor, not inventory.

**Steps**
1. **Provider incentive program** (mostly billing/reputation config):
   - Intro **0% platform fee for 90 days** for new providers (vs. the 15% in
     `billing.py`).
   - **Referral credits** for both sides.
   - **Fast-track reputation** — accelerate Bronze→Gold for verified Canadian
     hosts (`reputation.py`); Gold/Platinum/Diamond already get better economics.
   - **Prompt CAD/BTC payouts** — reduce time-to-cash; it's the #1 provider
     retention lever.
2. **Target Canadian idle GPUs:** gamers, ex-miners, university labs with
   3090/4090s. The `wizard/` one-command installer is the friction-reducer —
   invest in making it a 60-second setup.
3. **Tiny rented baseline:** rent **1–2 Canadian-region H100/A100** purely to
   guarantee availability for demos + SLA. **Treat as CAC, funded by the
   sovereign premium — not a profit center.** **[CALIBRATE]** monthly budget.
4. **Liquidity SLO:** define and monitor "% of requests with an available
   matching host within N seconds"; the rented floor backstops it.

**Exit criteria:** liquidity % ≥ target during business hours; ≥ 80% of GPU-hours
from converted providers; provider 30-day retention measured.

---

### Phase 5 — Sovereign premium as a product *(~6–12 weeks)*

**Goal:** turn the +43% sovereign multiplier into a real enterprise tier people
sign contracts for.
**Why now:** the multiplier exists, but the *product* (SLA, attestation, audit)
is what enterprises actually buy.

**Steps**
1. **Compliance pack:** data-residency attestation, PIPEDA/CLOUD-Act one-pager
   (lean on `jurisdiction.py`), audit-log export, DPA template.
2. **SLA tier:** uptime/availability guarantees backstopped by the rented floor
   (Phase 4) + Diamond providers (`reputation.py`).
3. **Segment-led pages:** hobbyist page leads with cheap spot 3060/3090;
   enterprise page leads with sovereign + SLA + serverless. (Ties to the
   marketing site already in `frontend/src/app/(marketing)/`.)
4. **Sales-assist:** invoice/PO billing, team workspaces (already in settings),
   procurement-friendly onboarding.

**Exit criteria:** first sovereign-tier contract(s); measurable sovereign attach
rate; SLA dashboard live.

---

### Phase 6 — Flywheel & moat *(ongoing)*

**Goal:** compounding liquidity + retention.
**Steps:** keep adding demand models (Phase 2 cadence), recycle margin into
provider incentives and the liquidity floor, lean into Canadian-residency
network effects, and report the north-star weekly. The flywheel: more models →
more demand → more provider revenue → more supply → better liquidity → more
demand.

---

## 4. KPIs & cadence

- **North star (weekly):** paid GPU-seconds + token revenue.
- **Leading:** supply liquidity %, signup→first-inference %, provider 30-day
  retention, sovereign attach %.
- **Margin guardrail:** blended serverless gross margin per model band.
- **Review cadence:** weekly metrics; phase exit gates as above.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Billing regression from token-meter change | Flag + dual-write + one-cycle reconcile before cutover (Phase 1) |
| Rented baseline becomes a cost sink | Hard monthly cap; classified as CAC; sunset once converted supply covers liquidity SLO |
| Catalog churn (model deprecations) | Treat presets as data; keep an upgrade cadence, not a fixed list |
| Undercutting our own providers | Never resell; rented floor is SLA-only, priced at/above provider economics |
| Sovereign claim challenged | Keep `jurisdiction.py` analysis current; ship attestation + audit export (Phase 5) |

## 6. Sequencing (at a glance)

```
Wk:  0   1   2   3   4   5   6   8   10   12
P0  ▓▓
P1      ▓▓▓▓▓▓
P2      ▓▓▓▓▓▓▓▓
P3              ▓▓▓▓▓▓
P4                  ▓▓▓▓▓▓▓▓▓▓▓▓▓
P5                          ▓▓▓▓▓▓▓▓▓▓
P6                                  ▓▓▓→ ongoing
```

Software-only, no-capex phases (0–3) front-loaded; supply + enterprise (4–5)
follow once the product and funnel are tightened.

## 7. Open decisions (need your calls)

1. **Token billing model** — blended (recommended) vs. per-token-only for
   serverless? (Phase 1, Option A/B.)
2. **Monthly budget for the rented liquidity floor** — sets the SLA we can
   promise. (Phase 4.)
3. **Launch segment priority** — lead with hobbyist (volume) or enterprise
   (margin)? Determines whether Phase 3 or Phase 5 gets the extra resource.
4. **Pricing aggressiveness** on the new token bands — win share (low) vs.
   protect margin (high)?

---

*Grounding note:* the prior version of this doc claimed token markup was already
a revenue lever and described reputation as "Bronze→Sovereign." Corrected here:
token costs are observability-only until Phase 1 productizes them, the take-rate
is 15% (reputation-discounted), and reputation runs Bronze→Diamond + Sovereign.
