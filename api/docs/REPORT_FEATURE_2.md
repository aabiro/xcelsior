<!-- residency-guard: documents-removal — original feature spec that defined the removed model -->
# Product and Feature Research Report for Xcelsior

## Executive Synthesis

Xcelsior's path to adoption aligns directly with the marketing synthesis: evolve from a credible residency-focused MVP (Canada-only scheduling with rebate-ready billing) to stronger sovereignty tiers (jurisdiction-aware trust controls), while anchoring supply with vetted operators and exploiting the AI Compute Access Fund as a demand accelerator. Research on competitors like RunPod, Vast.ai, and Lambda Labs shows that successful GPU platforms win by reducing onboarding friction (e.g., self-serve host registration with automated verification), building trust through transparent monitoring (e.g., real-time GPU metrics and logs), and creating retention loops via analytics and incentives (e.g., earnings dashboards for providers). For Xcelsior's small-team constraints and consumer/hardware focus, prioritize simplicity: no Kubernetes-scale complexity, but targeted features like CLI-first onboarding with web confirmation, agent-based telemetry for basic health (utilization, errors), and subsidy-aligned invoicing that auto-generates rebate documentation.

The absolute best ideas—evidence-backed and executable—are phased to produce "sellable truths":

1. **Phase A (Residency + Subsidy Fit)**: Self-serve provider onboarding with Canada-only defaults, rebate-ready usage reports, and basic monitoring to prove reliability for startups/research.

2. **Phase B (Sovereignty Tier)**: Host verification workflows (e.g., incorporation proof), jurisdiction tagging in scheduling, and audit logs for transparency reports.

3. **Phase C (Regulated Tier)**: Tamper-evident logging and SLA enforcement for public/health sectors, integrated with provincial compliance matrices.

This creates feedback loops: providers earn via spot pricing and see utilization analytics; users get subsidy rebates and trust signals (e.g., "Canada-only" badges). Avoid bloat—focus on 5-10 high-impact features per phase, each with CLI/API primacy and dashboard visualization.

## Research Methodology and Competitor Benchmarks

To inform recommendations, I researched leading GPU clouds (RunPod, Vast.ai, Lambda Labs, CoreWeave) and analogous systems (Kubernetes for orchestration, Upwork for marketplaces) using web searches and page browsing. Key queries included "RunPod provider onboarding process", "Vast.ai host monitoring features", "Lambda Labs billing and compliance", and "GPU cloud data sovereignty". Findings emphasize self-serve UX, real-time visibility, and tiered trust—adapted here for Xcelsior's Python/FastAPI stack and Canada focus.

- **RunPod**: Self-serve pod creation via web/CLI; providers add servers with auto-verification (ping, GPU detection); monitoring via Grafana-integrated metrics (utilization, temp); billing in USD with credits; no explicit sovereignty but region selection.
- **Vast.ai**: Marketplace-style; providers list rigs via CLI/agent with benchmarks; users bid/spot; monitoring shows live console/logs; trust via ratings/reviews; CAD/USD pricing but no subsidy tools.
- **Lambda Labs**: Enterprise-oriented; onboarding via sales/API; deep telemetry (Prometheus exports); compliance focus (SOC2, HIPAA-eligible); billing with detailed invoices but no auto-rebate mapping.
- **CoreWeave**: Kubernetes-based; advanced SLAs; monitoring via custom dashboards; sovereignty via dedicated regions but not Canada-specific.
- **Analogies**: Kubernetes' kubectl for CLI UX; Upwork's verified badges for trust.

Recommendations prioritize features that fit Xcelsior's simplicity (e.g., agent-pull over push-based metrics) and marketing standpoint (e.g., subsidy integration for demand, jurisdiction tags for trust).

## Phase A: Residency and Subsidy Fit (Core MVP for Startups/Research)

Focus: Make Xcelsior "rebate-ready" with low-friction onboarding and basic visibility to convert trials to paid usage. This phase sells "cheapest compliant compute in Canada" by auto-enforcing Canada-only and generating AI Compute Access Fund documentation.

### 1. Self-Serve Provider Onboarding (Priority: High)
   - **Research Insight**: Vast.ai and RunPod use CLI agents for host registration (e.g., Vast's CLI runs benchmarks and lists rigs); reduces support load vs. manual approval. Upwork verifies freelancers with ID/proof.
   - **Xcelsior Fit**: Align with Canada-first by requiring country tagging at registration; auto-verify via IP geolocation.
   - **Recommendation**: Extend CLI `host-add-ca` with interactive mode and web confirmation.
     - **Design**: `xcelsior host-add --interactive` prompts for ID, IP, GPU, VRAM, rate, country (default "CA"); generates SSH key if missing; auto-registers and pings for verification. Post-registration, email/SSE confirms with a one-time web link for terms acceptance. API endpoint `/hosts/register` accepts JSON with geolocation proof (e.g., IP-derived). Dashboard adds "Add Host" button linking to CLI docs.
     - **Trust Loop**: Auto-add to marketplace if verified; providers see earnings projection based on spot prices.
     - **Why Now**: Cold-start supply; targets miners pivoting to AI (e.g., Bitfarms-style operators) with bulk registration.

### 2. Job Submission UX with Subsidy Preview (Priority: High)
   - **Research Insight**: RunPod's web console previews costs before launch; Lambda provides API simulators. No direct subsidy integration in competitors.
   - **Xcelsior Fit**: Tie to AI Compute Access Fund by estimating rebates (67% for Canadian compute).
   - **Recommendation**: Add CLI flag `--estimate-rebate` and API param `simulate=true`.
     - **Design**: `xcelsior run --model llama --vram 16 --tier standard --estimate-rebate` outputs: "Estimated cost: CAD $0.50/hr; Rebate eligible: 67% (CAD $0.335 credit via AI CAF)". API `/jobs/submit?simulate=true` returns JSON with breakdown. Dashboard form for job submission with real-time preview. Enforce Canada-only via scheduler flag.
     - **Feedback Loop**: Post-submission, SSE pushes "rebate eligibility confirmed" if host is CA-tagged.
     - **Why Now**: Drives demand; directly exploits subsidy as marketing engine.

### 3. Basic Monitoring and Telemetry (Priority: Medium)
   - **Research Insight**: RunPod exposes nvidia-smi metrics via agent; Vast.ai provides live console. Kubernetes uses node-exporters for health.
   - **Xcelsior Fit**: Agent-pull model fits pull-based architecture; focus on essentials for consumer hardware (no full Prometheus).
   - **Recommendation**: Enhance agent to report GPU metrics every 5s.
     - **Design**: Agent endpoint `/agent/metrics/{host_id}` pulls JSON: {"utilization": 75, "temp": 65, "memory_errors": 0}. Scheduler stores in DB (add `metrics` JSONB column to hosts/jobs). Dashboard shows live gauges for utilization/temp; alert on thresholds (e.g., temp >80C). CLI `xcelsior host-metrics <id>` for debugging.
     - **Trust Loop**: Providers see utilization in dashboard to optimize listings; users get job progress via inferred metrics (e.g., "GPU active: 80%").
     - **Why Now**: Builds reliability perception; prevents "black box" complaints.

### 4. Rebate-Ready Invoicing and Analytics (Priority: High)
   - **Research Insight**: Lambda's invoices break down by resource; CoreWeave provides usage exports. No subsidy-specific tools.
   - **Xcelsior Fit**: Map to AI CAF categories (compute, storage); auto-generate attestation PDFs.
   - **Recommendation**: Extend billing to output subsidy forms.
     - **Design**: API `/billing/export?format=caf` generates CSV/PDF with fields: job_id, duration, cost_CAD, eligible_category ("Canadian cloud compute"), host_country. Dashboard "Export Rebate Docs" button filters by date/user. Analytics endpoint `/analytics/usage` returns aggregates (e.g., total CAD spent, rebate estimate).
     - **Feedback Loop**: Email weekly summaries: "You spent CAD $100; claim CAD $67 rebate".
     - **Why Now**: Turns subsidy into retention; differentiates as "rebate-ready marketplace".

## Phase B: Sovereignty Tier (Vetted Hosts and Transparency)

Focus: Add trust controls for enterprise; sell "credible sovereignty" with verifiable jurisdiction.

### 1. Host Verification Workflow (Priority: High)
   - **Research Insight**: Upwork's "verified" badges build trust; Lambda requires sales vetting for compliance.
   - **Xcelsior Fit**: Require proof for "Sovereign" tier (e.g., Canadian incorporation).
   - **Recommendation**: Add `/hosts/verify` endpoint and CLI `xcelsior host-verify`.
     - **Design**: Providers upload docs (PDF/JSON: incorporation cert, no-foreign-control affidavit). Admin dashboard approves/rejects; approved hosts get "Sovereign" tag in scheduler. Jobs can specify `tier=sovereign` for routing.
     - **Trust Loop**: Marketplace shows badges; providers get premium rates (e.g., 20% uplift).
     - **Why Here**: Anchors with large operators; mitigates over-claiming sovereignty.

### 2. Jurisdiction-Aware Scheduling (Priority: Medium)
   - **Research Insight**: AWS regions for residency; no direct provincial analogs.
   - **Xcelsior Fit**: Encode provincial rules (e.g., Québec PIA flag).
   - **Recommendation**: Add job param `jurisdiction=QC` to force CA-only and log PIA compliance.
     - **Design**: Scheduler filters hosts by tags (country, province); auto-logs "PIA check: adequate protection confirmed". Dashboard shows jurisdiction badges.
     - **Feedback Loop**: Users get compliance reports in exports.
     - **Why Here**: Directly from marketing—provincial fragmentation as opportunity.

### 3. Transparency Reports and Audit Trails (Priority: Medium)
   - **Research Insight**: Google publishes legal request reports; Lambda has SOC2 audits.
   - **Xcelsior Fit**: Start simple for foreign-demand handling.
   - **Recommendation**: Log all access/subpoenas in DB; API `/transparency/report`.
     - **Design**: Monthly JSON export: {"requests_received": 0, "complied": 0}. Dashboard visualization.
     - **Trust Loop**: Builds credibility for sovereignty claims.
     - **Why Here**: Aligns with CLOUD Act diligence.

## Phase C: Regulated Tier (High-Assurance for Public/Health)

Focus: Deep auditability for sensitive workloads; sell to regulated sectors.

### 1. Tamper-Evident Logging and SLAs (Priority: High)
   - **Research Insight**: CoreWeave's SLAs with credits; Kubernetes audit logs.
   - **Xcelsior Fit**: Prep for PHIPA (even if not in force).
   - **Recommendation**: Add job-level logs with hashing.
     - **Design**: Agent streams logs to scheduler (SSE); store with SHA256 hashes. SLA endpoint `/slas/enforce` checks uptime, credits bills on breach.
     - **Trust Loop**: Users query `/jobs/{id}/audit` for verifiable trails.
     - **Why Here**: Targets Ontario/Québec health/public.

### 2. Interactive Documentation and Analytics Dashboards (Priority: Medium)
   - **Research Insight**: RunPod's API docs with examples; Vast's earnings analytics.
   - **Xcelsior Fit**: Embed in dashboard for retention.
   - **Recommendation**: Add Swagger extensions and provider dashboards.
     - **Design**: Dashboard tab "My Earnings" with charts (utilization, revenue); interactive API playground.
     - **Feedback Loop**: Providers optimize based on data; users explore via examples.
     - **Why Here**: Scales adoption in regulated contexts.

## Risk Register and Implementation Notes

- **Friction Risk**: Overly complex onboarding kills trials. Mitigate: CLI-first, with web fallbacks.
- **Trust Risk**: Unvetted hosts erode sovereignty claims. Mitigate: Tiered listings.
- **Scale Risk**: Consumer hardware variability. Mitigate: Telemetry thresholds auto-delist flaky hosts.
- **Subsidy Risk**: Program changes (e.g., post-2027). Mitigate: Modular invoicing for flexibility.

Implementation: Leverage existing FastAPI for endpoints; add SQLite JSONB for logs/metrics. Total: ~15 new endpoints/CLIs, integrable in sprints aligned to phases.