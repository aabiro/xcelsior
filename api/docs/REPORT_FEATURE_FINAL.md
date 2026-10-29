<!-- residency-guard: documents-removal — original feature spec that defined the removed model -->
# Due Diligence: Highest‑Leverage Ideas from the Two Reports for Xcelsior

## Core conclusions and “best ideas” shortlist

The strongest ideas across both reports converge into a single production strategy: **make Xcelsior a Canada-first, security-hardened, verifiable GPU marketplace whose core product truth is “you can trust strangers’ hardware.”** That requires three foundations that are simultaneously technical, operational, and market-facing.

First, the **automated trust infrastructure** (verification → continuous monitoring → automatic “deverification”) is the highest-leverage concept in both reports, and it is strongly corroborated by how mature P2P GPU marketplaces operate. entity["company","Vast.ai","gpu cloud marketplace"] explicitly models a machine lifecycle that can move from “Verified” to “Deverified” based on health/performance degradation, with verification driven by automated checks and reliability thresholds. citeturn2search0turn2search2 This pattern is not cosmetic; it is the minimum viable mechanism that allows “strangers” to transact at scale without manual audits.

Second, a **defense-in-depth isolation stance** is not optional in a multi-tenant GPU marketplace. The container ecosystem has had recent high-impact escape classes in exactly the places Xcelsior would rely on (low-level runtimes and GPU container toolchains). For example, the runc escape CVE-2024-21626 is fixed in runc 1.1.12 and is described as enabling host filesystem access and container escape in affected versions. citeturn3search12turn3search14turn3search6 Separately, a critical NVIDIA container-toolchain escape (CVE-2025-23266) affected NVIDIA Container Toolkit versions up to 1.17.7 and was fixed in 1.17.8 per NVIDIA’s security bulletin; the risk is especially acute in “run your own container” GPU services. citeturn4search2turn4search1turn4search0 In short: Xcelsior must assume untrusted images and hostile tenants.

Third, Xcelsior’s most defensible differentiator is **Canada-first compute economics and compliance alignment**—but only if it is built into scheduling, storage, auditing, and invoicing in a way that matches Canadian programs and statutes. The AI Compute Access Fund program guide states that it covers **two-thirds of eligible costs for Canadian cloud-based compute services**, versus **half** for non-Canadian cloud compute (until March 31, 2027), and defines a “Canadian AI compute service provider” in corporate + physical-residency terms (incorporated/registered in Canada and operating physical data centres in Canada, with infrastructure and data staying in Canada). citeturn0search5 The Fund itself is positioned by entity["organization","Innovation, Science and Economic Development Canada","federal department canada"] as part of the broader Canadian Sovereign AI Compute Strategy and has been publicly described as up to $300M. citeturn0search0turn0search1turn0search7

Taken together, the “absolute best ideas” from the reports for your constraints (small team, self-host preference, limited hardware today, upgrade path later) are:

- **Automated host verification + auto-deverification**, plus an explicit machine lifecycle and reliability thresholds. citeturn2search0turn2search2  
- **Continuous GPU telemetry** (not just reachability), implemented with proven primitives (NVML/DCGM) and surfaced as customer-facing trust signals and internal SLO enforcement. citeturn6search12turn6search0turn6search6  
- **Stronger tenant isolation than “plain Docker”**, using sandboxing where feasible, and version-gating/patch enforcement where not. citeturn3search12turn4search2turn3search0  
- **Canada-residency routing + rebate-ready billing artifacts**, explicitly aligned to AI Compute Access Fund rules and definitions. citeturn0search5turn0search7  
- **S3-compatible artifact storage with presigned URLs** (battle-tested control pattern), but with a residency-aware choice between Backblaze B2 and Cloudflare R2. citeturn11search3turn11search1turn11search0  
- **Marketplace payouts/compliance offloaded to a mature provider** (Stripe Connect), unless you intentionally decide to postpone “real money marketplace” functionality. citeturn6search5  

## Current-state due diligence on the aabiro/xcelsior repository

A close reading of the selected repository indicates you already have a meaningful amount of scaffolding for a production control plane (FastAPI endpoints, database wiring, marketplace concepts, and a worker-agent concept). However, there are also **integration seams that should be treated as “production blockers” until reconciled**, because mismatches between the scheduler state model, the worker agent expectations, and the CLI/API layer create correctness and trust risks.

Key due-diligence findings (derived from inspection of the repository code and tests, not from external sources):

- A **state-machine mismatch** exists between the worker agent’s implied job lifecycle (e.g., “assigned”/“preempted”) and the scheduler’s validated statuses (e.g., “queued/running/completed/failed”). When state transitions are inconsistent, you get silent failure modes: agents may not receive jobs, or status updates may be rejected, breaking billing and reliability accounting.
- The “pull-based agent” approach is directionally excellent for NAT/firewall realities, but it needs the scheduler to expose a **clean “lease/claim” protocol** (assign → lease renewal → completion) rather than conflating “assigned” with “running.” The reports correctly emphasize trust and transparency; that only works if the control plane can reliably define “what happened” for a job.
- The repo already gestures at security hardening (including version checks and optional sandboxed runtimes), but any security posture must be explicitly aligned with current upstream advisories for the container stack and GPU hooks that underwrite multi-tenancy risk. citeturn3search12turn4search2turn4search0

The important meta-point: the best ideas from both reports become materially more valuable if you first make the platform’s “truth engine” coherent—meaning: a single authoritative job model, a single authoritative host model, idempotent accounting, and an auditable event history.

## Trust, verification, and security as the marketplace foundation

The two reports are effectively correct that “trust for strangers” is the gating function. The most defensible implementation is to copy the *shape* (not necessarily the UI) of mechanisms already proven in GPU marketplaces and modern secure compute systems.

### Automated verification and the deverification trigger

Vast.ai’s documentation makes the lifecycle explicit: machines begin unverified, can become verified, and can be **temporarily deverified** if sustained issues are detected—network instability, container launch failures, NVML failures, insufficient PCIe bandwidth, and other degradations are listed examples. citeturn2search0 Here, the “best idea” from Report #1 is not merely to run a benchmark once; it is to **institutionalize drift detection** and make verification revocable.

For Xcelsior, the production-ready version of this idea is:

- A host is issued a “Verified” badge only after it passes test suites that cover **GPU identity/VRAM**, **CUDA + driver readiness**, **PCIe bandwidth sanity**, **sustained thermal/power stability under load**, and **network throughput + loss/jitter**.
- The platform continuously monitors for regressions (driver changes, GPUs removed, bandwidth collapse, repeated job failures) and flips the host into “Deverified” automatically, removing it from premium search/routing until it requalifies. citeturn2search0turn2search2

This is the most leverage-per-engineering-hour feature because it simultaneously improves:
- marketplace quality,
- dispute resolution,
- user retention (jobs fail less),
- operational scalability (less manual support).

### Isolation design driven by real-world container escape classes

Report #1 recommends “least privilege containers” and optional sandboxing. The strongest due-diligence reinforcement for that recommendation is that container escape vulnerabilities are not hypothetical in 2024–2026; they have occurred in widely deployed components.

Two examples show why Xcelsior must plan for hostile tenants and untrusted images:

- **runc escape (CVE-2024-21626)**: NVD’s description notes that runc 1.1.11 and earlier could allow escape via file descriptor leak / cwd handling, and that runc 1.1.12 includes patches. citeturn3search12turn3search8 entity["company","Docker","container platform company"] also published a security advisory confirming the issue and fix in runc 1.1.12. citeturn3search14  
- **NVIDIA Container Toolkit escape (CVE-2025-23266)**: NVIDIA’s bulletin specifies affected versions and a fixed release (1.17.8), and entity["company","Wiz","cloud security company"] describes why this category is most dangerous in shared GPU container services. citeturn4search2turn4search0turn4search1  

The practical conclusion: Xcelsior needs a **node admission policy** that refuses to run customer workloads on hosts that are not patched to minimum safe versions (or that cannot demonstrate a known-safe configuration). This is exactly the kind of “battle-tested” mechanism enterprises expect in multi-tenant compute.

### Using sandboxed runtimes for a stronger security boundary

If you want a stronger barrier than “containers as isolation,” the best idea embedded in your repository directionally matches a modern approach: sandbox the container runtime. entity["company","Google","technology company"]’s gVisor documentation explains GPU support via `nvproxy`, describing how CUDA workloads can run unmodified inside a sandbox and listing supported GPU microarchitectures (with consumer GPUs on the same architectures “likely work”). citeturn3search0

For Xcelsior’s constraints, the production-robust approach is:

- Default to hardened container settings and strict node admission.
- Offer a “Secure tier” that requires a stronger isolation boundary (gVisor/Kata-style), knowing that GPU support may constrain hardware/driver combinations and will need a compatibility matrix. citeturn3search0turn4search2

This matches the reports’ “tiering” logic, but ties it to measurable security posture rather than marketing labels.

## Canada-first differentiation that is legally and economically real

The reports’ “Canadian sovereign AI advantage” is plausible, but it becomes compelling only when your product directly maps to Canadian rules and incentives (and can prove compliance, not just claim it).

### AI Compute Access Fund alignment as a purchase accelerator

The AI Compute Access Fund program guide is unusually explicit about reimbursement rates and what counts as “Canadian compute.” It states:

- **67% coverage (2:1) for eligible Canadian cloud-based compute costs**, versus **50% (1:1) for non-Canadian compute costs (until March 31, 2027)**. citeturn0search5  
- A “Canadian AI compute service provider” must be incorporated/registered in Canada and operate **physical data centres located within Canada**, with infrastructure and data hosted exclusively in Canada (with some nuance for multinationals with Canadian entities and Canada-contained infrastructure). citeturn0search5  

Separately, ISED’s announcements frame the Fund as part of a $2B broader strategy and describe the Fund as up to $300M. citeturn0search0turn0search1turn0search7

The best idea from Report #2—the “rebate-ready” MVP—is therefore not just a nice-to-have. It is a near-term demand lever: you can sell “effective cost after reimbursement” *if and only if* your system can produce auditor-friendly usage reports and your infrastructure choices don’t accidentally disqualify you.

### Privacy-by-default and governance hooks that should be “product DNA”

If Xcelsior is handling personal information (provider identities, billing/contact info, potentially job metadata), Canadian privacy expectations become design constraints.

Under PIPEDA, the 10 fair information principles include **Accountability** and explicitly require organizations to appoint someone accountable for compliance. citeturn2search3 The Office of the Privacy Commissioner of Canada notes organizations must protect personal information with appropriate safeguards and destroy it when no longer needed. citeturn2search10 entity["organization","Office of the Privacy Commissioner of Canada","canada privacy regulator"] is a primary source for interpreting these operational expectations.

In entity["state","Québec","province, canada"], Law 25-era guidance from entity["organization","Commission d'accès à l'information du Québec","quebec privacy regulator"] states that where organizations offer technological products/services with privacy settings, they must ensure default settings provide the highest level of confidentiality, and that identification/location/profiling functions cannot be enabled by default. citeturn2search4turn2search1 Related guidance from the Government of Québec similarly explains that identification/location/profiling functions must be disabled by default and the person must activate them. citeturn2search5

The “best idea” implication for Xcelsior is to encode privacy governance as built-in controls rather than policy documents:

- Default-minimal logging for job payloads; explicit opt-in for sensitive fields.
- A clear “privacy officer” field in org settings and provider onboarding flows.
- Job metadata redaction and retention policies (especially for logs/telemetry exports).

### CLOUD Act marketing: be precise, not hand-wavy

Report #1 mentions the U.S. CLOUD Act. The credible version of this argument is not “avoid the U.S.” in general, but “jurisdiction and provider control matter.” A entity["organization","United States Department of Justice","federal agency us"] white paper describes that U.S. legal process can compel disclosure by U.S.-jurisdiction cloud service providers **regardless of where data is stored**. citeturn2search48turn2search13

For Xcelsior, this primarily means: your sovereignty story is strongest when your **control plane entity**, **data plane locations**, and **storage choices** are all aligned to Canadian residency and Canadian corporate control—especially if you plan to target regulated or public-sector-adjacent workloads.

## Storage and artifacts: B2 vs R2, and the upgrade path to self-hosted

Your storage layer is not just a technical choice; it affects:

- AI Compute Access Fund “Canadian compute” eligibility narratives,
- unit economics (egress),
- speed-to-first-token (model weight downloads),
- and security boundaries (how you distribute data to hosts).

### What the evidence says about R2

entity["company","Cloudflare","internet infrastructure company"] R2 pricing explicitly states **no egress bandwidth charges** for Standard storage (and free tier includes 10 GB-month storage plus free operation quotas). citeturn0search3 This is extremely attractive for LLM weights and repeated downloads, where egress can dominate cost.

However, Cloudflare’s R2 data location documentation distinguishes between:
- “Location hints” (best-effort placement near a broad region like ENAM/WNAM), and
- “Jurisdictional Restrictions” (guarantees) which currently list supported jurisdictions as **EU** and **FedRAMP**. citeturn1search0

As of this documentation, there is **no Canada jurisdiction guarantee for R2 object residency**. If your product promise is “Canada-only storage,” you should not treat R2 as residency-guaranteed.

### What the evidence says about B2

entity["company","Backblaze","cloud storage company"] documentation notes a **Canada East region located in Toronto, Ontario**, and that account region choice governs where data is stored. citeturn1search6 That is directly aligned with your Canada-first positioning and is materially relevant if you want to market toward AI Compute Access Fund recipients and definitional “Canadian compute” expectations. citeturn0search5

Economics-wise, Backblaze’s transaction pricing page states free storage for the first 10GB and describes an egress model with “free egress up to 3x your monthly average storage,” with additional egress priced at $0.01/GB. citeturn0search2 This can be excellent for many workloads, but it can be punitive for “store once, download constantly” model-weight usage unless you build caching.

### Presigned URLs are the battle-tested artifact distribution primitive

Both reports recommend presigned URLs. The due-diligence reinforcement is that presigned URLs are a widely standardized S3 pattern for time-limited access:

- AWS’s S3 user guide describes presigned URLs as time-limited access without changing bucket policy, and emphasizes they are effectively bearer tokens limited by the creator’s permissions. citeturn11search3  
- Cloudflare documents presigned URLs in R2 as an “S3 concept,” supporting multiple operations and treating URLs as bearer tokens. citeturn11search0  
- Backblaze explicitly states its S3-compatible API supports presigned URLs. citeturn11search1turn11search8  

This is one of the strongest “best ideas” because it cleanly separates:
- control plane authentication (Xcelsior issues URLs),
- data plane access (workers pull/push directly to object store),
- and secret management (workers don’t need long-lived storage credentials).

### The pragmatic recommendation for your constraints

Given your “self-host preferred, no hardware right now, upgrade later” constraint, the most robust approach is a **two-tier storage strategy**:

- Use **Backblaze B2 in CA East (Toronto)** for anything that could plausibly fall under “Canada-only” claims (private artifacts, customer datasets, job outputs that might include sensitive IP), because it offers a Canada-based region. citeturn1search6turn0search5  
- Use **Cloudflare R2** for *non-sensitive/public or cacheable* objects where egress dominates (public model caches, non-sensitive layers, distributable artifacts), because egress is free and the pricing model is designed for heavy reads. citeturn0search3turn1search0  
- Build the platform such that both backends are treated uniformly as “S3-compatible object stores,” so you can later migrate “Canada-only storage” to self-hosted S3-compatible storage when you actually have hardware (or colocated Canadian infrastructure) without rewriting the application contract.

This preserves your long-term self-hosting goal while keeping near-term cost and execution practical.

## A cohesive production roadmap anchored in “sellable truths”

This roadmap is structured around the most valuable outputs you can truthfully sell at each stage while continuously improving security and reliability. It assumes you are willing to accept complexity, but you want to keep the system cohesive and production-grade.

### Establish the control plane truth engine

The first “battle-tested” move is to unify the platform around a consistent, auditable event story:

- Define a strict job state machine (queued → assigned/leased → running → completed/failed/preempted) and make every transition idempotent and recorded as an event.
- Make billing, SLA credits, and reputational scoring derive from events, not from best-effort heuristics.
- Treat “lease renewal” as the mechanism that differentiates “host is alive” from “host completed work,” so downtime is measurable.

This is the enabling step for almost every best idea from both reports: verification, reputation, SLA enforcement, and dispute resolution.

### Implement verification and telemetry as a single system

Use the Vast.ai pattern as your operational model: automated verification + continuous monitoring + reversible status. citeturn2search0turn2search2

Telemetry should be based on proven NVIDIA primitives:

- NVML is explicitly described by NVIDIA as the underlying library for `nvidia-smi` and a C-based interface for monitoring/managing GPU state. citeturn6search12turn6search15  
- For higher-fidelity operational monitoring, NVIDIA recommends DCGM Exporter for Prometheus-based telemetry stacks and documents it as a standard GPU metrics exporter. citeturn6search0turn6search6  

If you are self-hosting, this is aligned with mature operational patterns: Prometheus + Alertmanager + Grafana are explicitly described by NVIDIA as a common open-source monitoring stack, with DCGM Exporter providing GPU metrics. citeturn6search6turn6search0

### Treat security posture as a schedulable resource

A “production-ready” marketplace will not allow any host to run any workload. It will route based on:

- patch posture for container runtime/toolkit (version gating; host admission),
- isolation capabilities (sandboxed runtime availability),
- and workload sensitivity classification (customer chooses “standard” vs “sovereign” vs “secure” tier).

This is justified by the demonstrated severity of recent runtime and GPU-container-toolchain escapes. citeturn3search12turn4search2turn4search0

A practical routing rule is: “untrusted workloads require the strong isolation tier,” since gVisor positions itself as a sandbox that supports many CUDA workloads via `nvproxy`. citeturn3search0

### Make Canada-first capabilities measurable and exportable

If you want to be the obvious provider for Canadian startups and research-adjacent SMEs, tie your platform outputs to program mechanics:

- The AI Compute Access Fund program guide’s reimbursement ratios (Canadian vs non-Canadian compute) make it rational for customers to prefer Canadian compute when eligible. citeturn0search5  
- The Fund is a named, public program with published announcements and a due diligence framing by ISED. citeturn0search0turn0search1  

That means Xcelsior should automatically produce:
- a “Canada-incurred compute” spend breakdown,
- a provider attestation bundle,
- and a workload residency trace (where jobs ran; where data was stored).

The compliance framing is reinforced by Canadian privacy regulators emphasizing accountability and privacy-by-default expectations. citeturn2search3turn2search4turn2search5

### Decide early how you will handle payments and tax obligations

If you are building a real two-sided marketplace (providers paid out, users charged), do not build your own financial compliance stack unless you deliberately choose that as a differentiator.

- Stripe Connect explicitly positions itself as providing onboarding, verification, and payouts for marketplaces, and aims to reduce compliance overhead. citeturn6search5  
- Canadian GST/HST rules for digital-economy “distribution platform operators” can impose collection obligations depending on the supplier’s registration regime and how the supply is facilitated. citeturn6search4turn6search8  

These are not “later problems” in a marketplace; they affect whether you can safely scale revenue without compounding liabilities.

### Treat SHARCNET Nibi as an internal acceleration path, not the marketplace baseline

Your possible access to SHARCNET’s entity["organization","SHARCNET","ontario research computing consortium"] Nibi resource is strategically useful for bootstrapping reliability and performance learnings. Public descriptions indicate Nibi is hosted at entity["organization","University of Waterloo","Waterloo, Ontario, Canada"] and described as a large national system with substantial GPU capacity (including H100-class nodes), and is part of the refreshed national host infrastructure. citeturn5search1turn5search3turn5search15 The Digital Research Alliance lists Nibi as a resource with compute and GPU subsystems (nibi-compute / nibi-gpu). citeturn5search12 entity["organization","Digital Research Alliance of Canada","national research infrastructure org"] is the coordinating body referenced for national allocations and host site upgrades. citeturn5search12turn5search15

Pragmatically, Nibi can help you:
- validate scheduling semantics,
- run verification/benchmark suites on high-end GPUs,
- and harden telemetry/observability at scale.

But it should be treated as an internal/partner capacity source, because academic HPC usage and allocation constraints can limit commercial marketplace-style operation (this is something you will need to confirm under your account/allocation terms). Your connection via entity["organization","University of Western Ontario","London, Ontario, Canada"] can still be a meaningful path to access and collaboration, and the technical shape of integration points (e.g., Slurm adapters, data staging, job receipts) is consistent with the “control plane + adapters” model the reports implicitly recommend. citeturn5search16

## Final recommendations: the “best ideas” to commit to now

To reach a cohesive, production-ready strategy fastest, the best commitment set (i.e., ideas in the reports that are both essential and execution-worthy for a small team) is:

Build Xcelsior around a strict job/host event model, then ship **verification + deverification + telemetry** as the customer-visible trust layer, using NVML/DCGM-based metrics and automated lifecycle transitions modeled after established marketplaces. citeturn2search0turn6search0turn6search12

Make **security posture schedulable**: enforce node admission based on patched container runtimes and GPU container toolchains, and provide a strong-isolation tier for untrusted workloads, justified by recent real-world escape vulnerabilities. citeturn3search12turn4search2turn3search0

Make your Canada-first positioning economically real by building **AI Compute Access Fund–aligned billing exports and residency evidence** into the product, while adopting privacy-by-default and accountability hooks consistent with Canadian regulator guidance. citeturn0search5turn2search3turn2search4turn2search5

Adopt **S3-compatible artifact storage with presigned URLs** as your standard artifact contract, then choose storage backends by residency and egress economics: Backblaze B2 (Canada East) for Canada-only claims, Cloudflare R2 for egress-heavy non-sensitive caching, with a clean migration path to self-hosted S3-compatible storage when hardware arrives.