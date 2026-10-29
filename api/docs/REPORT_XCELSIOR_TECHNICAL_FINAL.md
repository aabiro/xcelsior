# Due Diligence on the Best Ideas for Xcelsior Under Your Constraints

## Context and evaluation criteria

Both reports correctly frame Xcelsior’s next set of bottlenecks as: (a) persistence under concurrency, (b) dispatch/orchestration across unreliable GPU nodes, (c) multi-tenant isolation and abuse resistance, and (d) real-time observability. Where they diverge is *fit* for your stated constraints: small team, no hardware right now, preference for self-hosted/free building blocks, and realistic upgrade paths later.

Two additional facts materially change prioritization:

* **SQLite WAL is “single-writer” by design**, even though WAL improves reader/writer coexistence. This makes it a predictable scaling ceiling once write contention becomes noticeable (heartbeats, job state transitions, bidding, accounting). citeturn5search0turn5search16  
* **Residential/edge GPU supply is often behind NAT/CGNAT**, which structurally breaks inbound “push” control patterns (SSH into node) and strongly favors “pull” agents or overlay networks. CGNAT is explicitly called out as commonly preventing port forwarding. citeturn7search10turn7search2turn7search7  

Your note about possible access to **SHARCNET / Alliance “Nibi”** (H100-scale cluster) adds a second, optional future deployment target that is *not* “consumer GPUs behind NAT,” and uses the HPC norm of batch scheduling (Slurm). Nibi is described as a general-purpose cluster with **288 H100 GPUs**, and Alliance documentation emphasizes **Slurm** for job scheduling and GPU requests. citeturn18search0turn18search14turn18search3  

That combination implies an architecture that can support **two backends** over time:
1) Xcelsior-managed heterogeneous edge/consumer nodes, and  
2) Optional HPC adapters (e.g., Slurm submission) for burst capacity or benchmarking—without forcing you into Kubernetes everywhere.

## The best ideas across the two reports, ranked for your situation

### Pull-based agent architecture plus NAT traversal is the single highest-leverage change

Report #2’s “pull-based agent” recommendation is the best foundational idea because it simultaneously addresses reliability, networking, and security posture:

* It removes the requirement for inbound SSH (fragile under CGNAT; increases exposed surface area). CGNAT’s structure often blocks customer-controlled port forwarding. citeturn7search10  
* It matches how robust systems usually work: nodes maintain outbound connectivity and “check in” with a coordinator; the coordinator does not rely on being able to dial into arbitrary nodes.
* It becomes the enabling layer for later features: preemption, local image caching, host benchmarking, runtime policy enforcement, egress filtering—none of which are pleasant over pure SSH. (This is a systems-design inference built from the mechanics of pull agents and NAT constraints.) citeturn7search10turn7search7  

For NAT traversal, the self-hosted/free option in Report #2—**Headscale**—is strongly aligned with your preferences. Headscale is explicitly “an open source, self-hosted implementation of the Tailscale control server,” designed for a single tailnet and self-hosters. citeturn4search3turn4search7turn4search10  
Headscale’s docs also specify you need a server with a public IP and that it is served via HTTPS on port 4431 (implementation detail that matters operationally). citeturn7search2  

This is also a good “upgrade later” story: you can stay self-hosted on Headscale, or later move to entity["company","Tailscale","mesh vpn company"]’s hosted control plane if you outgrow DIY (and still keep the same node client model). Tailscale’s own NAT traversal write-ups explain their use of a coordination server plus relay system (DERP) as part of traversal strategy. citeturn7search7turn7search11turn4search22  

### PostgreSQL migration with an “expand/contract” cutover is the right persistence strategy, but the triggers should be “write contention and operational needs,” not database size myths

Report #2’s Postgres plan (dual-write, then cutover) is the most credible approach for “zero or near-zero downtime” migration from SQLite, because **SQLite does not have a built-in replication mechanism analogous to Postgres logical replication**; the practical approach is application-level expand/contract patterns (dual writes, then switch reads/writes). (This is an inference based on SQLite’s architecture and the lack of native replication; the plan itself is widely used in practice.) citeturn5search0turn14search4  

Key due-diligence corrections to Report #1/#2 framing:

* SQLite can scale to very large *file sizes* in theory (SQLite itself documents ~281 TB max with max page size; and ~17.5 TB with default 4KiB pages). But this says nothing about write-concurrency suitability for a scheduler. citeturn5search1turn5search13  
* The real architectural constraint is still: **WAL allows multiple readers and a single writer**. citeturn5search0turn5search16  
* Postgres’ concurrency model (MVCC) is designed so **reads don’t block writes and writes don’t block reads** in the common case (though writers can still conflict with each other depending on locking). citeturn5search6turn5search2  

Given Xcelsior’s scheduler pattern (frequent state transitions; periodic heartbeats; queue updates), the “why Postgres” story should be:

* reduce “single writer” contention risk, citeturn5search0turn5search6  
* gain richer indexing and query capabilities, and  
* gain operational primitives (backups, replication, HA options) when you want them. citeturn14search3turn14search7  

The strongest detail in Report #2 is the recommendation to use **JSONB + GIN indexing** so the scheduler can query host capability payloads without Python-side full scans. Postgres documents GIN indexing for `jsonb` and the existence of dedicated operator classes/trade-offs. citeturn5search3turn5search7  

### Security posture improvements should start with “boring hardening + strict version gates,” then selectively add gVisor/Kata

Both reports are directionally right that multi-tenant GPU execution is *hostile* and container escape risk is real. The due diligence point is: **you’ll get most of your security ROI early from baseline hardening and patch discipline**, and only then should you pay the complexity tax of sandboxed runtimes.

Load-bearing security facts you should treat as “must design for”:

* `runc` had a serious escape class vulnerability **CVE-2024-21626**, with affected versions up to 1.1.11 and patched in 1.1.12. citeturn16view0turn0search20  
* The GPU stack itself has had critical issues; **CVE-2025-23266** in the **NVIDIA Container Toolkit** was addressed by updating to **1.17.8** (per NVIDIA’s own security bulletin), and the bulletin describes the issue as elevated-permission code execution via hooks. citeturn15view0turn0search5  

So, the “absolute best” security idea embedded in Report #2 is: **host admission control / version gating** (agent refuses to run workloads on nodes with known-vulnerable runc/toolkit/driver combinations). NVIDIA explicitly lists affected versions and updated versions for CVE-2025-23266, making it feasible to encode. citeturn15view0  

Then layer baseline container hardening (low operational burden):

* Prefer non-root and/or rootless container execution where feasible; Docker documents rootless mode as running the daemon and containers as non-root to mitigate vulnerabilities in the daemon/runtime. citeturn8search2  
* Avoid `--privileged`; Docker docs state it effectively grants all capabilities and host device access and loosens LSM protections. citeturn8search22  
* Add least-privilege controls: capability dropping, seccomp/AppArmor/SELinux, and “no new privileges.” NIST guidance for application container security explicitly calls out least privilege/least access and runtime profiling; NIST IR 8176 describes seccomp/AppArmor/SELinux as kernel hardening mechanisms to reduce attack surface. citeturn8search1turn8search5  
* OWASP’s Docker security guidance specifically recommends `--security-opt=no-new-privileges` to prevent privilege escalation via setuid/setgid binaries. citeturn17search11  

Only after those are in place does it make sense to evaluate sandbox runtimes:

* **gVisor with GPU (“nvproxy”)** is more plausible than many people assume, but has real constraints. gVisor documents that it supports running “most CUDA applications” on *preselected versions of NVIDIA’s open source driver*, and lists supported GPU families (T4/A100/A10G/L4/H100) while noting that other GPUs on the same microarchitectures (e.g., RTX 3090/4090) will “likely work,” but are not officially supported. citeturn9view0  
* gVisor is also explicit that nvproxy forwards ioctls to the host driver and therefore **does not protect you from vulnerabilities inside the NVIDIA driver**; timely driver updates remain imperative. citeturn9view0turn15view0  
* **Kata Containers** is a well-known VM-backed isolation approach; AWS describes Kata as running containers in OCI-compliant lightweight VMs for stronger isolation. citeturn8search3turn8search14  
* However, academic work shows microVM-based containers have their own attack surfaces (e.g., operation forwarding), reinforcing that “VM-backed” is not synonymous with “done”—it’s defense-in-depth, not a silver bullet. citeturn8search11  

Given your *current* “no hardware” constraint and likely early-stage user base, the best path is: baseline hardening + strict version gating now; gVisor/Kata as optional tiers later.

### Egress filtering + cryptomining detection is a top-tier idea for any GPU marketplace

Report #2 is correct to treat cryptomining abuse as a primary adversarial behavior for rented compute. Due diligence: the widely quoted “75%” figure is *context-specific*, but the general pattern is well-supported.

For example, entity["company","Palo Alto Networks","cybersecurity company"] Unit 42 ran a Docker-daemon honeypot experiment (March–April 2021) and reported that **more than 75%** of observed attacks were cryptojacking. citeturn3search2  
Separately, entity["company","Datadog","monitoring and security company"] security research describes threat actors leveraging Docker/Kubernetes/Docker Swarm for cryptomining campaigns, highlighting the operational reality that exposed container control planes are abused for mining. citeturn3search8  

That makes two ideas from Report #2 “best in class” for your context:

1) **Default-deny outbound networking with allowlists** for common package/model registries as needed (reduces Stratum/pool connections as a class). (This is a design inference; CGNAT/overlay networking does not change the need for outbound controls.)  
2) **Behavioral mining heuristics on the agent side**, using local GPU telemetry (e.g., utilization patterns). The reports’ exact heuristic is not a standard, but the principle—detect abnormal runtime behavior—aligns with runtime security approaches. citeturn17search0turn17search25  

If you want to go beyond heuristics, open-source runtime detection tools can help, but they add operational weight:

* Falco positions itself as a cloud-native runtime security tool that detects threats using kernel events and rules across hosts/containers/Kubernetes. citeturn17search0turn17search4  
* Tetragon is an eBPF-based security observability/enforcement project emphasizing filtering/policy in-kernel. citeturn17search1turn17search34  

For a small team, the “best” move is usually: start with egress restrictions + a small number of high-signal detections; only add Falco/Tetragon when you have staff/time to tune false positives.

### SSE is the right default for dashboards, but the report must account for browser limits and proxy behavior

Both reports push SSE for “server → browser” updates, which is directionally right: SSE is unidirectional and simpler than WebSockets for dashboards. citeturn11search21turn11search11  

However, due diligence shows two practical constraints you should design around:

* Under HTTP/1.1, browsers commonly enforce a **~6 concurrent connections per domain** limit; persistent SSE streams consume one slot. This is discussed repeatedly in developer references and Q&A, and is the core “multi-tab” gotcha. citeturn0search11turn0search15turn0search7  
* A strong mitigation is to serve the dashboard over HTTP/2, where multiplexing reduces the connection-limit pain (implementation details vary, but the architectural point stands). citeturn0search15turn7search11  

For *server-side fanout*, Report #2 suggests Redis Pub/Sub. That’s reasonable technically, but there’s a licensing/operational nuance relevant to your “free + self-hosted” preference:

* Redis changed licensing in March 2024 (RSAL/SSPL), and Redis itself acknowledges community frustration with that change; Redis 8 later added AGPL as a licensing option. citeturn4search11turn12search2  
* The BSD-licensed fork **Valkey** is explicitly positioned as open source and Linux Foundation-backed, and multiple sources emphasize it as a drop-in Redis-protocol-compatible option. citeturn4search1turn4search26turn12search1  

Finally, for a minimal stack, you can sometimes avoid an external broker entirely by using Postgres’ built-in pub/sub:

* Postgres documents `LISTEN`/`NOTIFY`, including that **NOTIFY payload defaults to < 8000 bytes**, notifications are queued in memory, and long transactions can prevent cleanup; it’s explicitly meant as interprocess signaling. citeturn13view0turn12search12  

So the “best” dashboard idea is: **one SSE stream per user session + lightweight events (IDs, deltas), backed initially by Postgres LISTEN/NOTIFY**; upgrade later to Valkey (or another broker) when throughput/fanout needs exceed what you want to risk inside the primary database.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Headscale self-hosted Tailscale control server diagram","Tailscale DERP NAT traversal diagram","Server-Sent Events dashboard architecture diagram","gVisor nvproxy GPU sandbox diagram"],"num_per_query":1}

## Deep dive on each pillar with “best idea” selections and what to avoid

### Persistence and database drivers

**Best idea:** migrate to Postgres with expand/contract dual-write, keep SQLite as a dev-mode default.

* SQLite WAL’s concurrency model (“only one writer at a time”) is a known mismatch for “many small writes from many actors.” citeturn5search0  
* Postgres’ MVCC model is built for concurrent transactional workloads without read/write blocking patterns typical of coarse locking. citeturn5search6turn5search2  
* If your host/job records are semi-structured, store capability blobs as JSONB and index the query-critical portions using GIN and/or expression indexes. Postgres documents JSONB indexing with GIN. citeturn5search3turn5search7  

**Migration tooling that is genuinely aligned with “small team”:**

* pgloader explicitly supports “SQLite to Postgres” migration with automatic schema discovery and index creation. citeturn14search0  
* Alembic is the standard lightweight migration tool around SQLAlchemy. citeturn14search1turn14search5  

**Driver choice due diligence:**

* SQLAlchemy 2.0+ added official dialect support for psycopg (psycopg3), including async support. citeturn6search4turn6search28  
* asyncpg’s project claims it can be ~5x faster than psycopg3 in their benchmarks (useful as a signal, but treat as workload-dependent and benchmark yourself). citeturn6search3  

**Decision that best fits your constraints:** Use psycopg (psycopg3) as the default for maintainability and SQLAlchemy integration; reserve asyncpg for isolated hot paths only if profiling proves you need it. citeturn6search4turn6search25turn6search3  

### Orchestration: consumer/edge marketplace vs HPC backends

**Best idea:** implement the pull-based “xcelsior-agent” as the universal execution substrate.

This is the most reusable foundation across both consumer GPUs and more structured environments because it:

* gives you a single place to enforce policies and collect telemetry, and  
* decouples “central scheduling decisions” from “how to start/stop a job on a node.”

**What to avoid early:** “Kubernetes everywhere” and “MIG-first” thinking.

* Report #1’s MIG + K8s framing is mismatched for consumer GeForce fleets; MIG is mainly relevant on specific data center/workstation-class GPUs and is explicitly documented as available only on certain GPU families (e.g., A100/H100 and RTX PRO 6000 class), not “any RTX card.” citeturn1search25turn1search0turn1search24  
* Also, gVisor’s nvproxy doc explicitly notes lack of integration support with the common NVIDIA k8s-device-plugin (outside GKE). That’s another reason not to hinge early strategy on K8s sharing plugins if you’re small-team and not on GKE. citeturn9view0  

**Best idea for cost/utilization later:** spot/interruptible + preemption, but only after the agent exists.

The concept is sound and well-proven in cloud economics:

* entity["company","Amazon Web Services","cloud provider"] explicitly markets Spot Instances as up to 90% discount vs on-demand. citeturn2search2turn2search27  
* entity["company","Vast.ai","gpu marketplace"] documents that interruptible instances are “often 50%+ cheaper” than on-demand. citeturn2search1turn2search22  

What’s “best” for Xcelsior is not the exact pricing curve from Report #2, but the architecture: design the job lifecycle to support interruption (checkpoint hooks, SIGTERM handling, requeue semantics). This is a systems-design inference supported by how spot/interruptible compute works operationally. citeturn2search27turn2search22  

**How Nibi changes this pillar (optional future):** treat Nibi as a separate backend, not a reason to overbuild Xcelsior’s own scheduler.

Nibi is documented as a large general-purpose cluster with H100 GPUs and is part of the Alliance ecosystem; Alliance docs emphasize Slurm as the scheduler and provide GPU usage guidance. citeturn18search0turn18search5turn18search14turn18search3  
That suggests a pragmatic “best idea” if you get access: build a **Slurm adapter** that translates Xcelsior job intents into Slurm submissions, rather than trying to replace Slurm on HPC. (This is an inference based on HPC norms and the presence of Slurm/MIG support docs.) citeturn18search14turn18search3  

### Container isolation and abuse prevention

**Best idea:** implement “defense in depth” in layers, starting with low-ops hardening and patch gates.

Layer 1: **Version gating / node admission control**
* Encode “refuse to run if vulnerable” for runc and NVIDIA Container Toolkit. runc’s advisory documents affected and patched versions for CVE-2024-21626. citeturn16view0  
* NVIDIA’s bulletin provides exact affected and fixed versions for CVE-2025-23266 (Toolkit fixed at 1.17.8). citeturn15view0  

Layer 2: **Least privilege container configs**
* Rootless Docker reduces risk by running daemon and containers as non-root. citeturn8search2  
* Avoid privileged containers; Docker documents privileged grants broad capabilities and host device access. citeturn8search22  
* Apply “no new privileges” to prevent privilege escalation via setuid/setgid where possible. citeturn17search11  
* NIST guidance emphasizes least privilege / least access principles and runtime threat profiling for container deployments. citeturn8search1turn8search5  

Layer 3: **Network controls + mining resistance**
* The Unit 42 honeypot result indicates cryptojacking dominates in that threat model, reinforcing the need for egress controls in a GPU marketplace. citeturn3search2  

Layer 4 (optional tier): **Sandboxed runtimes**
* gVisor’s GPU support is real but constrained: nvproxy requires selected GPU models/driver versions and uses passthrough forwarding of ioctls; it improves isolation against general Linux kernel vulnerabilities but cannot “magic away” NVIDIA driver bugs—driver updates remain mandatory. citeturn9view0turn15view0  
* Kata provides stronger isolation via lightweight VMs, but increases operational complexity and still has a known research surface area under microVM designs (do not treat it as risk-free). citeturn8search3turn8search11  

**What to avoid:** relying on a single mechanism (just gVisor; just Kata; just heuristics). Both gVisor and Kata docs/research implicitly support that these are components in layered security, not complete solutions. citeturn9view0turn8search11  

### Real-time dashboard and internal eventing

**Best idea:** SSE as default, with a single stream per UI session + lightweight message bus.

SSE’s strengths for dashboards are widely recognized: simpler than WebSockets for one-way updates and used at scale in real systems (e.g., entity["company","Shopify","ecommerce company"] engineering describes building scalable SSE servers for real-time visualization). citeturn11search11turn11search21  

Due diligence requirements:

* Design around the browser’s connection limits under HTTP/1.1 (~6 per domain); prefer HTTP/2 for multi-tab robustness. citeturn0search11turn0search15  
* Avoid pushing large payloads via Postgres NOTIFY because payload is <8000 bytes by default; publish IDs and fetch details from Postgres (or store payloads elsewhere). Postgres documents the 8000-byte default payload limit and the queue behavior. citeturn13view0  

**Broker choice with your licensing preference:**
* If you want Redis-like Pub/Sub without Redis licensing uncertainty, Valkey is Linux Foundation-backed, BSD-licensed, and positioned as Redis-protocol compatible. citeturn4search26turn12search1turn4search1  
* If you want the minimal stack early, Postgres LISTEN/NOTIFY is “good enough” for modest rates and is already in the database you’ll be running anyway. citeturn13view0turn12search12  

## Recommended implementation sequence with clear upgrade paths

This sequence is the distilled “absolute best” path implied by the two reports, corrected by due diligence and tuned to your constraints.

**Foundation**
1) **Build the pull-based agent + secure control channel** using entity["organization","Headscale","self-hosted tailscale control"] as the default self-hosted coordination option. Headscale positioning and requirements are explicit in its docs. citeturn4search3turn7search2turn4search22  
2) Replace inbound SSH orchestration with outbound agent polling / long-lived control streams (architectural inference supported by CGNAT limitations). citeturn7search10  

**Persistence upgrade**
3) Add a DB abstraction layer and implement Postgres schema + migrations (Alembic). citeturn14search1turn14search5  
4) Migrate using pgloader for initial backfill, then dual-write expand/contract cutover. citeturn14search0turn14search4  
5) Use JSONB for flexible payloads and add GIN indexes for the fields your scheduler actually queries. citeturn5search3turn5search7  

**Security baseline**
6) Implement node version gates for runc and NVIDIA Container Toolkit based on published advisories/bulletins. citeturn16view0turn15view0  
7) Enforce least privilege defaults (rootless where feasible, no privileged, no-new-privileges). citeturn8search2turn8search22turn17search11  
8) Add default-deny egress with allowlists, plus a small set of high-confidence mining detections. citeturn3search2turn3search8  

**Observability**
9) Implement SSE for dashboards with HTTP/2-aware deployment and a single stream per session; start with Postgres LISTEN/NOTIFY for notifications and upgrade to Valkey when you need more fanout/isolation from the DB. citeturn0search15turn13view0turn4search1turn12search1  

**Optimization and expansion**
10) Add interruptible/spot pricing and preemption *only once* jobs can be checkpointed and requeued cleanly (this is the step that monetizes utilization improvements). citeturn2search22turn2search2  
11) If/when you gain access to Nibi, treat it as an HPC backend for benchmarking and burst capacity; prototype a Slurm submission adapter rather than redesigning core orchestration around Kubernetes/MIG. Nibi’s scale and Slurm usage are documented, and MIG/GPU requests appear in the Alliance Slurm docs. citeturn18search0turn18search14turn18search3