# **Architectural Modernization of Xcelsior: From Prototype to Production Distributed Scheduler**

## **1\. Executive Summary and Architectural Vision**

The transition of Xcelsior from a single-node, SQLite-backed prototype to a distributed, production-grade GPU scheduling platform represents a significant inflection point in its architectural lifecycle. The current implementation, characterized by its reliance on synchronous SSH execution, atomic file-based locking via SQLite WAL mode, and a monolithic Python process, has successfully demonstrated the core value proposition: leveraging consumer-grade hardware for distributed compute. However, the constraints imposed by residential networking environments, the heterogeneity of consumer GPUs (RTX 2060–4090), and the imperative for multi-tenant security necessitate a comprehensive re-architecture.

This report provides an exhaustive technical analysis and roadmap for this transition. The analysis operates under strict constraints: the engineering team is small (1-3 developers), restricting the ability to adopt operationally complex solutions like Kubernetes (K8s) clusters across unreliable residential nodes. Instead, the recommendations prioritize robust, low-maintenance open-source components and incremental refactoring strategies that preserve backward compatibility with the existing CLI and API.

The architectural vision proposed herein centers on four critical pillars:

1. **Persistence Scalability:** A zero-downtime migration from the concurrency-limited SQLite architecture to a PostgreSQL backend utilizing JSONB for flexible schema evolution.  
2. **Orchestration Logic:** Moving from a "push-based" SSH model to a "pull-based" agent architecture that mimics the marketplace dynamics of Vast.ai and RunPod, incorporating spot pricing and robust queuing without the overhead of full K8s.  
3. **Adversarial Security:** Implementing defense-in-depth for untrusted consumer hosts, replacing standard Docker runtimes with sandboxed environments (gVisor) and implementing rigorous egress filtering to mitigate the pervasive threat of crypto-mining abuse.  
4. **Real-Time Observability:** Shifting from polling-based dashboards to event-driven architectures using Server-Sent Events (SSE) to ensure scalable state synchronization.

## ---

**2\. Database Persistence: The Migration from SQLite to PostgreSQL**

The current data layer of Xcelsior is built on SQLite in Write-Ahead Logging (WAL) mode.1 While SQLite is exceptionally fast for read-heavy workloads on a single node, its write concurrency model is fundamentally limited by the single-writer lock, even in WAL mode. As Xcelsior scales to manage hundreds of distributed nodes, the contention for the write lock during high-frequency heartbeat updates and job status transitions will become the primary system bottleneck.

### **2.1. Limitations of the Current SQLite Architecture**

The scheduler.py module currently utilizes an \_atomic\_mutation context manager that executes BEGIN IMMEDIATE.1 This locks the database file for writing, ensuring ACID compliance but forcing all other write operations to wait.

* **Write Contention:** In a distributed system, every compute node must report its health ("heartbeat") periodically (e.g., every 30 seconds). With 100 nodes, this results in \~3.3 writes per second purely for health checks. When combined with job status updates, marketplace bid insertions, and billing increments, the SQLite write queue will saturate, leading to database is locked errors and API timeouts.  
* **Data Integrity Risks:** The current "JSON-in-TEXT" pattern 1 relies on application-level serialization. This places the burden of data integrity entirely on the Python application. A crash during the serialization/write cycle, despite atomic transactions, can lead to corruption or lost updates if the WAL file is not properly checkpointed.  
* **Query Performance:** While the jobs and hosts tables have indexed columns 1, complex queries against the JSON payload (e.g., "Find all hosts with \>16GB VRAM and \<50ms latency") require either redundant columns or full table scans in Python, which is computationally expensive and slow.

### **2.2. PostgreSQL Implementation Strategy**

PostgreSQL is the industry standard for this class of problem due to its Multi-Version Concurrency Control (MVCC) architecture, which allows readers to not block writers and supports high write throughput.

#### **2.2.1. Deployment Patterns for Small Teams**

For a team of 1-3 developers, minimizing operational overhead is paramount.

* **Managed PostgreSQL (Recommended):** Services like AWS RDS, Google Cloud SQL, or specialized providers like Neon offer automated backups, point-in-time recovery, and high availability. This removes the burden of managing WAL archiving, replication slots, and OS-level patching.  
* **Self-Hosted (Fallback):** If cost is a primary constraint, a single-node Docker deployment using an NVMe-backed instance is viable for the initial scale. However, this requires setting up automated backups (e.g., pg\_dump to S3 via cron) and monitoring (Prometheus/Grafana).  
  * *Configuration Note:* Ensure shared\_buffers is set to 25% of RAM and wal\_level is set to replica to support potential future replication.2

#### **2.2.2. Schema Evolution: Leveraging JSONB**

The migration presents a critical opportunity to transition from SQLite's TEXT based JSON storage to PostgreSQL's binary JSONB format.3

* **Performance:** JSONB stores data in a decomposed binary format, allowing the database to index specific keys. Xcelsior can create GIN (Generalized Inverted Index) indexes on the payload column.  
  * *Example:* CREATE INDEX idx\_hosts\_gpu\_specs ON hosts USING GIN ((payload-\>'gpu\_specs')); allows the scheduler to query GPU capabilities directly in SQL with millisecond latency, bypassing the Python layer entirely.4  
* **Partial Updates:** Unlike SQLite, where the entire JSON blob must be rewritten for a single field change, PostgreSQL supports jsonb\_set, reducing I/O overhead for frequent updates like "last\_seen" timestamps.5

### **2.3. Migration Strategy: Zero-Downtime Dual-Write Pattern**

To migrate without service interruption, Xcelsior must adopt an "Expand and Contract" strategy, implementing a dual-write pattern within the application layer.6

#### **Phase 1: Expansion and Schema Mirroring**

1. **Schema Replication:** Use Alembic to create the PostgreSQL schema. The payload columns in jobs and hosts should be defined as JSONB types.  
   * *Mapping:* Use SQLModel or SQLAlchemy to define models that map Python dict types to JSONB columns using sa\_column=Column(JSONB).7  
2. **Connection Setup:** Instantiate an AsyncEngine for PostgreSQL alongside the existing SQLite connection in scheduler.py.

#### **Phase 2: Dual-Writing (The "Shadow" Phase)**

Modify the \_atomic\_mutation logic to perform writes to both databases.

* **Primary (SQLite):** The application continues to read from and write to SQLite as the source of truth.  
* **Secondary (Postgres):** Every write operation (Insert/Update/Delete) is mirrored to PostgreSQL asynchronously.  
  * *Error Handling:* Failures in the secondary write should be logged but **must not** fail the request. This ensures that the migration process does not degrade current system reliability.  
  * *Code Implementation:* Wrap the Postgres write in a try/except block or dispatch it to a background task queue (e.g., FastAPI BackgroundTasks) to prevent latency impacts on the API.8

#### **Phase 3: Data Backfilling and Validation**

1. **Backfill:** Use a tool like pgloader 2 or a custom Python script to copy historical data (records created before Dual-Writing started) from SQLite to Postgres.  
   * *Type Casting:* Ensure boolean values (stored as 0/1 in SQLite) are correctly cast to PostgreSQL BOOLEAN types.9  
2. **Validation:** Implement a "comparator" process that reads a random sample of records from both databases and verifies equality. This is crucial to detect JSON serialization discrepancies between SQLite's text driver and Postgres's binary driver.

#### **Phase 4: The Cutover (Contract)**

1. **Read Switch:** Toggle a feature flag to switch read operations to PostgreSQL.  
2. **Write Switch:** Once reads are stable, make PostgreSQL the primary writer and stop writing to SQLite.  
3. **Cleanup:** Archive the SQLite database and remove the dual-write code logic.

### **2.4. Connection Pooling: Asyncpg vs. Psycopg3**

The choice of database driver significantly impacts performance in an async FastAPI environment.

* **Asyncpg:** Known for raw speed, it is often 3-5x faster than other drivers but lacks full DB-API compliance, making integration with ORMs like SQLAlchemy complex.10  
* **Psycopg3:** The modern, fully async iteration of the standard Python driver. While slightly slower in synthetic benchmarks than asyncpg, it offers native integration with SQLAlchemy 2.0, superior type adaptation, and simpler connection pooling.12  
* **Recommendation:** Use **Psycopg3**. For a small team, the developer experience and stability of SQLAlchemy integration outweigh the marginal throughput gains of asyncpg.  
  * *Pooling:* Configure SQLAlchemy with pool\_size=20 and max\_overflow=10 to manage connections efficiently under load.

### **2.5. Table: Comparison of Database Drivers for Xcelsior**

| Feature | Asyncpg | Psycopg3 | Recommendation |
| :---- | :---- | :---- | :---- |
| **Protocol** | Custom binary protocol | Libpq (C library) | **Psycopg3** (Standard compliance) |
| **Performance** | Extremely High | High | **Psycopg3** (Sufficient for Xcelsior) |
| **ORM Support** | Limited (requires adapters) | Native SQLAlchemy 2.0+ | **Psycopg3** (Best for SQLModel) |
| **Type Handling** | Manual conversion often needed | Automatic adaptation | **Psycopg3** |
| **Complexity** | High | Low | **Psycopg3** |

## ---

**3\. GPU Orchestration: Managing the Consumer Cloud**

Orchestrating workloads on consumer-grade GPUs (RTX 3060/4090) distributed across residential networks presents fundamentally different challenges than managing A100s in a data center. Xcelsior must account for variable reliability, lack of hardware-level virtualization, and restricted bandwidth.

### **3.1. Industry Comparison: RunPod, Vast.ai, and Kubernetes**

* **RunPod/Lambda:** These platforms primarily use **Kubernetes** to manage data center hardware. They rely on reliable, flat networks and enterprise features like networked storage for "FlashBoot" (instant container starts).14  
* **Vast.ai:** Operates a **decentralized marketplace** model similar to Xcelsior's target. They handle heterogeneity by benchmarking host machines and exposing "Interruptible" (spot) instances. They heavily utilize Docker directly rather than Kubernetes to reduce overhead on consumer hosts.15  
* **Kubernetes (K8s):** While K8s offers powerful scheduling (via plugins like DRA for MIG 16), its operational complexity (etcd, control plane, CNI plugins) is excessive for a 1-3 dev team managing residential nodes behind NAT.  
* **Recommendation for Xcelsior:** Adopt a **hybrid agent-based model**. Instead of a full K8s node agent, deploy a lightweight Python agent (similar to Vast.ai) that polls the central API. This avoids the complexity of K8s networking while allowing sophisticated local management.

### **3.2. Advanced Allocation Algorithms**

The current algorithm (free\_vram \-\> latency \-\> cost) 1 is insufficient for the performance variance in consumer cards. An RTX 3090 and RTX 4090 both have 24GB VRAM, but the 4090 offers \~2x the TFLOPS.17

#### **3.2.1. Compute-Aware Scoring**

Xcelsior must introduce a "Compute Score" (XCU \- Xcelsior Compute Unit) to normalize hardware capabilities.

* **Benchmarking:** Upon registration, the host agent should run a standardized benchmark (e.g., a short PyTorch inference run or gpu-burn test) to establish a baseline performance score.  
* **Algorithm Update:** The scheduler should filter by VRAM requirements but sort by (Compute Score / Price). This ensures users get the best "bang for their buck" rather than just the cheapest or emptiest card.

#### **3.2.2. GPU Sharing: MPS vs. Time-Slicing vs. MIG**

Consumer GPUs (GeForce) do **not** support Multi-Instance GPU (MIG), which provides hardware-level isolation on A100/H100 cards.18

* **Time-Slicing:** Kubernetes uses this default. It context-switches processes on the GPU. This is safe but causes high latency for real-time inference.19  
* **NVIDIA MPS (Multi-Process Service):** Allows multiple CUDA processes to run concurrently on the same GPU, sharing the CUDA cores.  
  * *Pros:* Higher throughput for small jobs (e.g., 3 separate 4GB inference jobs on a 24GB card).20  
  * *Cons:* **Zero Fault Isolation.** If one MPS client crashes the GPU, all clients crash.21  
* **Recommendation:** Use **MPS** strictly for a specific "Economy/Shared" tier where users accept lower reliability for lower cost. For "Premium" tiers, enforce exclusive GPU access (1 job per GPU) to guarantee stability.

### **3.3. Preemption and Spot Pricing Mechanics**

To maximize utilization, Xcelsior needs a mechanism to reclaim resources from low-priority jobs when high-priority demand spikes.

#### **3.3.1. Spot Pricing Algorithm**

Instead of a complex continuous double-auction, implement a **supply/demand multiplier curve**.22

* **Formula:** ![][image1]  
  * ![][image2]: The host's minimum ask price.  
  * ![][image3]: Demand (active jobs \+ queued jobs).  
  * ![][image4]: Supply (total available GPUs of that type).  
  * ![][image5]: Sensitivity coefficient (e.g., 0.5).  
* **Mechanism:** Prices update periodically (e.g., every 10 minutes). If a user's max bid falls below the current spot price, their job is marked for preemption.

#### **3.3.2. Preemption Workflow**

1. **Signal:** The scheduler identifies a job to preempt.  
2. **Grace Period:** The agent sends SIGTERM to the container. The container has a configurable window (e.g., 30 seconds) to checkpoint its state to shared storage or S3.  
3. **Termination:** If the process is still running, the agent sends SIGKILL.  
4. **Requeue:** The job is returned to the queue with a "paused" status, waiting for the price to drop or a cheaper host to become available.23

### **3.4. Container Image Caching Strategies**

Residential bandwidth is a major bottleneck. Downloading a 10GB PyTorch image can take 20+ minutes, destroying the "on-demand" experience.

* **Current State:** docker run pulls from Docker Hub every time if not present.  
* **Distributed Caching (Spegel/Dragonfly):** Tools like **Spegel** create a P2P registry within a K8s cluster.24 For Xcelsior's non-K8s environment, a simpler approach is needed.  
* **Recommendation: Predictive Pre-Caching.**  
  * **Tiered Cache:** The agent should maintain a local LRU (Least Recently Used) cache of the top 10 most popular images (e.g., pytorch/pytorch, stable-diffusion-webui).  
  * **Idle-Time Pulls:** The scheduler instructs agents to pull these "base images" during idle periods (e.g., 3 AM \- 6 AM local time) to ensure readiness without impacting active jobs.25  
  * **Layer Sharing:** Leverage Docker's layer caching by encouraging users to build upon standard base images managed by Xcelsior.

## ---

**4\. Container Runtime Security & Isolation**

Running arbitrary code on third-party consumer hardware creates a hostile security environment. The standard docker run \--gpus all command exposes the host kernel and GPU driver to the container, creating a large attack surface for container escapes and privilege escalation.26

### **4.1. Runtime Isolation: gVisor vs. Kata Containers**

To mitigate kernel-level escapes, Xcelsior must move beyond standard runc.

#### **4.1.1. Kata Containers**

Kata runs each container in a lightweight Virtual Machine (microVM) using QEMU or Firecracker.27

* **Pros:** Hardware-level isolation (separate kernel).  
* **Cons:** Requires **Nested Virtualization** support on the host (VT-x/AMD-V). This is often disabled in consumer BIOSes, unstable on gaming PCs, or unavailable in some cloud instances. It also incurs higher memory/startup overhead.

#### **4.1.2. gVisor (runsc)**

gVisor intercepts application system calls in user space, acting as a kernel proxy.

* **Pros:** Does not require nested virtualization. Runs anywhere standard Docker runs. Significantly reduces the attack surface by preventing direct interaction with the host kernel.28  
* **GPU Support:** Google has added support for GPU passthrough in gVisor, proxying CUDA calls safely.29  
* **Recommendation:** **gVisor** is the superior choice for Xcelsior's target hardware (heterogeneous consumer PCs). It offers a robust security improvement without the hardware dependency hell of Kata.

### **4.2. GPU Security Risks & Mitigations**

* **NVIDIAScape (CVE-2025-23266):** A recent vulnerability allows container escape via the NVIDIA Container Toolkit.26  
  * *Mitigation:* The Xcelsior agent must enforce strict version checks on the host's NVIDIA driver and container toolkit. It should refuse to schedule jobs on vulnerable hosts until updated.  
* **IOMMU Groups:** On consumer motherboards, IOMMU groups are often poorly separated. If a user breaks out of the container, they could potentially DMA (Direct Memory Access) into host memory via the GPU.  
  * *Mitigation:* Since full IOMMU isolation is hard to guarantee on consumer gear, the primary defense must be preventing the initial escape via gVisor and dropping capabilities (--cap-drop ALL).

### **4.3. Preventing Crypto-Mining Abuse**

Crypto-mining is the \#1 threat to GPU marketplaces. Attackers will rent cheap instances to mine Monero or Ethereum.

#### **4.3.1. Egress Filtering**

Miners must communicate with a mining pool (Stratum protocol) to receive work and submit shares.

* **Network Limits:** By default, containers should have **no outbound internet access** except to whitelisted domains (HuggingFace, GitHub, PyPI, AWS S3).  
* **Implementation:** Use Docker network drivers or iptables to drop all traffic on non-whitelisted ports. Specifically, block ports commonly used by pools (3333, 4444, etc.).30  
* **Protocol Detection:** Advanced miners tunnel Stratum over port 80/443. Xcelsior can deploy a lightweight eBPF probe (using tools like Tetragon or Falco) to inspect packet headers for Stratum signatures without full DPI overhead.31

#### **4.3.2. Telemetry & Heuristics**

Miners have a distinct resource usage fingerprint.

* **Power/Compute Ratio:** Mining keeps the GPU Compute utilization at 100% constant, while memory transfer is periodic. Training/Inference is "bursty" with high PCIe bus usage.  
* **Detection Logic:** The agent should poll nvidia-smi every 10 seconds. If GPU\_UTIL \> 95% AND PCIE\_TX \< Threshold for \> 5 minutes, flag the job for manual review or auto-pause.32

### **4.4. Network Architecture: Solving NAT Traversal**

The current SSH solution requires the host to have Port 22 open, which is impossible for many residential users behind CGNAT (Carrier-Grade NAT).

* **Recommendation:** Integrate **Tailscale** (or Headscale for a self-hosted control plane).  
* **Architecture:**  
  1. The Xcelsior Agent joins a specific Tailscale network (Tailnet) upon startup.  
  2. This creates a flat, encrypted mesh network. The scheduler can connect to the agent via its Tailscale IP (e.g., 100.x.y.z) regardless of the host's NAT situation.  
  3. **Security:** This removes the need for exposing *any* ports to the public internet. All control traffic flows over WireGuard tunnels.33

## ---

**5\. Real-Time Communication: Dashboards & Observability**

Users expect a "live" dashboard showing job progress, console logs, and GPU availability. Polling the API every second is unscalable and creates unnecessary load on the database.

### **5.1. Protocol Selection: SSE vs. WebSockets**

* **WebSockets:** True bi-directional communication. Excellent for games or chat, but complex to scale (requires maintaining stateful TCP connections). Firewalls often drop idle WebSocket connections.34  
* **Server-Sent Events (SSE):** Unidirectional (Server \-\> Client) over standard HTTP. Ideal for "feed" style updates (logs, status changes).  
  * *Browser Limit:* HTTP/1.1 limits browsers to 6 concurrent connections per domain. This is a blocker for multi-tab dashboards.  
  * *Solution:* Use **HTTP/2**, which multiplexes multiple streams over a single TCP connection, effectively removing the 6-connection limit.35

### **5.2. Recommended Architecture: SSE \+ Redis Pub/Sub**

For Xcelsior, the data flow is primarily one-way (System \-\> User).

1. **State Change:** When a job status changes (e.g., PENDING \-\> RUNNING), the scheduler writes to Postgres and publishes a message to a **Redis Channel** (e.g., user:123:updates).  
2. **API Delivery:** The FastAPI backend exposes an endpoint /api/stream. When a user connects, it subscribes to their specific Redis channel.  
3. **Streaming:** As messages arrive in Redis, FastAPI yields them as SSE events to the browser.  
4. **Benefits:** This decouples the heavy scheduling logic from the lightweight view logic. Redis handles the fan-out, allowing the API to scale horizontally.36

### **5.3. Performance & Benchmarks**

Benchmarks indicate that FastAPI with uvicorn can handle thousands of concurrent SSE connections with minimal memory footprint compared to WebSockets, which often require ping/pong frames and complex handshake logic.37 For a dashboard refresh rate of 1-5 seconds, SSE provides the optimal balance of simplicity and performance.

## ---

**6\. Implementation Roadmap & Migration Plan**

### **Phase 1: Foundation (Weeks 1-4)**

* **Database:** Provision Managed Postgres. Implement Dual-Write logic in scheduler.py.  
* **Schema:** Port SQLite schema to Postgres using Alembic/SQLModel.  
* **Agent:** Build the xcelsior-agent skeleton (polling loop, heartbeat).

### **Phase 2: Security & Orchestration (Weeks 5-8)**

* **Runtime:** Integrate gVisor (runsc) and update Docker launch commands.  
* **Networking:** Integrate Tailscale/Headscale into the agent for NAT traversal.  
* **Allocation:** Implement "Compute Score" benchmarking in the agent.

### **Phase 3: Real-Time & Polish (Weeks 9-12)**

* **Observability:** Deploy Redis. Implement SSE endpoint in API. Update frontend to use EventSource.  
* **Cutover:** Validate data consistency. Stop SQLite writes. Remove legacy code.

## ---

**7\. Conclusion**

By executing this architectural transition, Xcelsior will evolve from a fragile prototype into a resilient platform capable of competing in the distributed compute market. The move to PostgreSQL eliminates the concurrency ceiling, while the shift to a pull-based agent architecture with Tailscale solves the residential networking quagmire. Most importantly, integrating gVisor and rigorous egress filtering transforms the security posture from "hope-based" to "defense-in-depth," enabling the safe utilization of untrusted consumer hardware. This roadmap balances technical rigor with the resource realities of a small team, ensuring steady, manageable progress toward production readiness.

## ---

**8\. Deep Dive: Technical Specifications**

### **8.1. PostgreSQL Dual-Write Implementation (Pseudo-Code)**

Python

\# scheduler.py \- Migration Logic

async def \_save\_job\_state(job\_id: str, data: dict):  
    """  
    Writes to SQLite (Primary) and Postgres (Secondary).  
    Failures in Postgres are logged but do not block the response.  
    """  
    \# 1\. Primary Write: SQLite (Blocking/Critical)  
    try:  
        with \_atomic\_mutation\_sqlite() as conn:  
            conn.execute("INSERT OR REPLACE INTO jobs...",...)  
    except Exception as e:  
        log.error(f"SQLite Write Failed: {e}")  
        raise e \# Fail the request if primary DB fails

    \# 2\. Secondary Write: Postgres (Non-Blocking/Best Effort)  
    try:  
        \# Ideally dispatched to a background task  
        await postgres\_db.upsert\_job(job\_id, data)  
    except Exception as e:  
        \# Log for reconciliation later, do not crash  
        log.warning(f"Postgres Dual-Write Failed for {job\_id}: {e}")

### **8.2. Spot Pricing Formula**

![][image6]

* Where ![][image7] is the utilization threshold (e.g., 0.8 or 80%). Pricing only surges when demand exceeds 80% of supply.

### **8.3. Crypto-Mining Detection Heuristic (Agent-Side)**

Python

def check\_mining\_abuse(gpu\_id):  
    stats \= nvidia\_smi.get\_stats(gpu\_id)  
      
    \# Mining Signature: High Compute, High Memory, Low PCIe  
    if (stats.utilization \> 98 and   
        stats.memory\_util \> 90 and   
        stats.pcie\_tx \< 5\_MB\_s):  
          
        return True \# Likely Mining  
    return False

This heuristic leverages the fact that legitimate training jobs typically involve significant data shuffling (high PCIe usage) and fluctuating compute, whereas hashing algorithms pin the core and memory controller constantly.

#### **Works cited**

1. scheduler.py  
2. How to migrate from SQLite to PostgreSQL \- Render, accessed February 12, 2026, [https://render.com/articles/how-to-migrate-from-sqlite-to-postgresql](https://render.com/articles/how-to-migrate-from-sqlite-to-postgresql)  
3. JSON vs. JSONB in PostgreSQL: A Complete Comparison \- DbVisualizer, accessed February 12, 2026, [https://www.dbvis.com/thetable/json-vs-jsonb-in-postgresql-a-complete-comparison/](https://www.dbvis.com/thetable/json-vs-jsonb-in-postgresql-a-complete-comparison/)  
4. Searching Postgres JSON columns using SqlAlchemy | by Mark Jones | Code on a Boat, accessed February 12, 2026, [https://medium.com/code-on-a-boat/searching-postgres-json-columns-using-sqlalchemy-aece6ae5b0e9](https://medium.com/code-on-a-boat/searching-postgres-json-columns-using-sqlalchemy-aece6ae5b0e9)  
5. Updating PostgreSQL JSON fields via SQLAlchemy \- \- Benjamin Shelton's Musings, accessed February 12, 2026, [https://bashelton.com/2014/03/updating-postgresql-json-fields-via-sqlalchemy/](https://bashelton.com/2014/03/updating-postgresql-json-fields-via-sqlalchemy/)  
6. Lessons from Migrating a Live Postgres Schema Without Downtime \- DEV Community, accessed February 12, 2026, [https://dev.to/leena\_malhotra/lessons-from-migrating-a-live-postgres-schema-without-downtime-338c](https://dev.to/leena_malhotra/lessons-from-migrating-a-live-postgres-schema-without-downtime-338c)  
7. Is it possible to make a field of type JSON · fastapi sqlmodel ... \- GitHub, accessed February 12, 2026, [https://github.com/fastapi/sqlmodel/discussions/1105](https://github.com/fastapi/sqlmodel/discussions/1105)  
8. Zero-Downtime PostgreSQL Migration: My Journey with DevCycle \- DEV Community, accessed February 12, 2026, [https://dev.to/jacksonkasi/zero-downtime-postgresql-migration-my-journey-with-devcycle-23lg](https://dev.to/jacksonkasi/zero-downtime-postgresql-migration-my-journey-with-devcycle-23lg)  
9. How to convert SQLite SQL dump file to PostgreSQL? \- Stack Overflow, accessed February 12, 2026, [https://stackoverflow.com/questions/4581727/how-to-convert-sqlite-sql-dump-file-to-postgresql](https://stackoverflow.com/questions/4581727/how-to-convert-sqlite-sql-dump-file-to-postgresql)  
10. Understanding benchmarking and setting up Async (Postgres+asyncpg vs Postgres+psycopg2) in FastAPI \#13732 \- GitHub, accessed February 12, 2026, [https://github.com/fastapi/fastapi/discussions/13732](https://github.com/fastapi/fastapi/discussions/13732)  
11. Psycopg 3 vs Asyncpg \- fernandoarteaga.dev, accessed February 12, 2026, [https://fernandoarteaga.dev/blog/psycopg-vs-asyncpg/](https://fernandoarteaga.dev/blog/psycopg-vs-asyncpg/)  
12. Psycopg2 vs Psycopg3 Performance Benchmark | Tiger Data, accessed February 12, 2026, [https://www.tigerdata.com/blog/psycopg2-vs-psycopg3-performance-benchmark](https://www.tigerdata.com/blog/psycopg2-vs-psycopg3-performance-benchmark)  
13. FastAPI, Pydantic, Psycopg3: The Ultimate Trio for Python Web APIs, accessed February 12, 2026, [https://spwoodcock.dev/blog/2024-10-fastapi-pydantic-psycopg/](https://spwoodcock.dev/blog/2024-10-fastapi-pydantic-psycopg/)  
14. Runpod vs Lambda vs Northflank: GPU cloud platform comparison | Blog, accessed February 12, 2026, [https://northflank.com/blog/runpod-vs-lambda-northflank](https://northflank.com/blog/runpod-vs-lambda-northflank)  
15. Runpod vs. Vast AI: Which Cloud GPU Platform Is Better for Distributed AI Model Training?, accessed February 12, 2026, [https://www.runpod.io/articles/comparison/runpod-vs-vastai-training](https://www.runpod.io/articles/comparison/runpod-vs-vastai-training)  
16. Kubernetes GPU Sharing: NVIDIA MIG \+ DRA on Amazon EKS \- \_CLOUD, accessed February 12, 2026, [https://blog.ediri.io/kubernetes-gpu-sharing-nvidia-mig-dra-on-amazon-eks](https://blog.ediri.io/kubernetes-gpu-sharing-nvidia-mig-dra-on-amazon-eks)  
17. GPU Benchmarks NVIDIA RTX 3090 vs. NVIDIA RTX 4090 \- Bizon-tech, accessed February 12, 2026, [https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3090-vs-NVIDIA-RTX-4090/579vs637](https://bizon-tech.com/gpu-benchmarks/NVIDIA-RTX-3090-vs-NVIDIA-RTX-4090/579vs637)  
18. Compare with MIG/MPS/Timeslicing | Maximize your GPU usage, powered by GPU virtualization and pooling, fractional GPU, Remote GPU Sharing, GPU over IP \- Tensor Fusion, accessed February 12, 2026, [https://tensor-fusion.ai/guide/comparison/compare-with-mig-mps](https://tensor-fusion.ai/guide/comparison/compare-with-mig-mps)  
19. Time-Slicing GPUs in Kubernetes — NVIDIA GPU Operator, accessed February 12, 2026, [https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)  
20. MPS better than MIG and Time-Slicing for performing inference on a shared GPU? \- Reddit, accessed February 12, 2026, [https://www.reddit.com/r/kubernetes/comments/1190utm/mps\_better\_than\_mig\_and\_timeslicing\_for/](https://www.reddit.com/r/kubernetes/comments/1190utm/mps_better_than_mig_and_timeslicing_for/)  
21. Share GPUs with multiple workloads using NVIDIA MPS | GKE AI/ML, accessed February 12, 2026, [https://docs.cloud.google.com/kubernetes-engine/docs/how-to/nvidia-mps-gpus](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/nvidia-mps-gpus)  
22. New Amazon EC2 Spot pricing model: Simplified purchasing without bidding and fewer interruptions | AWS Compute Blog, accessed February 12, 2026, [https://aws.amazon.com/blogs/compute/new-amazon-ec2-spot-pricing/](https://aws.amazon.com/blogs/compute/new-amazon-ec2-spot-pricing/)  
23. Instance Types \- Vast.ai Documentation – Affordable GPU Cloud ..., accessed February 12, 2026, [https://docs.vast.ai/documentation/instances/choosing/instance-types](https://docs.vast.ai/documentation/instances/choosing/instance-types)  
24. spegel-org/spegel: Stateless cluster local OCI registry mirror. \- GitHub, accessed February 12, 2026, [https://github.com/spegel-org/spegel](https://github.com/spegel-org/spegel)  
25. How to Implement Docker Layer Caching Strategies \- OneUptime, accessed February 12, 2026, [https://oneuptime.com/blog/post/2026-01-30-how-to-implement-docker-layer-caching-strategies/view](https://oneuptime.com/blog/post/2026-01-30-how-to-implement-docker-layer-caching-strategies/view)  
26. NVIDIAScape: Breaking Container Isolation with CVE-2025-23266 in NVIDIA Container Toolkit \- ZeroPath Blog, accessed February 12, 2026, [https://zeropath.com/blog/nvidiascape-cve-2025-23266-nvidia-container-toolkit-escape](https://zeropath.com/blog/nvidiascape-cve-2025-23266-nvidia-container-toolkit-escape)  
27. Kata Containers vs Firecracker vs gVisor: Which container isolation tool should you use?, accessed February 12, 2026, [https://northflank.com/blog/kata-containers-vs-firecracker-vs-gvisor](https://northflank.com/blog/kata-containers-vs-firecracker-vs-gvisor)  
28. Kata Containers vs gVisor? \- kubernetes \- Stack Overflow, accessed February 12, 2026, [https://stackoverflow.com/questions/50143367/kata-containers-vs-gvisor](https://stackoverflow.com/questions/50143367/kata-containers-vs-gvisor)  
29. gVisor: The Container Security Platform, accessed February 12, 2026, [https://gvisor.dev/](https://gvisor.dev/)  
30. Docker Engine v28: Hardening Container Networking by Default, accessed February 12, 2026, [https://www.docker.com/blog/docker-engine-28-hardening-container-networking-by-default/](https://www.docker.com/blog/docker-engine-28-hardening-container-networking-by-default/)  
31. Detecting Cryptojacking Containers Using eBPF-Based Security Runtime and Machine Learning \- MDPI, accessed February 12, 2026, [https://www.mdpi.com/2079-9292/14/6/1208](https://www.mdpi.com/2079-9292/14/6/1208)  
32. Monitoring GPU utilization for Deep Learning | DigitalOcean, accessed February 12, 2026, [https://www.digitalocean.com/community/tutorials/monitoring-gpu-utilization-in-real-time](https://www.digitalocean.com/community/tutorials/monitoring-gpu-utilization-in-real-time)  
33. The SSH tunnel is dead, long live Tailscale | blog.kronis.dev, accessed February 12, 2026, [https://blog.kronis.dev/blog/the-ssh-tunnel-is-dead-long-live-tailscale](https://blog.kronis.dev/blog/the-ssh-tunnel-is-dead-long-live-tailscale)  
34. WebSockets vs Server-Sent Events: Key differences and which to use in 2024 \- Ably, accessed February 12, 2026, [https://ably.com/blog/websockets-vs-sse](https://ably.com/blog/websockets-vs-sse)  
35. Real-Time Notifications in Python: Using SSE with FastAPI | by İnan DELİBAŞ | Medium, accessed February 12, 2026, [https://medium.com/@inandelibas/real-time-notifications-in-python-using-sse-with-fastapi-1c8c54746eb7](https://medium.com/@inandelibas/real-time-notifications-in-python-using-sse-with-fastapi-1c8c54746eb7)  
36. Real-Time Features in FastAPI: WebSockets, Event Streaming, and Push Notifications, accessed February 12, 2026, [https://python.plainenglish.io/real-time-features-in-fastapi-websockets-event-streaming-and-push-notifications-fec79a0a6812](https://python.plainenglish.io/real-time-features-in-fastapi-websockets-event-streaming-and-push-notifications-fec79a0a6812)  
37. SSE vs WebSocket Performance: Surprising Results Using 100 ..., accessed February 12, 2026, [https://www.youtube.com/watch?v=d0\_WugmrNBg](https://www.youtube.com/watch?v=d0_WugmrNBg)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARQAAAAYCAYAAAA7xQK2AAAHKElEQVR4Xu2cZ4glRRCAy5zPnBVPzxwOFcMf/wiiGFE4VBBFRcwR74cJMWDAhAlUDJgVxYSKYuLMeigYwCx6KIqnnp45a3/21G6/uu6emd1988LNB8XOVPV7M10zXd1d3W9FWlpausnVTv51cquTK5zcWZzvHhZqaWlpqcJC4gOIBd3CVtnSMmxsaBUtpeR8dpKT363S8Y2TC62yT9jfKgJWd7KgVba0xIj1pC15ynw2x8m5Vin+c7dbZZ+wnpM3rTKgrM4jXO5krvgPIL+Kd0iou2ek9PAx08lPMlrXn5186+THQHfeSOnh4isnK1tlwG1ONrHK+ZwynwHvzBJWKV6fGwk0wXQnR1llAfme66yyYFEn/1hlDm08lnXE65+whiGDOv5plY59xNtOs4YB5yAns63S8YL4F0ffh007zfM1KZ+FMDWItaMpEtc3wV1O/pDRZ3p0p7mD3D2+5ORSq0zBF71slQWpYDMsrCm+fhdYQ8Ew1p/6rGaVAVvL+APKi1Yx4JT5DE4W33gt6Pa0yh5QFlAYlaamPqlk8zwwDKPgLtYgfug2jA0q5Frx9VvaGhw7ibd9bQ0Ns71VRNjYKhKsIeXPcyICyitW0TBN+wyYKtsp8iwnhxpdrygLKMtKvp7YaBNZ3pH0lzwg3ra3NQwRuYDJnBnbOtbQMCw35uawqfuPwf6IWC8aMhEB5TWrGAPLO7nJycUSz0vkaNJndEZ/yei7xDHlyVEuEJTrNWUBBSgzySoL3hOfd8ySalA7iteTrBlmqGPsZTlOvI25cz9AYiz2nGK6HH87ucMqDRMRUF63iprQ299bHC8u/n50FL2Ik2nFcY4mfTYIVA0op1plAbnEUt9pQPneyXdOfivO33ayYlCuVzCvS8ktTm4W34vd6OR6J+v//6lqrCq+rvRk1B0faAJrhvTfRiTbQEofbgQ+E1vWDNGAspk11GA8AYVrMzoO0akp3BcaSmjKZ4MA9TjGKg2UoV3F0PRIEi3Q9BxvMavoEfqSsg6fgsz901LiyAbRBoKMZTjN5w6xSoMGlC2sIQJTkW0i8m5Ep5KDQBTztY6YgRFDHZrw2URi/ZWTOlAPRt452DqSyn+Rl4o9mxHel5ICXaIX14yhL1kZl4kPPv2ALk1Wue8YfO5gqzRoQJlqDRGWc7JXRD6K6FRycN3YEj5BHxtTD0aWdei2z/S7q0oZ1l85qQPXPt4qDezFSo0ut5WS+69awYmERFsuWWa5qKbUGaZXrT89YtmSYROE+xwqL+MZ+MzZVmnQgLKlNdQg9VKWwXVJZlpWEW9L9Z4pmvLZIEA9TrRKA2Xut8qC/aTEfxjpSapAzuJB6UxgMhVglMPvF8hjhHMvEmnsOmW78ReFThuwSi+ZLP4ebjD6GJRjX8WT4p2qTHfyiJNPnDwc6GFXJx84+Vw6VyjOFO9zVkFYpqsKQ3Xrs7E0EMrzLHNoQNnKGmownoBygFWK74iwbW4NGZr0WYy1g+M6z7pbUA/aag7KpDZyniIZ350u3niENURgCzZLZ6Cji2uKv3yHzkt/EJ/IZBktvDDD1POLY3p7Xo5e87j4e9zIGgyTpLMuenys+AALNvEHek7dNQi95eSw4hjOCI7LsN+vpHZnpuDn9CTec+wm/jv3sIYajDWgfCa+kwqhV/1Y/D0d6GQt8XtDykj5pRs+C+Gd+VJ8OyDPwaiq8i7TLrGS+DpfYg0GyqTaJws1b1jlVeKX5FjVmCN+zlQ2BeHXlVyIYMCNKTzUcL5Lmani1+GnBXoani4B1nmQ3YCdgAQ+6o4PfpF8ko9fiLJ6pOj9h/Ugqf1UcA7YkcOLc+0ZCWSzZTTAViG26TCEgM78tgqTJf0MuC/23jCqomHzF92ssFBFxhpQQJPgCPegORMSo+h0xJujKZ/FsGUZ3aQaabeh3bExM3ymPOPYVomlZN57D8FW5tdKLOlkGfHRLbwgPyYKt6zHGpuebyA+gVcWvPoNXt51i2Pdpq9/lU+d7BCc6xSHQKPltguOew330e0XfDwBpR+p6jPecfuc2Zg3CLDwkFqSj00fx8TO0hkE6N0VLsCIBOhxdUNMeGGGfPwvCDhHRqdKjxV/+x0SVJqQJcLrHD6sox4/ZM5zx0wN2TvTC04QX5duwrRimKjjM54zy6/k2DQdMAjkAgb5wyutcqyw4esZJ88ZPTfA/PBD8c5TaIDYSDyeFehpRASkGYFuEGBax8sU5luYOs4U/y//SDy/GtgYsuMXtiqHCdkjxQ8/mWqxB6iXMC1l1NlSnao+I+fIO0MbQJhK9DssFtjfISl0Drm0wIRABhsHtwwuuR6pJU5dnzG6T62a9AvkqOj8UnQ9mJC9fl58snUFY2sZHBiOk9tqqU7OZ+TJLIxUq6xI9ZLcZrcp0j8721ta5isedbKv0bGi2tLS0lKbueL3d5FTvNvJs53m4eM/XbAG9ozjU6oAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAYCAYAAAB5j+RNAAABiElEQVR4Xu2WTStFURSGV5FiYGIsJTIwMzAzMKJEMsBIUWLkD1DyD3wMGF4lAxMTRZgbiBL56F4ykIHkI/kY4V322tc6q3NKbuecyXnq6ez9rl173X33PRBlZKTLLHyGX+I7fDDZWnF1SvhGLHXk8m1bSBJuYM+GQlTjiTBIbvMOWwCVlHJzpxS9+Tq5Wq8tJEXUybSTy+dsIUl8c0/wEX7I/BjWqHWJ4+/biC0Ylin8dGPlgv62aQ88t2HcRN03yyEcsGHccGMFG4bA61bFGZV3wX14A+dVzjTCe3gEm1TOH5LzLdii8gCT5DYds4UQ9On6cSu5pmxu56NwUcY5uCJjZleNf1iAL+R+mfx39BV+BlYE6YR5NedNK+RZK1k9fCuucPgrs2SyTXhFwSb/DX+Vw2ruT0SfVA5OqXmZPPvIrWuWuT3dkpmA/TLeoN9rYL/qKnin5uUyvpanzzV8H0uGX9BnsFtlQ+ReRTtwHB7Aaqnxv2In8BI2SMa0kXvJ38JplWdkpMI3MIRqWri52sYAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAXCAYAAADtNKTnAAAAsUlEQVR4XmNgGAWEQBcQfwTi/1D8HYjfoYldh6smAGAasIGfDLjlUABI0SF0QSjgYYDIN6CJo4AIBogiR3QJJIDPpWBwjYGAAgYiDCGogIEINSDJA+iCSMCNAaIGZyzBwsMBTRwZ3GaAqBFDl4ABQs40ZIDI16FLIAOQAlA6wAVA8k/QBZGBCgNEUTO6BBDIMUDk1qFLwEAgEJ9kQHjlDhAfh+KzUDFQ0jeFaRgFIw4AAFhqNpdzGLpuAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAAqklEQVR4XmNgGLJADYhnArEvklgJEhsFsALxPyCeDcR8QGwHxP+BuAaIPyOpQwEgBTboggwQ8Sp0QRBYwACRxAZA4iBXYACQBD5NWAFMUy+6BD7QzYDQCMMzUFTgAHkMmBpvoaggAFwY8PuTIRhdAAoWM+DQ5AfEBeiCUFDKgEPTWSBehy4IBX8ZcAQGzN08aOJrGfAknSdAzATEHxggmt9D6QVIakbBwAEAIrItoSGpzDcAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAAAiklEQVR4XmNgGAXDAswG4v9AfABK16HIooFfQHwBTQykyRHK/o0scYMBIokOQGIgg0BgN7rEc2QBKPjHAJEzB+JomKADVNAdJoAEHjFA5FBsX40ugASuMUDkJJEFG6CC2MBFBhxyIEFVNLF7QLwWKgcCfUhyDEJA/JcB4d4ZSHL3oWLxSGKjgLoAACgOJkiAxK1GAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAXCAYAAADduLXGAAAAXUlEQVR4XmNgGAX0BpxA7AzEHkDsBcUgtguyIhDYDcT/8WBXmMJcIF4M4wDBRiCWQOKjAGs0PsgkooADAwmK7wHxAXRBXABkaiy6IDYgxQBRzIwugQuEowuMAlwAAOEOEm1iyv5jAAAAAElFTkSuQmCC>