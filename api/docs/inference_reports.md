<!-- residency-guard: documents-removal — dated report -->
Strategic Architecture and Implementation of Serverless Inference Endpoints1. Executive Summary and the Paradigm Shift in Cloud GPU ComputingThe proliferation of Large Language Models (LLMs), multimodal generation frameworks, and advanced artificial intelligence inference tasks has fundamentally altered the computational demands placed on cloud infrastructure. Historically, deploying a high-performance machine learning model required provisioning static, bare-metal graphics processing units (GPUs) or persistent virtual machines. This legacy paradigm inherently suffers from severe resource utilization inefficiencies. Because inference workloads are notoriously bursty—experiencing massive spikes in traffic followed by periods of complete dormancy—maintaining persistent GPU clusters results in exorbitant idle compute costs. The serverless GPU endpoint architecture emerged as the definitive solution to this utilization crisis, encapsulating containerized inference workloads behind scalable application programming interfaces (APIs) and dynamically modulating the active worker pool in strict accordance with real-time demand.The modern serverless inference ecosystem is characterized by an intricate orchestration of control planes, data planes, and worker execution environments. Unlike traditional serverless functions (such as AWS Lambda), which process lightweight, transient data manipulation tasks, serverless GPU workers must handle colossal model weights, persistent tensor states, and continuous token streams over prolonged execution windows. The engineering complexity lies in mitigating "cold starts"—the severe latency penalty incurred when downloading multi-gigabyte neural network weights into a newly provisioned container—while maintaining a highly concurrent, zero-drop request queue.To architect a state-of-the-art serverless inference platform, one must synthesize the optimal design patterns of established market competitors, analyze specialized decentralized topologies such as the xcelsior distributed scheduler repository, and deploy a rigorously tested technology stack. This exhaustive report dissects the technological landscape, recommending the definitive packages and frameworks necessary for production deployment. Furthermore, it provides a highly detailed, step-by-step implementation plan covering the overarching system architecture, the mathematical and operational mechanics of the autoscaler, the stateful API gateway, and the frontend integrations required to construct a comprehensive, enterprise-grade serverless inference endpoint platform.2. Exhaustive Competitor Analysis and Architectural PrecedentsDeriving a superior technological blueprint requires a meticulous forensic analysis of the architectural paradigms deployed by existing market leaders. The serverless GPU market is presently stratified into several distinct operational philosophies, ranging from centralized handler-based execution and disaggregated proxy routing to highly experimental decentralized scheduling protocols. Analyzing the specific implementation details, storage strategies, and scaling metrics of competitors like RunPod, Vast.ai, and Novita AI provides the foundational knowledge required to engineer a superior alternative.2.1 RunPod: Handler-Based Architectures and Persistent State CachingRunPod operates on a centralized architecture heavily reliant on a proprietary Python Software Development Kit (SDK) and a handler-based execution methodology. In this paradigm, developers define a specific Python handler function tasked with processing incoming payload dictionaries. The RunPod serverless framework wraps this handler function within a Dockerized worker, executing the logic sequentially as jobs are pulled from a centralized queue. The RunPod platform distinguishes itself through its aggressive optimization of storage persistence and container lifecycle management.To combat the cold start latency inherent to massive language models, RunPod implements persistent Network Volumes backed by high-performance Non-Volatile Memory Express (NVMe) solid-state drives. These network volumes provide persistent, portable storage that exists entirely independent of the compute resources. By mounting these volumes directly to the /runpod-volume directory within the serverless worker container, the architecture effectively eliminates the need to download model weights from external repositories on every container boot. The inference engine simply loads the tensors directly from the locally mounted network file system into the GPU's Video Random Access Memory (VRAM), a process that operates at transfer speeds between 200 megabytes and 10 gigabytes per second.Furthermore, RunPod employs a mechanism known as FlashBoot, which reduces cold starts by retaining worker state in system memory after the worker spins down, allowing for near-instantaneous revival compared to a fresh container boot. RunPod's API design also introduces a rigid dualism between asynchronous and synchronous execution. Synchronous requests (/runsync) hold the HTTP connection open and return the output directly, featuring a strict one-minute result retention policy. Asynchronous requests (/run), designed for longer generative tasks, return a unique job identifier immediately, allowing the client to poll a status endpoint while the platform retains the final results for thirty minutes. However, this architecture is not without flaws; developer discussions highlight architectural edge cases within the RunPod Python package, such as unhandled exceptions when asynchronous generators fail to stream errors correctly, or KeyError crashes triggered when a handler inadvertently returns an empty dictionary object.2.2 Vast.ai: Disaggregated Proxies and Benchmark-Driven ScalingIn stark contrast to the centralized payload routing of RunPod, Vast.ai leverages a disaggregated, multi-component architecture centered around a custom Python web server wrapper known as the PyWorker. The Vast.ai Serverless Engine acts strictly as a decision-making control plane, separating the routing logic from the actual data transmission.When a client initiates an inference request via the Vast.ai Serverless SDK, the client application first sends a lightweight routing request to the central serverless system. The Serverless Engine evaluates the current load, capacity, and queue times across all managed endpoints, selects an available GPU instance, and returns the direct IP address and connection parameters of that specific worker to the client. The client application then transmits the heavy inference payload directly to the selected worker's API endpoint, entirely bypassing the central control plane. This architectural decision prevents the central load balancer from becoming a network bandwidth bottleneck during high-throughput multimodal operations.The PyWorker utility, running on the target GPU instance alongside the machine learning model, acts as an intelligent proxy. It validates the incoming client payload, strips away the authentication wrapper, and forwards the pure inference parameters to the localized model server (such as Text Generation Inference or vLLM). Concurrently and independently of the inference workflow, the PyWorker periodically aggregates operational metrics—including hardware utilization and local queue depth—and reports these statistics back to the Serverless Engine.Vast.ai also innovates through its benchmark-driven autoscaling methodology. Rather than scaling indiscriminately across homogeneous hardware, the autoscaler automatically evaluates the price-to-performance ratio across Vast.ai's massive, mixed-hardware marketplace. It recruits the most cost-efficient GPUs available that meet the workload's minimum compute requirements, drastically reducing the financial overhead of burst scaling.2.3 Novita AI: High-Performance Infrastructure and Serverless for Frontend (SFF)Novita AI positions itself as a robust, enterprise-grade foundation model service platform, focusing heavily on reducing the operational complexity of deploying models like Llama 3. Novita's infrastructure emphasizes the elimination of maintenance overhead by allowing developers to specify public or private Docker image repositories (such as the official vllm/vllm-openai:latest image) and mapping them directly to standardized hardware profiles, predominantly leveraging RTX 4090 24GB instances for cost-effective inference.A critical architectural paradigm highlighted in the Novita AI ecosystem is the transition from the traditional Backend For Frontend (BFF) pattern to the Serverless For Frontend (SFF) model. Traditionally, managing Node.js applications to orchestrate data between backend microservices and frontend clients required dedicated virtual machines and continuous resource allocation. The SFF model replaces these persistent Node.js environments with transient, event-driven serverless functions operating on a run-to-completion basis. When a frontend client initiates a data request, the function trigger activates the serverless service, invokes the necessary backend metadata interfaces, processes the payload into a consumable format, and immediately terminates. This approach drastically reduces the architectural footprint of the API layer bridging the inference endpoint and the end-user application.2.4 The Xcelsior Model: Decentralized and Geographic-Aware SchedulingWhile hyperscalers dominate the commercial market, specialized open-source repositories provide invaluable insight into alternative scheduling philosophies. The xcelsior project, authored by user aabiro and described as a "distributed GPU scheduler for Canadians who refuse to wait," introduces a localized, partition-tolerant approach to task orchestration.The defining characteristic of a distributed scheduler like xcelsior is its rejection of the monolithic, centralized data center model in favor of geographic awareness and latency optimization. In environments where users (such as specific regional demographics) face unacceptable queue times on global platforms, a distributed scheduler pools decentralized hardware assets, routing requests based on topological proximity and immediate hardware availability rather than strict global load balancing.Furthermore, concepts adjacent to the xcelsior paradigm include tokenized, decentralized compute platforms such as the DeGPUhub hackathon project. Such architectures integrate smart contracts and tokenized governance (e.g., compute tokens) to manage trustless interactions between GPU providers listing their hardware and end-users submitting inference workloads. While implementing a full blockchain consensus mechanism may exceed the scope of a standard enterprise endpoint, the underlying principle of the xcelsior model—designing the autoscaler and queue manager to treat worker nodes as ephemeral, untrusted, geographically dispersed assets—is highly applicable to building a resilient, fault-tolerant serverless architecture.Architectural ParadigmPrimary Competitor/ProjectCore Operational MechanicsStrategic AdvantagesCentralized Handler RoutingRunPodCentral API ingress, Python handler wrappers, Network Volume caching.Zero-configuration for developers, instantaneous FlashBoot revivals, simplified state management.Disaggregated Proxy RoutingVast.aiCentral routing decisions, direct client-to-worker payload delivery, PyWorker proxy.Eliminates central bandwidth bottlenecks, enables benchmark-driven cost optimization across heterogeneous hardware fleets.Serverless For Frontend (SFF)Novita AIRun-to-completion intermediate functions replacing dedicated Node.js orchestration layers.Removes middle-tier infrastructure overhead, streamlines metadata processing before inference execution.Distributed Geographic SchedulingXcelsior RepositoryLatency-optimized routing to decentralized, potentially untrusted node clusters.Bypasses centralized queue bottlenecks, minimizes regional network latency, supports highly distributed hardware pooling.3. Foundational Technology Stack and Engine SelectionConstructing the serverless inference platform requires selecting software packages that offer unparalleled concurrent throughput, native asynchronous support, and robust state persistence without introducing unnecessary orchestration overhead.3.1 Inference Backend: vLLMFor the execution of Large Language Models, vLLM is unequivocally the industry standard and the optimal choice for the worker node inference engine. vLLM was engineered specifically to bridge the gap between experimental model research and high-throughput production deployment by solving the memory fragmentation crisis inherent to traditional inference scripts.The technological cornerstone of vLLM is PagedAttention. During autoregressive token generation, the model must cache the key and value (KV) vectors for all previously generated tokens to avoid recalculating them. In traditional setups, this KV cache is allocated in contiguous blocks of GPU memory based on the maximum possible sequence length. Because actual sequence lengths are highly variable and unpredictable, this static allocation results in massive internal memory fragmentation, severely limiting the number of requests a single GPU can process simultaneously. PagedAttention fundamentally resolves this by managing the KV cache utilizing methodologies derived from operating system virtual memory paging. It partitions the continuous logical KV cache into non-contiguous physical blocks within the VRAM. This dynamic allocation eliminates internal fragmentation, allowing the system to handle dramatically larger concurrent batch sizes.Coupled with PagedAttention is vLLM's implementation of continuous batching. Rather than waiting for all sequences within a static batch to yield an End-of-Sequence (EOS) token, continuous batching dynamically ejects completed sequences and instantly injects new pending requests into the active batch during the very next forward pass. This guarantees maximum GPU utilization at all times. Furthermore, vLLM provides native OpenAI API compatibility (vllm serve), enabling immediate integration with existing tooling, and supports an --enable-offline-docs flag to bypass internet requirements for the FastAPI Swagger UI in air-gapped serverless deployments.3.2 API Gateway and Concurrency Manager: FastAPIFastAPI is a modern, exceptionally fast Python web framework built upon Starlette and Pydantic, making it the premier choice for the central control plane API. FastAPI natively supports asynchronous request handling via Python's asyncio event loop, which is mandatory for managing the thousands of concurrent polling connections generated by clients checking the status of their serverless jobs.FastAPI allows for seamless integration of sophisticated background tasks, dependency injection for database connection pooling, and automatic generation of OpenAPI specifications. Crucially, FastAPI supports asynchronous generator functions and StreamingResponse objects, which are required to stream partial token generations back to the user via Server-Sent Events (SSE). Within the context of the vLLM integration, the architecture leverages specific decorators such as with_cancellation. This decorator binds the lifecycle of the inference generation task directly to the client's HTTP connection state. If a client abruptly disconnects or closes their browser before the generation is complete, the with_cancellation decorator captures the disconnect event and immediately propagates a cancellation signal to the vLLM engine, halting processing and instantly freeing the locked PagedAttention KV cache blocks.3.3 State Management and Queueing: PostgreSQL SKIP LOCKEDIn traditional microservice architectures, job queues are almost exclusively managed by dedicated in-memory brokers such as Redis, RabbitMQ, or Apache Kafka. However, for a serverless inference architecture where job payloads contain substantial context windows and require strict durability, utilizing a native PostgreSQL database as the primary queue mechanism provides superior architectural simplicity and robust Atomicity, Consistency, Isolation, and Durability (ACID) guarantees.The cornerstone of this approach is the FOR UPDATE SKIP LOCKED SQL extension. When multiple worker nodes simultaneously poll the database looking for pending jobs, standard row-level locking would cause severe lock contention, forcing workers to wait in a blocking queue while a single transaction processes. The SKIP LOCKED directive bypasses this entirely. When a worker executes a SELECT query to fetch the next available job, it places a transaction lock on that specific row. Any subsequent workers executing the exact same query will simply skip the locked row and retrieve the next available job in the queue. This enables massive, lock-free concurrency across distributed worker instances without the need to deploy and maintain a separate Redis cluster. Furthermore, utilizing PostgreSQL ensures that all job histories, execution times, failure states, and payload metadata remain fully queryable using standard relational analytics, providing profound operational observability.4. Comprehensive Architectural Implementation PlanThe physical deployment of the serverless inference platform is conceptually divided into three independent operational vectors: the Global Control Plane, the Stateful Data Plane, and the Distributed Worker Plane.4.1 Global Control Plane EngineeringThe Control Plane is the highly available, horizontally scaled ingestion layer built upon FastAPI. It is responsible for authenticating incoming API requests, validating the JSON schemas against expected prompt formats, and interacting with the PostgreSQL database. The Control Plane strictly performs state management and routing; it never touches the actual neural network execution.When a client submits an inference payload, the API gateway assigns a unique UUID to the transaction. This UUID mitigates collision risks across distributed microservices. The gateway inserts the payload into the jobs table with a PENDING status. If the request was asynchronous, the gateway immediately returns an HTTP 202 Accepted status along with the UUID, terminating the active connection and freeing the thread pool to process the next incoming request. If the request demands synchronous streaming, the FastAPI application utilizes an asyncio.sleep loop, continuously checking the database state or listening for PostgreSQL NOTIFY events, yielding the results back over the open connection only when the worker updates the corresponding row.4.2 The Data Plane: Model Weight Caching and Network StorageThe most significant bottleneck in serverless GPU scaling is the initialization time of the worker container. Downloading a 40-gigabyte Llama 3 model from the Hugging Face Hub during a scale-up event introduces unacceptable multi-minute latencies. To circumvent this, the architecture implements a persistent Data Plane utilizing high-throughput Network File System (NFS) or NVMe volumes, modeled directly after RunPod's network volume integration.The implementation requires provisioning a centralized, highly available network volume within the specific data center housing the GPU compute nodes. Prior to deploying the serverless endpoint, an administrator utilizes an S3-compatible API or a command-line interface utility to download the necessary safetensors model weights directly into this persistent storage volume.When the autoscaler provisions a new worker container, the container orchestration system (e.g., Kubernetes) mounts this network volume directly to the container's file system, utilizing a standardized path such as /runpod-volume. The container's environment variables are configured to override the default cache directories, strictly pointing the inference engine to the mounted volume (e.g., setting the environment variable HF_HOME=/runpod-volume/cache). Consequently, when the vLLM engine initializes, it detects the pre-populated model weights on the local file system and begins loading them directly into the VRAM over the PCIe bus, entirely bypassing the external network. This architectural decision guarantees that cold starts are constrained only by the hardware transfer speeds and the engine initialization time, routinely reducing cold starts to a matter of seconds.4.3 Worker Plane: Custom PyWorker and Proxy ContainerizationThe Worker Plane consists of the ephemeral GPU instances executing the inference workloads. Following the Vast.ai paradigm, the worker nodes do not expose the raw vLLM server to the external internet. Instead, they execute a specialized, lightweight Python wrapper known as the PyWorker, which acts as a localized proxy and telemetry agent.The container image packages the PyWorker script, the vLLM binary, and all requisite CUDA drivers. Upon initialization, the PyWorker executes a series of rigorous system fitness checks before declaring itself ready to accept traffic.Pythonimport torch
import shutil

def run_fitness_checks():
    if not torch.cuda.is_available():
        raise SystemError("CUDA runtime is inaccessible; hardware failure suspected.")
        
    disk_stat = shutil.disk_usage("/")
    free_gb = disk_stat.free / (1024**3)
    if free_gb < 10:
        raise SystemError(f"Insufficient local storage: {free_gb:.2f}GB remaining.")
If a fitness check fails, the PyWorker triggers a fatal exit code, instructing the orchestrator to instantly terminate the container and attempt provisioning on a healthier node. Once the fitness checks pass, the PyWorker spins up the internal vLLM process on a local port (e.g., localhost:8000). It then initiates its primary execution loop, opening a database connection to the PostgreSQL server and executing the SKIP LOCKED queries to claim PENDING jobs. Upon claiming a payload, the PyWorker strips away any platform-specific metadata, constructs the appropriate HTTP request, and transmits the payload directly to the internal vLLM server.5. Algorithmic Design of the AutoscalerThe autoscaler is the critical nervous system of the serverless infrastructure, responsible for navigating the fundamental tradeoff between minimizing client wait times and maximizing hardware cost efficiency. A poorly tuned autoscaler will either over-provision expensive GPUs, destroying profitability, or under-provision resources, resulting in unacceptable queue degradation.5.1 Real-Time Queue Depth and Hardware BenchmarkingThe autoscaler operates as an independent, highly resilient microservice that evaluates system state at predefined intervals (e.g., every three seconds). The decision engine ingests a matrix of operational variables:$Q_d$ (Queue Depth): The absolute count of jobs residing in the PENDING state within the PostgreSQL database for a specific model endpoint.$W_a$ (Active Workers): The current count of healthy, fully initialized GPU containers that are successfully passing their periodic PyWorker heartbeats.$C_w$ (Worker Throughput Capacity): The maximum number of concurrent requests a single instance can process before hitting Out of Memory (OOM) errors or violating latency Service Level Agreements (SLAs). This capacity is dictated by the vLLM continuous batching configuration and the specific VRAM available on the provisioned hardware class.$T_p$ (Average Processing Time): A rolling average of the time required to complete historical jobs, utilized to predict queue clearance times.5.2 Mathematical Modeling of Worker ProvisioningThe core logic of the autoscaler evaluates whether the current active worker pool possesses the necessary aggregate capacity to process the pending queue within a strict target timeframe. The basic mathematical representation determining the optimal target worker count ($W_t$) is defined as:$$W_t = \left\lceil \frac{Q_d}{C_w} \right\rceil + W_{reserve}$$In this formula, $W_{reserve}$ represents an optional configurable integer denoting the number of idle instances to maintain at all times. By maintaining a small reserve pool (e.g., $W_{reserve} = 1$), the system guarantees that sporadic, low-volume requests experience zero cold start latency.When the autoscaler calculates that $W_t > W_a$, it issues commands to the infrastructure provider (e.g., via the Kubernetes API) to deploy $(W_t - W_a)$ new instances. To prevent chaotic thrashing—where a brief spike causes massive scale-up followed immediately by a scale-down—the algorithm implements a hysteretic delay. A worker will only be marked for termination if it remains idle, and the calculated $W_t$ remains lower than $W_a$, for a continuous duration exceeding a predefined scale_down_delay threshold.5.3 Edge Cases: Time-To-Live (TTL) and Hardware MatchingThe autoscaler must also manage edge cases such as stalled workers and hardware routing. To prevent failed jobs from permanently occupying the queue, the system implements a strict Time-To-Live (TTL) parameter. If a job remains in the RUNNING state beyond its maximum allowed TTL (e.g., one hour), the system automatically updates the database status to FAILED, allowing the autoscaler to discount the job from capacity planning and enabling the client to receive a definitive error response.Furthermore, incorporating the Vast.ai benchmark-driven methodology, the autoscaler can be enhanced to query a hardware marketplace API. Instead of uniformly deploying identical virtual machines, the autoscaler dynamically evaluates the required $C_w$ against the real-time rental prices of available GPUs. It automatically identifies and recruits the optimal mix of hardware, seamlessly routing less demanding text generation tasks to affordable consumer-grade cards while reserving high-bandwidth enterprise GPUs exclusively for massive batch operations.6. API Construction and OpenAI Compatibility LayerThe commercial viability of a serverless inference endpoint relies heavily on its interoperability with existing developer ecosystems. Forcing developers to rewrite their application code to interface with a proprietary API schema creates massive friction. Therefore, the architecture strictly mandates full compatibility with the OpenAI Chat Completions API standard.6.1 Synchronous vs. Asynchronous Payload ExecutionThe FastAPI control plane exposes specialized routes to handle different execution paradigms.
The asynchronous execution flow (POST /run) is designed for integration into robust, fault-tolerant backend systems. When the client posts a payload, the API inserts the record into PostgreSQL and immediately returns the newly generated job_id. The client system is then responsible for implementing a polling mechanism against the GET /status/{job_id} endpoint to retrieve the results.The synchronous execution flow (POST /runsync) provides a simplified developer experience at the cost of holding network connections open. When invoked, the FastAPI endpoint internally awaits the completion of the job by continuously executing a lightweight database poll. A critical engineering consideration here is the handling of empty results. Documented bugs in older SDK iterations reveal that if the internal handler function yields an empty dictionary {}, the framework attempts to pop an non-existent output key, resulting in a fatal KeyError. The API gateway must incorporate defensive programming to sanitize and validate all worker responses before transmitting them back across the synchronous connection.6.2 OpenAI Chat Completions Schema IntegrationIncoming requests must support the standardized OpenAI messages array, which structures the prompt history into discrete role and content dictionaries.JSON{
  "model": "meta-llama/Llama-3-8B-Instruct",
  "messages":,
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true
}
When this payload reaches the PyWorker, the worker forwards it to the localized vLLM instance. Because various open-source models require drastically different prompt templates (e.g., Llama 3 formatting vs. Mistral formatting), the PyWorker invokes the vLLM engine using the --chat-template-content-format openai flag. This instructs the inference engine to automatically detect the incoming OpenAI schema, intercept the list of dictionaries, and internally render the messages into the exact proprietary string format expected by the loaded model's tokenizer.6.3 Server-Sent Events (SSE) and Generative StreamingWhen the stream: true parameter is detected in the payload, the standard HTTP request-response cycle is replaced by Server-Sent Events (SSE). As the vLLM engine performs its autoregressive generation, it yields partial output tokens sequentially. The PyWorker detects the presence of an asynchronous Python generator object (types.GeneratorType) and enters a streaming loop.For each generated token, the PyWorker updates the database's updated_at column to act as a heartbeat, preventing the TTL monitor from terminating the job. Simultaneously, it transmits the token back to the FastAPI gateway, which streams it directly to the client over the persistent HTTP connection. Architectural care must be taken to handle asynchronous exceptions during this phase; documented issues indicate that errors occurring mid-stream often fail to propagate through the streaming output, requiring clients to independently poll the status endpoint to retrieve the traceback. The PyWorker must explicitly catch exceptions within the generator block and yield a final, formatted error chunk before closing the stream.7. Database Engineering and State ManagementThe utilization of PostgreSQL as the fundamental state engine requires precise schema design to support high-velocity, concurrent transactions without degrading into deadlock scenarios.7.1 Schema Design for High-Concurrency Job QueuesThe structure of the jobs table dictates the efficiency of the entire queuing architecture.Column IdentifierSQL Data TypeOperational Purpose and ConstraintsidUUID PRIMARY KEYUniquely identifies the job; generated at ingestion to prevent distributed collisions.endpoint_idVARCHAR(255) INDEXEDFacilitates rapid filtering by PyWorkers polling for their specific deployed model.statusVARCHAR(50) INDEXEDConstrained enum (PENDING, RUNNING, COMPLETED, FAILED). Crucial for SKIP LOCKED queries.payloadJSONBStores the entire inference request (parameters, prompt history) dynamically.outputJSONBStores the finalized generated response or serialized error tracebacks.created_atTIMESTAMP WITH TIME ZONEEstablishes absolute queue ordering, ensuring First-In-First-Out (FIFO) processing.updated_atTIMESTAMP WITH TIME ZONEContinuously updated by the PyWorker; evaluated by the TTL monitor to detect stalled workers.7.2 Transactional Guarantees and Failure RecoveryThe operational integrity of the queue relies on the atomicity of the FOR UPDATE SKIP LOCKED transaction. When the PyWorker executes the polling query, it attempts to transition a row from PENDING to RUNNING. If the PyWorker experiences a catastrophic hardware failure (e.g., a GPU driver crash) immediately after locking the row, the job is stranded in the RUNNING state.To counter this, a background asynchronous cron task—operating similarly to the mechanisms found in the pgqueuer library—continuously scans the database. It identifies any row where status = 'RUNNING' and the updated_at timestamp exceeds the predefined TTL threshold. Upon detection, the cron task transitions the job back to the PENDING state, effectively implementing a dead-letter queue mechanism that automatically re-queues tasks that die mid-run, ensuring absolute operational durability.8. Frontend Integration and the Serverless SDKA high-performance backend architecture is irrelevant if it presents extreme friction to the end-user. Therefore, the deployment strategy mandates the creation of a robust Client SDK that perfectly mimics the development experience of interacting with localized APIs.8.1 Serverless For Frontend (SFF) OrchestrationThe integration of the serverless endpoint into consumer-facing applications relies heavily on the Serverless For Frontend (SFF) pattern. Developers no longer build heavy, monolithic Node.js middleware to manage data orchestration. Instead, edge functions (such as Vercel Serverless Functions or AWS Lambda) act as the intermediary bridging the user interface and the GPU endpoint.When a user submits a prompt in a React application, the edge function receives the request, constructs the appropriate messages array, and leverages the Client SDK to authenticate and communicate with the FastAPI control plane. Because the SFF edge functions are billed by execution time, it is vital that the SDK supports efficient asynchronous mechanisms to prevent the edge function from idling and accruing massive costs while waiting for the GPU to process the inference.8.2 Client-Side Polling and Exponential BackoffThe Client SDK, distributed as a standard Python pip package or an npm module, abstracts the complex logic of managing asynchronous jobs. When a developer invokes client.run_inference(payload), the SDK initiates the POST /run request.Recognizing that network volatility and massive inference queues can delay results, the SDK implements an automated status polling loop utilizing an exponential backoff algorithm. The SDK queries the GET /status/{job_id} endpoint after 500 milliseconds. If the status remains RUNNING, the SDK waits for 1000 milliseconds, then 2000 milliseconds, up to a defined ceiling. This approach drastically reduces the sheer volume of HTTP requests flooding the control plane while ensuring that the application remains responsive. The SDK also gracefully parses HTTP 404 Not Found errors, which the API throws if a job is purged from the database after its retention limit expires (e.g., querying an asynchronous job after the strict 30-minute retention policy).9. Deployment Strategies and System FitnessTransitioning the theoretical architecture into a production-ready deployment requires rigorous infrastructure configuration, validating container lifecycles, and engineering network topology for maximum fault tolerance.9.1 Container Lifecycle and Startup ValidationsDeploying the PyWorker involves constructing a highly optimized Dockerfile. The image must be built on top of an NVIDIA CUDA base image to guarantee the presence of the necessary runtime libraries. During the build process, extraneous dependencies are stripped out to minimize the image footprint, accelerating the initial container download sequence by the orchestration hypervisor.As the orchestrator boots the container, it executes the PyWorker entrypoint. The fitness check logic runs comprehensively, validating that the vLLM process successfully mounts the PCIe interfaces and binds to the GPU hardware. If the endpoint configuration stipulates a specific CUDA version, the orchestrator guarantees that the physical server running the hypervisor utilizes drivers matching or exceeding that requirement, ensuring deep backward compatibility without sacrificing access to newer hardware pools. Only after the vLLM engine fully parses the model weights from the /runpod-volume NFS mount and reports a ready state does the PyWorker begin executing its database polling loop.9.2 Geographic Failover and Partition ToleranceThe fundamental premise of the xcelsior distributed scheduler highlights the necessity of geographic distribution. Restricting a serverless endpoint to a single datacenter guarantees devastating downtime during localized power failures or routing issues.To achieve global resilience, the architecture must support multiple worker pools dispersed across distinct geographic regions. However, because NFS network volumes are physically tethered to the infrastructure of a specific datacenter, assigning multiple datacenters to an endpoint introduces data synchronization challenges. A worker booting in a European datacenter cannot access an NVMe volume physically located in a North American facility without incurring crippling transatlantic latency penalties.Consequently, administrators must provision independent network volumes in every targeted region. Utilizing the S3-compatible storage API, automation pipelines must replicate the exact model safetensors files across all regional volumes prior to deployment. The FastAPI control plane remains regionally agnostic; it inserts the job into the global PostgreSQL database, and whichever PyWorker claims the job first processes it utilizing its localized network volume, ensuring maximum availability and the lowest possible inference latency.10. Strategic Conclusions and Future TrajectoriesThe design and implementation of a serverless inference endpoint represent the apex of modern cloud orchestration. It requires the seamless integration of deeply optimized, hardware-accelerated binaries with highly concurrent, partition-tolerant web architecture. By analyzing the landscape, the optimal solution requires discarding legacy queue mechanisms in favor of the transactional supremacy of PostgreSQL's SKIP LOCKED architecture, replacing monolithic proxies with decentralized PyWorkers, and mitigating cold starts through the aggressive application of mounted NVMe network volumes.The integration of the vLLM engine provides the fundamental computational throughput necessary to operate profitably, utilizing PagedAttention to shatter the limitations of GPU memory fragmentation. The implementation of an OpenAI-compatible FastAPI gateway ensures that the platform seamlessly integrates into the vast, existing ecosystem of AI application development, while the transition toward Serverless For Frontend (SFF) architectures drastically simplifies the deployment of the overarching application stack.As the industry advances towards the concepts pioneered by decentralized repositories like xcelsior, the future of serverless inference will increasingly prioritize geographic distribution, treating worker nodes not as centralized components of a monolithic datacenter, but as autonomous, heavily verified computational units pulling from a globally synchronized, high-velocity queue. This rigorous, step-by-step implementation blueprint provides the definitive framework for engineering a platform capable of dominating that future landscape.


**Serverless Inference Endpoints — Tech Recommendations + Step-by-Step Implementation Plan (Study-First, No Edits Yet)**

I studied the existing xcelsior codebase thoroughly via the GitHub repository (flat Python structure with `api.py`/`split_api.py` + `routes/` modular routers, `inference.py` + `inference_store.py` + `routes/inference.py` already containing substantial serverless-like functionality, `scheduler.py` + `worker_agent.py` + SSH/Docker container lifecycle, `bg_worker.py` with periodic tasks including `inference_scaledown`, `db.py` raw SQL + dual-write + `owner_id`/`customer_id` tenancy patterns on volumes/user_images/billing_cycles, `events.py` + SSE broadcasting, `billing.py` + Stripe integration, `volumes.py` for NFS, Alembic + `migrations/`, `pyrightconfig.json` strict typing, `_deps.py` + `security.py`/`jurisdiction.py`/`privacy.py` for auth/scope/owner guards and 404 cross-tenant, `teams.py`, and `routes/autoscale.py` + scheduler autoscaling primitives).

The current `InferenceEngine` (in `inference.py`) + routes already implement:
- Endpoint CRUD with `min_workers`/`max_workers`/`scaledown_window_sec`/`docker_image`/`mode`/`api_format=openai|custom`.
- OpenAI-compatible surface (`/v1/chat/completions`, `/v1/inference*` with some streaming/SSE).
- Worker provisioning/deprovisioning via `scheduler.submit_job`/`kill_job`, warm/cold routing via `worker_model_cache`, reactive scale-down, per-token billing hooks, and `inference_jobs`/`inference_results` storage.
- Team/owner scoping and wallet pre-flight checks.

It is a strong foundation (not greenfield), but incomplete/stubbed for full competitor parity (missing robust queue-depth-driven scale-up + hysteresis, full async/sync `/run`/`runsync`/`status`/`cancel`/`stream` per-endpoint surface, dedicated per-GPU-second execution billing, formal custom handler SDK/contract + templates, idempotency/webhooks, scale-to-zero robustness with `min_workers=0`, NFS volume flashboot integration, per-endpoint keys/rate limits/quotas, comprehensive observability, and the Next.js dashboard/wizard).

**Competitor research summary (RunPod, Novita, Vast.ai)**: All converged on OpenAI-compatible surfaces for instant SDK adoption, async + sync job APIs with status polling + streaming + cancel + webhooks + idempotency, autoscaling (queue depth / request rate driven, scale-to-zero with `min`/`max`/`idle_timeout`, keep-warm), two modes (managed presets like vLLM-style + custom container/handler), and per-execution (GPU-second or token) billing from wallet. RunPod excels at simple `handler.py` + Dockerfile + GitHub deploy + Flash (pure Python). Novita emphasizes managed models + pay-per-token + auto-scale. Vast.ai leverages marketplace routing + workergroups + predictive scaling + SDK + pay-per-second.

### Recommended Technologies & Packages (Competitor Parity or Better)
Use these for the full task (platform backend + worker images). Prioritize reuse of existing stack; add minimally.

- **LLM Inference Engine (managed preset mode)**: **vLLM** (official `vllm/vllm-openai` or CUDA images). Best choice in 2026: native OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` + streaming SSE + tool calling + vision), continuous batching + PagedAttention for high throughput/variable load, broad model/quantization support (AWQ/GPTQ/FP8), easy containerization, active ecosystem. Superior adoption driver and simplicity vs. TGI (good HF integration but less flexible throughput) or TensorRT-LLM + Triton (max perf on fixed models but high complexity/maintenance — use only for later premium tier). SGLang strong alternative for prefix-caching/multi-turn but vLLM wins for broad compatibility.

- **Custom container / handler mode**: Follow/extend **RunPod handler pattern** for familiarity (easy porting from competitors) but improve it. Provide:
  - Base images (e.g., `xcelsior/serverless-base:cuda12.4-python3.12` with CUDA, uv, common deps, FastAPI/Starlette).
  - Lightweight Python SDK (`pip install xcelsior-serverless` or vendored): decorator `@endpoint` / `class Handler: async def __call__(self, job: dict) -> dict | AsyncIterator` + `serve(Handler())` entrypoint. Supports both simple handler-fn mode (platform wraps) and "bring-your-own-server" mode (container exposes HTTP on `$PORT` with `/healthz` + `/invoke` or full OpenAI surface).
  - Streaming: generator/yield or NDJSON/SSE callback for tokens.
  - Better than pure RunPod: standardized contract + error/timeout formats, env/secrets injection, heartbeat, and optional full OpenAI compat for advanced users. Ship templates + examples (in `docs/serverless/` or separate `xcelsior-workers` templates). Avoid heavy frameworks like BentoML/Modal unless scope expands.

- **Job queue / state / orchestration**: **Existing Postgres** (`jobs` + extend `inference_jobs`/`inference_results` or new `serverless_*` tables via Alembic; JSONB payloads excellent). Reuse scheduler for placement. Optional later: `redis` (redis-py) for high-throughput queues/pubsub/rate-limit counters/idempotency sets if Postgres hot paths saturate — but start without (reuse existing patterns).

- **Autoscaler / control loop**: Pure Python reconciliation loop (no new packages). Extend `bg_worker.py` pattern (`inference_scaledown` every 300s) + scheduler primitives. Simple, idempotent, crash-safe queue-depth + concurrency-target controller with hysteresis/cooldowns.

- **API layer / streaming / typing**: **FastAPI + Pydantic v2** (existing, already in use in `routes/inference.py`). `StreamingResponse` + async generators for SSE (`data: {...}\n\n` + `[DONE]`). Strict pyright (respect `reportCallIssue`/`reportArgumentType` etc. from `pyrightconfig.json`).

- **Auth / keys / rate limits / tenancy**: Reuse existing (`_deps.py`, OAuth/sessions, `_require_team_instance_write` / `_require_inference_endpoint_access` style guards, `owner_id`/`customer_id` scoping, 404 cross-tenant). Add per-endpoint API keys (new table or extend). Rate limiting: custom middleware or `slowapi` (lightweight) with DB/Redis counters; return standard `X-RateLimit-*` headers + 429.

- **Billing / metering**: Extend **existing `billing.py` + `billing_cycles`** (tie to `job_id` or new serverless job). Track `gpu_seconds` (execution time) + optional cold-start time at endpoint's GPU tier rate (reuse `gpu_pricing`). Wallet suspend → reject enqueue. Per-second or token hybrid; reconcile worker uptime + per-job.

- **Events / realtime / observability**: Reuse **existing `events.py` + SSE patterns** (broadcast `endpoint_created`, `worker_scaled`, `job_completed`, metrics snapshots). Structured logs per job. Enhance usage/health endpoints or add Prometheus client (optional, check if fits existing). Per-endpoint dashboard metrics (requests, error rate, queue/exec time, tokens/s or GPU util, active/idle workers, cost).

- **Model weights / image cache / cold starts**: Reuse **existing `volumes.py`** (NFS persistent volumes) + `user_images` / registry. Platform-managed or per-endpoint "model-cache" volumes (pre-populated weights) attached read-only for flashboot. Pre-pull images. Realistic cold-start budget: 20–90s p99 (image pull + vLLM model load on quantized weights from NFS cache on good host); document + optimize (keep-warm `min_workers≥1`, pre-warmed popular presets).

- **Migrations / schema**: **Alembic** (existing `alembic.ini` + `migrations/versions/`). Use for all new tables/columns/indexes (even if some `inference_*` tables currently use `db.py` `_ensure_*` DDL). Keep dual SQLite/Postgres parity during transition.

- **Frontend**: **Next.js 15** (existing in `frontend/`, likely app router). Reuse team context/banner, viewer read-only, i18n (en + fr strings), existing components/patterns. Add: multi-step create wizard (Zod + React Hook Form or equivalent), detail page with live metrics (EventSource/SSE + polling), test console (form + streaming preview + auto-generated curl/Python SDK snippets), keys management, cost view. No heavy new libs initially (TanStack Query/SWR if data fetching needs it; charts via existing or lightweight like recharts).

- **Other / infra**: Python ≥3.12 (existing), Docker (multi-stage for small worker images), existing gunicorn/uvicorn. Add `tiktoken` (optional, for accurate token counting in billing/observability). No K8s or external autoscalers (keep sovereign/simple, reuse scheduler).

These choices match or beat competitors on adoption (OpenAI surface + handler familiarity), performance (vLLM), cost (per-execution + marketplace routing + NFS cache), and integration (reuse 90%+ of existing subsystems).

### Refined Data Model (Builds Directly on Existing)
Extend (preferred for minimal disruption) or parallel the existing `inference_endpoints` / `worker_model_cache` / `inference_jobs` / `inference_results`:

- `inference_endpoints` (or `serverless_endpoints`): `id`, `owner_id`/`customer_id`, `name`, `mode` (preset|custom), `model_or_image_ref`, `gpu_tier`/`gpu_model`/`vram_gb`, `min_workers`, `max_workers`, `idle_timeout_s`, `max_concurrency`, `scaling_policy` (jsonb: queue_target_wait_s, scale_step, cooldown_s, etc.), `env` (jsonb), `secrets_ref`, `status`, `region`, `created_at`, `updated_at`, + indexes.
- `serverless_workers` (or extend `worker_model_cache`): `id`, `endpoint_id`, `host_id`, `scheduler_job_id`, `status` (booting|idle|busy|draining), `started_at`, `last_active_at`, `current_concurrency`, ...
- `serverless_jobs` (or extend `inference_jobs`): `id`, `endpoint_id`, `status`, `input_payload` (jsonb), `output` (jsonb), `error`, `queued_at`, `started_at`, `finished_at`, `gpu_seconds`, `cost_cad`, `idempotency_key`, `webhook_url`, + indexes on `(endpoint_id, status)`, partial unique on idempotency_key, etc.

Use Alembic migrations. Update `inference_store.py` + engine accordingly. Tenancy via `owner_id` + team guards exactly like volumes/instances.

### Autoscaler Control Loop (Reconciliation Algorithm)
New/extended `reconcile_serverless_endpoints()` (or enhance `InferenceEngine` + call from `bg_worker.py` like `inference_scaledown`):

1. Periodic (30–60s recommended; bounded queries, `LIMIT 100–200` per tick for safety).
2. For each active endpoint (filter suspended wallets via billing):
   - Compute: `queue_depth` (pending serverless_jobs), `in_flight` (busy workers/jobs), current active/idle workers, historical avg queue wait (from job timestamps or simple moving avg).
   - Desired count = clamp( f(queue_depth / max_concurrency + headroom_for_target_wait, rate), min_workers, max_workers ). Simple effective formula: scale up if queue > threshold or wait > target; scale down only if queue==0 and all idle workers > idle_timeout and current > min.
3. Hysteresis/cooldown: per-endpoint `last_scale_at`; min 60s between scale events; step size (e.g., +1 or +min(2, remaining)).
4. Scale up: create `serverless_worker`(booting), `submit_job`(special `job_type=serverless_worker` or payload with endpoint config + image + env + optional volume mount for cache) → scheduler places on suitable host (GPU tier match, jurisdiction, VRAM via existing `allocate`/`allocate_binpack`).
5. Worker boot: container starts runtime (vLLM for preset or handler server), loads model (NFS cache path), signals ready (callback/agent result → `mark_ready`, status=idle).
6. Scale down: pick idle (last_active + idle_timeout, concurrency==0), set draining → drain/finish current or timeout → `kill_job` → terminate.
7. Crash/edge: heartbeats update `last_active`; reaper for stuck booting (timeout → fail/retry); worker crash → requeue job + scale event; idempotent via status machine + transactions.
8. Cold-start mitigation: `min_workers≥1` keep-warm; platform pre-populates popular preset volumes (via `volumes.py`); pre-pull images; document realistic budget (20–90s p99 with flashboot/NFS).

Run in `bg_worker.py` (daemon thread, exception-safe loop, idempotent). Integrates with existing scheduler health/reconcile patterns.

### API Design Deliverables (Build on Existing `routes/inference.py`)
Enhance/consolidate under `/api/v2/inference/endpoints` or introduce `/v2/serverless` (recommend aligning on "Inference Endpoints" as the Serverless product name for minimal breaking change, or add alias). Full list (extend existing):

- `POST /api/v2/inference/endpoints` (create; full scaling config, mode, image/model, env).
- `GET /api/v2/inference/endpoints` (list, owner/team scoped).
- `GET /api/v2/inference/endpoints/{id}` + `/health` + `/usage` (metrics: requests, errors, queue/exec times, workers, cost, tokens/s or GPU util).
- `DELETE /api/v2/inference/endpoints/{id}`.
- OpenAI surface (existing + complete): `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` (per-endpoint routing or base_url + key; streaming SSE).
- Async/sync job API (RunPod-style, add/complete):
  - `POST /v2/{endpoint_id}/run` → enqueue → `{id, status: "IN_QUEUE"}` (idempotency-key header support).
  - `POST /v2/{endpoint_id}/runsync` → block (timeout) → output or error.
  - `GET /v2/{endpoint_id}/status/{job_id}` → full status + output/error + timings.
  - `POST /v2/{endpoint_id}/stream/{job_id}` or SSE on status.
  - `POST /v2/{endpoint_id}/cancel/{job_id}`.
  - Optional `webhook` per request.
- Internal worker callbacks (existing pattern): result/complete, ready, etc.
- Keys: `POST/GET/DELETE /api/v2/inference/endpoints/{id}/keys`.
- Rate-limit headers + 429 backpressure on queue full/max size.
- Auth: existing `_require_*` + per-endpoint key validation. Status codes: 200/201/202/400/401/403/404/429/500.

OpenAI mapping: platform routes (or proxies to warm worker port) or jobs → workers handle actual vLLM/custom execution. Streaming: async generator yielding OpenAI chunks.

**Worker handler SDK contract** (document + ship examples):
- Input: `job = {"id": "...", "input": {...}, "endpoint_id": "...", "timeout": ...}` (env/secrets mounted or injected).
- Output: `{"output": {...}}` or error dict; for streaming: yield chunks or callback.
- Lifecycle: `/healthz` (or platform ping), heartbeats, graceful drain on SIGTERM.
- Timeouts/errors: platform-enforced + reported.
- Presets: vLLM OpenAI server handles mapping automatically.

### Frontend (Next.js) Deliverables
New or expanded section under dashboard (e.g., `frontend/app/(dashboard)/inference/` or `/serverless/`):

- Create-endpoint wizard (multi-step: mode select → model/image picker or custom upload + handler validation → GPU tier (from pricing/offers) → scaling config sliders + policy preset → env/secrets → review/launch). Team-aware, validation, i18n.
- Endpoint list + detail: status, live metrics (SSE-driven cards/charts for requests/success/error, queue time, exec time/GPU util, active/idle workers, cost), actions (scale, pause, keys, delete). Live updates on `worker_scaled`/`job_completed` etc.
- Test console: "Try it" form (chat or custom payload) → calls endpoint → shows response + streaming preview. Auto-generated snippets (curl with key, Python openai SDK with base_url, JS). Copy buttons.
- Keys management tab (generate/revoke/list usage).
- Cost/usage view (breakdown, projections).
- Consistent with existing: team switch reloads context, viewer read-only disables mutates, en/fr i18n strings, loading/error states, responsive.

Use EventSource for SSE, existing form/UI patterns, and optimistic updates where safe.

### Comprehensive Step-by-Step Implementation Plan (Phased, File-Level Changes)
**Do NOT edit code yet.** Review this plan, answer open questions below, then we implement sequentially (migrations first, then backend, API, SDK/docs, frontend, integration/tests). Each phase produces testable increments. Match all conventions (typed Pydantic + pyright, owner_id/team guards from `_deps.py`, SSE broadcast patterns, Alembic migrations, scheduler reuse, billing tick integration).

**Phase 0: Alignment & Prep (no code changes)**
- Decide final naming (keep/enhance "Inference Endpoints" as the Serverless Inference product, or introduce `serverless_*` parallel + deprecation path?).
- Confirm frontend structure/details (list files in `frontend/app/`, components, i18n setup, UI lib, data fetching).
- Finalize handler contract details + cold-start SLA target to advertise.
- Audit existing `inference_endpoints` / `worker_model_cache` columns vs. proposed model; decide extend vs. new tables.

**Phase 1: Data Model, Migrations & Store (foundational, Alembic-first)**
- Create Alembic migration(s) under `migrations/versions/` (e.g., `20260608_xxxx_extend_inference_endpoints_for_serverless.py` or `add_serverless_tables.py`): add columns to `inference_endpoints` (or create `serverless_endpoints`), create/extend `serverless_workers` / `serverless_jobs` (or alter `inference_jobs` + `worker_model_cache`), add indexes (queue polling, per-endpoint metrics, idempotency partial unique), JSONB fields, FKs, timestamps. Include downgrade.
- Update `db.py` `_ensure_pg_tables` / SQLite equivalents only if needed for dev parity (prefer Alembic).
- Edit `inference_store.py`: extend CRUD for new fields/tables, add queue-depth queries, metrics helpers, idempotency checks, webhook queueing.
- Update any dependent models/schemas in `inference.py` / `routes/inference.py`.
- Test migration on fresh DB + backfill if data exists.

**Phase 2: Core Engine & Autoscaler Logic**
- Edit `inference.py`: Refactor/enhance `InferenceEngine` (or introduce `ServerlessEngine` subclass/extension) with:
  - Full `reconcile_autoscaler()` (queue-depth + policy logic, hysteresis, scale up/down via scheduler, worker lifecycle, crash reaper).
  - Per-GPU-second execution billing integration (track `gpu_seconds` from job timings, call into `billing.py` patterns; hybrid with existing token cost).
  - Custom mode support (handler contract routing vs. preset vLLM).
  - Webhook firing (on completion/failed, with retries?).
  - Idempotency-Key handling (dedupe on enqueue).
  - Improved cold-start / flashboot (integrate `volumes.py` for cache volumes, pre-warm logic).
  - Health/usage/metrics computation (active/idle workers, queue times, GPU util, cost).
  - SSE event emission for all key transitions (reuse existing broadcast helpers).
- Ensure all new code is fully typed, passes pyright gates.
- Edit `bg_worker.py`: add `serverless_reconcile` task (interval 30–60s, bounded, exception-safe, idempotent via status/tx) calling the new reconcile method. Follow exact pattern of `inference_scaledown` / `reconcile_paused_stopped`.

**Phase 3: API Surface Completion & Hardening**
- Edit `routes/inference.py` (primary file): 
  - Add/complete missing routes (`/v2/{endpoint_id}/run`, `/runsync`, `/status/{job_id}`, `/cancel/{job_id}`, `/stream/...`, per-endpoint keys CRUD, rate-limit handling).
  - Full Pydantic request/response schemas (match OpenAI shapes where applicable, add serverless job shapes).
  - Integrate new engine methods; enforce tenancy (`_require_inference_endpoint_*` style), wallet checks, idempotency, webhooks.
  - Enhance streaming SSE to full OpenAI spec + custom job streaming.
  - Add rate-limit headers, max queue size 429 backpressure, request size/timeout limits.
  - Internal routes/callbacks for worker ready/result (existing pattern).
- Add any new deps to `pyproject.toml` / `requirements.txt` if minimal (e.g., optional tiktoken).
- Update `routes/_deps.py` if new guards needed.
- Ensure SSE events match dashboard expectations.

**Phase 4: Worker SDK, Templates, Docs & Preset Support**
- Create `docs/serverless/` (or `templates/serverless/`): handler contract spec (inputs/outputs/streaming/errors/timeouts), example `handler.py` + `Dockerfile` for custom, vLLM preset example, quickstart.
- (Optional but recommended) Add lightweight SDK module or separate installable (or document `pip install` + examples). Provide GitHub-ready templates for easy deploy (like RunPod Hub).
- Update root `README.md`, `inference.py` docstrings, and any runbooks with serverless usage, cold-start expectations, billing details.
- Platform-side: support preset images (vLLM with specific models) + custom docker_image validation/pull via existing registry/user_images.

**Phase 5: Frontend Implementation**
- First: explore `frontend/` structure in detail (if not already) to match exact conventions (app router? components? i18n files?).
- Add serverless/inference section: wizard page/component (multi-step form, validation, team context), endpoint detail (live metrics via EventSource + polling, actions), test console (form + streaming response viewer + snippet generator), keys tab, cost view.
- Add i18n strings (en + fr) consistent with existing.
- Integrate SSE for real-time updates (endpoint/worker/job events).
- Ensure viewer read-only, reload on team switch, polished UX matching competitors (clean metrics, one-click test, copyable code).

**Phase 6: Integration, Optimization, Tests & Polish**
- Wire NFS volumes (`volumes.py`) into engine for model cache / flashboot (create/attach logic for endpoints).
- Enhance observability (structured logs, more metrics in usage/health, optional Prometheus).
- Add/update tests (in `tests/`) for new endpoints, engine logic, migrations.
- Run full pyright, lint, tests (`run-tests.sh`).
- Cold-start / scale testing (document realistic budgets).
- Billing reconciliation + suspended wallet tests.
- Update any systemd/Docker configs if new background tasks or images.

**Phase 7: Rollout & Iteration**
- Feature flag or gradual enable (min_workers etc.).
- Monitor (queue depths, cold starts, costs, errors).
- Iterate on policy tuning, handler ergonomics, frontend UX based on usage.
- Optional later: Redis for hot paths, advanced predictive scaling (like Vast), multi-GPU workers, GitHub deploy integration for custom.

**Estimated effort**: Phased over multiple iterations; start with Phase 1 (migrations + store) for safety. Each phase ends with reviewable diff + tests passing.

### Open Questions (Please Answer Before We Edit)
1. Naming preference: Enhance existing "Inference Endpoints" (minimal breaking) or introduce parallel `serverless_endpoints` / `ServerlessEngine` with migration path?
2. Exact handler SDK contract details (e.g., `async def handler(job) -> dict` vs. class + `serve()`, streaming mechanism, error format, timeout propagation)?
3. Billing exacts: pure GPU-seconds of execution (worker-reported or timed) + optional cold-start time, or hybrid with existing token pricing? Formula for gpu_seconds?
4. Request routing model: platform proxies `/v1/*` to warm worker's HTTP port (existing inference pattern), or uniform job-queue pull model for both preset and custom (simpler ports/security)?
5. Cold-start target/SLA to advertise and optimize for (e.g., <45s p99 with NFS flashboot + keep-warm)?
6. Frontend confirmation: exact structure/files in `frontend/`, preferred form/validation libs (Zod/RHF?), charting solution, and any existing SSE/EventSource hooks or i18n patterns to match precisely?
7. Add Redis now (for queues/rate limits/idempotency) or stay pure Postgres (reuse existing hot paths)?
8. Scope for first increment: full scale-up + billing + core API, or also custom SDK + frontend in one go?
9. Any constraints on multi-GPU per worker or specific regions/GPU tiers priority?
10. Idempotency-Key and webhook retry/security details (HMAC? max retries?).

This plan is production-ready, reuses >90% of existing subsystems (scheduler, billing, tenancy, events, volumes, bg_worker, inference foundation), matches competitor table-stakes UX while fitting the Canadian sovereign marketplace model, and respects every specified convention.

Once you confirm/answer the opens (and any adjustments), we start with **Phase 1 migrations** (never before). Ready when you are — let's build the best serverless inference layer on the marketplace.


# Serverless Inference Endpoints for Xcelsior

## What the existing codebase already gives you

Xcelsior already has the major platform primitives you want to reuse rather than replace. The backend is a modular FastAPI app that mounts route modules from `routes.__init__.py`, includes a dedicated `routes.inference` router today, runs periodic background tasks from `api.py`, enables OpenTelemetry when configured, and is wired for Alembic migrations under `migrations/versions/`. CI already enforces the exact pyright gate you called out: `reportCallIssue`, `reportArgumentType`, `reportAttributeAccessIssue`, and related checks are treated as errors in both `pyrightconfig.json` and the GitHub Actions workflow. citeturn30view0turn29view1turn29view2turn41view0turn41view1turn41view2turn42view0turn42view1

The compute control plane is also already pointed in the right direction. The scheduler persists `jobs`, `hosts`, `billing_cycles`, `volumes`, and `agent_commands` in Postgres, assigns queued jobs to hosts, and relies on pull-based worker agents rather than scheduler-initiated SSH for normal operation. The worker agent polls `/agent/commands/{host_id}`, and that command queue already uses `FOR UPDATE SKIP LOCKED` semantics for at-most-once draining. This is exactly the right substrate for a crash-safe serverless dispatcher and autoscaler without adding Redis, Celery, or Kafka as a second orchestration plane. citeturn24view2turn24view3turn25view0turn25view1turn36view4turn36view5turn34view1turn34view2turn33view0

The tenancy and realtime pieces are also present. `routes._deps.py` already has common auth/rate-limit helpers, owner-resolution helpers such as `_canonical_owner_id()` and `_user_team_id()`, an in-process SSE subscriber bus, and `broadcast_sse()`. Existing resource routes such as volumes apply `owner_id`-scoped access checks and distinguish read from write paths; team membership already uses `admin | member | viewer` roles, and the frontend already treats viewer access as read-only. Instance log streaming is already implemented with `StreamingResponse(..., media_type="text/event-stream")`, which is the right precedent for token streaming and live endpoint dashboards. citeturn18view0turn26view0turn27view0turn28view0turn28view4turn28view5turn38view4turn38view5

On storage and billing, the repo already includes the two hardest integration points. `volumes.py` is an NFS-backed persistent-volume engine with region-aware mounts, canonical mount options reused by `volumes.py`, `scheduler.py`, and `worker_agent.py`, and restart logic for reopening encrypted volumes after server reboots. `billing.py` already has wallets, wallet suspension, background billing ticks, and enforcement that stops running jobs for suspended wallets. Those should become the backbone for serverless weight caches and per-job GPU-second debits. citeturn20view3turn21view0turn21view1turn21view2turn35view0turn35view1turn35view2turn35view4

There is already a partial serverless/inference implementation, but it is not yet at the product bar you described. The current repo exposes `/api/inference`, `/v1/inference`, endpoint CRUD under `/api/v2/inference/endpoints`, and a basic OpenAI-ish `/v1/chat/completions` path. The engine can create an endpoint, provision a worker job, track simple warm/cold routing via `worker_model_cache`, and schedule a periodic idle-worker scaledown task. But the current OpenAI route returns an immediate “processing” shell instead of a true completion, and `InferenceEngine.complete_request()` still bills by token count through `compute_token_cost()` rather than by actual GPU-seconds. That makes the existing inference subsystem a good starting point, but not the final architecture. citeturn31view0turn32view2turn32view3turn23view2turn23view3turn29view1turn22view0

On the frontend, the repo is already a modern strict TypeScript Next.js app with React 19, Recharts, Sonner, Tailwind 4, and a cookie-authenticated `apiFetch()` client. The frontend auth model already includes team context on the `AuthUser` object, and the GPU selector can reuse the canonical catalog in `frontend/src/lib/gpu-models.ts`. That means the dashboard work is additive rather than foundational. citeturn17view0turn17view3turn37view1turn38view1turn38view5

## The best technologies and packages to use

The strongest day-one managed-model engine is **vLLM**. It gives you the broadest match to your target product shape in one package: an OpenAI-compatible server with `/v1/completions`, `/v1/chat/completions`, and `/v1/embeddings`; streaming outputs; tool-calling support; high-throughput scheduling via PagedAttention and continuous batching; prefix caching; quantization; and an operational model that strongly resembles what customers already expect from RunPod-class serverless LLM products. For Xcelsior, that means “managed preset” should default to a vLLM worker image. citeturn55view0turn55view1

The best secondary managed preset is **Hugging Face Text Generation Inference**. TGI’s Messages API is officially OpenAI-chat compatible and supports streaming with OpenAI client libraries, so it is a good second engine for customers who are already standardized on Hugging Face tooling or want TGI-specific behavior. I would not make it the default, because vLLM now covers more of the OpenAI surface you asked for in one server, including embeddings, but TGI is the right optional preset family to support later. citeturn55view2turn53search4

The best “future optimization, not day-one default” engine is **SGLang**. Its current documentation emphasizes low-latency, high-throughput serving, RadixAttention, prefix caching, multi-GPU support, and OpenAI compatibility. That makes it attractive for later premium endpoint classes, especially reasoning-heavy or multimodal workloads, but not necessary to land the first product if your goal is to fit tightly into Xcelsior’s current scheduler, billing, and tenancy model. citeturn55view3

For the control plane, the best technology choice is **the technology you already have**: FastAPI + Postgres + Alembic + the existing scheduler + the existing worker agent. RunPod, Vast, and Novita all converge on the same endpoint knobs—min/max workers, idle timeout, concurrency, request timeout, and queue- or latency-driven autoscaling—and Xcelsior already has all the underlying machinery needed to express those knobs. The strongest architectural move here is not adding Redis/Celery/Kafka, but letting Postgres own durable endpoint/job/worker state and using the scheduler to place serverless workers as managed GPU jobs. citeturn47view2turn44view4turn52view3turn24view2turn36view4turn34view2

For custom workloads, the best product pattern is to support **both** of the deployment modes competitors have converged on: a queue-based Python handler API and a direct HTTP/ASGI mode. RunPod’s docs explicitly separate queue-based handlers (`def handler(job)`) from load-balanced endpoints that can run FastAPI or Flask and expose custom REST paths. Xcelsior should copy that idea, but fit it to its own scheduler: queue-based handler mode for arbitrary async jobs, and an ASGI/container mode for low-latency direct HTTP and OpenAI-compatible preset serving. citeturn47view0turn47view1turn47view3

For observability, do not introduce a second telemetry stack. Reuse Xcelsior’s existing **SSE bus**, **OpenTelemetry instrumentation**, **Prometheus client hooks**, and **agent telemetry path**. On the frontend, stay with **native EventSource or existing SSE consumption patterns**, **Recharts** for time series, and the existing cookie-auth API client. citeturn18view0turn29view2turn34view4turn17view0turn38view1

## Target backend design and data model

The right modeling choice is a **hybrid design**: a serverless worker should be represented as an underlying scheduler job for placement and lifecycle, plus a new first-class `serverless_workers` row for endpoint semantics. In other words, do **not** invent a second compute primitive parallel to `jobs`; instead, reuse `submit_job()` and the worker-agent/job-log/billing/event infrastructure you already have, but add serverless-specific metadata so you can represent `booting | idle | busy | draining | crashed` without overloading the customer-facing instance model. This gets you scheduler reuse, host placement, log forwarding, telemetry, and agent commands “for free,” while still letting the dashboard think in endpoint/worker/job terms instead of raw instances. citeturn36view5turn36view4turn33view0turn28view3turn24view2

I would replace the current “inference endpoint plus inference job/result tables” concept with a richer serverless data model:

- `serverless_endpoints`: `id`, `owner_id`, `name`, `slug`, `mode` (`preset|custom`), `task_type` (`chat|completion|embedding|custom`), `managed_engine` (`vllm|tgi|null`), `model_ref`, `image_ref`, `startup_command`, `gpu_tier`, `gpu_count`, `region`, `min_workers`, `max_workers`, `idle_timeout_sec`, `max_concurrency`, `request_timeout_sec`, `max_request_bytes`, `max_queue_size`, `scaling_policy_type`, `scaling_policy_value`, `keep_warm`, `cache_volume_id`, `status`, `created_at`, `updated_at`, `deleted_at`.

- `serverless_workers`: `id`, `endpoint_id`, `scheduler_job_id`, `host_id`, `state`, `current_concurrency`, `max_concurrency`, `healthy`, `warm_model_ref`, `cache_state`, `started_at`, `ready_at`, `last_active_at`, `drain_requested_at`, `stopped_at`, `crash_reason`.

- `serverless_jobs`: `id`, `endpoint_id`, `owner_id`, `api_key_id`, `status`, `request_kind` (`run|runsync|chat|completion|embedding`), `request_json`, `response_json`, `error_json`, `webhook_url`, `idempotency_key`, `queued_at`, `dispatched_at`, `started_at`, `first_token_at`, `finished_at`, `queue_ms`, `execution_ms`, `gpu_seconds`, `cold_start_seconds`, `cost_cad`, `input_tokens`, `output_tokens`, `worker_id`, `cancel_requested_at`.

- `serverless_job_stream_events`: `job_id`, `seq_no`, `event_type`, `payload`, `created_at` for replayable token/progress streams and reconnect-safe SSE.

- `serverless_api_keys`: `id`, `owner_id`, `endpoint_id nullable`, `name`, `key_prefix`, `key_hash`, `scopes`, `rate_limit_rpm`, `status`, `last_used_at`, `created_at`, `revoked_at`.

This schema matches how Vast and Novita describe endpoint/worker/policy/image/health storage, but it stays consistent with Xcelsior’s `owner_id`-centric tenancy and scheduler-job reality. citeturn44view3turn44view4turn52view3turn31view0turn27view0

On tenancy, keep the current Xcelsior pattern exactly: `owner_id` is the billing and visibility anchor, team context is inherited from the caller, write operations are denied to viewers, and cross-account access stays a 404 rather than a discoverable 403. The current inference routes already import endpoint access helpers, and volumes demonstrate the same “find resource, then apply read/write guard” pattern you want to keep. citeturn31view0turn27view0turn26view0turn28view4turn38view4

On billing, the serverless product should **stop using token-based charging as the primary billing primitive**. The current inference engine literally computes token cost and charges the billing engine that way, which conflicts with your stated goal of billing on actual GPU execution time. The correct integration is to keep wallets, suspension, and `billing_cycles`, but write serverless usage as line items derived from actual worker execution spans—ideally from “job started on worker” to “job finished on worker,” with optional separate treatment of cold-start seconds if product wants to bill them. Tokens should remain for observability, cost explainability, and UI metrics, not as the authoritative debit basis. citeturn23view3turn35view0turn35view1turn35view2turn35view4

For weight and image caching, the cleanest integration is to reuse the existing NFS volume engine and treat warm cache as a managed platform volume attached to serverless workers. The volume engine is already designed for attachable NFS-backed storage, and both Novita and Vast expose network/cloud storage attachments in their endpoint config models. For managed presets, cache by `(model_ref, revision, region)`; for custom containers, optionally allow a platform-managed cache mount at a fixed path like `/xcelsior-cache`. citeturn20view3turn21view0turn52view3

## Autoscaling, scale-to-zero, and cold starts

The autoscaler should run as a **deterministic reconciliation loop** over endpoints, not as ad hoc logic on request submission. The competitor pattern is clear: Vast exposes queue-time and utilization-driven parameters, Novita exposes queue-latency or queue-concurrency policies, and RunPod calls out queue delay and concurrency/runsync-oriented scaling. Xcelsior should adopt that same control shape. citeturn44view4turn52view3turn46search14

The reconciliation algorithm I recommend is:

1. For each active endpoint, read durable state: queued jobs, in-progress jobs, active workers, worker concurrency in use, oldest queued age, recent arrival rate, and recent p95 execution time.
2. Compute free slots as `sum(worker.max_concurrency - worker.current_concurrency for ready workers)`.
3. Compute `desired_by_queue = ceil(max(0, queued_jobs - free_slots) / max_concurrency) + active_workers`.
4. If policy type is queue-wait, also compute `desired_by_wait` from oldest queue age or target queue wait budget.
5. Clamp `desired = max(min_workers, min(max_workers, max(active_workers, desired_by_queue, desired_by_wait)))`.
6. Scale up immediately when queued work exceeds free capacity or queue wait breaches SLO.
7. Scale down only when workers have been idle longer than `idle_timeout_sec`, no queued jobs remain, and a cooldown window has elapsed.
8. Drain before reap: mark worker `draining`, stop assigning new jobs, and terminate only after `current_concurrency == 0`.

That algorithm matches competitor behavior while fitting Xcelsior’s existing scheduler/agent design. citeturn44view3turn44view4turn52view3turn47view2

From a runtime-placement perspective, the autoscaler should *request placement* by creating or deleting scheduler jobs, not by directly manipulating hosts. Xcelsior already has a host-fleet autoscale monitor in `scheduler.py`, a background-task pattern in `api.py`, and durable queue-processing logic. The serverless reconciler should therefore live as a new periodic background task or separate long-running worker process that submits worker jobs through the scheduler, observes them through Postgres, and never bypasses the existing placement engine. citeturn36view1turn36view2turn29view1turn36view4

To make it crash-safe and idempotent, use the repo’s existing Postgres patterns: either a leader lease in the `state` table or a Postgres advisory lock for the reconcile loop, and `FOR UPDATE SKIP LOCKED` when claiming queued `serverless_jobs` rows for dispatch to workers. Xcelsior already uses state persistence and `SKIP LOCKED` in multiple places, so the implementation style is already established. citeturn25view0turn34view2turn36view3

For cold-start mitigation, copy the features customers now assume exist: **keep-warm**, **pre-pulled image**, and **persistent weight cache**. RunPod documents always-on workers, idle timeout, and FlashBoot-style faster restarts; Vast’s docs make clear that true first-deploy startup can be minutes long because workers need to start, download models, and pass health checks; Novita exposes worker timeouts, storage mounts, concurrency, and health configuration at the endpoint layer. Xcelsior already has the building blocks for a practical equivalent: the worker agent can surface popular images for pre-pull decisions, the scheduler can submit managed worker jobs, and the NFS volume engine can hold reusable model files. citeturn47view2turn44view2turn52view3turn34view4turn22view0turn21view0

A realistic cold-start statement for Xcelsior today should be explicit and conservative. Based on the current code, `InferenceEngine.provision_worker()` only waits up to about 60 seconds for a worker container to reach running state, which means synchronous first-hit latency will still be poor unless you add pre-pull and cache reuse. My best engineering estimate is: first deployment or uncached model load should be treated as **minutes**, not seconds; a pre-pulled image plus NFS-cached weights on an already-available host can plausibly bring smaller 7B–8B class models into a **tens-of-seconds** readiness band; larger models will still often be **one to several minutes** unless you keep workers warm. That is an inference from the current Xcelsior control plane and the competitor docs, not a measured benchmark, so it should be documented as a target rather than a promise. citeturn44view2turn47view2turn22view0turn21view0

## API surface, streaming contract, and worker SDK

The public API should expose **two complementary surfaces**, because that is the combination competitors have normalized and because it maps well onto Xcelsior’s architecture.

The first surface is the **queue-oriented endpoint API**, modeled directly on RunPod-style async/sync jobs:

- `POST /v2/{endpoint_id}/run`
- `POST /v2/{endpoint_id}/runsync`
- `GET /v2/{endpoint_id}/status/{job_id}`
- `GET /v2/{endpoint_id}/stream/{job_id}`
- `POST /v2/{endpoint_id}/cancel/{job_id}`

This is the guaranteed interface for all endpoints, including arbitrary custom handlers. It should accept `Idempotency-Key`, optional `webhook`, and return the status enum you specified: `IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | CANCELLED`. RunPod’s docs make the async `/run`, sync `/runsync`, and status retrieval pattern explicit, and Xcelsior already has the SSE and long-poll precedents needed to implement it cleanly. citeturn46search1turn46search11turn46search12turn28view0turn32view0

The second surface is the **OpenAI-compatible per-endpoint surface**. The cleanest day-one implementation is path-based, because it avoids vanity-domain work and still preserves “change base_url + api_key only” for SDK users:

- `POST /v2/{endpoint_id}/openai/v1/chat/completions`
- `POST /v2/{endpoint_id}/openai/v1/completions`
- `POST /v2/{endpoint_id}/openai/v1/embeddings`
- `GET /v2/{endpoint_id}/openai/v1/models`

That closely mirrors RunPod’s documented OpenAI base URL pattern and Novita’s “set base URL to our OpenAI-compatible endpoint” model. For managed vLLM presets, these routes should proxy directly into the local vLLM OpenAI server running inside the worker. For endpoints whose task type does not support a route—most importantly embeddings on a text-generation-only model—the API should fail explicitly rather than pretend universal support. vLLM’s own docs are clear that `/v1/embeddings` only applies to embedding models. citeturn47view4turn44view0turn50view0turn55view0

Streaming should be standard SSE: `text/event-stream`, `data: {...}\n\n` chunks, optional `: ping\n\n` heartbeats, and a terminal `data: [DONE]\n\n`. The current Xcelsior `/v1/inference` prototype already emits OpenAI-style `data:` chunks and `[DONE]`, and existing instance log streaming already uses native FastAPI `StreamingResponse` with the right SSE content type and anti-buffering headers. citeturn32view0turn28view0

For management APIs, add or expand the existing inference router so the full endpoint lifecycle is represented in one place:

- `POST /api/v2/serverless/endpoints`
- `GET /api/v2/serverless/endpoints`
- `GET /api/v2/serverless/endpoints/{endpoint_id}`
- `PATCH /api/v2/serverless/endpoints/{endpoint_id}`
- `DELETE /api/v2/serverless/endpoints/{endpoint_id}`
- `GET /api/v2/serverless/endpoints/{endpoint_id}/workers`
- `GET /api/v2/serverless/endpoints/{endpoint_id}/jobs`
- `GET /api/v2/serverless/endpoints/{endpoint_id}/metrics`
- `GET /api/v2/serverless/endpoints/{endpoint_id}/logs`
- `POST /api/v2/serverless/endpoints/{endpoint_id}/keys`
- `GET /api/v2/serverless/endpoints/{endpoint_id}/keys`
- `DELETE /api/v2/serverless/endpoints/{endpoint_id}/keys/{key_id}`

These should use the current Xcelsior session/team auth for dashboard management, preserve the 404-on-cross-account rule, and emit SSE events such as `serverless_endpoint.created`, `serverless_worker.scaled`, and `serverless_job.completed` through the existing event bus. citeturn31view0turn27view0turn18view0

The worker SDK should also follow the competitor pattern, but with two entry modes:

- **Queue handler mode**: `handler(job) -> output`, with support for sync return, async return, sync generator, or async generator. This is the direct analogue of RunPod’s handler contract and is the easiest way to support arbitrary custom jobs plus `/run`, `/runsync`, and `/stream`. citeturn47view0turn47view1
- **ASGI mode**: `serve_asgi(app)` for users who want to run FastAPI/Starlette/Flask-style custom HTTP apps inside their worker. This matches RunPod’s load-balancing guidance and is how you should package the managed vLLM preset as well. citeturn47view3

The SDK contract should normalize three things the platform needs but user containers should not have to reimplement: structured progress events, final output aggregation for sync callers, and canonical error envelopes with `status_code`, `code`, `message`, and `retryable`. That will make the async/sync APIs, webhooks, and UI test console all behave consistently even when custom containers vary internally. citeturn47view1turn46search13

## Frontend implementation plan

The frontend should add a **new Serverless section** inside the existing dashboard rather than inventing a separate app. The current frontend already has strict TypeScript, a reusable `apiFetch()` client, team/billing fields on `AuthUser`, Recharts for visualizations, and Sonner for inline user feedback. Those are exactly the tools you need for a serverless console. citeturn17view0turn17view3turn38view1turn38view5

The create-endpoint flow should be a guided wizard with two first-class modes: **Managed model preset** and **Custom container**. The managed model path should expose task type, model, GPU tier, region, min/max workers, idle timeout, concurrency, timeout, and cache/keep-warm toggles. The custom path should expose image, startup command, health path, environment/secrets reference, optional network cache mount, and advanced scaling controls. The GPU tier selector should reuse the canonical GPU catalog already checked into `frontend/src/lib/gpu-models.ts`. citeturn37view1

The endpoint detail page should have four panels that stay visible without excessive tab-hopping: a live status/metrics header, a request log and recent jobs panel, a worker fleet panel, and a “Try it” console. The test console should generate both cURL and official OpenAI SDK snippets using the endpoint-specific base URL. For managed vLLM presets, the page should prominently show the OpenAI-compatible base URL and API key instructions, because that is one of the biggest adoption levers competitors emphasize. citeturn47view4turn44view0

For realtime updates, the frontend should stay on SSE/EventSource rather than adding websocket-specific client infrastructure unless a concrete UX requires bidirectional transport. The backend already has an SSE bus, and the job-log stream already uses SSE semantics. That will be sufficient for endpoint status, queue depth, active workers, recent completions, and token streaming in the test console. citeturn18view0turn28view0

Team-awareness should be built in from the first commit. The auth model already exposes `team_id`, `team_role`, `team_plan`, `team_can_manage_billing`, and `team_can_write_instances`, and the frontend already has user-facing copy for “viewer access is read-only.” Serverless should follow the same interaction model: viewers can inspect endpoints, logs, and costs, but cannot create endpoints, delete them, rotate keys, or run mutating tests. citeturn38view4turn38view5turn28view4

For internationalization, add the new strings in the existing English and French locale dictionaries rather than embedding prose in components. I could confirm the repo’s frontend stack and API layer, but I could not fully enumerate the exact locale-file paths from the public raw-file inspection, so the final in-tree insertion points for `en` and `fr` strings should be confirmed before editing. citeturn17view0turn17view3

## Step-by-step file-level implementation plan

The files below are the ones I would change first because the current serverless logic already anchors there: `routes/inference.py`, `inference.py`, `inference_store.py`, `api.py`, `routes._deps.py`, `billing.py`, `worker_agent.py`, and the Alembic migration tree. The frontend already has an API client in `frontend/src/lib/api.ts` and a canonical GPU model catalog in `frontend/src/lib/gpu-models.ts`, so those are the right integration points on the UI side. citeturn31view0turn22view0turn23view4turn29view1turn18view0turn35view0turn33view0turn41view1turn38view1turn37view1

1. **Add the schema migration**
   - Create `migrations/versions/<revision>_serverless_endpoints_workers_jobs_keys.py`.
   - Add `serverless_endpoints`, `serverless_workers`, `serverless_jobs`, `serverless_job_stream_events`, and `serverless_api_keys`.
   - Add hot-path indexes for:
     - queued jobs per endpoint
     - active workers per endpoint
     - idempotency-key lookup
     - time-range metrics queries
   - If useful for observability, add a lightweight view or materialized view for endpoint-level rollups.
   - Do **not** wedge these tables into ad hoc bootstrap code; keep them Alembic-managed to match the repo’s migration workflow. citeturn41view0turn41view1turn42view1

2. **Replace the current inference engine with a richer service layer**
   - Keep `inference.py` as a backwards-compatible façade if other modules already import it.
   - Add a new package such as:
     - `serverless/service.py`
     - `serverless/repo.py`
     - `serverless/autoscaler.py`
     - `serverless/dispatcher.py`
     - `serverless/openai_proxy.py`
     - `serverless/metering.py`
     - `serverless/streams.py`
     - `serverless/keys.py`
   - Move endpoint CRUD, worker lifecycle, queue dispatch, OpenAI proxying, and metering out of the current monolithic `InferenceEngine`.
   - Either retire `inference_store.py` or reduce it to a compatibility wrapper that calls the new repository layer. citeturn22view0turn23view4

3. **Rewrite `routes/inference.py` around the final API contract**
   - Keep the route module name so `routes.__init__.py` does not need a structural change.
   - Replace the current prototype paths with:
     - endpoint management routes
     - queue-style `/run | /runsync | /status | /stream | /cancel`
     - path-based OpenAI routes under `/v2/{endpoint_id}/openai/v1/...`
   - Preserve or extend current scope checks rather than inventing a new auth style.
   - Keep the route module purely orchestration-focused: request models, auth, validation, response shapes, SSE formatting. Put business logic in the service layer. citeturn30view0turn31view0turn32view2turn32view3

4. **Extend `routes/_deps.py` instead of creating a parallel authorization subsystem**
   - Add serverless-specific scope helpers such as `serverless:read`, `serverless:write`, and endpoint-key auth helpers.
   - Add reusable resource guards:
     - `_require_serverless_endpoint_read`
     - `_require_serverless_endpoint_write`
     - `_serverless_scope_owner_id`
     - `_serverless_owner_ids_readable`
   - Reuse the existing rate-limit, hashing, and SSE helper style.
   - Return the same style of rate-limit headers the rest of the API uses. citeturn18view0turn26view0

5. **Integrate with the scheduler instead of bypassing it**
   - In `serverless/dispatcher.py`, submit worker jobs through existing scheduler functions.
   - Store the returned scheduler job ID on `serverless_workers.scheduler_job_id`.
   - Use existing job logs, agent callbacks, and host placement data instead of creating a new side channel.
   - If needed, add a small scheduler helper for “submit managed serverless worker” so worker-job payloads are standardized. citeturn36view5turn36view4turn33view0

6. **Add metering to `billing.py`**
   - Introduce a serverless line-item function such as `charge_serverless_execution(...)`.
   - Write usage into `billing_cycles` with a new `resource_type` like `serverless_gpu`.
   - Reuse wallet preflight on submit and the existing suspended-wallet enforcement model.
   - Remove token-count charging as the main debit path from the serverless completion flow; keep tokens as metadata only. citeturn23view3turn35view0turn35view1turn35view2turn35view5

7. **Teach the worker side about serverless state**
   - In `worker_agent.py`, add structured callbacks for:
     - worker ready / draining / exited
     - current concurrency
     - first token / completion
     - GPU utilization snapshot
     - model load / cache hit / cache miss
   - In `routes/agent.py`, add internal callbacks or reuse the current log/telemetry/report paths for serverless lifecycle events.
   - If implementing explicit image pre-pull or cache warmup, add a minimal new agent command rather than a second side channel. citeturn33view0turn34view4

8. **Use the existing volume engine for weight caching**
   - Add a helper module such as `serverless/cache.py` that provisions or resolves a managed NFS cache volume.
   - Mount that cache into managed vLLM workers and optionally custom workers.
   - Record cache binding on the endpoint so warm restarts can reuse it. citeturn20view3turn21view0

9. **Register the reconcile loop in `api.py`**
   - Add a periodic task such as `serverless_reconcile` beside the existing billing and inference background tasks.
   - Add a separate cleanup task for expired stream chunks and old completed jobs.
   - Keep the task idempotent and safe to disable in tests via existing background-task env flags. citeturn29view1turn42view1

10. **Add frontend API bindings and UI modules**
    - Extend `frontend/src/lib/api.ts` with:
      - endpoint CRUD
      - key CRUD
      - metrics fetchers
      - sync/async invocation helpers
      - EventSource/SSE helpers for live output
    - Add a new feature folder such as:
      - `frontend/src/features/serverless/types.ts`
      - `frontend/src/features/serverless/create-endpoint-form.tsx`
      - `frontend/src/features/serverless/endpoint-detail.tsx`
      - `frontend/src/features/serverless/metrics-panel.tsx`
      - `frontend/src/features/serverless/workers-panel.tsx`
      - `frontend/src/features/serverless/logs-panel.tsx`
      - `frontend/src/features/serverless/try-it-console.tsx`
      - `frontend/src/features/serverless/keys-panel.tsx`
    - Add route-entry files under the existing dashboard route tree for:
      - endpoint list
      - create endpoint
      - endpoint detail
    - Add `en` and `fr` strings in the repo’s existing locale dictionaries. citeturn38view1turn38view5turn37view1turn17view0

11. **Fill in tests before editing production behavior**
    - Backend:
      - `tests/test_serverless_routes.py`
      - `tests/test_serverless_tenancy.py`
      - `tests/test_serverless_billing.py`
      - `tests/test_serverless_autoscaler.py`
      - `tests/test_serverless_openai_compat.py`
      - `tests/test_serverless_idempotency.py`
    - Frontend:
      - Vitest coverage for forms, detail panels, and viewer read-only states
      - Playwright smoke for create → invoke → stream → key rotate
    - Workflow changes should be minimal because CI already runs pyright, migrations, and the full test suite. citeturn42view1

## Open questions and limitations

The largest product question is whether **every** endpoint must expose `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`, or whether the OpenAI surface is only guaranteed for endpoints whose model/task type actually supports those routes. vLLM’s own API support is model-type dependent, so forcing universal support would either require wrappers that fake capability or would create misleading APIs. citeturn55view0

The largest billing question is whether **cold-start seconds** are billable. Competitors frame pricing differently: Vast highlights paying for execution rather than rental time, while Novita documents worker-running-duration pricing. Xcelsior should make this explicit early because it affects both UX expectations and the metering design. citeturn44view1turn52view2

The largest UX question is whether day-one OpenAI compatibility should be **path-based only**—for example `.../v2/{endpoint_id}/openai/v1`—or whether product insists on vanity per-endpoint hostnames immediately. Path-based base URLs are enough for OpenAI SDK compatibility and match a documented RunPod pattern, so I would ship that first. citeturn47view4turn44view0

The main implementation uncertainty from repo inspection is the exact in-tree location of the existing locale dictionaries and final dashboard route files on the frontend. The public raw-file inspection was enough to confirm the frontend stack, API layer, and reusable GPU catalog, but not enough to enumerate every existing route and i18n filename. That is a small verification task before editing, not a blocker to the architecture above.


