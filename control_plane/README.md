# Xcelsior Control Plane

This directory contains the production control-plane components. For full architectural details, see the [Control Plane Blueprint](../docs/xcelsior-production-control-plane-mcp-blueprint.md).

## Core Control Loop

This high-level diagram illustrates the primary operational loop of the control plane, highlighting how intent becomes execution through the database as the sole source of truth.

```mermaid
flowchart TD
    subgraph Control Plane
        API["API Server\n(Gateway)"]
        DB[("PostgreSQL\n(Source of Truth)")]
        SCH["Scheduler\n(Placement)"]
        REC["Reconciler\n(Drift Repair)"]
    end

    subgraph Data Plane
        WA["Worker Agent"]
        GPU["GPU Container"]
    end

    User((User/Client)) -->|"1. Desires Execution"| API
    API -->|"2. Persists Intent"| DB
    
    SCH -->|"3. Claims Queue &\nReserves GPU"| DB
    SCH -->|"4. Writes Command"| DB

    WA <-->|"5. Polls Commands &\nReports Status"| API
    WA -->|"6. Executes"| GPU
    
    REC -->|"7. Reads Desired vs Observed"| DB
    REC -->|"8. Repairs State / Requeues"| DB
```

## Target Architecture

The full system topology, including observability, ingress, and background services.

```mermaid
flowchart TB
    subgraph Clients["Product surfaces"]
        UI["Next.js dashboard"]
        MC["MCP clients"]
        SDK["REST / SDK clients"]
    end

    subgraph Edge["Public edge"]
        NG["Nginx public ingress"]
        AGW["Agent mTLS gateway"]
    end

    subgraph Stateless["Stateless control-plane services"]
        MCP["MCP gateway replicas"]
        API["FastAPI replicas"]
        SCH["Scheduler replicas"]
        REC["Reconciler replicas"]
        OUT["Outbox dispatcher replicas"]
        MAINT["Maintenance scheduler/workers"]
        VOL["Privileged volume provisioner"]
    end

    subgraph Data["Authoritative data"]
        PG["HA PostgreSQL"]
        REDIS["HA Redis cache/rate limits"]
        OBJ["Object storage / audit checkpoints"]
    end

    subgraph Fleet["Provider GPU fleet"]
        WA["Worker agent + SPIFFE identity"]
        GPU["Container runtime + GPUs"]
    end

    subgraph Observe["Observability"]
        OTEL["OpenTelemetry Collector"]
        PROM["Prometheus + Alertmanager"]
        GRAF["Grafana"]
        LOKI["Loki"]
        TEMPO["Tempo or retained Jaeger transition"]
    end

    UI --> NG
    SDK --> NG
    MC --> NG --> MCP
    MCP --> API
    NG --> API
    API --> PG
    API --> REDIS
    API -->|"desired state + outbox"| PG
    SCH -->|"claim/reserve/bind"| PG
    REC -->|"desired vs observed"| PG
    OUT -->|"SSE/webhook/audit/billing intents"| PG
    MAINT --> PG
    API -->|"durable volume command"| VOL
    VOL --> PG
    WA --> AGW --> API
    WA -->|"work/commands"| AGW
    WA --> GPU
    API --> OBJ
    OUT --> OBJ
    MCP --> OTEL
    API --> OTEL
    SCH --> OTEL
    REC --> OTEL
    WA --> OTEL
    OTEL --> PROM
    OTEL --> LOKI
    OTEL --> TEMPO
    PROM --> GRAF
    LOKI --> GRAF
    TEMPO --> GRAF
```
