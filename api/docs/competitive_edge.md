<!-- residency-guard: documents-removal — pre-pivot positioning note -->
# Xcelsior Competitive Edge & Gap Analysis

> **Last updated:** March 30, 2026
> **Compared to:** RunPod · Lambda Cloud · Vast.ai · CoreWeave · Together.ai · Modal · Replicate

---

## 🏆 What Xcelsior HAS That Competitors DON'T

| Advantage | Impact |
|-----------|--------|
| **Canadian data sovereignty** | Province-level residency controls, PIPEDA/Law 25 compliance |
| **CAD-native billing** | Automatic GST/HST per province — no USD conversion friction |
| **AI Compute Access Fund (CAF)** | 67% rebate on Canadian compute — unique cost advantage |
| **Reputation-weighted marketplace** | Trust tiers (Bronze→Diamond) affect pricing power + visibility |
| **SLA auto-enforcement** | Uptime monitoring with automatic credits on breach |
| **Hardware isolation per job** | gVisor/Kata sandboxing — competitors share kernel between tenants |
| **Interactive GPU instances** | ✅ SSH-accessible long-running pods via Tailscale mesh |

---

## 🔴 Critical Gaps (every serious competitor has these)

| Gap | RunPod | Lambda | Vast.ai | Xcelsior Status |
|-----|:------:|:------:|:-------:|-----------------|
| SSH into running instances | ✅ | ✅ | ✅ | ✅ **Implemented** — Tailscale mesh + SSH |
| Long-running pods (keep-alive) | ✅ | ✅ | ✅ | ✅ **Implemented** — interactive mode |
| Persistent volumes | ✅ | ✅ | ✅ | ⚠️ NFS endpoints exist, not provisioned |
| Jupyter/notebook access | ❌ | ✅ | ❌ | 📋 Planned |
| Container templates (1-click) | ✅ | ❌ | ✅ | ✅ **Implemented** — 6 templates (PyTorch, TF, vLLM, ComfyUI, Jupyter, Ubuntu) |
| Serverless inference | ✅ | ❌ | ❌ | ⚠️ Stubbed (`inference.py`) |
| Web terminal | ✅ | ❌ | ✅ | 📋 Planned |
| Real-time GPU availability | ✅ | ✅ | ✅ | ✅ **Implemented** — /gpu-availability public page with live data |

---

## 🟡 Important Gaps (most competitors have)

| Gap | Details |
|-----|---------|
| **Fine-tuning as a service** | Together/Lambda offer managed fine-tuning pipelines |
| **Model API endpoints** | Replicate/Together: deploy model → get REST API endpoint |
| **Auto-scaling** | Modal: scale-to-zero + auto-scale. Xcelsior has stubs only |
| **CLI tool** | `cli.py` exists but isn't published. RunPod/Lambda have pip-installable CLIs |
| **Persistent storage volumes** | RunPod volumes, Lambda persistent filesystems |
| **Usage dashboards** | Detailed burn-rate, projections, cost alerts |

---

## 📊 Pricing Model Comparison

| Model | RunPod | Lambda | Vast.ai | Xcelsior |
|-------|:------:|:------:|:-------:|:--------:|
| On-Demand | ✅ | ✅ | ✅ | ✅ |
| Spot / Interruptible | ✅ | ❌ | ✅ | ✅ |
| Reserved (commitment) | ❌ | ❌ | ❌ | ✅ |
| Priority queue jump | ❌ | ❌ | ❌ | ✅ |
| Native currency (non-USD) | ❌ | ❌ | ❌ | ✅ CAD |
| Government rebate eligible | ❌ | ❌ | ❌ | ✅ CAF |

---

## 🎯 Strategic Priority

1. ~~Interactive instances (SSH + long-running pods)~~ ✅ **DONE**
2. ~~Container templates / 1-click deploy~~ ✅ **DONE** — 6 pre-built templates
3. Serverless inference endpoints
4. ~~Public GPU availability dashboard~~ ✅ **DONE** — /gpu-availability
5. Published CLI (`pip install xcelsior`)
