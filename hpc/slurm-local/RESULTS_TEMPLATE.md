# Xcelsior Slurm adapter: local integration test results

*Attachment to a research proposal. Replace every `[…]` and delete this line.*

## Summary

Xcelsior is a GPU compute marketplace whose orchestration layer is Slurm-native and integration-ready. It submits work into Slurm-run clusters through an adapter instead of replacing their scheduler. Slurm schedules GPUs inside a cluster; Xcelsior schedules across clusters.

On [date] we validated the adapter end to end against a two-node Slurm [25.11.2] cluster built on our own hardware. [N] of [N] test cases passed. This document reports what was tested, the results, and what a larger academic cluster would test that ours cannot.

The adapter has not been run on any Alliance or SHARCNET system. Its cluster profiles for those systems are written from public documentation.

## Test environment

| | Node 1 | Node 2 |
|---|---|---|
| Role | controller, accounting, compute | compute |
| CPU / RAM offered to Slurm | Ryzen 9 5950X; 8 CPUs/job, 8 GB | Ryzen 9 4900H; 8 CPUs/job, 4 GB |
| GPU | 1 × NVIDIA RTX 3060 12 GB (GRES `gpu:rtx_3060`, NVML autodetect) | none |
| OS / Slurm | Ubuntu 26.04.1, Slurm [25.11.2], cgroup v2 | same, config-less |
| Interconnect | 1 GbE + WireGuard overlay | same |

Isolation: `select/cons_tres`, `task/cgroup` with `ConstrainDevices`, and accounting through slurmdbd. The adapter reached the controller over one SSH key restricted by a forced-command allowlist. That is the access model of Alliance automation nodes.

## Results

| Area | What was tested | Cases | Passed |
|---|---|---|---|
| Placement | GPU job placed by GRES alone; CPU job pinned to a node; GPU invisible to non-GPU jobs | GPU-1, GPU-2, CPU-1 | [3/3] |
| Lifecycle | pending, running, completed, failed, cancelled (pending and running) and timeout, each mapped correctly from both `squeue` and `sacct` | LIFE-1 to LIFE-5 | [5/5] |
| Failure handling | unsatisfiable GPU count, memory and walltime; unknown partition, node and GPU type; hostile job metadata; unreachable submit host; cancel of an unknown job. Each returned a machine-readable error code, created no job, and never hung | FAIL-1 to FAIL-9 | [9/9] |
| Metering | per-second start/end from Slurm accounting, matching the job's own clock; non-billable and edge cases | METER-1 to METER-3 | [3/3] |
| Pod data model | the cluster described as a marketplace listing whose GPU, NVLink and NIC specs were verified by scheduler-run probe jobs; an overclaimed listing rejected | POD-1, POD-2 | [2/2] |

Measured values:

| Metric | Value |
|---|---|
| Submit-time rejection latency (FAIL-1 to FAIL-5, FAIL-8) | [max … s] |
| Metering agreement, accounting vs in-job clock | [± … s] |
| Unreachable-host detection | [… s] |
| Status fallback to `sacct` after MinJobAge | [verified / not] |

Evidence: per-case job IDs, state traces, accounting rows and outputs are in the run's `results.json` [run id].

## Limitations of a two-node test

- No high-speed fabric. NIC speed was verified at 1 Gb/s; NVLink domain size is 1.
- No shared filesystem. Output from the second node was read on that node.
- One GPU, so no multi-GPU or multi-node GPU jobs and no NCCL collectives.
- Our own scheduler policy. Real clusters assign partitions automatically, enforce allocation accounts and fair-share, and admit automation only through approved robot nodes.
- Throughput was not tested: one adapter, tens of jobs.

## What scale validation on an academic cluster would answer

1. **Policy compatibility.** Does the adapter's job translation hold under automatic partition assignment, allocation accounts and site QoS? The cluster profile would be corrected from the site's real behaviour.
2. **Automation path.** Can a restricted robot-node key carry the full lifecycle (submit, poll, cancel, fetch output) under the site's MFA policy?
3. **Accounting fidelity at scale.** Does per-second metering from `sacct` stay exact across multi-node, multi-GPU (H100) jobs, preemption and requeue?
4. **Verified specs on a real fabric.** Can scheduler-run probe jobs verify NVLink domains and fabric bandwidth (NCCL all-reduce)? This is the pod data model's hard case.
5. **Reconcile behaviour.** Does state reconciliation stay correct and cheap under thousands of jobs and controller restarts?

Use would be research and integration testing only, under the sponsoring PI's allocation. No commercial workloads would run on the cluster. Automation would be deterministic, scripted adapter traffic only: the fixed command set above, through the site's automation node and wrapper. Following Alliance's automation-node policy, no AI-agent-originated submissions would reach the cluster.

## Reproducibility

The configs, install and rollback scripts, test suite and a container rehearsal are in the Xcelsior repository under `hpc/slurm-local/` [commit …]. Every change to the test machines was recorded and reversed by script.
