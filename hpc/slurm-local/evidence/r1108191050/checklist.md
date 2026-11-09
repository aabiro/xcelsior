# Adapter validation run r1108191050

- Profile: `xcl-local`  ·  transport: ssh xcl-tower
- Adapter: 3.0.0  ·  started 2026-11-08 19:10:50  ·  finished 2026-11-08 19:19:18

| Case | Title | Result | Note |
|---|---|---|---|
| ENV-1 | Cluster shape seen through the adapter | **PASS** |  |
| GPU-1 | GPU job through the adapter runs nvidia-smi; GRES honoured | **PASS** |  |
| GPU-2 | CPU-only job on the GPU node cannot see the GPU | **PASS** |  |
| CPU-1 | CPU job targeted at the CPU node | **PASS** |  |
| LIFE-1 | Lifecycle: pending / running / completed / cancelled-while-pending | **PASS** |  |
| LIFE-2 | Lifecycle: failed (non-zero exit) | **PASS** |  |
| LIFE-3 | Lifecycle: cancelled while running | **PASS** | second cancel → already_terminal |
| LIFE-4 | Lifecycle: walltime exceeded | **PASS** |  |
| LIFE-5 | sacct fallback after the controller forgets the job | **PASS** |  |
| FAIL-1 | Unsatisfiable GPU count | **PASS** | error_code=unsatisfiable_request in 0.36s |
| FAIL-2 | Bad partition | **PASS** | error_code=invalid_partition in 0.34s |
| FAIL-3 | Unsatisfiable memory | **PASS** | error_code=unsatisfiable_request in 0.35s |
| FAIL-4 | Walltime over the limit | **PASS** |  |
| FAIL-5 | Unknown node | **PASS** | error_code=invalid_nodelist in 0.35s |
| FAIL-6 | Hostile job metadata is inert | **PASS** |  |
| FAIL-7 | Dead transport fails fast | **PASS** |  |
| FAIL-8 | GPU type the cluster does not have | **PASS** | error_code=unsatisfiable_request |
| FAIL-9 | Cancel an id Slurm has never seen | **PASS** |  |
| SEC-1 | A job can't widen the adapter's key | **PASS** |  |
| SEC-2 | The gate checks batch scripts | **PASS** |  |
| METER-1 | Accounting timestamps in billing's shape | **PASS** |  |
| METER-2 | Non-billable and edge cases | **PASS** |  |
| METER-3 | Active metering of a running job | **PASS** |  |
| POD-1 | Pod listing backed by the real cluster | **PASS** |  |
| POD-2 | An overclaimed listing is rejected | **PASS** |  |
| SYNC-1 | Reconcile loop against real jobs (optional) | **PASS** |  |
| SYNC-2 | A reconcile pass is batched (optional) | **PASS** |  |
| REG-1 | Reconciling through the cluster registry (optional) | **PASS** |  |
| PROG-1 | A job reports its own progress (optional) | **PASS** |  |
| VERDICT-1 | The success verdict reads DerivedExitCode (optional) | **PASS** |  |
| SIZE-1 | Output is read only when it changed (optional) | **PASS** |  |
| SCEN-1 | Requeue keeps the mapping open | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |
| SCEN-2 | Spot preemption requeues, never settles early | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |
| SCEN-3 | Out of memory settles as failed | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |
| SCEN-4 | A hold is held, not failed | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |
| SCEN-5 | Suspension is suspended, not pending | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |
| SCEN-6 | A node failure never settles the job early | **SKIP** | run with --scenarios (and --with-db): needs admin scontrol on the controller host |

Totals: {'PASS': 31, 'FAIL': 0, 'SKIP': 6, 'ERROR': 0}

Evidence for every case: results.json.
