# Adapter validation run r1108113648

- Profile: `xcl-local`  ·  transport: ssh xcl-tower
- Adapter: 3.0.0  ·  started 2026-11-08 11:36:48  ·  finished 2026-11-08 11:55:40

| Case | Title | Result | Note |
|---|---|---|---|
| SCEN-1 | Requeue keeps the mapping open | **PASS** | accounting runs: ['REQUEUED', 'COMPLETED'] |
| SCEN-2 | Spot preemption requeues, never settles early | **PASS** | accounting runs: ['PREEMPTED', 'COMPLETED'] |
| SCEN-3 | Out of memory settles as failed | **PASS** | settled as ('failed', 'OUT_OF_MEMORY', True) |
| SCEN-4 | A hold is held, not failed | **PASS** | held ('held', 'SPECIAL_EXIT', False), released running, settled ('cancelled', 'CANCELLED', True) |
| SCEN-5 | Suspension is suspended, not pending | **PASS** | suspended ('suspended', 'SUSPENDED', False), resumed running, settled ('cancelled', 'CANCELLED', True) |
| SCEN-6 | A node failure never settles the job early | **PASS** | state PENDING, mapping ('pending', 'PENDING', False), runs ['NODE_FAIL', 'PENDING'] |

Totals: {'PASS': 6, 'FAIL': 0, 'SKIP': 0, 'ERROR': 0}

Evidence for every case: results.json.
