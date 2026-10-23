# Final control-plane candidate checks

The clean repeat at `5b51d90e` finished with **3,336 passed, zero failures,
errors or skips**; three explicit baseline checks remain deselected. Its owned
PostgreSQL and Redis both cleaned up successfully. The earlier 3,332-pass run
found four obsolete test contracts/fixture shapes, corrected as recorded below.

This receipt describes the pre-operator candidate. The later native host
eviction source, schema 137, regenerated MCP contract and current handoff are
recorded in [the host eviction receipt](2026-10-23-native-host-eviction.md).
The current image and subsequent review/restore evidence are in the
[follow-up receipt](2026-10-23-native-review-and-full-restore.md).

| Original failure | Correction and passing evidence |
|---|---|
| Instance stop/start expected optimistic stopped/running state without an attempt | The API flow now reserves, claims and reports a real native attempt, waits for fenced stop confirmation, then verifies queued restart and a higher fence on the next attempt. `native-flow-volume-final`: 115 passed. |
| Empty-wallet start test never settled its stop | The fixture now confirms the stopped attempt before checking the 402 balance refusal. Same 115-test gate. |
| Volume test required detach during the termination request | The regression now prohibits pre-settlement detach and requires the fenced/unplaced domain paths. Same 115-test gate. |
| Historical refusal mock returned seven columns for a nine-column query | The mock now includes the selected spec/hash fields; the original actionable refusal assertion passes. `historical-refusal-fixture-final`: one passed. |

Before the later operator candidate, no runtime source changed after frozen
image commit `7757bdc3`. Earlier
qualification includes the schema-135 candidate's clean 3,299-test selection,
the later 309-test native lifecycle/wallet gate, 24 corrected wallet/reset
fixture checks, and 22 inventory/classification/isolation checks. These
selections overlap; their counts must not be added as unique coverage.

The clean combined run passed fresh bootstrap to 136, production-shaped migration
and rollback, history/disposition, native intent, archive permissions,
serializable placement, fairness, preemption and worker deadline/journal tests.
The production-derived scoped rehearsal preserved all 364 jobs and execution
history through 135 and rollback/replay, recovering four exact specs and
explicitly retaining 360 unavailable specs. Financial rows are outside that
private fixture; it is not a full production financial restore.

Five real isolated CUDA journeys passed, including terminal HTTP failures and
recovery across separate worker processes. Both frozen images booted as UID
10001 against their intended disposable schemas, passed readiness, verified
twelve signatures and exported `xcelsior_control_plane_metrics_available = 1`
through the configured Prometheus scrape path. The transfer archive contains
those exact images with retained checksums and Config/RootFS identities.
Public launch, full worker provisioning, deployed financial acceptance and
provider payment reconciliation are not implied by that local CUDA proof.

Black and Ruff passed for the 54 Python paths changed through `c0f4af05`; the
three subsequently corrected test files also passed. Targeted runtime Pyright
reported zero errors/warnings. All nine protected user files matched their
saved hashes. Every recorded gate shut down its owned PostgreSQL cluster, and
the combined runner removed its owned Redis container.

The final read-only production probe at 2026-10-23 12:32 UTC still found schema
125, zero active attempt pointers/leases/device allocations, dedicated roles
without login, and all eight repairs `report_only`. No production mutation
occurred. B16 is not marked complete. The staged production role/schema/worker
handoff, remaining observation adapters/module decomposition and three excluded
baseline gates remain explicit in the
[handoff procedure](2026-10-23-staged-control-plane-handoff.md).

The adjacent JSON retains exact arguments, original/focused manifests,
replacement-case mapping, source/image identity, cleanup receipts and protected
file hashes. Failed and superseded runs remain available as diagnostic evidence.
