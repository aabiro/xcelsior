# Native candidate acceptance

The frozen schema-135 candidate passed **3,299 selected tests**, with three
documented baseline exclusions and no failures, errors or skips. After the
later schema-136 lifecycle change, the schema/inventory selection passed 75
checks and found one stale generated inventory. Regeneration and updated
lifecycle classification notes passed **22 checks**, including fresh currency
comparison and isolation guards. No public OpenAPI change was produced.

The hardware harness now admits its workload explicitly, records its real GPU
device in native inventory, and calls the current worker monitor. Normal
running settlement must open exactly one uncharged meter before the harness
deletes that specific disposable meter to qualify missing-meter repair. This
is an intentional local incident injection, not a production repair action.

Five real local CUDA journeys passed: exit 0, exit 7, terminal HTTP failure
before commit, response loss after commit, and terminal recovery across two
separate worker processes. Each verifies CUDA result 32768, native placement,
claim, command delivery, observations, original/repair meter identity,
report-only refusal, exact usage/invoice agreement, settled lease/allocation,
stale-report rejection and container cleanup. The dedicated test database and
API shut down successfully. The first attempt retained as diagnostic evidence
found an obsolete harness argument; the corrected full hardware run passed.

This hardware test manually launches a bounded container after the real claim.
It does not prove public launch, full `run_job` provisioning, wallet debit,
provider payments or deployed private ingress. Source regression and local
hardware success are separate from production acceptance.

The read-only production probe at 2026-10-23 11:56 UTC still found schema 125,
364 historical jobs, no active attempt pointers/leases/device allocations,
dedicated service roles without login, and all eight repair policies set to
`report_only`. The obsolete task row is present but unclaimed. No production
schema, login, task, service or worker was changed. The final combined
schema-136 regression and staged production handoff remain required.
