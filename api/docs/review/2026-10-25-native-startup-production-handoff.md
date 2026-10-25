# Executed native worker startup handoff

The qualified b935d694 follow-up is deployed to API, scheduler, background worker,
SSH gateway and both signed GPU workers. Schema remains 141; this handoff ran no
migrations. All eight repairs remain report_only and preemption/deadline force
remain disabled. Existing credentials, worker environments, journals and unrelated
Tower/VTT workloads were preserved. Seven runtime database-role contexts pass.

The frozen source passed 3,897 tests with no failures/errors/skips and 1,129
unchanged source hashes. Its non-root image passed health, readiness, schema,
metrics and all twelve signatures. Importing worker helpers preserves the
caller's signal handlers; the executable defines artifact/fingerprint helpers
before entering the main loop. Explicit provider persistence, canonical host
read projections and the non-lowerable schema floor are included.

A fresh pinned CUDA job, 9e5d8a0e, succeeded on attempt
1e991915-5126-4672-897e-f74dd01e752b, fence 10. Its durable HTTP logs contain the
expected matrix result. It released the attempt, lease and GPU allocation. Across
five acceptance jobs, six usage meters, seven charged cycles and seven wallet
debits agree at 57,400 micro-CAD, with zero ledger mismatches. Invoice preview and
CSV agree with usage and do not create stored invoices. Public signed distribution,
public status, MCP readiness, Prometheus and recovery-timer checks pass.

The [sanitized receipt](2026-10-25-native-startup-production-receipt.json) records
these source, image and runtime checks. The prior schema-141 full restore remains
a dated 155-table recovery proof; this runtime-only handoff does not relabel it as
a new snapshot. External payment transfer and independent offsite/PITR/object
recovery remain unverified.

The v1 tombstone candidate bcec08cb is staged separately and is not deployed.
Its broad gate found 23 stale legacy-fixture failures among 3,923 checks; those
are being corrected with the native queue follow-up. Old placement walkers,
synthetic inventory, native fleet JSONB reads and historical invoice disposition
remain open. No complete control-plane retirement claim is made.
