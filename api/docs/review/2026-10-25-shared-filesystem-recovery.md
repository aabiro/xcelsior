# Shared filesystem exhaustion and release headroom

The production root filesystem reached 100% during repeated control-plane image
staging. PixelEnhance alerts at 06:34 EDT reported unavailable temporary
directories and Redis `MISCONF` write refusals. Redis's own log contains seven
`No space left on device` entries for the day. Candidate staging added pressure
to the same filesystem used by the other applications; the ad hoc transfer path
had not checked available space first.

Unused Docker build cache older than seven days was removed with bounded cache
retention. Four superseded, never-deployed candidate image tags and their remote
transfer archives were removed only after checking that no container referenced
them and that byte-identical archives remained local. Their extracted source and
receipts remain. Running and rollback images, database backups, volumes and
application data were preserved. Available space subsequently measured
19,019,137,024 bytes.

Authenticated Redis inspection reports a successful RDB snapshot and normal
persistence status. Both production PixelEnhance workers became ready again;
the worker log records subsequent successful periodic tasks. Write, fsync and
unlink checks in `/tmp` and `/var/tmp` passed as the services' actual UID 1001.
No Redis persistence protection was disabled and no queues were purged.

Headscale's replica log separately records a successful refresh at
2026-10-25 10:38:49 UTC, with one user and six nodes. The primary HTTPS health
check returns 200 and DNS still points to 46.225.20.97. The exact error behind
the pictured earlier replication alert was not recovered. This does not prove
standby TLS promotion; no promotion or DNS mutation was performed.

`api/scripts/check_release_disk.py` now refuses release work unless each affected
filesystem can retain at least 10% of its total capacity or 5 GiB, whichever is
larger, after the declared release budget. It uses space available to ordinary
service users, excluding root-reserved blocks. Missing capacity information or
an invalid budget refuses the operation. Deployment backup, source sync, mirror
sync and Docker build paths invoke it before release writes, with an 8 GiB
default budget. Already transferred runtime-only handoffs use a zero incoming
budget while preserving the same reserve. This probe never deletes anything.

The deployment/headroom selection passed **57 checks** through the isolated
security harness. Black, Ruff, Pyright and shell syntax checks pass. The actual
read-only production handoff preflight passed, retaining 16,092,108,391 bytes.
The failed-activation fixture also now injects migration failure at the real
`run_db_migrations.py` entrypoint and verifies that the live colour is preserved.

Runnable gates: `tests/test_release_disk_headroom.py`,
`tests/test_deploy_runtime_readiness.py`,
`tests/test_deploy_failed_activation_is_retried.py`,
`tests/test_deploy_cannot_lose_prod_config_or_ship_secrets.py` and
`tests/test_deploy_skips_what_did_not_change.py` through
`api/scripts/run_security_gates.py`. Control-plane runtime regression/release
qualification is separate; all repair policies remain `report_only`.
