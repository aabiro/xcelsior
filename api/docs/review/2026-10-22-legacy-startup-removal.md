# Legacy startup-adoption removal — 2026-10-22

The follow-up removes `worker_agent.adopt_running_containers` and its startup
call. A container name beginning with `xcl-` no longer causes discovery, old
lease acquisition, tracking, log forwarding, or monitoring. Startup recovery
uses `v2_adopt_attempts`: the durable attempt journal, exact container labels,
and fenced lease renewal remain the authority. Recovery without a journal
entry does not scan Docker for names, even if an older negotiation response
advertises legacy execution.

Unused scheduler SSH helpers `stop_container_graceful`,
`start_stopped_container`, and `wait_for_job` are physically removed. The
unused `settle_queued_cancel_without_attempt` function is also removed; it
could rewrite a non-queued historical row without enforcing the API's new
status/host CAS. The queued cancellation and fenced lifecycle gates remain
the supported behavior.

Tests that asserted the existence of deleted helpers or exercised old
name-based adoption are removed. The replacement startup gate proves there
is no Docker discovery or old lease request without journal authority.
Existing native recovery tests still exercise report-loss retries, restart,
mismatched labels, preserved container evidence, stale authority, and fencing.

The focused gate passed **266 tests**. The combined isolated PostgreSQL gate
passed **1,387 tests**, with no failures, errors, or skips. The same three
exclusions remain explicit: optional Compose smoke, future B16.2 contraction,
and the absent companion-prose source check. Ruff, changed-file Black,
Pyright, generated artifacts, and all twelve worker signatures passed. The
two existing duplicate operation-ID warnings and whole-tree formatting
backlog are not claimed fixed.

This is source removal in a review candidate. No application or worker was
deployed or restarted, and no historical job, lease, volume, event or state
record was changed. Repairs remain `report_only`. Other legacy lease,
queue, SSH lifecycle/serverless/checkpoint, and JSONB paths still have callers;
B16.2/B16.3 remain open under the
[schema and historical-data prerequisites](2026-10-21-legacy-control-plane-removal.md).
The [machine-readable receipt](2026-10-22-legacy-startup-removal.json) pins the
tested sources and the updated worker signature.
