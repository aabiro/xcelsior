# Control-plane production rollout — 2026-10-20

The approved control-plane candidate is deployed. The API, maintenance worker
and scheduler are healthy on the verified release image. Both GPU workers run
the signed candidate and have confirmed v2 negotiation. Native Prometheus
scrapes the active API and evaluates the retirement rule successfully.
Repairs remain `report_only`, scheduler scope remains the same two-host canary,
and schema remains 125. This closes the deployment gate for this increment;
it does not complete B16 or prove a legacy-use retention window.

## Exact package and preserved production changes

The implementation is `82c4a59bbce64b489776e72cd2f7b9f28e52dcd2`, packaged
with documentation at `b181c1cf06eb3890161c17813631f17c59fb55e0`.
Preflight compared 518 selected source paths and found four already-deployed
files missing from or different in the frozen image: `api/routes/billing.py`,
`billing/stripe_connect.py`, `core/meta_capi.py`, and `core/gitguardian.py`.
Their deployed contents exactly matched the pre-existing local edits.

A small image layer preserves those four exact deployed files. Their hashes
are recorded in the [machine-readable evidence](2026-10-20-control-plane-production-rollout-evidence.json).
No unrelated local edits were committed or rewritten. The resulting image is
`xcelsior-api:shared-metrics-82c4a59b-preserved-live`, local ID
`sha256:026984f3e0ab1cc4a4fde96e6c257fdce974040c0588e4ad533d3ba535d6ac4f`.
The imported engine reports ID
`sha256:07e7b4dceefad6044dd68640f5e6b0c971538a241f74d1c3bdbfe0c54f50af27`.
Before starting containers, the import's complete Config and RootFS layer lists,
creation time, architecture and OS were independently compared with the local
image and matched. The imported ID was then pinned for all handoff checks.

## Validation and handoffs

The isolated PostgreSQL gate passed **360 tests**, with zero failures, errors
or skips, using the preserved production files. This includes the real
two-worker Gunicorn aggregation acceptance, native monitoring planner,
fenced-agent/ingress checks, Meta CAPI and selected Stripe regressions.

A broader optional billing expansion passed 401 tests but exposed three
payment-intent tests that reach external Stripe with synthetic credentials.
Those same three failures reproduce on the unchanged candidate. They remain
an existing billing-test isolation issue; no billing behavior was changed to
make this deployment pass. The passing gate used the frozen test virtualenv.

The new API first ran as the blue standby, passed readiness, source identity,
report-only policy and metrics checks, then received nginx traffic. Green was
recreated and verified before traffic returned to port 9500. The standby was
stopped, leaving one active API scrape target and preserving internal callers'
existing port. Maintenance and scheduler handoffs followed, each with semantic
readiness and policy checks. Twenty selected source hashes match in each of
the three running containers. The effective Compose override pins the image
and launcher; fifteen approved host-source files were aligned afterward.
The production environment file and existing enrollment mounts were preserved.

Both installed GPU workers were updated at their actual systemd entrypoint,
`/home/aaryn/.xcelsior/worker_agent.py`, with all signed dependencies staged
and all twelve signatures checked before installation. Both confirmed
`v2=True legacy_execution=True`, as expected under the preserved canary.
Postflight shows both hosts admitted, ready and reporting observations.
Tower's service was already inactive immediately before its handoff; it was
started with the verified package. Other Tower services were not restarted.
The public distribution also served all twelve expected payloads with valid
signature headers and exact candidate hashes.

Public health, readiness, pricing rates and OAuth-provider endpoints returned
200. The protected pricing-model endpoint returned its expected 401, and the
public metrics path remained blocked with 404. These checks prove their HTTP
contracts, not a customer purchase or production GPU provisioning journey.

## Native monitoring is live

The applied configuration and isolated rules match the exact reviewed hashes:

- Configuration: `ad305f35e4f98345f30eaf62835d395f147c121984785130b7cee10978a0c781`.
- Rules: `0edb120e05e4e18e40669e7b4bd80e63fb38fb2cf74c9914c461bc3f0d0ab0d0`.

Full installed-config validation passed before the switch. Live verification
caught the new rules directory inheriting mode 0700 from the private deployment
umask; it was corrected to 0755 before accepting the reload. The API target is
up, aggregation readiness is 1, all 32 usage series and eight policy series are
present, and eight successive samples over two minutes show no scrape failure
or metrics-generation change. The retirement rule is healthy and inactive.
The eight prior scrape jobs and two prior rule groups remain present.
One active Alertmanager endpoint is configured; receiver delivery was not tested
and no synthetic notification was sent.

The sample already contains **43 work-poll requests and 44 command-drain
requests**. These are caller counts, including empty polls, not proof of legacy
container execution. All eight retirement policy values are 0 under the current
canary. A working scrape and healthy rule therefore do not establish retirement
or an agreed zero-use window. Native dashboard provisioning/verification and
the other B16 path measurements remain separate gates.

## Protected state and shared GPU check

All twelve stopped-job rows match their preflight row hashes. The missing
volume's selected metadata still matches the earlier inventory, with no
attachments or snapshots; it remains a recovery discrepancy. No job was
restored, requeued, retired or deleted, and no replacement volume was created.

At the user's request, VTT was checked read-only on both machines. Both APIs
returned HTTP 200, `status: healthy`, and `active_jobs: 0`; Tower's GPU was idle
at the sampled check. The worker logs showed no rental job starts. No VTT
service was restarted and no GPU acceptance or VTT inference was submitted
during this rollout. API health is not loaded-model or inference proof.

## Remaining gates and rollback

B16.1 remains partial: other legacy paths and native dashboard coverage still
need instrumentation/verification; real callers remain; receiver delivery and
the agreed continuous zero-use retention period are unproven. The stopped rows
and missing volume require an explicit recovery or retirement disposition.
B16.2/B16.3 contract migration and removal, plus dependent fairness/preemption,
remain open. No repair enforcement or scheduler-scope expansion was enabled.

Production rollback copies, prior image IDs, original configuration and source
files are retained under
`/opt/xcelsior-releases/shared-metrics-20261020/rollback`. Each worker retains
its prior installed files under
`/home/aaryn/.xcelsior/releases/shared-metrics-20261020-rollback`.
If reverting to an API without explicit execution negotiation, roll the signed
workers back first, then hand off the services using the prior images and
configuration. Restore the original native Prometheus config separately.
These are prepared rollback materials; no rollback or restore drill was needed
or claimed by this successful rollout.
