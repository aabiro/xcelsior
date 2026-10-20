# Shared legacy metrics and native monitoring plan — 2026-10-20

**Production follow-up:** the approved increment is now deployed, including
signed GPU workers and the exact native monitoring additions. Live checks pass;
repairs remain report-only and scheduler scope remains canary. See the
[production rollout evidence and remaining gates](2026-10-20-control-plane-production-rollout.md).
The planning and local acceptance record below describes the pre-rollout state.

The local candidate closes the observed Gunicorn counter-aggregation defect.
It also prepares and validates the two additions needed in the existing native
production Prometheus configuration. Production was inspected and validation
was performed through stdin; no service configuration was applied or reloaded.
Repair policies and scheduler scope were not changed.

## Worker aggregation

The image and both API Compose services now start through
`/bin/sh api/scripts/start_api.sh`. Before Python starts, the launcher allocates
a fresh mode-0700 multiprocess directory and records the master generation's
start time, then execs Gunicorn. All workers share that directory. Worker-exit
cleanup retains cumulative counter files. The API container's tmpfs owns their
lifetime; a new master gets a fresh generation rather than reusing stale totals.

Gunicorn refuses a missing or late setup, even with one worker: that worker can
be replaced and lose process-local counters too. Standalone development remains
available, with aggregation explicitly advertised as 0. The production launcher
advertises aggregation 1 after validating the shared setup.

`core/prometheus_export.py` merges counter, histogram and summary families with
the client library's multiprocess collector. Callback/default gauges and custom
collectors remain on their live registry, avoiding duplicate families and the
standard collector's unsupported callback-gauge behavior. The retirement policy
gauges therefore remain correct. This follows the library's
[multiprocess storage and lifecycle requirements](https://prometheus.github.io/client_python/multiprocess/).

The scrape fails with HTTP 503 for missing or malformed shared storage, missing
shared legacy counters, or collector failure. It never substitutes a local zero
or silently returns a partial HTTP 200. Generation start and aggregation status
are exported alongside usage. A zero-use window needs aggregation 1, complete
path series and continuous scrapes on every expected target, with a stable
generation that predates the window. A new generation invalidates the window.

## Executed acceptance

The initial six regressions failed against the prior code: exited workers'
requests/histograms were invisible, collector failure returned 200, and the
launcher/startup guard did not exist.

The expanded tests verify normal exit and SIGKILL before the first scrape,
retained counter/histogram totals, unchanged callback gauges, a new private
generation for each master, missing/invalid storage failing with 503, missing or
late setup rejected for one/two Gunicorn workers, and the actual exported
aggregation/generation contract.

A real two-worker Gunicorn server uses the real API and metrics route against
the disposable PostgreSQL database. Both workers record synthetic observations;
one is killed before any scrape; Gunicorn replaces it; six later HTTP scrapes
all retain the full count. Generation remains unchanged across replacement.
The harness uses a reserved loopback socket, synthetic credentials, a private
process group and bounded cleanup. No GPU or production workload is needed for
this metrics-only gate.

**336 selected tests passed**, with zero failures/errors/skips, including the
existing observability, auth/ingress, metrics, startup and fenced-agent checks.
Ruff, targeted Pyright and whitespace checks passed. Runtime sources are pinned
in the [machine-readable evidence](2026-10-20-shared-legacy-metrics-evidence.json).

## Native production monitoring plan

`api/scripts/plan_native_prometheus.py` reads the existing configuration, adds
only the requested loopback replica inventory and isolated retirement-rule
reference, and validates both with the installed native `promtool`. It reports
structural additions and hashes without exposing existing credentials. It never
reloads services or modifies the source. An explicit output option writes only
a new private mode-0600 candidate file.

The actual production run used native Prometheus **2.45.3**, whose syntax and
rule checks both exited 0. The plan has exactly two operations:

1. Add job `xcelsior-api`, path `/metrics/prometheus`, target
   `127.0.0.1:9500`, replica `green`.
2. Add rule-file reference
   `/etc/prometheus/xcelsior-rules/legacy-retirement.yml`, containing only the
   `xcelsior.legacy-retirement` group.

Existing scrape jobs, rule references, global settings and Alertmanager routing
are preserved. The original configuration hash is
`6841aa84c7ffd334f8eff7ed66072ff94c3630f6d7acb9c014913b6eb00beb7f`.
The candidate hash is
`ad305f35e4f98345f30eaf62835d395f147c121984785130b7cee10978a0c781`.
The complete reviewed plan and validation output digests are in the evidence.
No configuration credentials appear in the report. A syntax-only check does not
install referenced files or prove that targets/rules/notifications are live.

An unauthenticated request to the API's private loopback metrics socket returned
HTTP 200 and Prometheus text; the current deployed image does not yet expose the
legacy usage family. This confirms scrape reachability, not deployment of the
new counters.

## Committed candidate package

Implementation commit `82c4a59bbce64b489776e72cd2f7b9f28e52dcd2` built from a
clean worktree as `xcelsior-api:shared-metrics-82c4a59b`, image
`sha256:f937b0e132f817036f8f5dbc1df987213d98c80e5dc36afcd8fb227c32a66f4d`.
The packaged default command uses the new launcher and runs as `10001:10001`.
A network-disabled, read-only probe used the actual launcher and Gunicorn hooks,
recorded seven requests across normal worker exit and SIGKILL, and exported all
seven after both workers were gone. Aggregation readiness was 1, the directory
was mode 0700, all selected sources matched and all twelve distribution
signatures verified. The image contains neither the private environment file
nor the signing key. This is local package proof, not production deployment.

The commit scanner flagged an embedded synthetic password in a planner test.
The fixture now generates that value per process; all 19 planner tests passed
again, and the normal commit scan passed without bypassing it.

## Rollout and remaining gates

Deploy the verified candidate image and launcher/Compose
changes together in API-first order before upgrading signed workers. Preserve
the scoped canary and all report-only repair policies. Back up and verify the
native monitoring source hash before staging its isolated rule file and the
reviewed additions. Run full installed-config validation before the atomic
switch/reload; preserve rollback copies and the other host tenants' monitoring.

After rollout, verify exact source/signature identity, worker journeys, every
active API scrape, aggregation 1, all eight usage/policy paths, stable generation,
loaded retirement rules and receiver delivery. No notification was sent by this
read-only planning run. The agreed zero-use retention window has not started.

Other B16 instrumentation, explicit recovery/retirement disposition for the
twelve stopped rows and the missing volume, contract migration/code removal,
and dependent fairness/preemption remain separate gates. This metrics fix does
not restore workload data or complete B16.
