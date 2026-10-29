# Instance startup and terminal investigation — 2026-10-10

## Production evidence before deployment

SSH from the configured workstation to `prod-hetzner` works. The earlier cloud
environment's missing-key/connection-refused result does not describe this
workstation. Production is `/opt/xcelsior` on `46.225.20.97`; the local GPU
worker is `aaryn-tuf-rtx2060`.

Read-only database checks and API/worker logs showed:

- Jobs `a6049cb5` and `2e09afd9` were assigned and leases claimed. Startup then
  failed pulling `xcelsior/serverless-vllm:12.4` (repository unavailable).
- Interactive job `d7ee7261` failed mounting required volume
  `vol-d39579cb5073`. The NFS server answered that the export path did not exist.
  Cleanup then raised `UnboundLocalError` for `encrypted_ws_vid`, masking the
  original failure.
- The volume's database status was `available` despite its absent directory.
  Its data was not found in the examined October 7 Vultr export backup or
  October 9/10 Hetzner export backups. No empty replacement was created.
- Worker v2 observations were fresh while the payload's legacy `last_seen`
  timestamp was stale. Installation verification only checked `last_seen`.
- Background maintenance was executing, but its container health check tested
  a separate reconciler's heartbeat. The maintenance container was unhealthy.
- Telemetry retention repeatedly failed comparing epoch seconds with a SQL
  timestamp. Idle worker polls also logged `Response content longer than
  Content-Length` because a 204 response carried a JSON `null` body.

## Repairs

- Terminal progress counts as connection activity while an image is pulling.
  A socket is shown as connected only when the shell is ready. Terminal
  initialization and ticket requests discard obsolete async completions,
  including React Strict Mode cleanup/restart. Startup errors survive socket
  closure instead of being replaced by a generic close code.
- `leased` is a valid terminal waiting state. Waiting refreshes the shared
  session lease, and every terminal startup exit reports its actual failure.
- Required-volume and exclusive-GPU startup failures clean up worker state
  and leases without an uninitialized-variable exception.
- Managed vLLM uses the published `vllm/vllm-openai:v0.8.5` image, matching the
  version previously named by the custom Dockerfile. Engine arguments preserve
  the image entrypoint; the process binds its configured health-check port.
  Cache paths are explicit endpoint environment settings. The upstream
  [Docker contract](https://docs.vllm.ai/en/v0.8.5/deployment/docker.html) and live
  registry manifest were checked. Pulled digest:
  `sha256:6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33`.
- Host detail exposes the server's observation timestamp, and worker install
  accepts a fresh v1 heartbeat or v2 observation. It still distinguishes
  heartbeat receipt from marketplace admission.
- Network verification downloads a measured payload using the authenticated,
  host-scoped worker transport. HTTP error pages, incomplete downloads, and
  unavailable ping evidence cannot produce successful measurements.
- Background health checks the task it actually runs. Telemetry pruning uses
  epoch units. Idle polling returns an empty 204 response.
- Worker installers require a durable credential, preserve shell quoting, and
  never invoke platform SSH key generation as a lookup fallback.
- Wizard retry after a lost launch response only adopts an existing instance.
  An empty result is inconclusive and never replays the paid POST. Queued
  instances without a host assignment are recoverable. Durable server-side
  launch idempotency remains necessary for automatic replay.
- Deployment excludes the private signing key and local virtualenv. Docker
  also excludes the signing key. The current production source/image did not
  contain that key. Updated worker signatures were verified with the public key
  already trusted by the installer.

## Verification and remaining operational work

- Wizard: 632 tests across 39 files, build, bundled installer packaging passed.
- Frontend: 385 tests across 66 files, lint (zero errors, 19 existing warnings),
  translation validation, production build, and PWA verification passed.
- Production-build browser scenarios: all 11 passed, including both themes,
  secret modal, clipboard denial, theme changes without reloading, pagination,
  admin grant UI, sidebar return, keyboard/history, and narrow team switching.
  The toaster now mounts with the application; deferred loading could drop
  an action's first notification before the subscriber existed.
- Focused worker/managed-engine checks: 148 passed. Host freshness and fleet
  authorization: 13 passed. Corrected generated-contract/billing checks:
  71 passed.
- Full backend run: 6,772 passed, 70 skipped, one failure in the chat quota
  burst test. Its clock is now fixed so a slow run cannot roll past the
  one-minute quota window; all 41 chat and distributed-quota tests passed on
  rerun. No second full-suite green run is implied.
- Deployment packaging: 26 tests passed, including actual rsync/tar runs that
  exclude the private signing key while retaining its public key/signatures.
- Repository-wide Ruff, deployment shell syntax, and diff whitespace checks
  passed. Frontend tests/build/PWA were repeated after the toaster fix.
- The published vLLM image pulled successfully and performed a CUDA tensor
  operation on the RTX 2060 (vLLM 0.8.5, CUDA 12.4). Its entrypoint accepts the
  documented engine arguments. This is not an inference/model-load test.
- Postdeployment checks are pending at this record's creation. Browser
  fixtures prove UI behavior, not production ledger writes.

Existing serverless endpoint records still name the old image; changing the
default does not rewrite them. Their explicit repair must preserve model,
custom flags, configured port, environment, and scaling intent. GPU inference
readiness is separate from a successful image pull or CUDA initialization.

The missing volume needs its original backing data located or its unavailable
state reconciled. This change does not recover that data. A live interactive
launch, terminal command, cleanup, and fresh API/worker/maintenance log review
are required after deployment.
