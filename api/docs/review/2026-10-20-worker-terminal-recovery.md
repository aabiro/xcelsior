# Worker terminal-report recovery — 2026-10-20

An exited v2 batch container could be removed after its terminal HTTP report
failed, discarding the evidence needed to retry. Restart adoption could leave
that report pending until another restart. The worker now keeps the labeled
container and attempt journal until a terminal write succeeds or the API
definitively rejects the old authority.

Batch and interactive monitors share the retry path. The monitor stops renewal
before reporting; an in-flight renewal rejection cannot erase the pending
journal. Replies for old attempts cannot erase replacement attempt authority.
Cleanup inspects the exact attempt label again. A confirmed fenced completion
remains confirmed if its compatibility PATCH fails; an uncertain or rejected
fenced report never falls back to legacy mutation. API replay semantics remain
unchanged: a lost committed reply is retried and the resulting 409 ends retries.

Pending stop intent and the requested preserved-container result survive retry.
Interactive clean/signal exits remain `stopped`; crashes remain `failed`. The
interactive flag is persisted for newly journaled attempts. Running adoption
rearms both renewal and exit monitoring. Legacy startup adoption excludes any
v2-labeled container. Docker inspection errors are uncertain state rather than
evidence of missing execution; created, paused and restarting containers are
not mistaken for completed work.

## Executed evidence

- The original six regressions all failed against the previous implementation
  before the fix (`artifacts/terminal-recovery-evidence/red`).
- **330 passed, zero failures/errors/skips**, across 14 worker, agent API,
  billing, invoice, provider settlement, static distribution and installer
  suites (`artifacts/terminal-recovery-evidence/verified-20261020`). This includes
  28 terminal-recovery cases covering replacement authority, a renewal race,
  preserved stop commands, restart, interactive exit meaning, inspect errors and
  adoption isolation.
- Targeted Ruff and Pyright passed. All **12 distribution signatures** verified
  with the published Ed25519 trust root; the changed worker signature is included.
- The final CUDA gate passed using real local Docker, loopback HTTP, the actual
  API transactions and disposable PostgreSQL
  (`artifacts/terminal-recovery-evidence/cuda-final-20261020`). It used the cached
  image `sha256:e2a75ae5a502eceac46a1b29c8acfe8876bb2216d32cd04dd1193f7e76366652`
  on the RTX 2060. Every CUDA operation returned **32768.0**.

| Journey | Exit | Attempt result | Meter micro-CAD | Invoice delta micro-CAD |
| --- | ---: | --- | ---: | ---: |
| Normal success | 0 | succeeded | 800 | 800 |
| Normal failure | 7 | failed | 300 | 300 |
| HTTP 503 before terminal commit | 0 | succeeded | 1700 | 1700 |
| HTTP 503 after terminal commit | 7 | failed | 300 | 300 |
| HTTP 503, then worker-process restart | 0 | succeeded | 400 | 400 |

The fault middleware injects HTTP 503 around the real transaction. For the
post-commit case it consumes the real successful response before replacing it.
This exercises lost-success-response semantics; it does not simulate a TCP
connection reset. At each second terminal request the gate verifies that both
the journal record and exited Docker container still exist. The restart journey
ends the first worker process with SIGTERM and runs adoption in a distinct
process sharing only its persisted state and container. Each fault journey
requires exactly two requests. Every journey releases its lease/allocations,
settles the original single meter, refuses stale running reports, matches exact
invoice-line micro-CAD and cleans up its container. No wallet debits or provider
payments occur.

The [machine-readable evidence](2026-10-20-worker-terminal-recovery-evidence.json)
contains journey IDs, captured amounts, final source hashes, gate exits and
references/checksums for the full local manifests. Earlier broad runs exposed a
stale artifact-download exemption in the HTTP seam guard and two Docker fixtures
using the former boolean inspect format. The exemption and fixtures now match
their actual code paths; the final suite passes without weakened assertions.

Reproduce from the repository root with its frozen Python 3.12 environment:

```sh
python api/scripts/run_security_gates.py --worker-billing-acceptance \
  --gpu-device 0 --output /tmp/xcelsior-terminal-recovery-cuda
python api/scripts/run_security_gates.py \
  --output /tmp/xcelsior-terminal-recovery-tests \
  tests/test_worker_terminal_recovery.py tests/test_worker_agent.py \
  tests/test_worker_agent_v2.py tests/test_agent_v2.py \
  tests/test_active_meter_repair.py tests/test_billing_controller.py \
  tests/test_invoice_reads_match_billed_usage.py tests/test_provider_settlement.py \
  tests/test_worker_billing_acceptance_guard.py tests/test_static_endpoints.py \
  tests/test_install_clean_host.py tests/test_worker_api_calls_go_through_one_seam.py \
  tests/test_worker_startup_regressions.py tests/test_worker_v2_paths_match_the_server.py
python api/scripts/sign-static-agent.py --verify
```

## Deployment boundary and remaining gates

Read-only production checks on October 20 found all six application containers
healthy. The maintenance service explicitly configured all five current repair
actions as `report_only`. No production deployment, migration, worker update or
policy mutation was performed for this candidate.

This closes the local batch/interactive terminal-report retry proof while B2.9
remains partial. The next gates are production worker rollout and failure drills,
public launch/full `run_job` provisioning, deployed service identities/private
ingress, active-workload false-positive measurement, and wallet/provider payment
reconciliation. Serverless callback recovery remains separate. Existing journals
without the new interactive flag keep the prior batch interpretation and need
explicit rollout handling. The journal plus labeled container is the retained
evidence; deletion of both, host loss, or an outage beyond lease authority is not
an exactly-once completion guarantee. Fenced authority is never resurrected.
Enforcement still requires per-remedy qualification and authorization.
