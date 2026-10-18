# Isolated CUDA worker-to-invoice acceptance — 2026-10-17

The local hardware acceptance passed for successful and failed containers using
the current repository code. This advances the B2.9/B3.3 execution proof without
claiming a production rollout or closing the remaining lifecycle work.

Read-only readiness checks found the production API, scheduler, maintenance,
MCP, SSH gateway and frontend containers healthy. The production API was in
scheduler canary mode, but `control_plane/active_meter.py` was absent and the
active-meter/orphan-meter/job-lease repair flags were unset. Production therefore
could not exercise the new repair. No production code, flags or data were changed.

The workstation reported an RTX 2060 with 6 GiB total VRAM, about 1.1 GiB used
and zero GPU utilization at readiness. The gate used the cached PyTorch image
`sha256:e2a75ae5a502eceac46a1b29c8acfe8876bb2216d32cd04dd1193f7e76366652`.
Its tiny CUDA matrix operation returned **32768.0** in both containers. Images
were not pulled. Unrelated services and workloads were left in place.

| Actual container exit | Attempt result | Lease | Meter micro-CAD | Invoice-line delta micro-CAD |
| --- | --- | --- | ---: | ---: |
| 0 | succeeded | released | 500 | 500 |
| 7 | failed | released | 700 | 700 |

These amounts are test ledger usage, not wallet debits or real payment activity.
Each attempt retained its original active meter, recorded zero cost while open,
released every active allocation on exit, and rejected a stale running report.
Repeated billing repair changed no meter. The invoice displayed a zero-cent
subtotal for these tiny amounts while preserving the exact micro-CAD line data.

Executed evidence:

- `/tmp/xcelsior-worker-billing-verified-20261017/worker_billing.json`: both real
  GPU journeys, image/GPU identity, command/attempt/lease/fence/spec/observation/
  container/meter IDs, server timestamps, exact amounts and pass result.
- The same directory contains the full API/worker log and the runner manifest
  with source hashes, Python 3.12 version and successful database bootstrap,
  migration, hardware acceptance and PostgreSQL shutdown exits.
- `/tmp/xcelsior-worker-billing-regression-20261017`: **194 passed, zero skips**
  across eight suites: isolation refusal, worker v2, agent API v2, host tokens,
  active repair, billing controller, invoice reads and provider settlement.
- Ruff, targeted Pyright and diff whitespace checks passed.

The first hardware run reached completion and settlement but failed an incorrect
harness assertion comparing exact micro-CAD to the cent-rounded display subtotal.
The corrected gate compares invoice lines exactly and checks display rounding
separately. This was a harness error, not an invoice defect. Subsequent success
and failure journeys passed; no application billing behavior was changed here.

The new `scripts/worker_billing_acceptance.py` runs only under the disposable
runner. It validates the database socket/identity/state path and rejects remote
addresses, libpq service/host-address overrides, mismatched DSNs, production mode
and uncontrolled state. Worker credentials and journals are isolated for both
hardware acceptance and the runner's normal pytest mode. Docker uses a fresh CLI
config and the explicit local Unix socket, so saved remote contexts are ignored.

The harness owns one network-disabled, read-only container at a time, with no
host mounts, dropped capabilities, no-new-privileges, a 1 GiB memory limit, one
CPU, a PID limit and an internal 90-second deadline. It launches only after the
real worker claims the scheduled lease and command. The real worker observation
collector and exit monitor inspect the actual container; status/observation and
compatibility requests travel over HTTP to the real API using a newly issued
host token. No HTTP, Docker, CUDA, database or billing response is mocked.

Reproduce with the command in `docs/reconciliation.md` under “Isolated
worker-to-invoice acceptance.” The normal security runner remains pytest-only
unless `--worker-billing-acceptance` is explicitly selected. Missing hardware,
failed authority, absent observations, unexpected exit, duplicate/unsettled
meters, invoice disagreement or incomplete container cleanup fail the gate.

Remaining gates are explicit: public launch and full `run_job` provisioning;
lost-terminal-report recovery; built/deployed service identity and private
ingress; wallet debit/provider payment reconciliation; and staged production
measurement/enforcement. The API and billing sweep here use the disposable DB's
bootstrap identity; the prior active-meter change separately proved a real API
runtime login. This test does not combine those proofs into a deployed-runtime
claim. Production must receive and validate the committed code before its
active-meter flag can be enabled. B2.9 remains partial.
