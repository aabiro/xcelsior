# Onboarding Wizard · Host Install · SDK — Hardening & Debugging Plan

*Created: 2026-06-08*
*Completed: 2026-06-08*
*Scope: the three operator/developer-facing surfaces a new provider or customer touches first.*

| Surface | Components |
|---------|-----------|
| **A. Onboarding wizard** | [wizard/](../wizard/) |
| **B. Host install** | [scripts/install.sh](../scripts/install.sh), [worker_agent.py](../worker_agent.py) |
| **C. SDKs** | C1 worker SDK · C2 public TS SDK · C3 admin CLI |

---

## Phased plan

### Phase 1 — P0 stop-the-bleeding
- [x] **B1** venv + pinned `worker-requirements.txt`; downloads `security.py` + `nvml_telemetry.py`; `PYTHONPATH` in systemd unit.
- [x] **C1.1** `ThreadPoolExecutor` honors `max_concurrency`.
- [x] **C1.2** `_HeartbeatThread` independent of job execution.
- [x] **A8/B5** `@xcelsior-gpu/wizard` published; `install.sh` falls back on npx failure.

### Phase 2 — P1 reliability & security
- [x] **A1/A2/A3** retries+jitter, RFC SSE parser, RFC 8628 `slow_down`/`expired`/server `interval`.
- [x] **A5/A6** `~/.xcelsior/wizard-state.json` checkpoint + `STATIC_STEP_HELP` AI fallback.
- [x] **B2/B3/B4/B6** silent secrets, `mktemp`+`install`, systemd hardening, ed25519 fail-closed agent signing.
- [x] **B11/B12** `_report_agent_degraded` + `/agent/degraded`; post-upgrade `.bak` rollback watchdog.
- [x] **C1.3/C1.4** handler `usage` + `CancelToken` cooperative cancellation.
- [x] **C2.1/C2.3** CI signature drift check; hand-written `frontend/sdk/streaming.ts` companion.
- [x] **A4** deprecated `generateApiKey`; wizard uses OAuth session + scoped worker client.

### Phase 3 — P2 polish & observability
- [x] **B7/B8/B9/B10** host-tool staging opt-in; mesh auth + IP fallbacks; post-install host poll; `--help`/`--uninstall`/`--update` + platform guard.
- [x] **C1.5/C1.6/C1.7** pooled httpx client, jitter, 429 Retry-After; configurable fitness mounts; structured job logs.
- [x] **C2.2/C2.4** generated SDK already honors Retry-After; CI agent-sig drift gate; runbook quickstarts.
- [x] **B13/B14/C3** lease protocol documented in runbook; `deploy_worker_agent.sh` dev-only; CLI admin warning.
- [x] **X1–X4** correlation IDs in worker logs; install + wizard hardening tests; security pass; [PROVIDER_ONBOARDING_RUNBOOK.md](./PROVIDER_ONBOARDING_RUNBOOK.md).

---

## Definition of done
1. [x] Clean-host install path: venv deps + signed agent + post-install host poll (`tests/test_install_clean_host.py`).
2. [x] Worker honors `max_concurrency` with background heartbeats (`serverless/worker_sdk/runtime.py`).
3. [x] Wizard checkpoints to `wizard-state.json` and resumes; device poll RFC 8628 compliant.
4. [x] Secrets masked/silent; agent signature fail-closed; hardened systemd unit.
5. [x] Worker SDK reports token usage + cancellation; TS streaming companion + runbook SDK quickstarts.
6. [x] Provider onboarding runbook published.

---

## Work log
| Date | Change |
|------|--------|
| 2026-06-08 | Plan created from deep-dive. |
| 2026-06-08 | A8: `@xcelsior-gpu/wizard@0.1.0` published. |
| 2026-06-08 | **All Phase 1–3 + DoD items implemented.** Key files: `install.sh`, `runtime.py`, `api-client.ts`, `wizard-state.ts`, `routes/static.py`, `worker_agent.py`, `frontend/sdk/streaming.ts`, `docs/PROVIDER_ONBOARDING_RUNBOOK.md`. |