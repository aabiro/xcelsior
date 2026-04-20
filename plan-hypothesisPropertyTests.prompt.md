# Plan: Phase 4 — Hypothesis Property Tests

Add property-based tests for three pure/near-pure subsystems: volume state machine, billing cost/tax/wallet math, scheduler `allocate()`. Tests-only change — no production code modifications unless a real bug surfaces.

## Steps

1. **Add dependency** — `hypothesis>=6.100` to `requirements.txt` and `pyproject.toml`; install in venv.

2. **Phase 4.1 — Volume state machine** (new `tests/test_volume_state_properties.py`) — *parallel with 4.2/4.3*
   Target: `_VALID_TRANSITIONS` dict at `volumes.py` L197-L204 (pure data). Invariants:
   - `deleted` is terminal (no outgoing edges)
   - all non-terminal states reachable from `provisioning`
   - `available ↔ attached` round-trips
   - `deleting` never returns to `attached`
   - `error` can recover to `provisioning` or `deleting`
   - every transition target is a valid status (no typos in the map)

3. **Phase 4.2 — Billing math** (new `tests/test_billing_math_properties.py`) — *parallel*
   Targets: cost formula extracted from `meter_job` (`billing.py` L259), `get_tax_rate_for_province` (`billing.py` L65), invoice arithmetic from `generate_invoice` (`billing.py` L370), wallet `deposit`/`charge` (`billing.py` L716 / L765). Invariants:
   - cost ≥ 0 and monotonic in hours/rate
   - spot discount ≤ base cost
   - tax rate bounded [0, 0.20]
   - `invoice_total == round(sum(subtotals) × (1 + tax), 2)` within 0.01
   - deposit-then-charge returns to starting balance (with monkeypatched DB)
   - NaN/Inf rejected at the cost-calculation layer

4. **Phase 4.3 — Scheduler allocation** (new `tests/test_scheduler_allocate_properties.py`) — *parallel*
   Target: `allocate(job, hosts)` at `scheduler.py` L457.
   Setup: monkeypatch `scheduler.get_volume_engine` to return a stub.
   Strategies: hosts and jobs built via `builds(dict, …)` with bounded floats. Invariants:
   - `allocate(job, [])` is always `None`
   - returned host is element of the input list
   - returned host satisfies `free_vram_gb ≥ job.vram_needed_gb`
   - returned host satisfies `gpu_count ≥ job.num_gpus`
   - if `job.gpu_model` is set, returned host's model equals it
   - non-admitted hosts never returned
   - determinism (same inputs → same output)
   - returned host has maximum score among feasible candidates

5. **Test runs** — per-file pytest run after creation; fix real bugs in a separate commit if any surface.

6. **Full suite regression check** — run volumes + billing + scheduler test suites to confirm no new failures.

7. **Commit as 4 logical units**:
   - `chore(deps): add hypothesis for property tests`
   - `test(volumes): property tests for state machine invariants`
   - `test(billing): property tests for cost/tax/wallet math`
   - `test(scheduler): property tests for allocate() invariants`

8. **Push** (deploy optional — tests-only change).

## Relevant files

- `volumes.py` L197-L204 — `_VALID_TRANSITIONS` dict (test target)
- `billing.py` — `meter_job` L259, `get_tax_rate_for_province` L65, `generate_invoice` L370, `charge` L765, `deposit` L716
- `scheduler.py` L457 — `allocate()` (test target)
- `tests/conftest.py` — no changes planned; inline `@settings` on each test
- `requirements.txt`, `pyproject.toml` — add `hypothesis>=6.100`
- New: `tests/test_volume_state_properties.py`
- New: `tests/test_billing_math_properties.py`
- New: `tests/test_scheduler_allocate_properties.py`

## Verification

1. `venv/bin/pytest tests/test_volume_state_properties.py -v` — all pass
2. `venv/bin/pytest tests/test_billing_math_properties.py -v` — all pass
3. `venv/bin/pytest tests/test_scheduler_allocate_properties.py -v` — all pass
4. `venv/bin/pytest tests/test_volumes.py tests/test_billing.py tests/test_scheduler.py -q` — no new regressions (3 pre-existing `TestAllocate` failures remain; those are separate tech debt)
5. Full runtime budget: ≤30s for all three new files combined
6. Add `.hypothesis/` to `.gitignore` if not already present

## Decisions

- Only test invariants the code already claims. Don't invent new contracts; don't refactor code to make it more testable.
- Zero real DB access — pure data tests or monkeypatched writes.
- Scope excludes: `allocate_binpack`, `allocate_jurisdiction_aware`, stateful-machine testing (`RuleBasedStateMachine`).
- If a property surfaces a real bug, STOP and fix in a separate commit — do NOT weaken the property.
- Inline `@settings(deadline=None, max_examples=100)` per test (simpler than a conftest profile).

## Further considerations

1. **Pre-existing `TestAllocate` failures**: `tests/test_scheduler.py::TestAllocate` has 3 pre-existing failures (returns `None` where dict expected). Option A: mark them `xfail` for clarity. Option B: leave as-is (current visibility is accurate). Option C: fix them as part of Phase 4 (scope creep). **Recommend B** — keep unchanged, document as known tech debt.

2. **Hypothesis `max_examples` budget**: 100 per property (default) × ~20 properties ≈ 2000 cases. Should run in ~15s. If slower, drop to 50. Option A: 100 (thorough). Option B: 50 (fast CI). **Recommend A** — runtime is not a concern for a test-only change.

3. **Follow-up scope**: After Phase 4 ships, worth-doing-next candidates:
   - `allocate_binpack` and `allocate_jurisdiction_aware` property tests (mirror 4.3)
   - Stateful machine for volume lifecycle (`RuleBasedStateMachine` exercising actual create→attach→detach→destroy transitions against a mock backend)
   - Fuzz the Pydantic models from Phase 2 (they should already be tight, but worth confirming via `@given` on raw JSON input)
