# API and MCP data-contract audit — 6 November 2026

This is an ongoing audit. Passing schema or unit checks does not prove every operation's business behavior or live integration. Unreviewed operations remain open.

## Scope and inventory

- 581 mounted operations: 269 POST, 256 GET, 31 DELETE, 13 PATCH, 8 PUT, 2 HEAD, and 2 WebSockets.
- The inventory reads the mounted application, including dynamic worker aliases and multi-method routes.
- 86 MCP tools. Static tracing maps 84 registrations and 106 direct HTTP calls; all 106 match mounted API routes.
- The two loop-registered tools were reviewed in `api/mcp/src/tools/operator.ts`: `retry_instance` and `reconcile_instance` call their corresponding POST `/api/v1/instances/{job_id}/retry|reconcile` routes through `mutate()`, with scope, expected-version, and idempotency checks.
- Route reachability is qualified separately from input validation, authorization, durable effects, and response semantics.

## Verified changes in this work

| Area | Behavior and evidence |
| --- | --- |
| Physical inventory | Counts unallocated physical devices from the scheduler snapshot. Provider offers supply pricing, never extra machines or alternate host locations. Missing or ambiguous host geography is excluded. |
| Host location | A regular country/subdivision form saves with one action. No move wizard or pause/resume controls. Version fencing and native persistence prevent stale worker updates from undoing a saved location; active workloads receive an inline conflict. |
| Account data | New registrations do not invent a name, country, or province. Missing required fields come from stored profile validation. Schema 150 adds empty address and onboarding-choice fields; existing customers must supply them. |
| Postal validation | All 249 ISO countries are available by name. Country and subdivision come from a shared catalog. Country-specific postal requirements validate both manual and captured addresses; an Ontario address cannot use a British Columbia postal prefix. |
| Account onboarding | Registration opens dedicated account/address pages and preserves the original return destination. Rent/provide/both and personal/business choices are saved. A business account requires its legal name. |
| Settings | Account/address details remain editable. Saving an address does not resend and overwrite a hidden, stale name. |
| Provider onboarding | Seller details are collected outside Earnings. Address fields use the same country rules. Saved seller geography supplies a new payout account's country; mismatches and truncated country values are rejected. |
| Billing address | Credit purchases require the current saved profile and send its complete address to tax calculations. IP and Ontario fallbacks are removed. Existing Stripe customers are updated from the saved account; administrator actions resolve the requested customer. |
| Trust and Stripe | Provider registration does not manufacture email verification. Temporary Stripe failures preserve the existing payout account. A new payout account is created only after the user starts verification. Stripe handles its required identity documents. Seller completion requires recorded terms acceptance and a valid tax-identifier suffix. |
| Address lookup | Server-side Loqate Find/Retrieve requests keep the key and address out of URLs. No key is configured yet, so manual entry is available. Tests cover provider errors, country mismatches, malformed data, missing province, keyboard focus, and late replies after manual edits. Live lookup remains unverified. |
| Private data | Delegated identity responses omit postal addresses and tax/business fields. The legacy `api` scope does not bypass identity consent. |
| Session handling | Protected profile/session requests refresh expired cookies and retry their original payload. A failed session on onboarding returns through sign-in with its original query/hash. Credential failures stay single requests. |
| UI and production assets | Desktop/mobile and light/dark setup pages were inspected in a real browser. Horizontal overflow was corrected. Stripe appearance updates without requesting another account session. Production build and PWA checks pass. |

## Validation receipts

Evidence is retained locally under `/tmp/xcelsior-api-audit-20261105/`.

- Focused backend gate: 209 passed, no skips or errors, on fresh disposable PostgreSQL with no live credentials (`onboarding-integrity-gates/`). Later handoff fixes also run in the full audit.
- Frontend suite: 446 passed (`frontend-tests-final-release.json`), including the simplified host form and unobstructed mobile host/Settings forms.
- MCP unit/integration suite: 316 passed, 2 skipped (`mcp-trace-tests.log`), with generated API types and MCP type checking passed.
- Optimized frontend build and service-worker/PWA verification: passed (`onboarding-build-receipt.json`). Final UI refinements require a fresh receipt before release.
- Real production-build browser journey: account/address completion, manual save, all 249 countries, Settings persistence, provider branch, and no page errors (`browser-r2/acceptance.json`).
- Real registration form: automatic onboarding, original return target, missing-name collection, business-name validation, and persisted business/provider choices (`browser-r2/signup-acceptance.json`).
- Full backend run: 8,695 passed, 12 failed, 64 skipped, no setup errors (`full-isolated-r1/`). Nine feature-related failures were corrected and covered by the final 482-test combined gate. Three failures also existed in the baseline: terminal-meter evidence in an aged runtime fixture, non-root failback status, and the installer wizard version pin. There is no repository-wide passing claim.
- Combined onboarding/host/billing/migration gate: 482 passed, no skips or errors (`final-onboarding-host-billing/`). The later simplified host edit also passes 81 integrity cases (`simple-host-location-gates-r2/`).
- Real expired-session browser case preserves the original query/hash and private no-store response (`browser-r2/expired-session-acceptance.json`).
- Real single-save host location journey: passed against the optimized build, with unchanged admission/status and account address, no extra move controls, no mobile overflow and no page errors (`browser-r2/host-move-acceptance.json`).
- MCP real-stack end-to-end journey: all thirteen steps passed against the actual API, disposable PostgreSQL, isolated Redis and three MCP replicas (`browser-r2/mcp-real-stack-external-1793971068.json`). It covers approval/execution/replay, physical placement, worker fencing, trace/audit continuity, serverless invocation, tenant/scope isolation, expiry, Redis failure and operator separation. No live funds or hardware were used.
- MCP traces now correlate API calls with their durable audit/outbox records even when no telemetry exporter is configured; concurrent tool traces remain isolated.

## Production data findings

The aggregate audit used one repeatable read-only snapshot, verified database/port and administrator row visibility, and bounded each table scan by 15 seconds and lock acquisition by 1 second. It counted all 2,164 columns across all 184 application tables without exporting records or credentials.

- Six stored hosts; five lack a province. Prior physical-inventory inspection identified two current physical GPUs, both attached to legacy hosts whose broad `ca-central` label does not establish a province.
- One application seller record lacks legal name, province, street, city, postal code, tax identifier, and accepted-terms timestamp.
- 145 account rows have country/province values under the old defaults. Presence alone does not establish provenance or confirm those locations. Schema 150 leaves new address/choice fields empty so customers complete them explicitly.
- Stripe MCP inspection found five live Express accounts with incomplete requirements and no enabled charges/payouts. No accounts were changed or deleted. These facts do not establish identity verification.

Missing/zero/empty counts are observations. Optional values and lifecycle fields require semantic review before repair. No location, identity, or address was backfilled from a guess.

## Open work and release gates

1. Resolve the three named baseline failures and repeat the full audit after remaining semantic repairs.
2. Verify the simplified host location form in the hosted release; the built local app passed. Account mailing address does not establish a machine's site.
3. Audit remaining region defaults in serverless deployment, volume placement, marketplace offers, and autoscale enrollment. Distinguish physical compute geography from storage-provider region names.
4. Verify hosted billing-profile completion and Stripe address synchronization without moving live funds.
5. Registration and email-change validation now use the same email rule as completion; retain the regression gate.
6. Seller completion now checks recorded terms and country-specific address data; legacy incomplete profiles require user completion.
7. Review reservation persistence and failure behavior; a response cannot claim an active commitment when its required durable record was not saved.
8. Complete per-operation semantic/authorization review and the real MCP end-to-end journey. Source reachability and test counts are not substitutes.
9. Regenerate API/MCP contracts after further API changes; keep the route classification current.
10. Configure a real Loqate key and verify a live search/retrieve transaction. Until then, describe and test manual entry accurately.
11. Commit and deploy only qualified changes, preserve other work, apply schema 150 before the API reads its new fields, and verify the hosted onboarding and inventory behavior.

These receipts establish local validation. Hosted release verification is recorded separately. The broad per-operation audit is not complete.
