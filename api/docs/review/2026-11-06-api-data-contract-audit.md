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

1. The terminal-meter fixture chronology and unreadable failback-state failures are repaired and covered by focused regression checks. The installer still pins wizard 0.1.7 while the source is 0.1.9 and the registry only contains releases through 0.1.8. Qualify a real wizard release rather than installing an unpublished version. Repeat the full audit after remaining semantic repairs.
2. Onboarding, the simple host-location form and schema 150 are deployed at `4bfbed2363d3ced126e3aac5c3d0f8c5a8e768c2`; hosted discovery, schema, redirect-cache and service checks are recorded in `release-receipt-20261106.json`. Account mailing address does not establish a machine's site.
3. Region defaults in serverless deployment, volumes, inference and marketplace offers are removed: an omitted region means any region (`f521f3be`, schema 151 clears the labels defaults had stamped). Autoscale enrollment and storage-provider region names still need review.
4. Verify hosted billing-profile completion and Stripe address synchronization without moving live funds.
5. Registration and email-change validation now use the same email rule as completion; retain the regression gate.
6. Seller completion now checks recorded terms and country-specific address data; legacy incomplete profiles require user completion.
7. Pricing-commitment persistence now fails the request if its durable write fails. Separately review marketplace reservations: exact GPU/rate sourcing, removal of invented prices, count validation, concurrent cancellation and financial effects remain open.
8. Complete per-operation semantic/authorization review. The initial real MCP end-to-end journey passed; repeat affected flows after further changes. Source reachability and test counts are not substitutes.
9. Regenerate API/MCP contracts after further API changes; keep the route classification current.
10. Configure a real Loqate key and verify a live search/retrieve transaction. Until then, describe and test manual entry accurately.
11. Commit and deploy only qualified changes and preserve other work. Initial onboarding release gates are complete; the billing/telemetry batch below needs its own final build, browser and hosted receipts.

These receipts establish local validation. Hosted release verification is recorded separately. The broad per-operation audit is not complete.

## Billing and telemetry continuation

The next batch is under verification and is not yet a hosted-release claim.

- Credit deposits reject booleans, numeric strings, fractional cents and non-finite amounts. Decimal conversion produces integer cents. When enabled, tax must have a valid provider calculation before a payment intent can be created; errors cannot silently charge a pretax amount or echo a submitted address.
- Usage spends pretax wallet credit. Launch quotes and usage statements no longer apply a second, guessed provincial tax. Tax is calculated on credit deposits using the saved billing address and appears on their payment receipts. Legacy usage tax overrides are rejected; unavailable tax fields are explicitly null. This does not establish country-aware provider payout-tax treatment, which remains open.
- Pricing commitments require a durable record before returning success or publishing the success event. They are pricing agreements, not a Stripe charge or a physical-capacity reservation.
- The wire-contract audit now authenticates real disposable users, administrators and pending-MFA identities. Unexpected statuses and response shapes fail rather than skipping protected operations. It found and corrected the telemetry `hosts` envelope and burst-budget response types. Unused legacy key helpers targeting permanently retired endpoints were removed.
- Host telemetry reads require `hosts:read` and host ownership, including a provider's pending onboarding hosts; fleet operators retain their authorized view. PostgreSQL read/write failures return an error rather than acknowledging unsaved samples or serving process-local data.
- Telemetry validates finite measurements, ranges, exact counts and used/total memory consistency. Shared frontend parsing follows the actual nested wire data; missing sensors display as unavailable, reported zero remains zero, memory is in GB and outages expose a retryable error. Full per-GPU worker aggregation is still open.
- Validation responses include field location, type and message without raw input. Passwords and non-finite submitted values cannot turn a form error into a serialization failure.
- Intermediate evidence: `financial-integrity-r3/` (261 passed), `wire-telemetry-integrity-r5/` (319 passed), `telemetry-input-validation-r1/` (148 passed), and `telemetry-frontend-tests-r2.log` (7 passed). These overlap and must not be summed into a unique test count. Final combined receipts are required before release.

The 184-table snapshot and 581-operation inventory define the audit scope, not completed semantic coverage. Remaining work includes country-aware provider settlement taxes, autoscale region enrollment, marketplace financial correctness, worker telemetry aggregation and the remaining per-field/per-operation lifecycle checks.
