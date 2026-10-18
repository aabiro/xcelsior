# Xcelsior security audit and remediation — 2026-10-17

## Decision and scope

**Security acceptance remains blocked.** Public HTTP destination controls, database-role machinery, artifact-copy protections, dependency updates and installer signature enforcement have local evidence. Production cutover, SSH/MCP and machine-client session revocation, deployed worker authority and adversarial network isolation still have high-priority gaps. No exception has been accepted, and no owner/expiry waiver is implied by this report.

This is the detailed findings and gate record for the user-approved security plan. It supplements [the implementation record](2026-10-17-security-completion.md), [worker billing evidence](2026-10-17-worker-billing-acceptance.md) and the [Track B checklist](../track-b-implementation-checklist.md). It does not close the whole control-plane programme. CI cockpit and unrelated applications are outside scope.

**Tower is read-only.** This continuation did not access tower, deploy production changes, create real accounts, change real balances, send notifications, rotate secrets or run external attack traffic. Tests used synthetic data, fresh local PostgreSQL clusters and local HTTP/TLS receivers. One local, network-disabled container ran `promtool`; no worker workload was launched during this continuation. The editor configuration and concurrent GX0 work were preserved.

Coverage uses [OWASP ASVS 5.0.0](https://owasp.org/projects/asvs) Level 2, including Level 1 requirements, with additional worker, tenant, money and administrative boundaries. [The requirement index](security-audit-2026-10-17/asvs-l2-coverage.csv) records 253 requirements across 17 chapters. A complete requirement-level ASVS pass is **not** asserted: individual application paths have partial evidence and each chapter has remaining checks below. This is not certification.

## Evidence and provenance

The tested tree is commit `582859dfdbf35318c25accad203655cd4c7d2bf6` plus the uncommitted patch. Earlier implementation commits include `1e62e045` (dependency compatibility), `f35c1c0c` (HTTP/database/ingress) and subsequent lease/meter fixes. The [evidence record](security-audit-2026-10-17/evidence.json) identifies the runs, exit codes and hashes of the security-relevant files. No deployed image hash or live configuration fingerprint was refreshed in this continuation; both are outstanding deployment gates.

- Final broad isolated Python run: **755 passed, 0 failed, 0 skipped**, pytest exit 0, runner exit 0, 70.25 seconds. It includes real PostgreSQL grants/logins, HTTP/TLS receivers, artifact file IO, authentication and tenant boundaries, synthetic payment/budget/hold checks, recovery cases, installer verification and the stream-authorization tests present at the start of that run. External payment providers, Docker lifecycle operations in unit tests, and some auth principals are mocked; those cases are not real payment or worker acceptance.
- After that run, `scripts/security.py.sig` was refreshed so the concurrent gVisor capability restriction matches the published Ed25519 trust root. Public-key verification of all 12 worker distribution files then exited 0. A follow-up isolated run of the static, clean-host install and gVisor capability tests passed **43 tests, 0 skipped**.
- A further stream run, including revocation before the job-log connected frame, passed **16 tests, 0 skipped**.
- One post-stream run was killed before completion (pytest exit -9). It is not a pass. An earlier focused stream run failed 1 of 29 tests because job-log continuation still produced a chunk after revocation, and the live database poll treated authorization failure as a transient database error. That handler now re-raises authorization failures. The later 755-test and 16-test runs include the correction.
- Earlier focused artifact/role/monitoring run: **112 passed, zero skipped**. The previous attempt failed when its temporary PostgreSQL filesystem exhausted space. PostgreSQL logged write/checkpoint failures; that run was terminated, its scratch cluster removed, and it is not a pass. Subsequent databases used the storage volume.
- The first broad run had **722 passed and one skipped**. The skip was an empty host-auth parameter list. It was replaced with actual anonymous requests to `/hosts` and `/host/{id}` before the final run.
- MCP: **299 passed, two skipped**, TypeScript typecheck exit 0. The skipped checks are the complete hosted MCP/API/PostgreSQL/Redis journey and real citation URL reachability. These remain unverified; the MCP suite is not an overall release pass.
- Dependency checks: 110 exported Python runtime records, MCP production dependencies and the resolved worker requirements report **zero known advisories** at audit time. This is scanner/database evidence, not proof of absence of vulnerable application behavior. Worker dependency ranges still resolve at installation time; retain the resolved artifact when releasing.
- Prometheus: the cached local `prom/prometheus:v3.5.0` image ran `promtool check rules` with network disabled and a read-only rule mount: **56 rules, exit 0**. Collection, scrape configuration and alert delivery are not deployed acceptance.

The October 16 live inspection is historical supporting evidence: the reported account was suspended across its linked identifiers and had no remaining sessions, jobs, deposits or spend; web/MCP/agent ingress includes were present; runtime services shared the `xcelsior` database identity with `CREATEDB` and schema creation; tower ran as the operator with Docker-group authority and weaker systemd restrictions; existing tower egress status reported success. The API and maintenance containers had an API signing token but no dedicated serverless webhook HMAC key. No credential values or account identifiers are included here. These facts require a read-only refresh before rollout; a listening socket is not evidence of public reachability.

## Findings and remediation backlog

Owner names below identify the responsible component/team function, not a person's acceptance of risk. All deployment states remain pending unless specifically stated. Confidence describes the evidence, not exploit likelihood.

### SEC-01 — Untrusted HTTP destination changes (high; high confidence)

**Surface/evidence:** The original webhook path validated DNS and then used an ordinary opener, allowing another resolution and redirects. OAuth client-metadata fetching had the same class of destination-policy concern. Current implementation: [`public_http.py`](../../public_http.py), [`serverless/webhooks.py`](../../serverless/webhooks.py), [`oauth_registration.py`](../../oauth_registration.py), `tests/test_public_http.py`.

**Impact/conditions:** An attacker controlling a callback or client-metadata URL and DNS/redirect behavior could reach a prohibited destination from the API. This was a code finding; no production SSRF exploit was demonstrated. Existing hostname/IP checks reduced straightforward private-address requests but did not bind the subsequent connection.

**Remediation:** Resolve each attempt once; validate every returned address; reject the entire set if any is prohibited; connect to the approved numeric address; preserve original TLS SNI, certificate hostname and HTTP Host. Reject URL credentials, malformed ports, unsupported schemes, metadata/private/mesh and IPv4-mapped bypasses. No redirects or environment proxies. Bound DNS concurrency, total request duration, payload and response bytes. Webhook field/signature compatibility is retained, errors are sanitized, production without any signing key fails closed.

**Acceptance/owner/status:** API/serverless. Local positive HTTP/HTTPS signature and hostname cases plus negative DNS rebinding, mixed DNS, redirects, body bounds, slow headers/body and key absence are covered. Release the tested image and prove real public receivers and production error counters before closing. Dedicated key separation is SEC-08. Artifact/model download exceptions are SEC-10.

### SEC-02 — Shared database authority and incomplete cutover (high; high confidence)

**Surface/evidence:** October 16 production role inspection; [`db_roles.py`](../../control_plane/db_roles.py), [`db_role_audit.py`](../../control_plane/db_role_audit.py), [`task_identities.py`](../../control_plane/task_identities.py), [`provision_db_roles.py`](../../scripts/provision_db_roles.py), [`maintain_db_partitions.py`](../../scripts/maintain_db_partitions.py), `tests/test_db_service_roles.py`.

**Impact/conditions:** Compromising a runtime service using the old shared owner/DDL identity can affect other domains and schema. Service roles do not enforce tenant ownership inside a domain. The API intentionally still writes multiple domains; task-specific pools do not prevent a compromised maintenance process from reading other credentials present in that process.

**Remediation implemented:** Explicit per-service DSNs, fail-closed strict mode, actual login/effective privilege audit, separate task pools, named task authority, restricted SSH gateway reads of jobs/hosts, and an external migration-login partition task. Runtime migration-stamp writes, DDL, unintended memberships/ownership, column-level grants and sequence reset/cross-domain access are checked. Standalone fencing sequences and child partitions are classified. PUBLIC and future-object blanket grants are removed during provisioning; newly migrated objects require convergence. Cached legacy pools cannot silently become strict-mode pools.

**Acceptance/owner/status:** Control plane/database operations. Local tests use actual runtime logins, negative DDL/DML, positive required statements, task switching and failure cleanup, partition maintenance twice, and restricted SSH login. Production remains on the old configuration until every service/task startup and ordinary workflow is qualified. Inventory actual object owners; transfer ownership deliberately to the migration identity; provision unique secrets; schedule external maintenance; cut over service by service with `XCELSIOR_DB_ROLES_REQUIRED=1`; remove shared credentials from runtime environments. Split mixed-purpose processes where compromise isolation is required. No live credential or ownership change was made.

### SEC-03 — Artifact promotion follows tenant-controlled filesystem paths (high; high confidence)

**Surface/evidence:** [`worker_agent.py::_promote_artifacts`](../../worker_agent.py) formerly opened predictable `final_path + '.part'` with `open(..., 'wb')`; destination components followed symlinks. Resume trusted `already_done` plus existence. Copy length was not bounded by manifest size. `tests/test_promotion_copies_and_verifies.py` exercises actual files and local HTTP.

**Impact/conditions:** A tenant able to change the destination volume could plant a symlink to a file writable by the worker, cause an out-of-volume write, or have changed bytes treated as already verified. An object response exceeding the manifest could consume disk. Host impact depends on volume layout and worker permissions; no tower exploit was attempted.

**Remediation:** Validate the volume path component; open every tenant-controlled directory with `O_NOFOLLOW` relative to a pinned parent descriptor; create an exclusive random private staging directory/file; check exact byte count and SHA-256 before atomic placement; remove failed partial copies; refuse object-store redirects; rehash resumed files and refuse symlink reads. Error logs do not include presigned URLs. The configured mount root remains an operator-controlled trust boundary.

**Acceptance/owner/status:** Worker/storage. Local tests cover private/final/directory symlinks, traversal, malformed or mismatched size, corrupt content, redirect refusal, changed resume data, valid copies and preserved valid resume files. This does not make tenant-writable data immutable after publication, prove protection against a process with the same host privileges, or qualify every filesystem/mount topology. Signatures were refreshed locally. Disposable-worker install and upload–promote–mount–read remain gates; **tower deployment is prohibited**.

### SEC-04 — Long-lived transport authority and global event disclosure (high; high confidence)

**Surface/evidence:** Initial inspection found account suspension enforced at credential issuance and admission while ongoing terminal, instance and job-log loops retained the admitted principal. `/api/stream` had no route authentication and broadcast/replayed the global event feed without tenant filtering. No public exploit was attempted. `ssh_gateway.py::query_running_instances` selects running/starting jobs; `close_listener` drains but does not force-close established relay tasks after the drain timeout.

**Impact/conditions:** An anonymous observer could receive fleet/job events on the unguarded dashboard stream. A user admitted before suspension, token/session revocation or an ownership/scope change could keep an existing stream or terminal while its instance remained alive. The incident account had no surviving jobs/sessions, so this finding does not establish surviving incident access. New admissions and durable suspension were existing controls.

**Local remediation:** Dashboard SSE now requires authentication and filters live and replayed events against current job/host access, with unclassified administrative payloads withheld from non-operators. Global and job-specific SSE re-resolve credentials. The job-log generator checks authority before the connected frame, before each replayed line, and on every live poll; a database error is no longer allowed to hide an authorization failure. Terminal and instance WebSockets recheck suspension, current ownership/scopes and user records, including quiet connections. Tickets carry a hashed OAuth-cache reference instead of a new bearer secret, allowing individual access-token and session revocation to end a stream. Agent-key revocation and legacy-session deletion are checked; WebSocket admission now shares REST's refusal of permanently disabled API keys. Initial scopes are checked before terminal access or instance data. The WebSocket watchdog polls every two seconds with a five-second lookup deadline; runtime scheduling adds latency, so this is not instantaneous revocation.

**Acceptance/owner/status:** Identity, streaming routes and SSH gateway. New tests use actual PostgreSQL users/jobs, token cache entries, real suspension/revocation and an established WebSocket. They cover anonymous refusal, live/replay cross-tenant filtering, quiet-terminal watchdog closure, token/session revocation, revocation before the job-log connected frame, object transfer, policy-store failure and disabled-key refusal. The isolated stream file passed 16 tests with no skips after that fix. Existing terminal/replay tests provide regression coverage. These are local application tests; Docker terminal IO in the legacy terminal suite is mocked, and the quiet-terminal case tests the real watchdog with a socket fixture.

**Remaining gate:** Track/close real SSH relay writers when their route or principal loses authority, filter suspended/abuse-blocked routes through a restricted inventory boundary, and prove new and established connections fail on store outages. Complete hosted MCP, machine-client JWT rotation/revocation and all direct-cookie/legacy credential cases. Browser OAuth continuation has local evidence; those other transport/credential combinations remain explicit verification gaps. Recheck the full matrix in a built staging image, including unaffected users. **Not deployed; high finding remains open until the residual paths pass.**

### SEC-05 — Deployed worker privileges exceed a hardened service boundary (high; high confidence)

**Surface/evidence:** October 16 read-only tower inspection: operator user, Docker group and missing installer systemd restrictions. Installer grants Docker membership even to the dedicated worker user; `scripts/install.sh` and `worker_privileged.py` define intended helper/sandbox behavior.

**Impact/conditions:** Worker compromise may exercise host-equivalent Docker authority and access operator resources. A dedicated user or `NoNewPrivileges` alone cannot constrain a privileged Docker daemon. Existing container argument checks, helper validation and egress policies help but are not a complete worker/host boundary.

**Required remediation/owner/status:** Worker/platform operations. Qualify dedicated identity, protected updates and credential paths, helper callers/arguments, allowed devices/mounts/namespaces and constrained daemon access on a disposable worker. Test legitimate GPU, volume, SSH and published-port behavior alongside refusals. Local helper/sandbox tests passed; deployed resistance is unverified. **Tower drift remains outstanding under read-only restriction.**

### SEC-06 — Egress and service identity enforcement remain deployment gates (high priority; high confidence in the gap)

**Evidence/impact:** Existing tower status reported protection, while production host-token mode was `allow` and SPIFFE strict mode disabled. These settings do not alone prove anonymous access: authenticated gateway/fallback paths must be evaluated. Fresh tenant-to-host/tenant/LAN/mesh/metadata/IPv6/DNS/proxy/direct-egress tests were not run on tower.

**Remediation/acceptance/owner:** Worker identity/network. Use a disposable worker to reject forged gateway headers, wrong/revoked host identities, stale attempts and replayed acknowledgements; test all prohibited paths and approved downloads/SSH/ports/GPU uses. Prove abuse-terminated jobs stay terminal through lease expiry, restart and reconciliation, including paused containers and legitimate sustained GPU load. Source/unit evidence and historical egress status are partial. Strict identity rollout requires compatibility proof; no tower enforcement change is permitted here.

### SEC-07 — Ingress deployment checks and monitoring need live qualification (medium; high confidence)

**Evidence/controls:** Installer preflight now verifies abuse includes in every public server block, including TLS; comments and nested-location includes do not satisfy it. The read-only host collector checks enabled web/MCP/agent vhosts, active deny-snippet content and an abuse map declaration. Tests reject comment-only or missing policies. Trusted proxy attribution tests reject caller-controlled forwarded headers.

**Limits/impact:** This is configuration verification, not proof of live Nginx reload state, current blocked-client traffic, correct entries in the map or direct-origin isolation. Collector scheduling, expected-host inventory and alert delivery are not deployed. A collector that has never emitted can escape a timestamp-only stale rule; target registration must cover this case.

**Acceptance/owner:** Ingress/observability. Read-only rendered configuration and firewall review; controlled blocked/unrelated-client tests against every public ingress; non-spoofable attribution; missing/stale collector and removed-policy alarms. Do not change deny lists solely on email-domain or shared-IP similarity.

### SEC-08 — Webhook signing secret is shared with API authority (medium; high confidence)

**Evidence/conditions:** Production key-presence check found no dedicated webhook HMAC secret, while the API token existed. Local code preserves that API-token fallback for receiver compatibility; production development-secret fallback is removed. Receiver operators knowing a shared signing token would create a broader credential-separation risk.

**Remediation/owner/acceptance:** API operations. Establish a distinct signing key, enumerate receivers, provide a coordinated key-ID/overlap migration if needed, and verify old/new signatures before retirement. Do not rotate unilaterally and break receivers. Dedicated-secret metric exists; rollout and key lifecycle remain open. No secret values are recorded.

### SEC-09 — Known dependency advisories and unsigned worker requirements (high priority; high confidence)

**Evidence/impact:** Initial scans found advisories in nine Python packages and three MCP production packages. Reachability varies: the [MCP SDK OAuth advisory](https://github.com/advisories/GHSA-6qxp-vccf-f47h) concerns client OAuth behavior and excludes SDK servers; its presence is not proof that this server is exploitable. [Proxy address handling](https://github.com/advisories/GHSA-jqcg-44mw-7w3h) depends on proxy trust configuration. Package severity is not automatically application severity.

The worker dependency file previously had only a server-provided checksum while executable modules required Ed25519 signatures. An attacker able to alter unsigned distribution bytes and matching headers could change dependency installation. TLS and signed Python modules were existing controls. The original installer trust bootstrap itself still depends on its authenticated distribution channel.

**Remediation:** Updated Python lock/export and MCP manifest/lock; worker requests/cryptography minimums now cover patched releases. Requirements are included in all installer/server/signer signature lists. CI verifies signatures with the public key rather than attempting to use a release private key. Tests verify exact dependency bytes, refuse a missing signature and detect tampering using a disposable signing key. A concurrent `security.py` change rejects gVisor GPU driver capabilities outside `compute` and `utility` and pins an unset request to `compute`; native `runc` still preserves an explicit `video` capability. That file was unsigned relative to the published trust root, so its signature was refreshed with the existing release key. The private key was not printed or copied.

**Acceptance/owner/status:** Supply chain/release engineering. Public-key verification of the 12 worker distribution files exited 0 after the signature refresh. The follow-up isolated run passed the static-byte, missing-signature, clean-host install and gVisor capability tests (43 tests, 0 skipped). No rebuilt/deployed-image SBOM or OS-package scan is claimed. Release signatures, dependency resolution, isolated worker installation, image scans and rollback compatibility remain mandatory. The private signing key was not printed, copied or modified.

### SEC-10 — Internal download integrations need explicit origin/integrity policy (medium; medium confidence)

**Evidence/scope:** `artifacts.py` creates object-store grants from operator-configured endpoints; those endpoints may deliberately be private. Promotion obtains them via an authenticated manifest. `serverless/registry_models.py` resolves local presets without runtime downloads; `scripts/fetch_preset_models.py` invokes operator-run HF/B2 tools. `serverless/b2_model_sync.py` accepts an existing nonempty destination without an independent digest check. Preset and object-store paths must not inherit an indiscriminate public-only block that breaks approved internal service access.

**Remediation/owner/acceptance:** Storage/model operations. Record approved origins and redirect policy, private endpoint ownership and credential scopes; bind expected digest/size to an independently trusted manifest; verify existing/resumed files; constrain proxies and total streaming time. Promotion now refuses redirects and size mismatch, but does not implement DNS pinning or a total slow-stream deadline for all private integrations. Generic artifact import and archive-expansion coverage remains a verification gap. No model downloads were launched.

### SEC-11 — Recovery, money and workflow acceptance is partial (high priority verification gap)

**Confidence:** High that the listed acceptance evidence is incomplete.

**Evidence/controls:** Local tests cover wallet holds, synthetic payment idempotency, endpoint budgets, exact billing arithmetic, lease settlement, queue recovery, cross-tenant plans and approval version binding. Current semantic-cache runtime reuse is retired; the old objective's `SequenceMatcher` diagnosis is stale. The separate October 17 CUDA worker record proves two bounded local worker-to-invoice journeys, with explicit limits.

**Remaining conditions/owner:** Billing/control plane/operations. Validate concurrent distributed wallet/client/endpoint enforcement, chargebacks/refunds, provider payouts, cache/DB outages, process crashes, rolling deployment and rollback under actual runtime identities. Preserve ledger/holds, terminal abuse state and authorization throughout. Test real provider sandbox signatures/replay and upload–mount–read separately. No real payment, payout, whole-stack recovery or production cutover is asserted. Saved card plus settled C$5 still gates the welcome bonus; existing promotional/administrative credits remain usable for rentals.

## Authorization matrix

“Local” means the listed code/test boundary, not every endpoint of that class. Omitted endpoint/principal combinations remain unverified rather than implicitly allowed.

| Principal | REST / object ownership | MCP / OAuth | WebSocket / SSE | SSH / worker |
| --- | --- | --- | --- | --- |
| Anonymous | Local infrastructure, SLA and inventory denial; public installer/reference endpoints allowed | Connector admission and discovery contracts tested | Ticket/origin helpers exist; full cross-site journey open | SSH authenticates inside container; public relay is not an HTTP login |
| Renter | Two synthetic accounts: volume/artifact/team/key/ticket cross-account refusal | Scope and budget/approval contracts tested | Admission plus ongoing browser OAuth, suspension and ownership checks tested locally; residual SEC-04 paths open | Key/port ownership and established-session revocation require live proof |
| Provider | Role guards and visible-host policy inspected | Provider journey/payout not repeated | Operator/provider stream matrix unverified | Host-token and helper unit tests; disposable provider lifecycle open |
| Operator | Infrastructure requires admin; approval/version tests | Operator-scope refusal tested | Admin authority and revocation duration not fully qualified | Privileged maintenance must be separate from customer/worker identity |
| Suspended account | Minting, cached bearer, legacy session and scheduler refusal tested | Shared suspension policy applies; hosted client test open | Browser OAuth sockets/SSE close locally after suspension/revocation; remaining credential paths SEC-04 | Existing instance/relay termination unproven; incident account had none |
| OAuth client | Advertised/granted scopes and ownership tested selectively | PKCE, redirect/consent/audience contracts; hosted flow and distributed refresh replay incomplete | Browser token/ticket continuation tested; machine-client JWT rotation still unverified | Cannot substitute a customer OAuth token for host-bound identity |
| Worker | Host-bound observations/commands tested locally | Consumer tool authority is distinct | Gateway header and host matching tests are partial | Stale/replay/lifecycle units passed; deployed strict identity and isolation open |

[MCP security guidance](https://modelcontextprotocol.io/docs/draft/tutorials/security/security_best_practices) informs separate checks for confused-deputy consent, token audience/passthrough, session ownership and untrusted tool/document inputs. Current unit suites do not establish that a connected agent cannot be induced to authorize a privileged action; an adversarial hosted client/approval journey remains required.

## ASVS chapter coverage

| ASVS 5.0 chapter | Current evidence | Explicit remaining verification |
| --- | --- | --- |
| V1 Encoding/sanitization | URL parsing and filesystem containment negatives | Browser/HTML/template/command/SQL injection across the full input inventory |
| V2 Validation/business logic | Holds, idempotency, budgets, approval/version cases | Distributed concurrency, promotion/registration abuse and real payment/provider paths |
| V3 Web frontend | Proxy/origin/ticket code and route auth tests | Rendered CSP/CORS/cookie/CSRF, DOM injection, redirects and cross-site sockets |
| V4 API/web services | REST ownership, MCP auth and bounded webhook transport | Complete route/principal matrix, protocol limits and actual ingress behavior |
| V5 File handling | Digest/finalization, owner downloads, secure promotion | Archives, expansion quotas, all import paths, replica/deletion failure and mounted journey |
| V6 Authentication | Signup/login/MFA and account-hold tests | Distributed abuse, verification/reset races, real passkey/device and recovery journeys |
| V7 Sessions | Cached bearer hold; live browser stream token/session revocation and quiet-socket suspension | SSH/MCP and machine JWT continuation, distributed refresh replay and logout races |
| V8 Authorization | Tenant objects and live/replayed dashboard events, operator refusal, service roles | Every route/job/action principal combination and live multi-tenant acceptance |
| V9 Self-contained tokens | Host-token/identity machinery and dependency scan | Full algorithm/key/audience/expiry inventory and rotation/revocation across services |
| V10 OAuth/OIDC | Connector/operator/metadata and client contract tests | Hosted client journey, malicious consent substitution and real distributed replay |
| V11 Cryptography | TLS hostname verification, Ed25519 verification, patched packages | Key custody/rotation, at-rest configuration and deployed algorithms |
| V12 Communication | HTTPS receiver tests, proxy attribution, prior firewall evidence | Built-image TLS configuration and fresh network/mesh/private-origin enforcement |
| V13 Configuration | Strict DB mode, ingress parsing, worker installer tests | Production identity/ownership cutover, image/OS scans, worker drift and expected-host monitoring |
| V14 Data protection | Secret-redacted errors, opaque cache keys, artifact retention cases | Full logs/exports/backups/privacy retention and deletion audit |
| V15 Secure architecture | Separated task pools and runtime role denial | Process credential boundaries, Docker authority, supply-chain/update and recovery design |
| V16 Logging/errors | Refusal, abuse revival and identity signals; unavailable evidence distinguished | Collector rollout, absent-target detection, alert routing and incident runbook drill |
| V17 WebRTC | No WebRTC implementation was qualified | Confirm applicability from deployed route/component inventory; not silently marked N/A |

## Required gate matrix

| Gate | Positive and negative evidence now | Outstanding release acceptance |
| --- | --- | --- |
| Webhooks | Real HTTP/TLS delivery/signature; no redirected or rebound prohibited connect; slow/large responses bounded | Staged image plus real public receivers and compatible dedicated signing key |
| Database roles | Real service logins, permitted operations and denied DDL/writes; task/pool/partition/SSH checks | Full workflow qualification, ownership transfer, secrets, individual service cutover, no shared fallback |
| Suspension | Old/new bearer refusal; established browser WebSocket/SSE token/session revocation and quiet-terminal watchdog; unrelated account valid | Built-image direct terminal IO, MCP/SSH and machine JWT continuation; repeat registration under concurrency |
| OAuth/MCP | Scope/operator/connector/metadata and 299 TypeScript cases | Two skipped external checks; real client and adversarial consent/audience/replay journeys |
| Ingress | Per-server includes, comment bypass refusal, proxy attribution tests | Every live ingress, blocked and unrelated clients, direct-origin and IPv6 behavior |
| Worker control | Local host identity, privileged argument and command lifecycle cases | Disposable worker replay/forgery/revocation, terminal abuse state across restart |
| Egress/sandbox | Source/unit checks and dated existing tower status | Fresh disposable-worker network matrix and legitimate GPU/download/SSH/volume/port cases |
| Billing | Synthetic idempotency, holds/budgets/arithmetic and prior isolated worker proof | Distributed money invariants, real sandbox events, chargebacks, recovery and deployed identities |
| Artifacts | Cross-tenant denial, verified finalization, real HTTP/file promotion and negative symlink/size/digest cases | Authorized upload–promote–mount–read, replicas/retention/generation races and archive/import tests |
| Recovery | Local lease/queue/task recovery cases | Real DB/cache outages, process restarts, rolling deployment and rollback preserving security/ledger state |

Skipped tests are not passes. Historical passing evidence is not current deployment acceptance. No high finding has an approved exception; owners must either close it or obtain an explicit scoped, dated acceptance with expiry.

## Monitoring and response

Use `scripts/audit_runtime_db_roles.py --service SERVICE` under each actual runtime environment; it refuses a missing dedicated DSN and emits no DSN/password. Connection application names are self-reported monitoring hints, not authentication. A local restricted-role test confirms that connection identity metadata needed by the existing metric is visible without granting broad statistics privileges. Effective-role audit remains the authority.

Run `scripts/collect_host_security_status.py --ingress` on the gateway and `--egress` on an authorized disposable worker. It reads state and prints metrics; it never repairs rules. Publish output atomically and retain failed-check metrics even when exit status is nonzero. Define expected hosts/targets so a never-started collector is detectable, monitor both scrape absence and stale timestamps, and test alert routing. Do not deploy a collector or timer to tower under this restriction.

For an abuse-blocked job becoming runnable, preserve findings and attempt/lease history, stop the responsible writer through the approved incident procedure and keep the durable block. For ingress drift, retain denies while comparing enabled/rendered vhosts. For egress failure, stop new tenant admission until qualification passes. For identity fallback, fail the service rollout rather than restoring the shared credential. For destination refusals, inspect sanitized counts without logging full URLs, signed grants, tokens or HMAC material.

## Staged rollout and rollback gates

1. Freeze the reviewed patch and record commit, source/lock hashes, image digest and sanitized configuration fingerprint. Run the isolated gate runner, signature verification, current dependency/image scans and required CI. Preserve unrelated work.
2. Qualify webhook/CIMD transport and signing compatibility in staging. Release independently of database cutover where possible. Observe delivery failures and latency; preserve destination refusal on rollback.
3. Exercise every runtime/background task with its intended identity on an isolated production-shaped database. Transfer DDL ownership deliberately, converge existing/future grants, schedule external partition maintenance and verify recovery. Stage secrets; switch services individually; remove shared credentials after each accepted cutover. A health check alone is insufficient.
4. Qualify the signed worker package and constrained helper/daemon/network boundary on disposable workers. Verify approved workloads and prohibited paths, plus monitoring and rollback. Tower remains excluded until the user explicitly changes its read-only boundary.
5. Close residual SEC-04 paths with built-image terminal, SSH/MCP and machine-client established-session revocation tests; finish ingress, money, artifacts, workflow and outage journeys. A prior passing unit suite cannot waive these gates.
6. Exercise rollback with suspensions, deny maps, exact ledger/holds and terminal abuse state preserved. Never restore broad database credentials or weaken isolation silently to recover availability.

## Reconciliation with the original goal

| Programme area | Security result | Remaining programme work |
| --- | --- | --- |
| Track A identity/database sign-off | Runtime role/host-token machinery has local evidence | Actual service identities, strict admission and worker rollout remain distinct |
| B2 launch/reconciliation | Terminal-state/lease tests and newer repair commits exist | Per-path lifecycle and outage/production enforcement gates remain |
| B3 billing | Synthetic budgets/holds/idempotency and local CUDA settlement evidence | Concurrent distributed money, provider payouts and production qualification |
| B4 events/audit | Events/role/pipeline approval checks support authorization | Immutable external audit delivery and cross-process retention/recovery proof |
| B7/B8 operations | New evidence/monitoring helpers and local rule validation | Replicas, rolling restart, collector rollout and real recovery drills |
| B9 artifacts; P3 promotion | Promotion filesystem/size/resume flaw fixed locally | Real second-instance mount/read, replicas, deletion and retention acceptance |
| B10/B12 cache consistency | Current inference no longer uses the legacy approximate cache | Full context/version-bound cache acceptance and deletion/export lifecycle |
| B13 service identity/keys | Shared credentials and signing separation explicitly tracked | Private topology, ownership/secret cutover and key lifecycle |
| B14/B15/B17 acceptance/runbooks | Runnable isolated evidence and explicit gaps recorded | Worker adversarial testing, load/chaos/recovery and exercised operational runbooks |
| B16 legacy retirement | Compatibility remains explicitly visible | Measure usage and qualify replacements before removing legacy authority |
| Agent P2/P4/P5/P6/P7 | Selected auth/approval controls tested | Hosted SSH, full pipelines, two-host migration, provider payouts and multi-node sweep remain independent journeys |

No programme checkbox was marked complete on the strength of an unrelated unit pass. Complete audit coverage means a result or explicit gap is recorded; **security acceptance requires the outstanding high findings and deployment gates to be closed or explicitly accepted.**
