# Dashboard, credentials, billing, MCP marketing, and onboarding completion plan

Updated: 2026-10-08 (America/Toronto)

## Objective and working status

Complete every item in the original dashboard/onboarding request, including the visual polish and the actual user journeys behind it. Existing source changes are a starting point; a feature is complete only when its behavior and relevant rendered UI have been verified.

This plan was checked against commit `fdb5a6a`. The worktree was clean before adding this document. The original goal remains active.

Current evidence:

- The focused backend run passed **50 tests** across `test_admin_credit.py`, `test_event_pagination.py`, `test_mcp_quick_connect.py`, and `test_quick_connect_can_reach_what_it_promises.py`.
- Wizard lifecycle and renter/payment changes passed **598 tests across 35 files** and `npm run build`. The 15 added tests cover actual controller races, storage failure, checkpoint cleanup, expired sessions, selected GPU pricing, payment cancellation, and private-file replacement.
- An earlier session run reported 379 frontend tests passing. Changes followed that run, so it is historical evidence and must be rerun after the remaining fixes.
- The quick-connect reachability tests include static scope comparisons. Those passing does not establish that a real MCP connection or installed CLI skill can complete its workflow.
- Full browser verification, complete wizard journeys, and terminal animation review remain open. No production validation is implied by local tests.

## Scope tracker

| ID | User-visible requirement | Current source state | Remaining completion evidence |
| --- | --- | --- | --- |
| D1 | Paginate events | Newest-first cursor API and page controls implemented; focused backend tests pass | Browser traversal, filtering, incoming events, request failures, and session refresh |
| D2 | Avatar menu navigation works | Same-settings-page hash navigation handling implemented | Every menu destination, keyboard use, browser Back/Forward |
| D3 | Team menu opens without washing out or blanking the page | Full-screen backdrop removed; local dismissal handling implemented | Reproduce original hover sequence in a real browser, including theme and viewport changes |
| D4 | AI sidebar is closed after leaving and returning | Dashboard lifecycle reset implemented | Navigate out/back, reload, and browser history with previously open state |
| D5 | Avatar ring is visible in dark mode | Explicit ring and inner background styling implemented | Light/dark screenshots for image and initials avatars |
| V1 | Client secret modal is prominent and never wraps the secret | Wider modal, single-line scrolling field, copy, and acknowledgement implemented | Long secret, narrow viewport, clipboard failure, and all close paths |
| V2 | Notifications receive a substantial redesign and correct themes | Theme-aware toaster and new card styles implemented | All notification types, live theme switching, long content, actions, stacking, mobile |
| B1 | Admin grants credits without payment in test and production | Admin endpoint, credit ledger operation, and UI implemented; focused tests pass | Full UI-to-ledger round trip and broader billing/authorization regression |
| C1 | MCP calls its durable credential an API key | UI naming implemented | Consistency across instructions, dialogs, copied prompts, and translations |
| C2 | CLI shows the skill command and has its own working credential | Separate surface/client/key, independent audience and rotation; command restored | Real endpoint use, stable credential lifecycle, copied setup in an isolated project |
| M1 | MCP pixies remain behind the footer | Page isolation and footer stacking implemented | Scroll/hover screenshots with animation running in both themes |
| M2 | Cost guardrails card gets a complete facelift with cohesive SVGs | Card redesign and `guardrails-light.svg` / `guardrails-dark.svg` implemented | Compare visual language with existing assets and polish both themes at responsive sizes |
| W1 | Worker modal button says “View Manual Setup” | Label implemented | Destination and every advertised installation method work |
| W2 | Entire onboarding journey works robustly | Authentication/save/exit lifecycle repaired; renter price and payment cancellation regression added; resume and full-track gaps remain | Integrated renter, provider, both, SDK, retry, cancel, and resume journeys |
| W3 | Hexara visibly dances with coherent choreography and rendering | Half-block renderer, stage, and animation director implemented | Actual terminal motion, resizing, interruptions, input responsiveness, and a viewable recording |

## Execution order

1. Establish isolated wizard test/configuration storage and repair asynchronous lifecycle behavior.
2. Complete renter, provider, both, and SDK journeys using the real flow controller.
3. Finish credential lifecycle and admin-credit integration checks; fix any contract gaps they expose.
4. Verify dashboard navigation, team switching, events, and sidebar lifecycle in the browser.
5. Finish the shared visual pass: secret modal, notifications, avatar, MCP card, footer, and SVGs.
6. Tune Hexara against the corrected journey states and capture the running terminal animation.
7. Run the integrated regression/build matrix, regenerate affected contracts, and produce the completion record.

This order puts confirmed workflow defects first, gives the UI stable behavior to reflect, and avoids tuning the animation around incorrect success states. Related fixes should remain reviewable in small groups. Update this tracker after each group with exact evidence and outstanding limitations.

## 1. Wizard lifecycle and reliable progress

Primary files: `wizard/src/useWizardFlow.ts`, `wizard-state.ts`, `config-files.ts`, `wizard-guards.ts`, and the flow tests.

Gaps identified at the planning baseline, with implementation status:

- Fixed: authentication attempts now invalidate requests already in flight on method switch, retry, step exit, and unmount. Controller regressions exercise delayed device-code, token, and profile responses.
- `ensureWorkerOAuthClient()` writes shared answer state after an asynchronous operation and catches all failures by returning the old answers. This can overwrite newer state or hide a missing credential.
- Fixed: authentication verifies the account before writing a credential, and storage failure blocks progression. Ordinary sign-in no longer writes the current project's environment file; SDK setup owns project configuration changes.
- Resume initialization only explicitly restarts device authentication. Restoring an automatic check, marketplace fetch, or payment gate needs its own initialization.
- Partly fixed: delayed initialization checks mount, step, and flow identity before starting; automatic check completions also check those identities. Individual provider/SDK operations and AI streams still need a full cancellation audit.
- Fixed: required configuration writes complete before the success state; failures remain retryable. Completion/cancellation clears scheduled checkpoint writes and does not recreate a checkpoint on unmount.
- Fixed: `XCELSIOR_CONFIG_DIR` is shared by token/config/checkpoint storage. Every wizard test gets a temporary directory; existing tests no longer reassign `HOME`. Private writes atomically replace files with mode 0600.
- Fixed: checkpoint hydration runs once per mount and an expired checkpoint starts a fresh journey. Repeated input cannot change an answer during its transition.
- Fixed: wallet checks read the current GPU selection instead of an initial empty list. Missing prices are errors, payment polls cannot overlap, and skipping payment cancels the launch intent and invalidates late poll responses.
- Open: `/instance` currently creates a new job on each POST. Reliable resume/retry after an ambiguous launch response needs a durable request identity and server-side deduplication/recovery before automatic replay is enabled.
- Integration lead: the existing `/api/v1/launch-plans` service already supports durable execution by plan ID. Its current canonical launch specification does not include the wizard's selected `host_id`. Resolve the host-selection and quote contract before migrating the wizard; a migration must preserve the GPU/host the user selected.

Planned work:

1. Introduce a task-specific configuration directory override, such as `XCELSIOR_CONFIG_DIR`, shared by token, config, and checkpoint helpers. Keep the normal user default. Tests must use temporary directories and clean only their own files.
2. Give authentication attempts and step operations explicit identities. Check those identities after every asynchronous boundary before changing state, advancing a step, opening a browser, or writing credentials.
3. Invalidate active attempts on method switch, retry, cancellation, step change, completion, and unmount. Clear scheduled transitions as well as polling timers. Abort requests where supported; otherwise discard obsolete results.
4. Make credential provisioning return its result without secretly changing flow state. Distinguish optional setup from credentials actually required by the selected track. Report actionable failures.
5. Verify the authenticated user/workspace before accepting and persisting a sign-in. Handle expiry, denial, malformed responses, and persistence failures as recoverable states.
6. Centralize step entry so normal progression and checkpoint resume run the same initialization. Resume must respect the service gate and revalidate assumptions that can expire.
7. Prevent duplicate Enter presses, retries, and stale completions from launching work twice. Give mutating operations stable retry identities where the API supports them.
8. Await required configuration writes before declaring success. Cancel pending checkpoint writes before clearing a completed checkpoint. Show truthful cancellation and partial-completion messages.
9. Restrict project environment-file changes to the SDK integration path. Ordinary renter/provider sign-in should save the credential in the wizard configuration directory.

Acceptance:

- Switching from device to manual authentication while a request is pending cannot reopen the old flow or save its credential.
- Repeated Enter, Retry, or resume cannot create duplicate clients, hosts, or instances.
- Resume works at each automatic step as well as interactive steps; it does not hang waiting for work that never restarted.
- Unwritable storage gives an actionable failure instead of “all tasks complete.”
- Cancelled/failed/pending admission states are distinct from successful completion.
- Tests never read, overwrite, or clear the developer's real credentials or checkpoint.

## 2. Complete each onboarding track

Primary files: `wizard-flow.ts`, `useWizardFlow.ts`, `api-client.ts`, `sdk-checks.ts`, `provider-checks.ts`, and dashboard host installation surfaces.

The existing flow-walk tests mainly traverse step definitions. The Ink smoke tests cover service gates and initial mode selection. Add integrated tests that drive the actual controller through completion while controlling API, clock, filesystem, and process boundaries. Retain separate contract tests for the real API adapters so mocks cannot hide a mismatch.

| Track | Journey to exercise | Required end state |
| --- | --- | --- |
| Rent, no launch | Sign in → verify connection → decline launch → save | Setup is saved; no instance or charge is created |
| Rent, launch | Sign in → SSH setup → marketplace → GPU/environment selection → review → wallet → launch → status/access | One intended launch, accurate status, usable connection details when ready |
| Provide | Docker/GPU prerequisites → sign in → versions/network → benchmarks/verification → pricing → registration → admission status → worker install | Worker install is verified; server admission state is reported accurately |
| Both | Complete provider path, then renter path or decline launch, then save | Credentials and answers survive the track transition; neither path repeats side effects |
| SDK | Detect supported project → sign in → install/verify package → save credentials → real SDK request → starter example | Existing project settings survive and the generated example actually runs |

Renter work:

- Cover empty inventory, stale selections, changing availability/prices, insufficient funds, payment cancellation, expired authentication, and retry after a network timeout.
- Keep selected workspace, wallet owner, key permissions, and launch ownership consistent throughout.
- Confirm SSH setup uses actual supported keys and preserves existing user configuration.
- Verify payment polling stops on exit and can resume without duplicating payment or launch actions.
- Distinguish request accepted, provisioning, ready, and failed. Connection instructions should reflect actual readiness.

Provider work:

- Trace every advertised install method from the dashboard button or wizard entry to the actual worker runtime. Inventory the methods first; verify the exact artifacts, dependencies, commands, and permissions each uses.
- Audit the current systemd path, which downloads a single Python file and reports “running” after `systemctl start`. Align it with the real supported worker package and verify service health plus server registration/heartbeat before claiming readiness.
- Replace shell-interpolated installer arguments with structured process arguments where applicable. Use unique temporary paths, bounded execution, surfaced output, and correct permissions on existing credential files.
- Confirm the onboarding credential can call required host-registration, platform-key, verification, and heartbeat endpoints without granting unrelated administrator capabilities.
- Make host registration retries stable. Review how detected networking and pricing are carried into registration; avoid substituting unreported defaults when detection or lookup fails.
- Keep local compatibility checks, benchmark evidence, worker verification, and server admission separate. Preserve the protections tested by `test_returning_from_onboarding_proves_nothing.py`.
- Exercise prerequisite failures and remediation: missing Docker/NVIDIA tooling, unavailable mesh networking, insufficient privileges, failed download, inactive service, and missing heartbeat.
- Separate real hardware validation from mocked unit/integration results in the completion record.

SDK work:

- Make the package step match the product promise. Detect the project's package manager and perform the intended installation; verify a usable package import, not merely a dependency string in `package.json`.
- Report unsupported projects and installation failures clearly. Do not claim Python SDK integration from detection of `pyproject.toml` when the path installs a TypeScript SDK.
- Keep the preserving, atomic environment-file writer; cover existing unrelated values/comments, duplicate credential definitions, quoting, permissions, and write failure.
- Ensure the generated example uses the actual SDK constructor, authorization headers, environment, and return types. Compile and run it in a temporary project against the intended API.
- Verify the chosen environment-file name is loaded by the selected framework, or provide the exact loading step for plain Node. Keep credentials in server-side configuration.
- Validate both supported sign-in tokens and durable API keys against their actual lifetime and permissions. Persistent integrations must not silently depend on an expiring credential with no renewal path.

Acceptance: all four tracks complete through the real controller; their retry/cancel/resume branches are exercised; each side effect is attributable to a user step; installed/runtime status is supported by observed evidence.

## 3. MCP and CLI credentials that match the copy

Primary files: `oauth_service.py`, `routes/auth.py`, `frontend/src/components/dashboard/mcp-connect-card.tsx`, API helpers/translations, and quick-connect tests.

Keep these product terms explicit:

- **MCP API key:** the durable key issued for the MCP connection.
- **CLI API key:** a separately issued key for the CLI skill. Separate client/key identity and independent revocation are required.
- **OAuth access token:** retain this term for an actual OAuth token. Do not rename protocol response fields merely to change visible copy.
- **Client secret:** the confidential OAuth client's secret, distinct from both quick-connect keys.

Planned work:

1. Keep `npx skills add xcelsior-gpu/skill` prominent in a code font with exact command copying. Preserve the full setup prompt and make the command and key visually distinguishable within it.
2. Verify that the installed skill uses the CLI key successfully. Retain `XCELSIOR_ACCESS_TOKEN` where its existing contract requires it even though the interface says “CLI API key.” Include the correct API environment in the copied setup for non-production connections.
3. Exercise mint, masked reload, copy, rotation, revocation, workspace switch, and account switch for each tab independently.
4. Resolve the current copied-but-unused-key hazard: quick-connect may remint an unused key on a later GET. Loading another page/tab must not silently invalidate a credential the user just copied. Make creation/replacement semantics explicit and test concurrent requests.
5. Verify actual route use with the issued credentials: marketplace/instance operations for CLI and a real MCP handshake/tool call for MCP. Assert the intended audience and scope policy at each boundary.
6. Prove rotating MCP leaves CLI working and vice versa; the superseded key must fail. Confirm the orphaned-key migration preserves unrelated keys.
7. Check session refresh, network failures, stale responses during account/team changes, clipboard denial, and repeated React effects.

Acceptance: each copied setup works from an isolated client; labels describe the actual credential; neither tab exposes or rotates the other's key; refreshes and retries have predictable effects.

## 4. Admin credits with accurate wallet accounting

Primary files: `billing.py`, `routes/billing.py`, admin-users and billing UI, `frontend/src/components/billing/admin-credit.tsx`, and `tests/test_admin_credit.py`.

The implemented path records a ledger credit and leaves paid-deposit totals unchanged. It requires a platform administrator, an existing customer wallet, an amount, a reason, and an idempotency key. Current validation permits positive CAD amounts in cents up to CAD 10,000 per grant.

Planned work:

1. Walk the real admin UI → authenticated endpoint → ledger → updated wallet/history path. Verify the intended customer/workspace is obvious before submission.
2. Confirm the same administrator capability works with production configuration and test configuration, independently of payment checkout and relaxed-auth settings.
3. Verify ordinary users and team administrators cannot grant credits by calling the endpoint directly.
4. Cover concurrent submissions, retry after a lost response, reuse of a key with changed amount/reason, invalid amounts, and missing wallets. Preserve exact currency precision.
5. Verify history/audit attribution includes actor, recipient, amount, reason, and transaction identity. Review failure behavior between balance mutation and audit recording so retry does not create a second credit or an unexplained record.
6. Confirm billing pages, wallet headers, and launch affordability refresh after a grant; distinguish granted credits from cash paid in reports.

Acceptance: one administrator action produces one spendable ledger credit and accurate audit/history; no payment processor is invoked; non-administrators are rejected; repeating the same request cannot increase the balance again.

## 5. Dashboard navigation, events, and lifecycle

Primary files: `dashboard-shell.tsx`, `team-switcher.tsx`, dashboard settings/events pages, `routes/events.py`, `events.py`, and theme/styles.

Navigation and team menu:

- Exercise every avatar item from another dashboard page and while already on settings. Verify destination/tab, menu dismissal, keyboard focus, Escape, and browser history.
- Reproduce opening and hovering the team menu with the page scrolled and unscrolled. Keep the rest of the page visually stable and interactive as intended, without the unwanted full-page overlay.
- Verify outside click, keyboard selection, workspace loading/error states, rapid team switching, and narrow-screen placement.
- Inspect stacking and backdrop/filter behavior in real browsers, since DOM-only tests cannot reproduce compositor defects.

Events:

- Verify the newest 25-item page, older/newer controls, boundaries, empty state, loading/retry, and reset behavior after changing filters.
- Preserve deterministic ordering for equal timestamps and stable older pages as new events arrive. New events should offer a return to latest instead of moving rows under the user.
- Confirm visible timestamps use the correct units and that export means the current page unless explicitly designed otherwise.
- Test stale request cancellation, expired-session refresh through the shared authenticated client, and authorization rejection.
- Inspect query/index support and response time with representative event volume if the new order/filter query is not already covered by an appropriate index.

AI sidebar and avatar:

- Open the sidebar, leave the dashboard, and return via link, Back, and reload. It must start closed after the requested lifecycle transition.
- Verify listener cleanup and intended behavior when navigating between dashboard pages.
- Check the avatar ring for image/initials variants in light and dark themes, including hover and focus.

Acceptance: the original click/hover failures are no longer reproducible, pagination remains stable under updates, and navigation state follows the user's requested lifecycle.

## 6. Shared visual polish and MCP marketing

Primary files: `oauth-secret-reveal-modal.tsx`, `ClientToaster.tsx`, theme/global styles, MCP `content.tsx`, `marketing-theme.css`, and `frontend/public/mcp/guardrails-{light,dark}.svg`.

Secret modal:

- Make the secret the visual focus with strong hierarchy, an obvious copy action, and a clear one-time reveal message.
- Keep the entire value intact in a single-line monospace field. Use horizontal scrolling on small screens; never split or truncate the copied value.
- Verify copy success/failure, long client IDs/secrets, acknowledgement, Escape/outside click/close behavior, focus handling, mobile layout, and English/French text.

Notifications:

- Review success, error, warning, info, loading, and promise transitions as a coherent card family with readable title/body and restrained status accents.
- Ensure theme selection comes from the active application theme, including system-theme changes. Review contrast and surface opacity in both themes.
- Check action/close buttons, long messages, multiple stacked notifications, placement with the sidebar open, mobile safe areas, focus, and reduced motion.
- Trigger notifications through real product actions during verification rather than shipping a temporary preview route.

MCP card and footer:

- Refine the guardrails card's hierarchy and composition around what the product actually enforces. Avoid visual claims or example counters that imply unsupported live behavior.
- Match the existing GPU/bolt artwork's geometry, line treatment, spacing, and restrained glow; finish dedicated light/dark SVG variants with compatible proportions.
- Inspect the illustration at actual card size and on mobile, not only as a standalone SVG.
- Verify particles remain behind the footer throughout scrolling and theme changes. Check painting order, background opacity, click targets, and overlap with other cards.

Acceptance: a reviewed screenshot set covers each changed component in both themes and desktop/mobile layouts; the secret never wraps; toasts follow the selected theme; the footer consistently covers the particles.

## 7. Hexara rendering and choreography

Primary files: `wizard/src/HexaraStage.tsx`, `useWizardAnimation.ts`, `hexara-choreography.ts`, `capability.ts`, the frame generator, and animation/stage tests.

Rendering work:

- Keep the Ink-compatible coloured half-block renderer and verify it in actual terminal output. The current source no longer relies on Sixel support to show the wizard.
- Check palette/contrast on light and dark terminal backgrounds, floor/shadow placement, sprite anchoring, transparent regions, and vertical headroom.
- Test wide, compact, and narrow layouts plus resizing during motion. Hexara must not clip, overwrite prompts, leave stale cells, or force continuous scrollback.
- Verify terminal capability detection, explicit opt-out, and non-interactive output. Input and status rendering must remain responsive during movement and long-running operations.

Choreography direction:

| Journey state | Intended motion | Transition rule |
| --- | --- | --- |
| Arrival/mode selection | Entrance, wave, short dance, settle | Give the greeting room to read; do not restart on every render |
| User choosing/typing | Quiet idle, occasional glance or gesture | Avoid competing with the active prompt |
| Authentication/waiting | Looking around or gentle pacing | React promptly to authorization, expiry, or cancellation |
| Long checks/provisioning | Deliberate casting/pacing/levitation sequence | Loop smoothly without blocking progress updates |
| Routine successful check | Brief acknowledgement | Return to a calm waiting pose |
| Verified milestone | Full celebratory dance with visible lateral motion | Let the movement resolve cleanly; do not celebrate pending/failed outcomes |
| Error/retry | Clear error reaction, then attentive waiting | Recover naturally when the user retries |
| Successful completion | Celebration followed by bow/exit | Show only after required writes/operations succeed |
| Cancellation | Neutral acknowledgement/exit | Do not reuse the “everything succeeded” finale |

Tune pacing, travel distance, frame timing, and interruption rules by watching the animation. Unit assertions on frame counts alone cannot establish that the dance looks good.

Deliver a short terminal recording or playable local preview showing entrance, dance, casting, error recovery, and exit. If a dedicated preview entry is added, make it explicitly an animation demo with no authentication, billing, host registration, or fake onboarding completion.

Acceptance: the wizard visibly dances in a supported normal terminal, transitions are coherent, text remains usable, and the user has an artifact they can actually watch.

## 8. Final validation and completion record

Run focused tests while repairing each group, then run the broader suites against the final source once the changes settle.

| Area | Verification |
| --- | --- |
| Backend | Focused event/credit/key tests; broader billing, authorization, event, OAuth, onboarding-admission, and migration coverage; configured non-live backend regression suite |
| Frontend | Unit/component suite, lint, i18n checks, production build and existing PWA verification |
| Browser behavior | Real navigation/hover/copy/filter/team-switch flows, failure states, desktop/mobile, light/dark, system theme |
| Wizard | Isolated controller journeys, adapter contracts, filesystem/process failure cases, tests, build, and installed-package smoke check |
| Worker onboarding | Install-method contract checks plus actual service/heartbeat verification in a suitable test environment; hardware-dependent checks reported separately |
| Credentials | Issued-key use against real permitted endpoints, MCP connection, CLI skill setup, independent rotation and revocation |
| Animation | Terminal recording, resizing, keyboard responsiveness, and all choreography states |
| Generated contracts | Regenerate affected `public/openapi.json`, `fern/openapi.json`, and `docs/generated/endpoint-inventory.md`; compare for unintended drift |

Use the repository's existing commands and CI configuration. Current entry points include frontend `npm test`, `npm run lint`, `npm run i18n:check`, `npm run build`, `npm run verify:pwa`; wizard `npm test` and `npm run build`; and `.venv/bin/pytest` for backend tests. Check installed test-runner versions/configuration before relying on project separation or mocks.

Browser fixtures must block unintended external writes. Test doubles are appropriate for deterministic UI/failure checks, but credential correctness and install readiness require separate integration evidence. Use headless browsers in this environment because a desktop display is unavailable.

Final deliverables:

1. All scope-tracker rows resolved with source changes and relevant verification.
2. A screenshot set covering the redesigned dashboard components and MCP card/footer in both themes.
3. A viewable Hexara motion recording/preview.
4. A test/build record with exact commands, outcomes, skips, and any environment-dependent limitations.
5. A concise handoff explaining what changed and separating local completion from anything requiring deployed or hardware verification.

The goal stays open while a requested journey or visual remains unverified. A passing focused suite or a successful build alone does not close the full request.
