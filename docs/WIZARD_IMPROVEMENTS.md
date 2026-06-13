Done with all three tasks — packages fixed, 21 screenshots absorbed, and the plan rewritten and roughly tripled. Here's everything.

---
1. Did the wizard set up PostHog correctly?

Yes on code quality, no on completeness — and I just fixed the gap. The integration is actually more thorough than PostHog's own manual instructions, but it shipped broken because the packages were never installed. I ran npm install posthog-js posthog-node in frontend/; they're now in package.json (posthog-js@^1.386.6, posthog-node@^5.21.2) and node_modules. A typecheck is finishing in the background to confirm the imports resolve.

What it got right (verified against the files):
- instrumentation-client.ts uses the Next.js 15.3+ pattern — correct for your Next 16 / React 19 setup — and goes beyond the docs by using a reverse proxy (api_host: "/ingest") instead of hitting us.i.posthog.com directly.
- next.config.ts has the matching /ingest/static, /ingest/array, /ingest/* rewrites, and the CSP connect-src already lists us.i.posthog.com + us-assets.i.posthog.com.
- Server singleton (src/lib/posthog-server.ts), identify/reset in auth.tsx, 11 events with captureException error tracking, a remote dashboard (project 397400 / dashboard 1707940), and the .env.local token correctly gitignored.
- The agent skills it claimed are real: .claude/skills/integration-nextjs-{app,pages}-router/ at repo root (source: github.com/PostHog/context-mill).

Review before you commit:
- Build/run check — the typecheck I kicked off; I'll report if anything's off.
- Pre-existing vulns — npm install flagged 20 audit issues in the wider tree. Don't npm audit fix --force blindly; that's a separate, deliberate task.
- ui_host toolbar — instrumentation-client.ts sets ui_host: https://us.posthog.com, which isn't in script-src/connect-src. Only matters for the PostHog toolbar overlay, not capture; add it only if you want the toolbar.
- Confirm each instrumented file is client-side ('use client' or in an event handler) — they appear to be, but worth a glance since posthog-js is browser-only.

Net: it's good work that was 95% done; you were one npm install away, which is now run.

---
2. What the screenshots taught me — the caliber bar

PostHog's wizard is essentially a branded, narrated, two-pane TUI that turns an ~8-minute agent run into a product demo. The full inventory:

Frame / chrome
- Branded title bar (logo + v2.21.0 + persistent Feedback: wizard@posthog.com), hog-purple background, amber accent.
- Context-sensitive keybind footer everywhere (↑↓ navigate · enter select, esc skip, ↔ switch tab · s toggle status).

Opening
- Startup card: value prop + privacy reassurance (".env* contents will not leave your machine") + spinner "Detecting project framework…".
- Framework fallback: two-column keyboard-nav picker when auto-detect fails.
- Browser OAuth: org/project dropdowns + granular permission checklist, with a terminal fallback ("browser didn't open → paste this URL", "remote machine or devbox? Press R → paste auth code") and a "login complete, return to your terminal" browser page.

The working screen (the heart of it) — three regions:
1. Left "Learn"/"Tips" pane — a slideshow that advances while the agent works: sets a time expectation ("about eight minutes, grab a coffee ☕"), teaches concepts, and renders animated ASCII data-viz — an architecture diagram (App → capture() → SDK → Cloud → Dashboards), a trend line chart (signups → 9,575), a conversion funnel (app_launched 100% → ride_requested 72%, −28% drop), and a two-column product grid marketing every PostHog feature. "Tips" mode adds interactive nudges ("Use PostHog in Slack", "track your LLM costs — Press L to enable").
2. Right "Tasks" pane — a live checklist with ☐ todo → ▶ active (blue) → ■ done (green), multiple concurrent active items, and Progress: N/M completed.
3. Bottom status strip — typewriter log of granular sub-actions ("payment-method-modal.tsx — adding payment_method_added event", "Finding and correcting errors") with tabs that appear as content exists: Status · Event plan · Tail logs · HN (yes, a Hacker News reader to pass the wait).

After the core work — interactive upsells:
- MCP install — auto-detects installed agents ("Detected: Claude Code, Claude Desktop/Web, Codex, VS Code, Zed"), offers Install-all / Customize / No thanks.
- Slack integration — connect link + Open setup / Skip.
- "Keep the skills?" — shows the .claude/skills/ tree it dropped + Keep / Remove.
- Completion summary mirroring the report: "What the agent did", "Events added" (each with file), dashboard URL, the install nudge.

The takeaway: their "amazing" isn't graphics — it's a narrative that (a) hides latency behind education, (b) markets the whole product while doing one job, (c) shows honest, granular progress including self-correction, and (d) ends by deepening the relationship (MCP/Slack/skills). Our advantage they don't have: Hexara, an actual animated mascot. We can hit their narrative bar and out-charm them.

---
3. Expanded plan — gate + full polish/UX sweep to PostHog caliber

Grounded in our files: wizard/src/index.tsx (single-column renderer), useWizardFlow.ts (2235-line state machine; checkApi at :192 already pings /healthz, aiAvailable = !!apiToken at :671), wizard-flow.ts (WIZARD_STEPS + StepType), steps.tsx (per-step components), WizardLine.tsx/sprite system (Hexara animation), and backend routes/health.py (/healthz, /readyz).

Translation principle: ours onboards a user onto the GPU marketplace, it doesn't integrate an SDK. So PostHog's "Learn about events" becomes "Learn about the marketplace" (spot instances, XCU scoring, hardware verification, provider earnings); their "Tasks: install package" becomes our real onboarding steps (auth → detect GPU → benchmark → verify → register → launch). Same shapes, our substance.

Part A — Preflight service-health gate

Unchanged from before, refined: a pre-render gate driven by fetchServiceStatus(baseUrl, token?) → { services[], verdict: ok|degraded|blocked }. Operational → no panel, fall into flow. Degraded → show badges, Enter proceeds. Required service (API/DB) down → hard-block with status link + "Continue anyway (best-effort)". AI down but API up → soft: set aiAvailable=false, disable ?, surface STATIC_STEP_HELP. New preflight.ts, new StatusGate component, new "status-gate" StepType at index 0. Status source: start client-side (reuse checkApi's /healthz + add /readyz + AI ping) so it ships without backend work; keep the seam so a backend /api/status aggregator (shared with the dashboard) is a one-function swap later.

Part B — Two-pane "Learn + Tasks" layout (the biggest visual lift)

Today index.tsx renders one centered column. Re-architect the working screen into PostHog's three regions while keeping Hexara on top:
- Left "Learn" pane — slideshow component (Part C).
- Right "Tasks" pane — live checklist (Part D).
- Bottom status strip — tabbed log (Part E).
- Hexara's sprite/WizardLine stays as the title-row mascot (our differentiator), with the brand gradient already in steps.tsx driving the accent system.
- New WorkScreen.tsx composing the three panes with Ink <Box flexDirection="row">; gate this layout to the long-running auto-check steps (benchmark/verify/register/launch) where we actually have latency to fill, and keep simple single-prompt steps (mode select, pricing) clean and centered.

Part C — "Learn/Tips" slideshow engine + ASCII data-viz

A LearnSlides component cycling Xcelsior-flavored cards on a timer while long steps run:
- Concept cards: how the marketplace matches jobs to GPUs; what XCU scoring means; how spot/interruptible pricing works; provider earnings & reputation tiers.
- ASCII data-viz (the wow factor): a marketplace price/availability sparkline, a GPU utilization bar chart, a provider earnings trend line, and a job-lifecycle pipeline diagram (submit → schedule → run → settle) — reusing the BRAND_GRADIENT palette already in steps.tsx. Build a tiny asciichart-style renderer (line + horizontal-bar + funnel) so these are data-driven, not static art.
- Time expectation: "Benchmarks take ~60s — here's how XCU scoring works while you wait."
- Interactive nudges (PostHog's "Press L"): e.g. "Press S to enable spot instances on this host" or "Press A to ask Hexara anything" surfaced inline during waits.
- Product grid card marketing Xcelsior surfaces (marketplace, spot, serverless inference, volumes, verification, earnings dashboard).

Part D — Live task checklist + progress model

Our flow already knows its steps (WIZARD_STEPS + conditions + getNextStep), but renders them sequentially with a thin progress bar. Add a real task model:
- Derive the task list from the mode-filtered steps (the totalSteps logic in index.tsx:137 already computes this set).
- Per-task state todo → active → done → failed with glyphs (☐ ▶ ■ ✗) and Progress: N/M, supporting concurrent active tasks (e.g. "Installing worker agent" + "Configuring network" run together for providers).
- Drive it from the existing checkResults/stepIndex transitions in useWizardFlow.ts so it's a view over real state, not a parallel fiction.

Part E — Tabbed status pane + animated log + long-wait affordances

- Tabbed bottom strip: Status · Tail logs · Ask Hexara (our HN-equivalent is the AI escape hatch — more useful than a news feed, and we already have the streaming streamChat plumbing in api-client.ts).
- Typewriter status log: stream granular sub-actions from the real checks — provider-checks.ts/checks.ts already produce per-item results ("runc v1.1.12 ✓", "FP16 matmul 312 TFLOPS"); surface them line-by-line as they resolve instead of a single spinner.
- "Tail logs" tab: the raw check/benchmark output for power users.

Part F — Honest, self-correcting status messaging

PostHog shows "Finding and correcting errors" — honesty as polish. Mirror it: when an auto-check fails and auto-retries (the checkCanRetry/retryCheck path), say so plainly ("Driver too old — guiding you through the fix") rather than a generic spinner. Ties into the user's "no bugs, works well, sensical" bar: never show a dead spinner; every wait states what's happening and the ETA.

Part G — Post-flow value-add prompts (translated)

After the core onboarding completes (the DoneStep), add PostHog-style optional deepeners:
- Provider: "Install the worker agent as a service? (recommended)" — we already have worker-install mid-flow; offer the systemd/background-service path here with detect-and-confirm.
- Renter: "Connect your repo / set up SSH config?" and surface the dashboard + first-instance links (the DoneStep already prints SSH + dashboard URLs — make them an interactive menu).
- Notifications: "Get job/earnings alerts" (Slack/email/web-push — backend already has web-push per routes/health.py metrics).
- MCP/agent: if we ship an Xcelsior MCP, the same "Detected: Claude Code/…" auto-install prompt.

Part H — "Works well, no bugs" hardening pass (you emphasized this)

Quality is a first-class deliverable here, not an afterthought:
- Fold in the original polish items: aiAvailable reflects real AI health not just token presence (useWizardFlow.ts:671); front-load the API check via the gate so device-auth isn't wasted when the control plane is down (the api-check-is-step-4 problem, with the telling comment at :1657); route AI stream errors through mapHttpError remediation (api-client.ts:259,301); surface WizardError.remediation/url on failed checks (today AutoCheckStep shows only detail); resume notice names the step (index.tsx:175).
- Layout robustness: the new two-pane layout must degrade on narrow terminals (reflow to single column < ~100 cols), survive resize, and not corrupt the Hexara scroll-region setup (setupWizardRegion/resetWizardRegion).
- Test coverage: extend the strong existing __tests__ suite (vitest) — preflight.test.ts, learn-slides.test.ts (timer/advance/teardown), task-model.test.ts (state transitions, concurrent active), plus an ink-testing-library render smoke for WorkScreen at narrow/wide widths.
- End-to-end smoke: a dev-mode run against a stubbed API covering each mode (rent / provide / both) with forced degraded + blocked gate verdicts, so we see it works, not just unit-pass.

Part I — Brand & motion polish (the "looking amazing" bar)

- Design tokens: centralize the BRAND_GRADIENT + state colors (STATE_COLORS) into one theme module; consistent glyph set (◆ ▸ ■ ☐ ✓ ✗ ⋮); persistent title bar (version + "Ask Hexara: type ?" hint) and context-sensitive keybind footer like PostHog's.
- Motion: lean into Hexara — idle/thinking/success/error choreography already exists (wizardBranch in index.tsx:151); sync sprite mood to task state (celebrate on each ■, levitate while thinking, error pose on ✗) so the mascot narrates progress. This is the thing PostHog literally cannot copy.
- Truecolor/sixel fallback: ensure graceful degradation on terminals without 24-bit color (there's already a test-sixel.tsx — wire a capability check).

Sequencing

- Phase 1 — Hardening first (Part H + A): ship the preflight gate and fix the correctness/polish bugs. Foundation must be solid before decoration; this also directly answers your "no bugs, works well" priority.
- Phase 2 — Layout + task model (Parts B, D): the two-pane WorkScreen and live checklist — the structural change everything else hangs on.
- Phase 3 — Narrative (Parts C, E, F): the Learn slideshow + ASCII viz, tabbed status log, honest messaging — the latency-hiding "demo" layer.
- Phase 4 — Deepeners + polish (Parts G, I): post-flow prompts and the brand/motion pass that makes it look amazing.

One open call before I build: status source for the gate — I recommend client-side first (ships now, reuses checkApi) with the seam for a backend /api/status later. If you're good with that, say go and I'll start Phase 1 (gate + hardening). I'll also report the background typecheck result the moment it lands.

✻ Worked for 11m 33s

● Background command "Verify packages recorded and imports resolve" completed (exit code 0)

The background typecheck finished (exit 0). Let me read its output to confirm the imports resolve cleanly.

  Read 1 file

Confirmed — the fix holds. The typecheck surfaced no PostHog-related errors, which means the import posthog from "posthog-js" sites across the 11 instrumented files now resolve (a missing module would have thrown Cannot find module 'posthog-js', which would have shown up). So the integration that the wizard left 95% done is now complete and compiles.

That closes out the correctness question: the wizard did the setup correctly and well — it just hadn't installed the packages, which is now done. Everything else (the screenshot-informed UX inventory and the expanded gate + polish plan) is in my previous message. The one decision I need from you before building is the gate's status source — I recommend client-side first — then I'll start Phase 1 (gate + hardening).

❯ ok plan approved start implementation A few quick thoughts:
  •  The Hexara mascot is your ace. PostHog can’t touch that. Leaning hard into her reactions synced to task states is going to feel magical in a terminal.
  •  Client-side status gate first is smart — ships now, validates the concept, easy to swap later.
  •  The ASCII data-viz + Learn pane is ambitious in Ink but worth it. Nail the sparkline and utilization bar first; those will pop. just do the backend now for the status source for the gate. ok go. new packages (the very best) and new migrations are not out of scope.