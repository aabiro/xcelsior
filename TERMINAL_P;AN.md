Plan: Web Terminal Complete Rewrite
TL;DR: The current terminal has two root causes — a stale container name in the DB payload (causing No such container before any shell starts) and missing xterm.js configuration/addons causing glyph corruption. The rewrite is a 4-phase effort targeting the frontend (addons + config + protocol), backend (container resolution + binary streaming + rate limit), and infrastructure (tmux session persistence). No new services needed; existing SSH mesh, PTY, and WebSocket infrastructure is sound.

Phase 1 — Frontend Packages (~30 min, parallel with nothing)
Add to package.json devDependencies:

@xterm/addon-webgl — GPU-accelerated renderer (hardware-accelerated canvas, fixes texture atlas corruption)
@xterm/addon-search — in-terminal search (Ctrl+Shift+F)
@xterm/addon-unicode11 — correct wide-char (CJK/emoji) cell-width calculation
Run npm install.

Phase 2 — Frontend Rewrite: WebTerminal.tsx (depends on Phase 1)
2.1 — Terminal configuration fixes (resolves glyph overlap + cursor jump)

Add to new Terminal({...}):

Option	Value	Why
customGlyphs	true	Draws box/block chars via canvas instead of font fallback — eliminates the blue-outlined glyph artifact
convertEol	false	Do NOT patch raw line feeds clientside; the backend already allocates a full PTY that handles CRLF line discipline natively. Setting this true would break vim/htop cursor positioning
macOptionIsMeta	true	Makes Option key work as Meta on macOS (word-jump, etc.)
2.2 — WebGL addon with texture atlas lifecycle

Dynamically import WebglAddon, call term.loadAddon(webglAddon) inside a try/catch (falls back to Canvas renderer on WebGL context failure — e.g., on some Linux VMs)
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') webglAddon.clearTextureAtlas(); }) — clears corrupted GPU texture after tab switch or sleep/wake
In the existing ResizeObserver callback: also call webglAddon?.clearTextureAtlas() — clears artifacts after viewport resize
2.3 — Unicode11 addon

Load Unicode11Addon and call term.unicode.activeVersion = '11' — corrects cell-width for wide Unicode characters so they don't overflow into adjacent cells.

2.4 — Binary WebSocket protocol (output only)

The current JSON-wrapped output (encode UTF-8 → JSON string → decode JSON → write string) has encoding edge cases with binary escape sequences and adds unnecessary overhead. Switch to:

socket.binaryType = 'arraybuffer'
socket.onmessage: branch on event.data instanceof ArrayBuffer → term.write(new Uint8Array(event.data)) for terminal output; else parse JSON for control messages (error, exit, pong)
Sending direction is unchanged: client still sends text JSON {type: "input"/"resize"/"ping"}
2.5 — Exponential backoff reconnect

Add reconnectAttempts and reconnectTimer refs. In ws.onclose: if not user-initiated and reconnectAttempts.current < 8, schedule connect() after Math.min(1000 * 2 ** reconnectAttempts.current, 30_000) ms. Update toolbar badge to show "Reconnecting (N/8)". Reset count on successful onopen. Manual "Reconnect" button (already in toolbar) resets count and calls connect() immediately. Write timeout message to xterm on final failure.

2.6 — Keyboard interceptors

Add term.attachCustomKeyEventHandler((e: KeyboardEvent) => {...}):

Ctrl+C: if term.hasSelection() → navigator.clipboard.writeText(term.getSelection()); return false (prevent sending \x03). Else: send \x03 interrupt signal through WebSocket.
Ctrl+Shift+V / Ctrl+V on paste: navigator.clipboard.readText() then ws.send(JSON.stringify({type:"input", data})).
Return true for everything else (pass through to xterm default handling).
2.7 — Focus management

Call term.focus() immediately after term.open(containerRef.current) on mount.
Call term.focus() inside ws.onopen — ensures the terminal captures keystrokes as soon as the connection is established, preventing the "click jumps to new line" behavior.
Add onClick handler on the container <div> that calls term.focus().
2.8 — Search panel UI

Add a search icon button in the toolbar. Toggle a small overlay <div> with an <input> wired to searchAddon.findNext(query, {caseSensitive, regex}) / findPrevious(...). Include clear/close.

2.9 — Connecting state

Add "connecting" to the state enum alongside connected/error/disconnected. Set before calling connect(), clear on ws.onopen. Show a spinner badge in the toolbar during connection.

2.10 — Cleanup

webglAddon?.dispose() before term.dispose() on unmount.
Clear reconnect timer in useEffect cleanup.
Phase 3 — Backend Rewrite: instances.py (ws_terminal, lines 983–1241) (parallel with Phase 2)
3.1 — Container name resolution (root cause fix)

The error No such container: xcelsior-a6c44638 proves the stale container_name payload value is being used. Current code: container_ref = instance.get("container_name") or instance.get("container_id") or f"xcl-{instance_id}" — tries old format first.

Replace with a priority list:


candidates = [    f"xcl-{instance_id}",                              # current naming convention    instance.get("container_name", ""),                # stored name (may be stale)    instance.get("container_id", ""),                  # short hash    f"xcelsior-{instance_id}",                        # legacy format]container_ref = next((c for c in candidates if c), f"xcl-{instance_id}")
Add a pre-flight container check: before spawning the subprocess, SSH in with a quick docker inspect --format='' {ref} (or check exit code). If it fails, try next candidate; if all fail, send a structured {"type": "error", "message": "Container not found — it may still be starting. Retry in a few seconds.", "code": 4410} and close gracefully.

3.2 — Binary output protocol

In _stdout_relay(), replace:


await websocket.send_json({"type": "output", "data": text})
with:


await websocket.send_bytes(chunk)   # raw PTY bytes, no encode/decode roundtrip
Keep send_json for: error, exit, pong control messages.

3.3 — Rate limit increase

Raise _TERMINAL_RATE_LIMIT_BYTES from 10_240 (10 KB/s) to 524_288 (512 KB/s). Dense ML compilation logs and tail -f of training output easily exceed 10 KB/s, causing visible throttle stuttering during active workloads. 512 KB/s is still protective against DoS flooding.

3.4 — SSH command quoting fix

The remote docker exec arguments are currently passed as separate list elements — SSH joins them with spaces to form the remote shell command. The shlex.quote(container_ref) is correct, but "-e", "TERM=xterm-256color" in list-form will reach the remote shell unquoted. Fix by constructing the remote command as a pre-quoted string via shlex.join([...]) and passing it as a single arg to SSH:


remote_cmd = shlex.join(["docker","exec","-e","TERM=xterm-256color","-it", container_ref, shell])docker_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", ..., "-tt", f"{SSH_USER}@{host_ip}", remote_cmd]
3.5 — tmux session wrapping (session persistence)

Wrap the shell command in tmux to survive network drops:


# Try tmux first; shell fallback handled by exit-code detectiontmux_cmd = f"tmux new-session -A -s term-{instance_id} {shlex.quote(shell)}"
On reconnect, tmux new-session -A attaches to the existing session instead of spawning a new shell. Training scripts, vim sessions, and long-running processes persist across browser refreshes.

Add a fallback: if the PTY output immediately contains tmux: command not found, kill and re-exec with direct shell. This handles containers that don't include tmux.

On WebSocket disconnect: instead of process.kill(), send the tmux detach key sequence (\x02d = Ctrl+B D) to the PTY stdin before closing to gracefully detach the session instead of destroying it.

3.6 — Session timeout extension / idle detection

Extend _TERMINAL_SESSION_TIMEOUT from 1800 to 14400 (4 hours) — appropriate for long ML training sessions. Add idle detection: track last input message timestamp; send a warning at 3h55m, close at 4h if no recent input.

Phase 4 — Instance Detail Page: page.tsx (depends on Phase 2)
Expand default WebTerminal height from h-[400px] to h-[500px] in WebTerminal.tsx
The existing dynamic() import is already correct (SSR disabled)
Show terminal panel only when instance.status === 'running' (already done, confirm it persists for tmux sessions after job completes)
Relevant Files
WebTerminal.tsx — full rewrite
package.json — add 3 addons
instances.py — ws_terminal() ~lines 983–1241 (container resolution, binary protocol, rate limit, SSH quoting, tmux)
frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx — minor height/condition adjustments
No changes needed: xcelsior.conf (WebSocket upgrade headers already present, 86400s timeout set), db.py, scheduler.py, docker-compose.yml, auth system.

Verification
Glyph test: Open terminal on a running instance with Nerd Font glyphs (e.g., echo $PS1 with powerline prompt) — verify no blue rectangle artifact at top of viewport after tab switch and resize
Focus test: Click into the terminal after connection — cursor should NOT advance a line. Verify $ prompt is visible immediately.
Container fallback test: Manually set a stale container_name in a test instance's JSONB payload (e.g., xcelsior-fakeid) — verify terminal still connects by falling through to xcl-{instance_id}
Reconnect test: Kill the WebSocket proxy mid-session (e.g., nginx -s reload) — verify frontend reconnects automatically within 2–5 seconds and resumes the shell
tmux persistence test: Start a long sleep 3600 in the terminal, close the browser tab, reopen the instance page — verify the sleep is still running in the reattached tmux session
Binary protocol test: Run a command with heavy output (e.g., cat /dev/urandom | head -c 1M | xxd) — verify no encoding artifacts and throughput > 100 KB/s
Rate limit test: Tail a rapidly-updating training log — verify no stuttering or artificial delays at the new 512 KB/s limit
Keyboard test: Run vim, press Ctrl+C — verify interrupt (not copy) when no text selected; highlight text then Ctrl+C — verify clipboard copy
Decisions
SSH subprocess kept (not replaced with docker-py over TCP/SSH): the existing Headscale mesh SSH approach works correctly and hosts don't expose Docker daemon on TCP. Migration to docker-py over ssh:// would require paramiko and introduces new failure modes without sufficient benefit.
convertEol: false is the correct setting despite Report 2's snippet showing true — the backend already has a proper PTY handling CRLF, so a frontend patch would break vim/htop cursor control
tmux is conditional: wrapped in try/not-found fallback so containers without tmux work transparently
Out of scope: gVisor/Kata container isolation changes, auth system changes, new microservices, Prometheus terminal metrics

Folluw-ups:

1.

To give you an honest, unbiased opinion: you are absolutely right to be suspicious. While the other AI did a fantastic job diagnosing the frontend rendering bugs, it significantly cut corners on the backend architecture and dismissed critical production features as "out of scope." 

Here is a breakdown of what the plan got right, where it cut corners, and exactly how you should change it.

### What the Plan Got Right
The frontend recommendations are exceptionally good. The AI correctly identified that `customGlyphs: true` fixes the overlapping artifacts, that `convertEol: false` is necessary when a true pseudo-terminal is used, and that hooking into the browser's visibility lifecycle to clear the WebGL texture atlas prevents corrupted graphics after a tab wakes from sleep `[1]` `[2]` `[3]`. The suggestion to use `tmux` for session persistence is also a highly effective industry standard `[4]`.

### Where They Cut Corners (The "Decisions" and "Out of Scope" Sections)

The plan falls apart in its backend architectural decisions. By trying to avoid modifying your existing Python backend too much, the AI recommended a "band-aid" approach rather than a true enterprise-grade rewrite.

**1. Keeping the SSH Subprocess**
The decision to spawn a local `ssh` subprocess to connect to the container instead of using the Docker SDK (`docker-py`) over TCP is a massive corner cut. Relying on `subprocess` is not horizontally scalable; it consumes excessive memory and introduces process-management complexities (like orphaned processes) as user counts grow `` ``. Standard implementations use Docker's native attach API to stream the pseudo-terminal directly to the WebSocket `[5]` `[6]`.

**2. Excluding Auth System Changes**
Leaving authentication changes "out of scope" demonstrates a lack of understanding of WebSocket mechanics. Standard browser WebSocket APIs (`new WebSocket()`) do not support sending custom HTTP headers like `Authorization: Bearer` ``. If your current platform relies on standard HTTP header authentication, the terminal connection will fail. You absolutely must modify the auth system to accept JSON Web Tokens (JWTs) via query parameters or WebSocket subprotocols ``.

**3. Ignoring gVisor/Kata Isolation**
In a multi-tenant GPU marketplace, users are executing untrusted code on shared host nodes. Leaving gVisor or Kata container isolation out of scope is a critical security risk. The terminal bridge must be explicitly designed to respect these microVM security boundaries to prevent container escape attacks `` ``. 

**4. Excluding Prometheus Metrics**
For a cloud provider, real-time telemetry is not a "nice to have." WebSockets are persistent, stateful connections that behave differently than standard HTTP requests ``. You must monitor connection counts, drop rates, and handshake latency with Prometheus to ensure the service remains stable under load ``.

### How to Change the Final Plan

To achieve the best outcome possible for the Xcelsior web terminal, integrate the frontend fixes from their plan, but override their backend decisions with the following changes:

*   **Change the Transport Mechanism:** Scrap the `subprocess` SSH wrapper. Update the FastAPI backend to use the Docker Engine API (via `docker-py`) to execute `exec_create` and `exec_start` with `tty=True`. Pipe this raw binary stream directly to the WebSocket.
*   **Bring Auth In-Scope:** Modify your FastAPI WebSocket route to authenticate incoming connections by validating a short-lived, single-use token passed via the URL query string or as the first frame of the WebSocket subprotocol.
*   **Enforce Security Boundaries:** Ensure the new terminal routing explicitly integrates with your existing gVisor/Kata sandbox configurations. The terminal process must spawn inside the restricted network namespace, not the host namespace.
*   **Implement Telemetry:** Add Prometheus metric collectors to the WebSocket route to track active terminal sessions, bytes transferred per second, and sudden disconnection events.

2.

**Honest, no-holds-barred verdict: This is a genuinely strong plan (8.5/10). It is *not* lazy or half-assed.**

The team that produced this did an excellent job synthesizing the two reports you fed them. They correctly identified the *real* root causes (stale `container_name` + no true PTY + glyph texture corruption + missing line discipline), incorporated the deepest technical advice from Report 1 (customGlyphs, convertEol=false, WebGL visibility/resize clearTextureAtlas, tmux persistence, binary protocol), and kept the pragmatism of Report 2 (reuse existing SSH mesh, FastAPI/Next.js stack, no unnecessary new services).

They did **not** cut corners on the things that actually matter for user experience:
- Glyph artifact fix via `customGlyphs + WebGL lifecycle` is textbook correct (xterm.js issues confirm this is the exact workaround).
- `convertEol: false` + real PTY is the *right* call (my earlier snippet in Report 2 was slightly wrong; they caught it).
- tmux conditional wrapper is exactly what Vast.ai does and what Report 1 demanded.
- Binary WS, reconnect backoff, keyboard interceptors, focus management, rate-limit bump, container fallback logic — all high-quality, production-ready details.

The repo confirms their approach is realistic: current terminal is basically "SSH over Headscale + some output streaming" with no xterm.js or WebSockets yet. Sticking with SSH subprocess (instead of forcing docker-py over TCP:2375) is the *pragmatic* choice for v1 — it avoids exposing the Docker socket, requires zero infra changes, and works with your private mesh. The quoting fix they added is necessary and smart.

### Where It Falls Short (The Fair Criticisms)
It is *not* a "complete rewrite" in the full architectural-blueprint sense of Report 1. It's more of a "targeted, high-impact upgrade to the existing terminal pipeline." That's fine for speed, but it leaves a few gaps that prevent it from being *best-in-class* (Runpod/Vast.ai/Novita parity + future-proof):

1. **Missing @xterm/addon-fit** — This is a glaring omission. Report 1 and every professional implementation call it non-negotiable for dynamic resizing. Without it you get static grids or manual hacks.

2. **Resize handling is incomplete** — They mention it indirectly but never explicitly say "forward resize events → SIGWINCH or exec_resize". tmux needs this too.

3. **"Out of scope" section is too absolute** — You are right to flag this. gVisor/Kata isolation, auth polish, container readiness checks, and monitoring *are* relevant to a "complete" terminal rewrite even if they already exist elsewhere. Putting them fully out of scope feels dismissive for the scope you asked for ("best outcome possible").

4. **Minor polish misses**:
   - No explicit "container starting / retry" UX (the #1 user frustration right now).
   - No dedicated `terminal_service.py` — dumping everything into `instances.py` will make the file even bigger.
   - No mention of including tmux in base images (makes the fallback cleaner).
   - Timeline is unrealistically optimistic ("30 min for packages" is fantasy).

5. **Decisions section** — The SSH choice is defensible *for now*, but long-term docker-py (or docker exec over SSH with proper PTY) is cleaner. They acknowledge this implicitly.

**Bottom line**: This plan is excellent *pragmatic engineering*. It will fix 95% of your current pain immediately and get you to competitor parity fast. It is *not* lazy — it just prioritizes shipping over perfection. With 4-5 targeted refinements it becomes a 10/10 "complete rewrite" plan.

### How I Would Change Their Final Plan (My Refined Version)
Here is the **unified, improved plan** I recommend you adopt. I kept 90% of their structure and decisions (because they are good) but upgraded it to be truly comprehensive, future-proof, and aligned with both reports.

**TL;DR (Updated)**: 5-phase rewrite. Fixes every bug, adds tmux persistence + full xterm.js addons (including Fit), binary WS, container fallback, and production hardening. Keeps SSH mesh for v1 (smart), but adds migration note. Total realistic effort: 7-10 days.

#### Phase 1 — Frontend Dependencies & Core Setup (1 day)
- Add to `frontend/package.json`:
  ```json
  "@xterm/xterm": "^5.5.0",
  "@xterm/addon-fit": "^0.10.0",          // ← they missed this — critical
  "@xterm/addon-webgl": "^0.18.0",
  "@xterm/addon-search": "^0.15.0",
  "@xterm/addon-unicode11": "^0.8.0"
  ```
- Create / rewrite `frontend/src/components/WebTerminal.tsx` exactly as they described **plus**:
  - Load `FitAddon` and call `fitAddon.fit()` on mount + ResizeObserver + window resize.
  - `webglAddon.clearTextureAtlas()` on `visibilitychange` + resize (as they said).
  - `term.unicode.activeVersion = '11'`.
  - `customGlyphs: true`, `convertEOL: false`, `macOptionIsMeta: true`, `cursorBlink: true`.
  - Full keyboard handler + focus management exactly as written.

#### Phase 2 — Backend Rewrite: Terminal Service (2-3 days)
- **Extract** the terminal logic into new `backend/terminal_service.py` (do **not** bloat `instances.py`).
- Implement their container resolution priority list + pre-flight `docker inspect` check.
- Switch to **binary output** (`send_bytes`) + JSON control messages.
- Wrap shell in tmux (`tmux new-session -A -s term-${instance_id} ...`).
- **Add explicit resize handling**: On resize message from frontend → send `SIGWINCH` or `tmux send-keys` equivalent (or fallback to docker exec_resize via SSH).
- Rate limit → 512 KB/s, session timeout → 4 hours + idle warning.
- On disconnect: send tmux detach sequence (`\x02d`) instead of killing process.
- Container readiness polling (retry every 2s up to 30s with "Container still starting..." message).

#### Phase 3 — Integration & Polish (2 days)
- Embed in `frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx`.
- Add loading states: "Connecting...", "Container starting...", "Reattaching to session...".
- Search panel, reconnect UI, toolbar exactly as they planned.
- Make tmux **mandatory** in all base GPU images (documented in wizard).

#### Phase 4 — Testing & Verification (1-2 days)
Use their verification list + add:
- Resize test (vim/htop redraw correctly).
- tmux persistence + browser close/reopen.
- Heavy output (no encoding artifacts).
- Container-not-found fallback.
- FitAddon + WebGL on tab sleep/wake/resize.

#### Phase 5 — Production Hardening (1 day — this was their "out of scope")
- JWT/auth on WS (already mostly there — just enforce).
- Rate-limit + payload validation.
- Container isolation reminder (gVisor/Kata already in `security.py` — add terminal-specific audit log).
- Prometheus metrics for active terminals + errors.
- Feature flag rollout.
- Documentation + "how to use tmux" tooltip in UI.

**Decisions (Updated)**:
- SSH subprocess kept for v1 → Yes, correct. No need to expose Docker TCP. Future migration path to docker-py over SSH is noted in code comments.
- convertEol: false → Correct (PTY + tmux handles it).
- tmux conditional with fallback → Perfect.
- Nothing is truly "out of scope" — we just sequenced it.

**Relevant Files (Same as theirs + additions)**:
- `frontend/src/components/WebTerminal.tsx` (full rewrite)
- `backend/terminal_service.py` (new)
- `backend/instances.py` (only calls the new service)
- `frontend/package.json`
- Base Docker images / wizard (add tmux)

This refined plan gives you **the best outcome possible**: immediate bug fixes + native-like experience + session persistence + scalability, while staying realistic with your existing Headscale/SSH architecture.

It is now a true "complete rewrite" without over-engineering. Implement this and your terminal will be noticeably better than Runpod/Vast.ai/Novita in interactivity and reliability.

If you want me to write the full `WebTerminal.tsx` or `terminal_service.py` code right now, just say the word. This is the one to ship. Let's lock it in. 🚀

Final Plan:


## Plan: Web Terminal Complete Rewrite (v2 — Final)

**TL;DR**: 6-phase rewrite. Root causes: stale container name in JSONB + missing xterm.js config/addons. Full rewrite of WebTerminal.tsx, extraction of backend terminal logic to a dedicated `routes/terminal.py`, binary output protocol, tmux session persistence, container readiness polling, Ctrl+F inline search, Ctrl+R native history, brand-color state badges, and Prometheus metrics. Nothing is "out of scope." Realistic effort: 7–10 days.

---

### Phase 1 — Frontend Packages *(~30 min)*

Add to package.json (FitAddon already installed at `^0.11.0` — do not duplicate):

- `@xterm/addon-webgl` — GPU-accelerated renderer, fixes texture atlas artifact
- `@xterm/addon-search` — in-terminal text search
- `@xterm/addon-unicode11` — correct CJK/emoji cell-width calculation

---

### Phase 2 — Backend: Extract to `routes/terminal.py` *(parallel with Phase 3)*

Extract the entire `ws_terminal()` function and its constants/helpers out of instances.py into a new `routes/terminal.py`. Register the new router in __init__.py.

**2.1 — Container resolution (no legacy fallbacks)**
This is a rewrite. No stale `xcelsior-` fallback chain. Canonical name is `xcl-{instance_id}`:
```
container_ref = instance.get("container_name") or f"xcl-{instance_id}"
```

**2.2 — Container readiness polling**
Before spawning the PTY, preflight with `docker inspect --format='' {ref}` via SSH. If not found: send `{"type": "status", "message": "Container starting...", "retry": true}` to the client, retry every 2s, up to 15 attempts (30s). After 30s with no container: send `{"type": "error", "message": "Container did not start within 30s", "code": 4410}` and close. No more silent `No such container` errors.

**2.3 — SSH command quoting fix**
Construct the remote docker exec portion as a pre-quoted single string via `shlex.join([...])` passed as one argument to SSH — prevents `TERM` env var and container ref from being mangled by the remote shell interpreter.

**2.4 — tmux session wrapping**
Wrap the shell invocation:
```
docker exec -e TERM=xterm-256color -it {ref} tmux new-session -A -s xcl-{instance_id} {shell}
```
`-A` attaches to the existing tmux session if present, creates new if not. Training scripts survive browser close/refresh. Fallback: buffer first 200ms of PTY output; if it contains `tmux: command not found`, kill and re-exec with direct shell. On WebSocket disconnect: write `\x02d` (Ctrl+B D) to PTY stdin to detach the session gracefully instead of killing the process.

**2.5 — Binary output protocol**
In `_stdout_relay()`: `await websocket.send_bytes(chunk)` — raw PTY bytes, zero encode/decode overhead, accurate binary escape sequences. Control frames (`error`, `exit`, `status`, `pong`) remain JSON text.

**2.6 — Rate limit + session timeout**
- `_TERMINAL_RATE_LIMIT_BYTES`: `10_240` → `524_288` (512 KB/s)
- `_TERMINAL_SESSION_TIMEOUT`: `1_800` → `14_400` (4 hours)
- Idle detection: at 3h55m with no input, send `{"type": "status", "message": "Session closing in 5 min due to inactivity"}` into the terminal text; close at 4h.

**2.7 — Explicit resize → SIGWINCH**
Resize message from client → `fcntl.ioctl(master_fd, termios.TIOCSWINSZ, ...)`. Already partially present in current code — verify it fires correctly. tmux propagates SIGWINCH to all panes automatically when the PTY dimensions change; no extra step needed.

**2.8 — Security boundary audit**
`docker exec` on a gVisor/Kata-sandboxed container stays inside the sandbox — no security.py changes needed. Add audit log entry per session open/close: `log.info("terminal.session.open user=%s instance=%s", user_id, instance_id)`.

**2.9 — Prometheus metrics**
```python
_active_terminal_sessions = Gauge("xcelsior_terminal_sessions_active", "Active terminal sessions")
_terminal_bytes_sent = Counter("xcelsior_terminal_bytes_sent_total", "Bytes sent to clients")
_terminal_errors = Counter("xcelsior_terminal_errors_total", "Terminal connection errors", ["reason"])
```
Increment `_active_terminal_sessions` on accept, decrement in `finally`. Counters in relay loops. Prometheus scrape endpoint already exists in the app.

---

### Phase 3 — Frontend Rewrite: WebTerminal.tsx *(parallel with Phase 2)*

**3.1 — Terminal configuration**

| Option | Value | Why |
|---|---|---|
| `customGlyphs` | `true` | Draws box/block chars via canvas — eliminates the blue-rectangle artifact |
| `convertEol` | `false` | Backend PTY handles CRLF natively; `true` breaks vim/htop cursor positioning |
| `macOptionIsMeta` | `true` | Option key as Meta on macOS for word-jump etc. |
| `cursorBlink` | `true` | Visual feedback |
| `scrollback` | `10000` | Up from 5000 |

**3.2 — WebGL addon with lifecycle management**
Dynamic import inside `try/catch`, fall back to Canvas renderer silently on context failure. Hook:
- `document.addEventListener('visibilitychange', ...)` → `webglAddon?.clearTextureAtlas()` when tab becomes visible — clears GPU corruption after sleep/wake
- ResizeObserver callback → `webglAddon?.clearTextureAtlas()` — clears artifacts on resize
- `webglAddon.dispose()` before `term.dispose()` on unmount

**3.3 — Unicode11 addon**
Load `Unicode11Addon`, call `term.unicode.activeVersion = '11'` immediately after.

**3.4 — Binary + mixed protocol**
`socket.binaryType = 'arraybuffer'`. In `onmessage`: `event.data instanceof ArrayBuffer` → `term.write(new Uint8Array(event.data))`. Else parse JSON for control frames (`error`, `exit`, `status`, `pong`). Handle `{type: "status", retry: true}` by showing "Container starting..." in the badge. Client still sends text JSON for input/resize/ping.

**3.5 — Exponential backoff reconnect**
Refs: `reconnectAttempts`, `reconnectTimer`, `userClosedRef`. `ws.onclose` → if not user-closed and `attempts < 8`: schedule `connect()` after `Math.min(1000 * 2^n, 30_000)` ms. On success: reset count. On final failure: write `[Could not reconnect. Click Reconnect to try again.]` to terminal. Manual Reconnect button resets count and calls `connect()` immediately.

**3.6 — Keyboard interceptors**
`term.attachCustomKeyEventHandler((e: KeyboardEvent) => { ... })`:

| Key | Behavior |
|---|---|
| **Ctrl+C** (no selection) | Pass through → sends `\x03` interrupt to shell |
| **Ctrl+C** (with selection) | `navigator.clipboard.writeText(term.getSelection())` → return false (no interrupt) |
| **Ctrl+Shift+V** / paste | `clipboard.readText()` → `ws.send({type:"input",...})` |
| **Ctrl+F** | Open inline search panel; `return false` (prevent browser find) |
| **Ctrl+R** | Send `\x12` to shell → triggers bash/zsh native `reverse-i-search`; `return false` |
| All others | `return true` (pass through to xterm) |

Ctrl+R sends the raw ANSI reverse-search signal — the shell handles history natively. No clientside history reimplementation.

**3.7 — Inline search panel (Ctrl+F — no toolbar button)**
Positioned as an absolute overlay in the bottom-right of the terminal container, above the scrollbar. Contains: autofocused text `<input>`, Enter/Shift+Enter to cycle next/previous, case-sensitive toggle, regex toggle, match count display, Escape to close. Wired to `searchAddon.findNext(query, opts)` / `findPrevious(query, opts)`. Escape closes and returns focus to terminal.

**3.8 — Focus management**
Three-point focus enforcement that kills the "click advances a line" bug:
1. `term.focus()` immediately after `term.open(containerRef.current)` on mount
2. `term.focus()` inside `ws.onopen`
3. `onClick` on the terminal container `<div>` → `term.focus()`

**3.9 — Status badges with brand colors**
Use existing `@/components/ui/badge` variants — no new colors:

| State | Badge variant | Extra |
|---|---|---|
| `connected` | `variant="active"` | Wifi icon (emerald) |
| `connecting` / `reconnecting` | `variant="warning"` | `<Loader2 className="animate-spin text-sky-400" />` — matches `#38bdf8` cursor color |
| `container starting` | `variant="warning"` | spinner + "Starting..." text |
| `error` | `variant="failed"` | WifiOff icon |
| `disconnected` | default/muted | WifiOff icon |

**3.10 — Cleanup**
On unmount: `userClosedRef.current = true`, clear `reconnectTimer`, `ws.close()`, `webglAddon?.dispose()`, `term.dispose()`.

---

### Phase 4 — Instance Detail Page *(depends on Phase 3, minor)*

In [`frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx`](frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx):
- Expand terminal panel height from `h-[400px]` to `h-[500px]`
- Terminal panel shown only when `status === 'running'` (tmux is meaningless on a stopped container)
- Existing `dynamic()` import with `ssr: false` is already correct — no change needed

---

### Phase 5 — Base Images: tmux availability *(parallel with Phases 2–4)*

Add `RUN apt-get install -y tmux` (or equivalent) to Dockerfile and any GPU image templates referenced by the wizard. When tmux is present in every container the fallback code in Phase 2.4 rarely if ever fires. Add a brief onboarding note in the wizard/dashboard: "Your terminal session persists — reconnecting reattaches your previous session."

---

### Phase 6 — Testing *(depends on Phases 2–4)*

1. **Glyph test**: Nerd Font `echo $PS1` → no blue artifact after tab switch + resize
2. **Focus test**: click terminal → no line advance; prompt is immediately visible
3. **Binary protocol**: `cat /dev/urandom | head -c 1M | xxd` → no encoding artifacts, sustained throughput >100 KB/s
4. **Container readiness polling**: hit terminal before container is running → see "Container starting..." → auto-connects when ready
5. **Ctrl+F**: opens bottom-right search overlay; Enter/Shift+Enter cycles; Escape closes and refocuses terminal
6. **Ctrl+R**: triggers bash/zsh native `reverse-i-search` without any custom history implementation
7. **Ctrl+C (no selection)**: sends `^C` interrupt to active command
8. **Ctrl+C (with selection)**: copies text; does NOT send interrupt
9. **Reconnect backoff**: kill nginx WS proxy mid-session → badge shows "Reconnecting (N/8)" with spinner → re-enable → session resumes automatically
10. **tmux persistence**: `sleep 3600` in terminal → close browser → reopen instance page → sleep still running in reattached session
11. **Rate limit**: `tail -f` on heavy training log → no throttle stutter at 512 KB/s
12. **Resize**: open `vim` → resize browser window → vim redraws correctly (SIGWINCH propagation)
13. **Idle warning**: fast-forward session clock → confirm 3h55m warning appears as terminal text
14. **Prometheus**: `xcelsior_terminal_sessions_active` increments on connect, decrements on close; bytes counter increments

---

### Relevant Files

- package.json — add webgl, search, unicode11
- WebTerminal.tsx — full rewrite
- [`frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx`](frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx) — height + status condition
- `routes/terminal.py` — **NEW**: extracted ws handler + all helpers + Prometheus
- instances.py — remove ws_terminal block (~L971–1241)
- __init__.py — register terminal router
- Dockerfile — add tmux

**No changes needed**: xcelsior.conf, db.py, scheduler.py, docker-compose.yml, security.py, auth system (`_validate_ws_auth` already handles query param tokens correctly).

---

### Decisions

- **SSH subprocess kept** — hosts don't expose Docker TCP. docker-py `ssh://` transport (paramiko) is a future migration option noted in code comments; adds no benefit for v1.
- **No legacy container name fallbacks** — this is a rewrite; `xcl-{instance_id}` is canonical. Stale JSONB values are a DB hygiene issue.
- **`convertEol: false`** — correct; the backend PTY handles CRLF. Setting `true` would break vim/htop.
- **Ctrl+R → `\x12` to shell** — do not reimplement history clientside; bash/zsh native reverse-i-search is superior.
- **Auth already correct** — `_validate_ws_auth()` reads both cookie and query param. No auth system changes.
- **gVisor/Kata already enforced** — `docker exec` on a sandboxed container stays inside the sandbox; terminal adds audit logging only.
- **Nothing is "out of scope"** — Prometheus metrics, security audit, tmux in images, container readiness UX are all included above.