// capability.ts — Terminal capability detection for the Hexara sprite.
//
// Two layers:
//   1. heuristicCapable() — synchronous, conservative (TTY / TERM / opt-out).
//   2. detectSixelSupport() — an async DA1 round-trip that actually asks the
//      terminal whether it speaks Sixel, with a short timeout and a safe
//      fallback to the heuristic. Run once at startup (before Ink takes over
//      stdin); the resolved value is cached and read synchronously thereafter
//      via spriteCapable().

export interface TtyLike {
    isTTY?: boolean;
    isRaw?: boolean;
    setRawMode?: (mode: boolean) => void;
    resume?: () => void;
    pause?: () => void;
    on?: (event: "data", cb: (d: Buffer) => void) => void;
    removeListener?: (event: "data", cb: (d: Buffer) => void) => void;
    write?: (s: string) => boolean;
}

/**
 * Conservative synchronous capability: paint by default, but skip on non-TTY
 * output, TERM=dumb, or an explicit opt-out — the cases where Sixel writes
 * would emit garbage. Pure; exported for tests.
 */
export function heuristicCapable(
    env: NodeJS.ProcessEnv = process.env,
    stdout: { isTTY?: boolean } = process.stdout,
): boolean {
    if (env.XCELSIOR_NO_SPRITE === "1") return false;
    if ((env.TERM ?? "").toLowerCase() === "dumb") return false;
    if (stdout && stdout.isTTY === false) return false;
    return true;
}

/**
 * Parse a Primary Device Attributes (DA1) response for Sixel support.
 * A DA1 reply looks like ESC [ ? 62 ; 4 ; 6 c — feature code 4 means Sixel.
 * Pure; exported for tests.
 */
export function parseDa1Sixel(response: string): boolean {
    // eslint-disable-next-line no-control-regex
    const m = response.match(/\x1b\[\?([0-9;]+)c/);
    if (!m) return false;
    return m[1].split(";").includes("4");
}

/**
 * Ask the terminal (via DA1) whether it supports Sixel. Resolves to the
 * heuristic when there's no usable TTY, raw mode is unavailable, or no DA1
 * reply arrives within `timeoutMs`. Never throws; always restores stdin state.
 */
export function detectSixelSupport(opts: {
    stdin?: TtyLike;
    stdout?: TtyLike;
    env?: NodeJS.ProcessEnv;
    timeoutMs?: number;
} = {}): Promise<boolean> {
    const stdin = opts.stdin ?? (process.stdin as unknown as TtyLike);
    const stdout = opts.stdout ?? (process.stdout as unknown as TtyLike);
    const env = opts.env ?? process.env;
    const timeoutMs = opts.timeoutMs ?? 200;
    const fallback = heuristicCapable(env, stdout as { isTTY?: boolean });

    // Honor the explicit opt-out / non-TTY without touching the terminal.
    if (env.XCELSIOR_NO_SPRITE === "1") return Promise.resolve(false);
    if (!stdin?.isTTY || !stdout?.isTTY || typeof stdin.setRawMode !== "function" || typeof stdin.on !== "function") {
        return Promise.resolve(fallback);
    }

    return new Promise<boolean>((resolve) => {
        let done = false;
        const wasRaw = stdin.isRaw === true;
        let buf = "";

        const onData = (d: Buffer) => {
            buf += d.toString("latin1");
            // eslint-disable-next-line no-control-regex
            const m = buf.match(/\x1b\[\?[0-9;]+c/);
            if (m) finish(parseDa1Sixel(m[0]));
        };

        const finish = (result: boolean) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            try { stdin.removeListener?.("data", onData); } catch { /* ignore */ }
            try { stdin.setRawMode?.(wasRaw); } catch { /* ignore */ }
            resolve(result);
        };

        const timer = setTimeout(() => finish(fallback), timeoutMs);

        try {
            stdin.setRawMode!(true);
            stdin.resume?.();
            stdin.on!("data", onData);
            stdout.write?.("\x1b[c"); // DA1 query
        } catch {
            finish(fallback);
        }
    });
}

// ── Cached accessor used by the sprite renderer ──────────────────────

let _cached: boolean | null = null;

/** Run detection once and cache the result. Safe to call multiple times. */
export async function initSpriteCapability(opts?: Parameters<typeof detectSixelSupport>[0]): Promise<boolean> {
    _cached = await detectSixelSupport(opts);
    return _cached;
}

/** Synchronous capability for render paths — cached detection, else heuristic. */
export function spriteCapable(): boolean {
    return _cached ?? heuristicCapable();
}

/** Test helper: reset the cache. */
export function __resetSpriteCapability(): void {
    _cached = null;
}
