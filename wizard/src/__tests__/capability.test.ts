// Tests for terminal-capability detection: sprite (Sixel) gating + truecolor.

import { describe, it, expect } from "vitest";
import { heuristicCapable, parseDa1Sixel, detectSixelSupport } from "../capability.js";
import { supportsTruecolor, gradientAt, BRAND_GRADIENT } from "../theme.js";

const tty = { isTTY: true };
const notTty = { isTTY: false };

describe("heuristicCapable", () => {
    it("defaults to true on a normal TTY", () => {
        expect(heuristicCapable({ TERM: "xterm-256color" }, tty)).toBe(true);
    });

    it("is false when explicitly opted out", () => {
        expect(heuristicCapable({ XCELSIOR_NO_SPRITE: "1", TERM: "xterm" }, tty)).toBe(false);
    });

    it("is false on a dumb terminal", () => {
        expect(heuristicCapable({ TERM: "dumb" }, tty)).toBe(false);
    });

    it("is false when stdout is not a TTY (piped/CI)", () => {
        expect(heuristicCapable({ TERM: "xterm-256color" }, notTty)).toBe(false);
    });
});

describe("parseDa1Sixel", () => {
    it("detects Sixel (feature code 4) in a DA1 reply", () => {
        expect(parseDa1Sixel("\x1b[?62;4;6;9;22c")).toBe(true);
    });
    it("returns false when code 4 is absent", () => {
        expect(parseDa1Sixel("\x1b[?62;6;9;22c")).toBe(false);
    });
    it("returns false for a non-DA1 string", () => {
        expect(parseDa1Sixel("garbage")).toBe(false);
        expect(parseDa1Sixel("")).toBe(false);
    });
});

describe("detectSixelSupport", () => {
    it("falls back to the heuristic on a non-TTY without touching the terminal", async () => {
        const ok = await detectSixelSupport({
            stdin: { isTTY: false } as any,
            stdout: { isTTY: false } as any,
            env: { TERM: "xterm-256color" },
            timeoutMs: 10,
        });
        expect(ok).toBe(false); // non-TTY stdout → heuristic false
    });

    it("honors the opt-out immediately", async () => {
        const ok = await detectSixelSupport({ env: { XCELSIOR_NO_SPRITE: "1" }, timeoutMs: 10 });
        expect(ok).toBe(false);
    });

    it("resolves true when the terminal replies with a Sixel DA1", async () => {
        const listeners: Array<(d: Buffer) => void> = [];
        const fakeStdin: any = {
            isTTY: true, isRaw: false,
            setRawMode() {}, resume() {}, pause() {},
            on(_e: string, cb: (d: Buffer) => void) { listeners.push(cb); },
            removeListener() {},
        };
        const fakeStdout: any = {
            isTTY: true,
            write() {
                // Reply asynchronously as a real terminal would.
                setTimeout(() => listeners.forEach((cb) => cb(Buffer.from("\x1b[?62;4;6c"))), 0);
                return true;
            },
        };
        const ok = await detectSixelSupport({ stdin: fakeStdin, stdout: fakeStdout, env: { TERM: "xterm" }, timeoutMs: 200 });
        expect(ok).toBe(true);
    });

    it("times out to the heuristic when no reply arrives", async () => {
        const fakeStdin: any = {
            isTTY: true, isRaw: false,
            setRawMode() {}, resume() {}, pause() {},
            on() {}, removeListener() {},
        };
        const fakeStdout: any = { isTTY: true, write() { return true; } };
        const ok = await detectSixelSupport({ stdin: fakeStdin, stdout: fakeStdout, env: { TERM: "xterm-256color" }, timeoutMs: 30 });
        expect(ok).toBe(true); // heuristic true for a 256color TTY
    });
});

describe("supportsTruecolor", () => {
    it("true for COLORTERM=truecolor", () => {
        expect(supportsTruecolor({ COLORTERM: "truecolor" })).toBe(true);
    });
    it("true for 256color TERM", () => {
        expect(supportsTruecolor({ TERM: "xterm-256color" })).toBe(true);
    });
    it("true for known terminal programs", () => {
        expect(supportsTruecolor({ TERM_PROGRAM: "iTerm.app" })).toBe(true);
        expect(supportsTruecolor({ WT_SESSION: "x" })).toBe(true);
    });
    it("false for a bare/dumb terminal", () => {
        expect(supportsTruecolor({ TERM: "dumb" })).toBe(false);
        expect(supportsTruecolor({})).toBe(false);
    });
});

describe("gradientAt", () => {
    it("clamps to the gradient endpoints", () => {
        expect(gradientAt(0)).toBe(BRAND_GRADIENT[0]);
        expect(gradientAt(1)).toBe(BRAND_GRADIENT[BRAND_GRADIENT.length - 1]);
        expect(gradientAt(-5)).toBe(BRAND_GRADIENT[0]);
        expect(gradientAt(5)).toBe(BRAND_GRADIENT[BRAND_GRADIENT.length - 1]);
    });
    it("returns the first color for non-finite input", () => {
        expect(gradientAt(NaN)).toBe(BRAND_GRADIENT[0]);
    });
    it("maps the midpoint into the middle of the gradient", () => {
        expect(BRAND_GRADIENT).toContain(gradientAt(0.5));
    });
});
