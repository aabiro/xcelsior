// Tests for terminal-capability detection: sprite gating + truecolor.

import { describe, it, expect } from "vitest";
import { heuristicCapable } from "../capability.js";
import { supportsTruecolor, gradientAt, BRAND_GRADIENT } from "../theme.js";

const tty = { isTTY: true };
const notTty = { isTTY: false };

describe("heuristicCapable", () => {
    it("draws Hexara on a 256-colour or truecolor TTY", () => {
        expect(heuristicCapable({ TERM: "xterm-256color" }, tty, 2)).toBe(true);
        expect(heuristicCapable({ TERM: "xterm-256color" }, tty, 3)).toBe(true);
    });

    it("does not need Sixel: VS Code's terminal qualifies", () => {
        expect(heuristicCapable({ TERM: "xterm-256color", TERM_PROGRAM: "vscode" }, tty, 3)).toBe(true);
    });

    it("skips 16-colour and colourless terminals, where the pixel art turns to mud", () => {
        expect(heuristicCapable({ TERM: "xterm" }, tty, 1)).toBe(false);
        expect(heuristicCapable({ TERM: "xterm" }, tty, 0)).toBe(false);
    });

    it("is false when explicitly opted out", () => {
        expect(heuristicCapable({ XCELSIOR_NO_SPRITE: "1", TERM: "xterm" }, tty, 3)).toBe(false);
    });

    it("is false on a dumb terminal", () => {
        expect(heuristicCapable({ TERM: "dumb" }, tty, 3)).toBe(false);
    });

    it("is false when stdout is not a TTY (piped/CI)", () => {
        expect(heuristicCapable({ TERM: "xterm-256color" }, notTty, 3)).toBe(false);
        expect(heuristicCapable({ TERM: "xterm-256color" }, {}, 3)).toBe(false);
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
