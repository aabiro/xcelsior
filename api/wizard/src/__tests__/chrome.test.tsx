// Tests for chrome.tsx — title bar render + context-sensitive footer hints.

import React from "react";
import { describe, it, expect } from "vitest";
import { render } from "ink-testing-library";
import { TitleBar, KeybindFooter, footerHints, WIZARD_VERSION, type FooterContext } from "../chrome.js";

const base: FooterContext = {
    gatePhase: "passed",
    stepType: "select",
    isComplete: false,
    aiAvailable: true,
    isWorkStep: false,
    canRetry: false,
    awaitContinue: false,
    showAiPrompt: false,
    aiResponseOpen: false,
};

const keys = (ctx: FooterContext) => footerHints(ctx).map((h) => h.key);

describe("footerHints", () => {
    it("select step shows navigate/select + ask Hexara + quit", () => {
        expect(keys(base)).toEqual(["↑↓", "enter", "?", "q"]);
    });

    it("omits Ask Hexara when AI is unavailable", () => {
        expect(keys({ ...base, aiAvailable: false })).toEqual(["↑↓", "enter", "q"]);
    });

    it("confirm step shows y/n", () => {
        expect(keys({ ...base, stepType: "confirm" })).toContain("y");
        expect(keys({ ...base, stepType: "confirm" })).toContain("n");
    });

    it("device-auth shows manual paste", () => {
        expect(keys({ ...base, stepType: "device-auth" })).toContain("m");
    });

    it("work step with retry shows tab switch + retry/skip", () => {
        const k = keys({ ...base, stepType: "auto-check", isWorkStep: true, canRetry: true });
        expect(k).toContain("Tab/←→");
        expect(k).toContain("enter");
        expect(k).toContain("s");
    });

    it("work step awaiting continue shows continue", () => {
        const k = keys({ ...base, stepType: "auto-check", isWorkStep: true, awaitContinue: true });
        expect(k).toContain("Tab/←→");
        expect(k).toContain("enter");
        expect(k).not.toContain("s");
    });

    it("blocked gate shows recheck + continue anyway", () => {
        const k = keys({ ...base, gatePhase: "blocked" });
        expect(k).toEqual(["r", "c", "q"]);
    });

    it("ready gate shows continue + recheck", () => {
        expect(keys({ ...base, gatePhase: "ready" })).toEqual(["enter", "r", "q"]);
    });

    it("done screen shows navigate/select/exit", () => {
        expect(keys({ ...base, isComplete: true })).toEqual(["↑↓", "enter", "q"]);
    });

    it("open AI prompt takes over the footer", () => {
        expect(keys({ ...base, showAiPrompt: true })).toEqual(["type", "esc"]);
    });

    it("open AI response takes over the footer", () => {
        expect(keys({ ...base, aiResponseOpen: true })).toEqual(["enter", "?"]);
    });
});

describe("TitleBar / KeybindFooter render", () => {
    it("title bar shows brand + version + ask hint", () => {
        const { lastFrame } = render(<TitleBar />);
        const out = lastFrame() ?? "";
        expect(out).toContain("Xcelsior Setup");
        expect(out).toContain(`v${WIZARD_VERSION}`);
        expect(out).toContain("Ask Hexara: type ?");
    });

    it("footer renders the contextual keys", () => {
        const { lastFrame } = render(<KeybindFooter ctx={base} />);
        const out = lastFrame() ?? "";
        expect(out).toContain("navigate");
        expect(out).toContain("ask Hexara");
        expect(out).toContain("quit");
    });
});
