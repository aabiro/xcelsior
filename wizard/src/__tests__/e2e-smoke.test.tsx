// End-to-end smoke (Part H): render the full App against the api-client/checks
// mocks + a stubbed /api/status, and verify it progresses across gate verdicts
// (operational / degraded / blocked) and mode selection (rent / provide / both).
// This proves the wired flow works, not just the unit pieces.

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render } from "ink-testing-library";
import { App } from "../index.js";
import { clearWizardCheckpoint } from "../wizard-state.js";

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const DOWN = "[B";

function fakeResponse(body: unknown, status = 200) {
    return {
        ok: status >= 200 && status < 300,
        status,
        json: async () => body,
        text: async () => JSON.stringify(body),
    } as unknown as Response;
}

function installFetch(statusBody: unknown) {
    const fn = vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.includes("/api/status")) return fakeResponse(statusBody);
        return fakeResponse({ ok: true }); // /healthz and friends
    });
    vi.stubGlobal("fetch", fn);
    return fn;
}

const svc = (name: string, state: string, required = false) => ({ name, state, detail: "", required });
const OPERATIONAL = { ok: true, verdict: "operational", services: [svc("API", "operational", true), svc("Database", "operational", true)] };
const DEGRADED = { ok: true, verdict: "degraded", services: [svc("API", "operational", true), svc("AI (Hexara)", "degraded", false)] };
const BLOCKED = { ok: false, verdict: "blocked", services: [svc("API", "operational", true), svc("Database", "down", true)] };

beforeEach(() => {
    process.env.XCELSIOR_NO_SPRITE = "1";
    process.env.XCELSIOR_NO_BROWSER = "1";
    clearWizardCheckpoint(); // deterministic fresh start
});
afterEach(() => {
    vi.unstubAllGlobals();
});

describe("e2e — gate verdicts", () => {
    it("operational falls straight into mode select (no gate panel)", async () => {
        installFetch(OPERATIONAL);
        const { lastFrame } = render(<App />);
        await delay(80);
        const out = lastFrame() ?? "";
        expect(out).toContain("Rent GPUs");
        expect(out).toContain("Provide GPUs");
        expect(out).not.toContain("degraded mode");
    });

    it("degraded shows the gate, then Enter proceeds to mode select", async () => {
        installFetch(DEGRADED);
        const { lastFrame, stdin } = render(<App />);
        await delay(80);
        expect(lastFrame()).toContain("degraded");
        expect(lastFrame()).toContain("AI (Hexara)");
        stdin.write("\r"); // proceed
        await delay(40);
        expect(lastFrame()).toContain("Rent GPUs");
    });

    it("blocked shows continue-anyway, then c proceeds to mode select", async () => {
        installFetch(BLOCKED);
        const { lastFrame, stdin } = render(<App />);
        await delay(80);
        const out = lastFrame() ?? "";
        expect(out).toContain("continue anyway");
        expect(out).toContain("Database");
        stdin.write("c");
        await delay(40);
        expect(lastFrame()).toContain("Rent GPUs");
    });
});

describe("e2e — mode selection (rent / provide / both)", () => {
    async function startAtModeSelect() {
        installFetch(OPERATIONAL);
        const h = render(<App />);
        await delay(80);
        expect(h.lastFrame()).toContain("Rent GPUs");
        return h;
    }

    it("selecting Rent transitions", async () => {
        const { lastFrame, stdin } = await startAtModeSelect();
        stdin.write("\r"); // first option = rent
        await delay(40);
        expect(lastFrame()).toContain("Great choice");
    });

    it("selecting Provide transitions", async () => {
        const { lastFrame, stdin } = await startAtModeSelect();
        stdin.write(DOWN); // → provide
        await delay(20);
        stdin.write("\r");
        await delay(40);
        expect(lastFrame()).toContain("Great choice");
    });

    it("selecting Both transitions", async () => {
        const { lastFrame, stdin } = await startAtModeSelect();
        stdin.write(DOWN);
        await delay(20);
        stdin.write(DOWN); // → both
        await delay(20);
        stdin.write("\r");
        await delay(40);
        expect(lastFrame()).toContain("Great choice");
    });
});

describe("e2e — title bar & footer chrome render", () => {
    it("shows the persistent title bar and a contextual footer", async () => {
        installFetch(OPERATIONAL);
        const { lastFrame } = render(<App />);
        await delay(80);
        const out = lastFrame() ?? "";
        expect(out).toContain("Xcelsior Setup");
        expect(out).toContain("navigate"); // footer hint for the select step
    });
});
