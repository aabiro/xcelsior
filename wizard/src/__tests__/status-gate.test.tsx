// Render smoke tests for StatusGate — verifies each phase renders without crashing.

import React from "react";
import { describe, it, expect } from "vitest";
import { render } from "ink-testing-library";
import { StatusGate } from "../StatusGate.js";
import type { StatusReport } from "../preflight.js";

const noop = () => {};

const degraded: StatusReport = {
    verdict: "degraded",
    fallback: false,
    services: [
        { name: "API", state: "operational", detail: "reachable", required: true },
        { name: "AI (Hexara)", state: "degraded", detail: "no provider key", required: false },
    ],
};

const blocked: StatusReport = {
    verdict: "blocked",
    fallback: true,
    services: [
        { name: "API", state: "operational", detail: "reachable", required: true },
        { name: "Database", state: "down", detail: "unreachable", required: true },
    ],
};

describe("StatusGate", () => {
    it("checking phase shows a spinner label", () => {
        const { lastFrame } = render(
            <StatusGate phase="checking" report={null} statusUrl="https://x/dashboard"
                onProceed={noop} onRecheck={noop} onContinueAnyway={noop} />,
        );
        expect(lastFrame()).toContain("Checking");
    });

    it("ready phase lists services and the Enter affordance", () => {
        const { lastFrame } = render(
            <StatusGate phase="ready" report={degraded} statusUrl="https://x/dashboard"
                onProceed={noop} onRecheck={noop} onContinueAnyway={noop} />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("AI (Hexara)");
        expect(out).toContain("degraded");
        expect(out).toContain("Enter");
    });

    it("blocked phase shows the status link and continue-anyway", () => {
        const { lastFrame } = render(
            <StatusGate phase="blocked" report={blocked} statusUrl="https://x/dashboard"
                onProceed={noop} onRecheck={noop} onContinueAnyway={noop} />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Database");
        expect(out).toContain("https://x/dashboard");
        expect(out).toContain("continue anyway");
        expect(out).toContain("basic probe"); // fallback marker
    });
});
