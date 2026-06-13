// Test that AutoCheckStep surfaces STATIC_STEP_HELP on failure when AI is down.

import React from "react";
import { describe, it, expect } from "vitest";
import { render } from "ink-testing-library";
import { AutoCheckStep } from "../steps.js";
import type { AutoCheckResults } from "../useWizardFlow.js";

const failed: AutoCheckResults = {
    items: [{ name: "nvidia_driver", ok: false, detail: "not found — needs ≥550.0" }],
    allPassed: false,
};

describe("AutoCheckStep static help (Part A)", () => {
    it("shows static help on failure when provided", () => {
        const { lastFrame } = render(
            <AutoCheckStep
                results={failed}
                canRetry
                staticHelp={["Upgrade NVIDIA driver, Container Toolkit, Docker, and runc to minimum versions."]}
            />,
        );
        const out = lastFrame() ?? "";
        expect(out).toContain("Hexara is offline");
        expect(out).toContain("Upgrade NVIDIA driver");
    });

    it("does not show static help when none is provided (AI available)", () => {
        const { lastFrame } = render(<AutoCheckStep results={failed} canRetry />);
        expect(lastFrame()).not.toContain("Hexara is offline");
    });

    it("does not show static help on success", () => {
        const passed: AutoCheckResults = { items: [{ name: "x", ok: true, detail: "ok" }], allPassed: true };
        const { lastFrame } = render(
            <AutoCheckStep results={passed} awaitContinue staticHelp={["irrelevant"]} successTitle="Done!" />,
        );
        expect(lastFrame()).not.toContain("Hexara is offline");
    });
});
