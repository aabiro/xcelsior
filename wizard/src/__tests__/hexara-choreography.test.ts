import { describe, it, expect } from "vitest";
import {
    computeWizardMood,
    computeWizardBranch,
    stepEntryBranch,
    checkSuccessBranch,
    type HexaraContext,
} from "../hexara-choreography.js";
import type { WizardStep } from "../wizard-flow.js";

function baseCtx(overrides: Partial<HexaraContext> = {}): HexaraContext {
    const step: WizardStep = { id: "mode", type: "select", prompt: "pick" };
    return {
        exiting: false,
        isComplete: false,
        wizardState: "idle",
        step,
        mode: "rent",
        transitioning: false,
        aiStreaming: false,
        showAiPrompt: false,
        aiResponseOpen: false,
        checkResults: {},
        browseError: null,
        deviceAuthStatus: "waiting",
        gateOpen: false,
        stepPulseKey: "mode",
        ...overrides,
    };
}

describe("computeWizardMood", () => {
    it("maps thinking to working", () => {
        expect(computeWizardMood(baseCtx({ wizardState: "thinking" }))).toBe("working");
    });

    it("maps auto-check in flight to working", () => {
        const step: WizardStep = { id: "docker-check", type: "auto-check", prompt: "x", checkId: "docker" };
        expect(computeWizardMood(baseCtx({ step, wizardState: "idle" }))).toBe("working");
    });
});

describe("computeWizardBranch", () => {
    it("dances on mode select when idle", () => {
        expect(computeWizardBranch(baseCtx())).toBe("dance");
    });

    it("peeks while browsing GPUs", () => {
        const step: WizardStep = { id: "browse-gpus", type: "auto-fetch", prompt: "browse" };
        expect(computeWizardBranch(baseCtx({ step, wizardState: "idle" }))).toBe("peek");
    });

    it("types while thinking on sdk-credentials", () => {
        const step: WizardStep = { id: "sdk-credentials", type: "auto-check", prompt: "x", checkId: "sdk-credentials" };
        expect(computeWizardBranch(baseCtx({
            step, mode: "sdk", wizardState: "thinking",
        }))).toBe("type");
    });

    it("nods when AI panel is open", () => {
        expect(computeWizardBranch(baseCtx({ aiResponseOpen: true, wizardState: "idle" }))).toBe("nod");
    });

    it("sdk snippet celebrates", () => {
        const step: WizardStep = { id: "sdk-snippet", type: "confirm", prompt: "ready" };
        expect(computeWizardBranch(baseCtx({ step, mode: "sdk", wizardState: "idle" }))).toBe("celebrate");
    });
});

describe("stepEntryBranch", () => {
    it("sdk track uses peek and type", () => {
        expect(stepEntryBranch("sdk-detect", "sdk")).toBe("peek");
        expect(stepEntryBranch("sdk-credentials", "sdk")).toBe("type");
    });

    it("skips provider steps in rent mode", () => {
        expect(stepEntryBranch("benchmark", "rent")).toBeNull();
        expect(stepEntryBranch("browse-gpus", "rent")).toBe("cast");
    });

    it("provider benchmark casts on entry", () => {
        expect(stepEntryBranch("benchmark", "provide")).toBe("cast");
    });
});

describe("checkSuccessBranch", () => {
    it("maps sdk milestones", () => {
        expect(checkSuccessBranch("sdk-install")).toBe("nod");
        expect(checkSuccessBranch("sdk-verify")).toBe("eureka");
    });

    it("defaults unknown checks to eureka", () => {
        expect(checkSuccessBranch("unknown")).toBe("eureka");
    });
});