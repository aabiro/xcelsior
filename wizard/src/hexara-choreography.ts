// hexara-choreography.ts — Mood + branch mapping for Hexara across every wizard stream.

import type { BranchId, WizardMood } from "./useWizardAnimation.js";
import type { WizardState } from "../sprites/wizard/wizard-sprite.js";
import type { WizardStep } from "./wizard-flow.js";
import type { AutoCheckResults } from "./useWizardFlow.js";

export interface HexaraContext {
    exiting: boolean;
    isComplete: boolean;
    wizardState: WizardState;
    step: WizardStep;
    mode?: string;
    transitioning: boolean;
    aiStreaming: boolean;
    showAiPrompt: boolean;
    aiResponseOpen: boolean;
    checkResults: Record<string, AutoCheckResults>;
    browseError?: string | null;
    deviceAuthStatus: "loading" | "waiting" | "authorized" | "error" | "manual";
    gateOpen: boolean;
    stepPulseKey: string;
}

/** Step entry pulse — fires once when a step becomes active. */
const STEP_ENTRY: Record<string, BranchId> = {
    // Shared
    mode: "dance",
    "device-auth": "levitate",
    "api-check": "cast",
    done: "celebrate",

    // SDK
    "sdk-detect": "peek",
    "sdk-install": "peek",
    "sdk-credentials": "type",
    "sdk-verify": "cast",
    "sdk-snippet": "nod",

    // Renter
    workload: "wave",
    "ssh-key-setup": "type",
    "gpu-preference": "peek",
    "browse-gpus": "cast",
    "gpu-pick": "celebrate",
    "image-pick": "nod",
    "confirm-launch": "cast",
    "wallet-check": "levitate",
    "payment-gate": "sleep",
    "launch-instance": "cast",

    // Provider
    "docker-check": "cast",
    "gpu-detect": "peek",
    "version-check": "cast",
    "network-setup": "cast",
    benchmark: "cast",
    "network-bench": "cast",
    verification: "cast",
    pricing: "wave",
    "custom-rate": "type",
    "spot-enabled": "nod",
    "spot-min-cents": "type",
    "host-register": "cast",
    "admission-gate": "cast",
    "provider-summary": "wave",
    "worker-install": "type",
    "confirm-setup": "nod",
};

/** Auto-check success reactions keyed by checkId. */
const CHECK_SUCCESS: Record<string, BranchId> = {
    "sdk-detect": "nod",
    "sdk-install": "nod",
    "sdk-credentials": "type",
    "sdk-verify": "eureka",
    docker: "eureka",
    api: "eureka",
    gpu: "peek",
    versions: "nod",
    benchmark: "dance",
    network: "nod",
    verify: "eureka",
    "host-register": "celebrate",
    admission: "eureka",
    wallet: "nod",
    launch: "celebrate",
    "network-setup": "nod",
    "worker-install": "dance",
    "ssh-key-setup": "nod",
};

/** Steps where long thinking should show typing instead of generic cast. */
const TYPING_STEPS = new Set([
    "sdk-credentials", "ssh-key-setup", "worker-install", "custom-rate", "spot-min-cents",
]);

/** Steps where idle inspection should peek instead of generic wave. */
const PEEK_STEPS = new Set([
    "sdk-detect", "sdk-install", "gpu-detect", "gpu-preference", "browse-gpus",
]);

function stepApplies(stepId: string, mode?: string): boolean {
    if (!mode) return true;
    const sdkOnly = stepId.startsWith("sdk-");
    const provideOnly = [
        "docker-check", "gpu-detect", "version-check", "network-setup", "benchmark",
        "network-bench", "verification", "pricing", "custom-rate", "spot-enabled",
        "spot-min-cents", "host-register", "admission-gate", "provider-summary",
        "worker-install", "confirm-setup",
    ].includes(stepId);
    const rentOnly = [
        "workload", "ssh-key-setup", "gpu-preference", "browse-gpus", "gpu-pick",
        "image-pick", "confirm-launch", "wallet-check", "payment-gate", "launch-instance",
    ].includes(stepId);

    if (mode === "sdk") return sdkOnly || !provideOnly && !rentOnly;
    if (mode === "rent") return rentOnly || stepId === "device-auth" || stepId === "api-check" || stepId === "mode" || stepId === "done";
    if (mode === "provide") return provideOnly || stepId === "device-auth" || stepId === "api-check" || stepId === "mode" || stepId === "done";
    return true;
}

export function stepEntryBranch(stepId: string, mode?: string): BranchId | null {
    if (!stepApplies(stepId, mode)) return null;
    return STEP_ENTRY[stepId] ?? null;
}

export function checkSuccessBranch(checkId: string | undefined): BranchId | null {
    if (!checkId) return null;
    return CHECK_SUCCESS[checkId] ?? "eureka";
}

export function computeWizardMood(ctx: HexaraContext): WizardMood {
    if (ctx.exiting) return "idle";
    if (ctx.isComplete || ctx.wizardState === "finishing") return "success";
    if (ctx.wizardState === "error") return "error";
    if (ctx.wizardState === "excited") return "presenting";
    if (ctx.wizardState === "success") return "success";
    if (ctx.wizardState === "waiting") return "waiting";
    if (ctx.wizardState === "thinking" || ctx.aiStreaming) return "working";
    if (ctx.transitioning || ctx.showAiPrompt) return "presenting";
    if (ctx.gateOpen) return "waiting";

    if (ctx.step.type === "auto-check" && !ctx.checkResults[ctx.step.id]) return "working";
    if (ctx.step.type === "auto-fetch" && !ctx.browseError) return "working";
    if (ctx.step.type === "device-auth" && (ctx.deviceAuthStatus === "loading" || ctx.deviceAuthStatus === "waiting")) {
        return "waiting";
    }
    if (ctx.step.type === "payment-gate") return "waiting";
    if (ctx.aiResponseOpen) return "presenting";

    return "idle";
}

function thinkingBranch(step: WizardStep): BranchId {
    if (TYPING_STEPS.has(step.id)) return "type";
    if (PEEK_STEPS.has(step.id)) return "peek";
    if (step.checkId === "benchmark" || step.checkId === "launch") return "cast";
    return "cast";
}

export function computeWizardBranch(ctx: HexaraContext): BranchId | null {
    if (ctx.exiting) return null;

    if (ctx.wizardState === "finishing") return "bow";
    if (ctx.isComplete && ctx.wizardState === "success") return "celebrate";

    if (ctx.wizardState === "excited") {
        if (ctx.step.id === "confirm-launch" || ctx.step.checkId === "launch") return "cast";
        if (ctx.step.id === "sdk-snippet") return "nod";
        return "dance";
    }
    if (ctx.wizardState === "success") return "eureka";
    if (ctx.wizardState === "error") return "error";
    if (ctx.wizardState === "thinking") return thinkingBranch(ctx.step);
    if (ctx.wizardState === "waiting") return "sleep";

    if (ctx.wizardState === "idle") {
        if (ctx.step.type === "device-auth") {
            if (ctx.deviceAuthStatus === "waiting") return "sleep";
            if (ctx.deviceAuthStatus === "authorized") return "wave";
            if (ctx.deviceAuthStatus === "error") return "error";
        }

        if (ctx.step.type === "auto-check") {
            const result = ctx.checkResults[ctx.step.id];
            if (result?.allPassed) return checkSuccessBranch(ctx.step.checkId);
            if (result && result.items.length > 0 && !result.allPassed) return "error";
        }

        if (ctx.step.type === "auto-fetch") {
            if (ctx.browseError) return "error";
            return "peek";
        }

        if (ctx.showAiPrompt) return "peek";
        if (ctx.aiResponseOpen) return "nod";

        if (ctx.step.type === "select") {
            if (ctx.step.id === "mode") return "dance";
            if (ctx.step.id === "gpu-pick") return "celebrate";
            if (ctx.step.id === "pricing" || ctx.step.id === "spot-enabled") return "nod";
            if (PEEK_STEPS.has(ctx.step.id)) return "peek";
            return "wave";
        }

        if (ctx.step.type === "confirm") {
            if (ctx.step.id === "sdk-snippet") return "nod";
            if (ctx.step.id === "confirm-launch") return "cast";
            if (ctx.step.id === "provider-summary") return "nod";
            if (ctx.step.id === "confirm-setup") return "nod";
            return "wave";
        }

        if (ctx.step.type === "payment-gate") return "sleep";
        if (ctx.step.type === "done") return "celebrate";

        return stepEntryBranch(ctx.step.id, ctx.mode);
    }

    return null;
}