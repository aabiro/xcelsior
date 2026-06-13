// task-model.ts — Live task checklist derived from the real wizard flow (Part D).
//
// This is a *view* over the existing WIZARD_STEPS + collected answers + check
// results, NOT a parallel state machine. buildTaskList() filters the canonical
// steps by mode (same rule as index.tsx's totalSteps); computeTaskStates()
// projects the current stepIndex / completedStepIds / checkResults onto a
// todo → active → done → failed lifecycle so the right pane stays honest.

import { WIZARD_STEPS, type WizardStep } from "./wizard-flow.js";
import type { AutoCheckResults } from "./useWizardFlow.js";

export type TaskState = "todo" | "active" | "done" | "failed";

export interface WizardTask {
    id: string;
    label: string;
    state: TaskState;
}

/** Human-friendly task labels (PostHog-style imperative phrasing). */
const TASK_LABELS: Record<string, string> = {
    "mode": "Choose how you'll use Xcelsior",
    "docker-check": "Check Docker environment",
    "device-auth": "Authenticate your account",
    "api-check": "Verify control-plane connection",
    "gpu-detect": "Detect local GPUs",
    "version-check": "Check component versions",
    "network-setup": "Set up mesh networking",
    "benchmark": "Benchmark GPU compute",
    "network-bench": "Test network quality",
    "verification": "Verify hardware (7-point)",
    "pricing": "Set your pricing",
    "custom-rate": "Set a custom rate",
    "spot-enabled": "Configure spot instances",
    "spot-min-cents": "Set spot price floor",
    "host-register": "Register host on the marketplace",
    "admission-gate": "Check admission & runtime",
    "provider-summary": "Review provider summary",
    "worker-install": "Install the worker agent",
    "workload": "Choose your workload",
    "ssh-key-setup": "Set up SSH access",
    "gpu-preference": "Set GPU preference",
    "browse-gpus": "Browse the marketplace",
    "gpu-pick": "Pick a GPU",
    "image-pick": "Choose your environment",
    "confirm-launch": "Confirm launch",
    "wallet-check": "Check wallet balance",
    "payment-gate": "Add funds",
    "launch-instance": "Launch your instance",
    "confirm-setup": "Save configuration",
    "sdk-detect": "Detect project framework",
    "sdk-install": "Install @xcelsior-gpu/sdk",
    "sdk-credentials": "Configure API credentials",
    "sdk-verify": "Verify API connection",
    "sdk-snippet": "Copy starter code",
};

export function labelForStep(step: WizardStep): string {
    return TASK_LABELS[step.id] ?? step.id;
}

/**
 * Build the ordered task list for a given mode by filtering the canonical steps.
 * The terminal "done" step is excluded (it's a farewell, not a task).
 * Conditional steps whose condition can't yet be evaluated (e.g. custom-rate,
 * payment-gate) are included only when their mode-level condition holds, mirroring
 * the progress-count logic so the denominator stays stable.
 */
export function buildTaskList(mode: string | undefined): WizardTask[] {
    if (!mode) return [];
    const stableAnswers: Record<string, string> = { mode };
    return WIZARD_STEPS.filter((s) => s.type !== "done")
        .filter((s) => !s.condition || s.condition(stableAnswers))
        // Steps gated on a *runtime* answer (not just mode) are optional branches;
        // keep the mode-stable set so the denominator doesn't jump around.
        .filter((s) => !OPTIONAL_RUNTIME_STEPS.has(s.id))
        .map((s) => ({ id: s.id, label: labelForStep(s), state: "todo" as TaskState }));
}

/** Steps that only appear based on a runtime answer, excluded from the stable denominator. */
const OPTIONAL_RUNTIME_STEPS: ReadonlySet<string> = new Set([
    "custom-rate", // only if pricing === custom
    "spot-min-cents", // only if spot-enabled === yes
    "payment-gate", // only if wallet insufficient
]);

export interface TaskStateInput {
    /** Current step id (the in-flight step). */
    currentStepId: string;
    /** Step ids already completed/advanced past. */
    completedStepIds: string[];
    /** Check results keyed by step id (for failed detection). */
    checkResults?: Record<string, AutoCheckResults>;
    /** True when the current auto-check failed and is awaiting retry/skip. */
    currentFailed?: boolean;
    /** Extra step ids to mark active (concurrent work). */
    alsoActive?: string[];
}

/**
 * Project run state onto the task list. A task is:
 *   done   — in completedStepIds, or its check results all passed
 *   failed — it's the current step and its check failed (currentFailed) OR its
 *            recorded check results contain a failure
 *   active — it's the current step (or in alsoActive) and not failed
 *   todo   — otherwise
 */
export function computeTaskStates(tasks: WizardTask[], input: TaskStateInput): WizardTask[] {
    const completed = new Set(input.completedStepIds);
    const active = new Set<string>([input.currentStepId, ...(input.alsoActive ?? [])]);
    return tasks.map((task) => {
        let state: TaskState = "todo";
        const result = input.checkResults?.[task.id];
        const checkFailed = result ? !result.allPassed : false;

        if (completed.has(task.id)) {
            state = "done";
        } else if (active.has(task.id)) {
            state = (input.currentFailed && task.id === input.currentStepId) || checkFailed
                ? "failed"
                : "active";
        } else if (result && result.allPassed) {
            state = "done";
        } else if (checkFailed) {
            state = "failed";
        }
        return { ...task, state };
    });
}

export interface TaskProgress {
    done: number;
    total: number;
}

export function taskProgress(tasks: WizardTask[]): TaskProgress {
    return { done: tasks.filter((t) => t.state === "done").length, total: tasks.length };
}
