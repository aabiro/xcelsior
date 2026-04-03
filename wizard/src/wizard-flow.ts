// wizard-flow.ts — Deterministic step definitions for the Xcelsior CLI wizard.
// Same questions, same order, every time. PostHog-style structured flow.
//
// Each step has a type (select, checklist, auto-check, text, confirm, done)
// and a condition function that decides if the step runs based on prior answers.

export type StepType = "select" | "checklist" | "auto-check" | "text" | "confirm" | "done";

export interface SelectOption {
    label: string;
    value: string;
}

export interface WizardStep {
    id: string;
    type: StepType;
    /** Wizard sprite message when this step is active */
    prompt: string;
    /** For select/checklist steps */
    options?: SelectOption[];
    /** For text steps — placeholder text */
    placeholder?: string;
    /** Skip this step if condition returns false. Receives all collected answers. */
    condition?: (answers: Record<string, string | string[]>) => boolean;
    /** For auto-check steps — the check function ID (resolved at runtime) */
    checkId?: string;
    /** For confirm steps — what we're confirming */
    confirmLabel?: string;
}

// ── The Flow ─────────────────────────────────────────────────────────
// Every run walks this exact sequence. Steps with conditions may be skipped.

export const WIZARD_STEPS: WizardStep[] = [
    // ── Step 1: Mode ───────────────────────────────────────────────────
    {
        id: "mode",
        type: "select",
        prompt: "Welcome to Xcelsior! I'm your setup wizard — let's get you started. What would you like to do?",
        options: [
            { label: "🖥️  Rent GPUs — launch jobs on the marketplace", value: "rent" },
            { label: "🔌 Provide GPUs — earn by sharing your hardware", value: "provide" },
            { label: "🔄 Both — rent and provide", value: "both" },
        ],
    },

    // ── Step 2: Docker checks (providers only) ─────────────────────────
    {
        id: "docker-check",
        type: "auto-check",
        prompt: "Checking Docker environment...",
        checkId: "docker",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 3: API endpoint ───────────────────────────────────────────
    {
        id: "api-url",
        type: "text",
        prompt: "What's your Xcelsior API URL?",
        placeholder: "https://xcelsior.ca (press Enter for default)",
    },

    // ── Step 4: API key ────────────────────────────────────────────────
    {
        id: "api-key",
        type: "text",
        prompt: "Paste your API key (from Dashboard → Settings → API Keys):",
        placeholder: "xcel_...",
    },

    // ── Step 5: Validate connection ────────────────────────────────────
    {
        id: "api-check",
        type: "auto-check",
        prompt: "Connecting to Xcelsior API...",
        checkId: "api",
    },

    // ── Step 6: Provider — GPU detection ───────────────────────────────
    {
        id: "gpu-detect",
        type: "auto-check",
        prompt: "Detecting GPUs...",
        checkId: "gpu",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 7: Provider — Pricing ─────────────────────────────────────
    {
        id: "pricing",
        type: "select",
        prompt: "How would you like to price your GPU time?",
        options: [
            { label: "💰 Recommended — use marketplace average pricing", value: "recommended" },
            { label: "📊 Competitive — slightly below average", value: "competitive" },
            { label: "✏️  Custom — set your own rate", value: "custom" },
        ],
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 8: Provider — Custom rate ─────────────────────────────────
    {
        id: "custom-rate",
        type: "text",
        prompt: "Enter your hourly rate in CAD (e.g. 2.50):",
        placeholder: "2.50",
        condition: (a) =>
            (a.mode === "provide" || a.mode === "both") && a.pricing === "custom",
    },

    // ── Step 9: Renter — Workload type ─────────────────────────────────
    {
        id: "workload",
        type: "select",
        prompt: "What will you be running?",
        options: [
            { label: "🧠 Training — fine-tuning or pre-training models", value: "training" },
            { label: "⚡ Inference — serving models for predictions", value: "inference" },
            { label: "🔬 Research — Jupyter notebooks, experiments", value: "research" },
            { label: "🎮 Other — rendering, simulation, etc.", value: "other" },
        ],
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 10: Renter — GPU preference ───────────────────────────────
    {
        id: "gpu-preference",
        type: "select",
        prompt: "GPU preference?",
        options: [
            { label: "🏎️  Best available — highest performance", value: "best" },
            { label: "💵 Cheapest — minimize cost", value: "cheapest" },
            { label: "🎯 Specific — I know what I need", value: "specific" },
        ],
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 11: Confirm & save ────────────────────────────────────────
    {
        id: "confirm-setup",
        type: "confirm",
        prompt: "Ready to save your configuration?",
        confirmLabel: "Save config to ~/.xcelsior/config.toml",
    },

    // ── Done ───────────────────────────────────────────────────────────
    {
        id: "done",
        type: "done",
        prompt: "Setup complete! You're ready to go.",
    },
];

/**
 * Get the next step index, respecting conditions.
 * Returns -1 if flow is complete.
 */
export function getNextStep(
    currentIndex: number,
    answers: Record<string, string | string[]>,
): number {
    for (let i = currentIndex + 1; i < WIZARD_STEPS.length; i++) {
        const step = WIZARD_STEPS[i];
        if (!step.condition || step.condition(answers)) {
            return i;
        }
    }
    return -1;
}
