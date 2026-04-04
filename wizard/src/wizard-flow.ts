// wizard-flow.ts — Deterministic step definitions for the Xcelsior CLI wizard.
// Same questions, same order, every time. PostHog-style structured flow.
//
// Each step has a type (select, auto-check, text, confirm, device-auth, auto-fetch, done)
// and a condition function that decides if the step runs based on prior answers.

export type StepType =
    | "select"
    | "auto-check"
    | "text"
    | "confirm"
    | "done"
    | "device-auth"
    | "auto-fetch"
    | "payment-gate";

export interface SelectOption {
    label: string;
    value: string;
}

export interface WizardStep {
    id: string;
    type: StepType;
    /** Wizard sprite message when this step is active */
    prompt: string;
    /** For select steps */
    options?: SelectOption[];
    /** For text steps — placeholder text */
    placeholder?: string;
    /** Skip this step if condition returns false. Receives all collected answers. */
    condition?: (answers: Record<string, string | string[]>) => boolean;
    /** For auto-check steps — the check function ID (resolved at runtime) */
    checkId?: string;
    /** For confirm steps — what we're confirming */
    confirmLabel?: string;
    /** For text steps — validate input. Return error string or null if valid. */
    validate?: (value: string) => string | null;
    /** If true, auto-check failures cannot be skipped (must retry or fix) */
    checkRequired?: boolean;
}

// ── Docker image templates ───────────────────────────────────────────
// Mapped by workload type for auto-selection, with full list for overrides.

export interface ImageTemplate {
    label: string;
    value: string;
    vram: number; // minimum recommended VRAM in GB
}

export const IMAGE_TEMPLATES: ImageTemplate[] = [
    { label: "🧠 PyTorch (NVIDIA)", value: "nvcr.io/nvidia/pytorch:24.01-py3", vram: 24 },
    { label: "📊 TensorFlow (NVIDIA)", value: "nvcr.io/nvidia/tensorflow:24.01-tf2-py3", vram: 24 },
    { label: "⚡ vLLM (inference)", value: "vllm/vllm-openai:latest", vram: 24 },
    { label: "🎨 ComfyUI", value: "comfyanonymous/comfyui:latest", vram: 12 },
    { label: "📓 Jupyter + PyTorch", value: "quay.io/jupyter/pytorch-notebook:cuda12-latest", vram: 8 },
    { label: "🐧 Ubuntu + CUDA", value: "nvidia/cuda:12.4.1-devel-ubuntu22.04", vram: 8 },
];

export const WORKLOAD_IMAGE_MAP: Record<string, string> = {
    training: "nvcr.io/nvidia/pytorch:24.01-py3",
    inference: "vllm/vllm-openai:latest",
    research: "quay.io/jupyter/pytorch-notebook:cuda12-latest",
    "fine-tuning": "nvcr.io/nvidia/pytorch:24.01-py3",
    other: "nvidia/cuda:12.4.1-devel-ubuntu22.04",
};

// ── The Flow ─────────────────────────────────────────────────────────
// Every run walks this exact sequence. Steps with conditions may be skipped.

export const WIZARD_STEPS: WizardStep[] = [
    // ── Step 1: Mode ───────────────────────────────────────────────────
    {
        id: "mode",
        type: "select",
        prompt: "Welcome! I'm Hexara, your setup wizard. What would you like to do?",
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

    // ── Step 3: Device-code auth ───────────────────────────────────────
    {
        id: "device-auth",
        type: "device-auth",
        prompt: "Let's get you authenticated. Opening your browser...",
    },

    // ── Step 4: Validate connection ────────────────────────────────────
    {
        id: "api-check",
        type: "auto-check",
        prompt: "Verifying your connection...",
        checkId: "api",
        checkRequired: true,
    },

    // ── Step 5: Provider — GPU detection ───────────────────────────────
    {
        id: "gpu-detect",
        type: "auto-check",
        prompt: "Detecting GPUs...",
        checkId: "gpu",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 5b: Provider — Version check ──────────────────────────────
    {
        id: "version-check",
        type: "auto-check",
        prompt: "Checking component versions (runc, Docker, NVIDIA driver, toolkit)...",
        checkId: "versions",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 5c: Provider — Compute benchmark ──────────────────────────
    {
        id: "benchmark",
        type: "auto-check",
        prompt: "Running GPU benchmarks — FP16 matmul, PCIe bandwidth, thermal soak (≈60s)...",
        checkId: "benchmark",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 5d: Provider — Network benchmark ──────────────────────────
    {
        id: "network-bench",
        type: "auto-check",
        prompt: "Testing network quality to the scheduler...",
        checkId: "network",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 5e: Provider — Verification ───────────────────────────────
    {
        id: "verification",
        type: "auto-check",
        prompt: "Running 7-point hardware verification...",
        checkId: "verify",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 6: Provider — Pricing ─────────────────────────────────────
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

    // ── Step 7: Provider — Custom rate ─────────────────────────────────
    {
        id: "custom-rate",
        type: "text",
        prompt: "Enter your hourly rate in CAD (e.g. 2.50):",
        placeholder: "2.50",
        condition: (a) =>
            (a.mode === "provide" || a.mode === "both") && a.pricing === "custom",
        validate: (v) => {
            const n = parseFloat(v);
            if (isNaN(n) || n <= 0) return "Enter a positive number (e.g. 2.50)";
            if (n > 1000) return "Rate seems too high — max $1000/hr";
            return null;
        },
    },

    // ── Step 7b: Provider — Host registration ──────────────────────────
    {
        id: "host-register",
        type: "auto-check",
        prompt: "Registering your host on the marketplace...",
        checkId: "host-register",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 7c: Provider — Admission gate ─────────────────────────────
    {
        id: "admission-gate",
        type: "auto-check",
        prompt: "Checking admission status and security runtime...",
        checkId: "admission",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 7d: Provider — Summary ────────────────────────────────────
    {
        id: "provider-summary",
        type: "confirm",
        prompt: "Your GPU is verified and listed on the marketplace!",
        confirmLabel: "Continue to save configuration",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 8: Renter — Workload type ─────────────────────────────────
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

    // ── Step 9: Renter — GPU preference ────────────────────────────────
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

    // ── Step 10: Renter — Browse GPUs ──────────────────────────────────
    {
        id: "browse-gpus",
        type: "auto-fetch",
        prompt: "Searching the marketplace for available GPUs...",
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 11: Renter — Pick GPU ─────────────────────────────────────
    {
        id: "gpu-pick",
        type: "select",
        prompt: "Choose a GPU to launch your instance on:",
        options: [], // dynamically populated in useWizardFlow
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 12: Renter — Pick Docker image ────────────────────────────
    {
        id: "image-pick",
        type: "select",
        prompt: "Choose your environment:",
        options: [], // dynamically populated with pre-selected default
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 13: Renter — Confirm launch ───────────────────────────────
    {
        id: "confirm-launch",
        type: "confirm",
        prompt: "Ready to launch your instance?",
        confirmLabel: "Launch this instance",
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 14: Renter — Wallet check ─────────────────────────────────
    {
        id: "wallet-check",
        type: "auto-check",
        prompt: "Checking your wallet...",
        checkId: "wallet",
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 15: Renter — Payment gate ─────────────────────────────────
    {
        id: "payment-gate",
        type: "payment-gate",
        prompt: "Your balance is too low for this GPU. Let's add funds.",
        condition: (a) => {
            if (a.mode !== "rent" && a.mode !== "both") return false;
            // Only show if wallet balance is insufficient (set by wallet-check)
            return a["_wallet_insufficient"] === "true";
        },
    },

    // ── Step 16: Renter — Launch instance ──────────────────────────────
    {
        id: "launch-instance",
        type: "auto-check",
        prompt: "Launching your instance — this may take a few minutes to spin up...",
        checkId: "launch",
        condition: (a) => a.mode === "rent" || a.mode === "both",
    },

    // ── Step 17: Provider — Confirm & save ─────────────────────────────
    {
        id: "confirm-setup",
        type: "confirm",
        prompt: "Ready to save your configuration?",
        confirmLabel: "Save config to ~/.xcelsior/config.toml",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Done ───────────────────────────────────────────────────────────
    {
        id: "done",
        type: "done",
        prompt: "You're all set! Hexara will be here whenever you need — just run `xcelsior setup` to summon the wizard again.",
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
