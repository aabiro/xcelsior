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

/**
 * How long the sign-in screen shows the code before opening the browser.
 * It was 15s, which read as the wizard having stalled; Enter still opens it now.
 */
export const DEVICE_BROWSER_DELAY_MS = 3_000;

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

/** Static per-step help when AI is unavailable (A6). */
export const STATIC_STEP_HELP: Record<string, string[]> = {
    "api-check": ["Check your internet connection, then try again.", "On a work network? Make sure xcelsior.ca isn't blocked."],
    versions: ["Update your NVIDIA driver, NVIDIA Container Toolkit and Docker to at least the versions shown above."],
    benchmark: ["Close anything else using the GPU, let it cool down, then try again."],
    verify: ["Fix the items marked ✗ above, then try again."],
    "host-register": ["Check you're signed in and that this machine is online, then try again."],
    "worker-install": ["To install only the agent, run the installer with --agent-only."],
    "sdk-detect": ["Run the wizard from your app's main folder (the one with package.json).", "Python app? Use the REST API instead, or press b to choose a different path."],
    "sdk-install": ["Install it yourself with: npm install @xcelsior-gpu/sdk", "Then press Enter to check again."],
    "sdk-credentials": ["The wizard creates an app key for your project.", "You can also find it in Dashboard → Settings → API."],
    "sdk-verify": ["Make sure you finished signing in.", "If you set XCELSIOR_API_URL yourself, check it's correct."],
};

export const WORKLOAD_IMAGE_MAP: Record<string, string> = {
    training: "nvcr.io/nvidia/pytorch:24.01-py3",
    inference: "vllm/vllm-openai:latest",
    research: "quay.io/jupyter/pytorch-notebook:cuda12-latest",
    other: "nvidia/cuda:12.4.1-devel-ubuntu22.04",
};

function isRenterMode(a: Record<string, string | string[]>): boolean {
    return a.mode === "rent" || a.mode === "both";
}

function shouldLaunchInstance(a: Record<string, string | string[]>): boolean {
    return isRenterMode(a) && a["want-launch"] !== "no";
}

// ── The Flow ─────────────────────────────────────────────────────────
// Every run walks this exact sequence. Steps with conditions may be skipped.

export const WIZARD_STEPS: WizardStep[] = [
    // ── Step 1: Mode ───────────────────────────────────────────────────
    {
        id: "mode",
        type: "select",
        prompt: "Greetings! I'm Hexara, your trusty setup wizard. What sort of magic would you like to cast today?",
        options: [
            { label: "🖥️  Rent GPUs — launch jobs on the marketplace", value: "rent" },
            { label: "🔌 Provide GPUs — earn by sharing your hardware", value: "provide" },
            { label: "🔄 Both — rent and provide", value: "both" },
            { label: "📦 Build with the SDK — use Xcelsior from your own app", value: "sdk" },
        ],
    },

    // ── SDK track: detect project ──────────────────────────────────────
    {
        id: "sdk-detect",
        type: "auto-check",
        prompt: "Looking at your project...",
        checkId: "sdk-detect",
        checkRequired: true,
        condition: (a) => a.mode === "sdk",
    },

    // ── Step 2: Docker checks (providers only) ─────────────────────────
    {
        id: "docker-check",
        type: "auto-check",
        prompt: "Checking that Docker is installed and running...",
        checkId: "docker",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 3: Device-code auth ───────────────────────────────────────
    {
        id: "device-auth",
        type: "device-auth",
        prompt: "Let's get you signed in...",
    },

    // ── Step 4: Validate connection ────────────────────────────────────
    {
        id: "api-check",
        type: "auto-check",
        prompt: "Checking your connection to Xcelsior...",
        checkId: "api",
        checkRequired: true,
    },

    // ── SDK track: install package ─────────────────────────────────────
    {
        id: "sdk-install",
        type: "auto-check",
        prompt: "Installing the Xcelsior SDK in your project...",
        checkId: "sdk-install",
        checkRequired: true,
        condition: (a) => a.mode === "sdk",
    },

    // ── SDK track: credentials + .env ──────────────────────────────────
    {
        id: "sdk-credentials",
        type: "auto-check",
        prompt: "Creating an app key and adding it to your project's settings...",
        checkId: "sdk-credentials",
        condition: (a) => a.mode === "sdk",
        checkRequired: true,
    },

    // ── SDK track: verify API ──────────────────────────────────────────
    {
        id: "sdk-verify",
        type: "auto-check",
        prompt: "Making a test call with the SDK...",
        checkId: "sdk-verify",
        condition: (a) => a.mode === "sdk",
        checkRequired: true,
    },

    // ── SDK track: starter snippet ─────────────────────────────────────
    {
        id: "sdk-snippet",
        type: "confirm",
        prompt: "Your SDK integration is ready!",
        confirmLabel: "Finish setup",
        condition: (a) => a.mode === "sdk",
    },

    // ── Step 5: Provider — GPU detection ───────────────────────────────
    {
        id: "gpu-detect",
        type: "auto-check",
        prompt: "Finding your GPUs...",
        checkId: "gpu",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 5b: Provider — Version check ──────────────────────────────
    {
        id: "version-check",
        type: "auto-check",
        prompt: "Checking your NVIDIA driver and Docker versions...",
        checkId: "versions",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 5c: Provider — Network setup ──────────────────────────────
    {
        id: "network-setup",
        type: "auto-check",
        prompt: "Connecting this machine to Xcelsior's private network...",
        checkId: "network-setup",
        checkRequired: true,
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 5d: Provider — Compute benchmark ──────────────────────────
    {
        id: "benchmark",
        type: "auto-check",
        prompt: "Testing your GPU's speed and temperature (about a minute)...",
        checkId: "benchmark",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 5d: Provider — Network benchmark ──────────────────────────
    {
        id: "network-bench",
        type: "auto-check",
        prompt: "Testing your internet connection to Xcelsior...",
        checkId: "network",
        checkRequired: true,
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 5e: Provider — Verification ───────────────────────────────
    {
        id: "verification",
        type: "auto-check",
        prompt: "Checking your hardware is ready to rent out...",
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
        prompt: "Enter your hourly rate (e.g. 2.50 for $2.50/hr):",
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

    // ── Step 7a: Provider — Spot instances ─────────────────────────────
    {
        id: "spot-enabled",
        type: "select",
        prompt: "Also accept spot jobs? They pay less and can stop early, but keep your GPU busy.",
        options: [
            { label: "✓ Yes — accept spot workloads (recommended)", value: "yes" },
            { label: "✗ No — on-demand only", value: "no" },
        ],
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },
    {
        id: "spot-min-cents",
        type: "text",
        prompt: "Lowest price you'll take for a spot job, in cents per hour:",
        placeholder: "10",
        condition: (a) =>
            (a.mode === "provide" || a.mode === "both") && a["spot-enabled"] === "yes",
        validate: (v) => {
            const n = parseInt(v, 10);
            if (isNaN(n) || n < 0) return "Enter a non-negative integer (cents per hour)";
            if (n > 100000) return "Floor seems too high";
            return null;
        },
    },

    // ── Step 7b: Provider — Pending host registration ──────────────────
    {
        id: "host-register",
        type: "auto-check",
        prompt: "Adding your machine to the marketplace (hidden until it's verified)...",
        checkId: "host-register",
        condition: (a) => a.mode === "provide" || a.mode === "both",
        checkRequired: true,
    },

    // ── Step 7c: Provider — Compatibility and admission boundary ───────
    {
        id: "admission-gate",
        type: "auto-check",
        prompt: "Checking whether your machine has been approved...",
        checkId: "admission",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 7d: Provider — Summary ────────────────────────────────────
    {
        id: "provider-summary",
        type: "confirm",
        prompt: "Your machine passed our checks. It goes live once verification finishes.",
        confirmLabel: "Continue",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 7e: Provider — Worker agent install ───────────────────────
    {
        id: "worker-install",
        type: "auto-check",
        prompt: "Installing the Xcelsior agent so jobs can run on this machine...",
        checkId: "worker-install",
        condition: (a) => a.mode === "provide" || a.mode === "both",
    },

    // ── Step 7.5: Renter — Want to launch? ─────────────────────────────
    {
        id: "want-launch",
        type: "confirm",
        prompt: "Would you like to launch your first instance right now?",
        confirmLabel: "Yes, launch an instance",
        condition: isRenterMode,
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
        condition: shouldLaunchInstance,
    },

    // ── Step 8b: Renter — SSH key setup ────────────────────────────────
    {
        id: "ssh-key-setup",
        type: "auto-check",
        prompt: "Setting up a key so you can log in to your instance...",
        checkId: "ssh-key-setup",
        condition: shouldLaunchInstance,
    },

    // ── Step 9: Renter — GPU preference ────────────────────────────────
    {
        id: "gpu-preference",
        type: "select",
        prompt: "Which GPU should we look for?",
        options: [
            { label: "🏎️  Best available — highest performance", value: "best" },
            { label: "💵 Cheapest — minimize cost", value: "cheapest" },
            { label: "🎯 Specific — I know what I need", value: "specific" },
        ],
        condition: shouldLaunchInstance,
    },

    // ── Step 10: Renter — Browse GPUs ──────────────────────────────────
    {
        id: "browse-gpus",
        type: "auto-fetch",
        prompt: "Searching the marketplace for available GPUs...",
        condition: shouldLaunchInstance,
    },

    // ── Step 11: Renter — Pick GPU ─────────────────────────────────────
    {
        id: "gpu-pick",
        type: "select",
        prompt: "Choose a GPU to launch your instance on:",
        options: [], // dynamically populated in useWizardFlow
        condition: shouldLaunchInstance,
    },

    // ── Step 12: Renter — Pick Docker image ────────────────────────────
    {
        id: "image-pick",
        type: "select",
        prompt: "Choose your software setup:",
        options: [], // dynamically populated with pre-selected default
        condition: shouldLaunchInstance,
    },

    // ── Step 13: Renter — Confirm launch ───────────────────────────────
    {
        id: "confirm-launch",
        type: "confirm",
        prompt: "Ready to launch your instance?",
        confirmLabel: "Launch this instance",
        condition: shouldLaunchInstance,
    },

    // ── Step 14: Renter — Wallet check ─────────────────────────────────
    {
        id: "wallet-check",
        type: "auto-check",
        prompt: "Checking your balance...",
        checkId: "wallet",
        condition: shouldLaunchInstance,
    },

    // ── Step 15: Renter — Payment gate ─────────────────────────────────
    {
        id: "payment-gate",
        type: "payment-gate",
        prompt: "Your balance is too low for this GPU. Let's add funds.",
        condition: (a) => {
            if (!shouldLaunchInstance(a)) return false;
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
        condition: shouldLaunchInstance,
    },

    // ── Step 17: Both mode — Confirm & save (provider-only confirms at provider-summary) ──
    {
        id: "confirm-setup",
        type: "confirm",
        prompt: "Save these settings on this computer?",
        confirmLabel: "Save settings",
        condition: (a) => a.mode === "both",
    },

    // ── Done ───────────────────────────────────────────────────────────
    {
        id: "done",
        type: "done",
        prompt: "Huzzah! All tasks are complete. You are ready to shape the GPU cosmos. I, your wizard, shall take my leave!",
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
