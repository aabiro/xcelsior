// useWizardFlow — Hook that drives the structured wizard step-by-step.
// Manages step progression, answer collection, auto-checks, device auth,
// marketplace browsing, payment gating, instance launch, and AI escape hatch.

import { useState, useCallback, useRef, useEffect } from "react";
import { existsSync, readFileSync } from "node:fs";
import { WIZARD_STEPS, getNextStep, type WizardStep, IMAGE_TEMPLATES, WORKLOAD_IMAGE_MAP } from "./wizard-flow.js";
import {
    streamChat, confirmAction, type ApiClientConfig,
    requestDeviceCode, pollDeviceToken, type DeviceCodeResult,
    getMe, generateApiKey, searchMarketplace, type MarketplaceListing, type MarketplaceFilters,
    getWallet, claimFreeCredits,
    launchInstance, getInstance, type InstanceInfo,
    registerHost, reportVersions, reportBenchmark, reportVerification,
    type HostInfo, type VerifyResult,
} from "./api-client.js";
import type { WizardState } from "../sprites/wizard/wizard-sprite.js";
import { checkDocker, type CheckResult } from "./checks.js";
import {
    checkVersions, detectGpuFull, runComputeBenchmark,
    runNetworkBenchmark, runVerificationChecks, buildVerificationReport,
    type GpuInfo, type BenchmarkResult, type NetworkBenchResult,
    type VersionCheck, type VerificationReport,
} from "./provider-checks.js";

// ── Config ───────────────────────────────────────────────────────────

const API_BASE_URL = process.env["XCELSIOR_API_URL"] || "https://xcelsior.ca";
const CONFIG_HOME = process.env["HOME"] ?? "/tmp";
const TOKEN_FILE = `${CONFIG_HOME}/.xcelsior/token.json`;
const CONFIG_FILE = `${CONFIG_HOME}/.xcelsior/config.toml`;
const DEVICE_POLL_MS = 5_000;
const DEVICE_CODE_EXPIRY_MS = 15 * 60 * 1000;
const WALLET_POLL_MS = 5_000;
const CHOREOGRAPHY_DELAY_MS = 8_000;
const ADVANCE_DEBOUNCE_MS = 300;

// ── Types ────────────────────────────────────────────────────────────

export interface AutoCheckResults {
    items: CheckResult[];
    allPassed: boolean;
}

export interface DeviceAuthState {
    status: "loading" | "waiting" | "authorized" | "error" | "manual";
    userCode: string | null;
    verificationUri: string | null;
    token: string | null;
    email: string | null;
    errorMessage: string | null;
}

export interface UseWizardFlowReturn {
    step: WizardStep;
    stepIndex: number;
    answers: Record<string, string | string[]>;
    wizardState: WizardState;
    wizardMessage: string;
    checkResults: Record<string, AutoCheckResults>;
    aiResponse: string | null;
    aiStreaming: boolean;
    submitAnswer: (value: string | string[]) => void;
    askAi: (question: string) => Promise<void>;
    dismissAi: () => void;
    isComplete: boolean;
    /** Device auth state for the auth step */
    deviceAuth: DeviceAuthState;
    /** Switch device-auth to manual paste mode */
    switchToManualAuth: () => void;
    /** Retry device auth */
    retryDeviceAuth: () => void;
    /** Open browser now — skip the 15s countdown */
    openBrowserNow: () => void;
    /** Manual token submission */
    submitManualToken: (token: string) => void;
    /** GPU marketplace listings for gpu-pick step */
    gpuListings: MarketplaceListing[];
    /** Dynamic select options for gpu-pick */
    gpuOptions: { label: string; value: string }[];
    /** Dynamic select options for image-pick */
    imageOptions: { label: string; value: string }[];
    /** Launched instance info */
    instanceInfo: InstanceInfo | null;
    /** Validation error for current step */
    validationError: string | null;
    /** Confirm error (e.g. pressed Enter instead of y/n) */
    confirmError: string | null;
    /** Launch summary lines for confirm-launch */
    launchSummary: string[];
    /** Payment gate state */
    paymentGate: { balance: number; required: number; polling: boolean; billingUrl: string };
    /** Skip payment gate */
    skipPayment: () => void;
    /** Browse GPU error for retry */
    browseError: string | null;
    /** Whether auto-check can be retried */
    checkCanRetry: boolean;
    /** Whether auto-check passed and awaiting Enter to continue */
    checkAwaitContinue: boolean;
    /** Retry failed auto-check */
    retryCheck: () => void;
    /** Skip failed auto-check */
    skipCheck: () => void;
    /** Continue after auto-check passed */
    continueFromCheck: () => void;
    /** Continue after device-auth authorized (Enter) */
    continueFromAuth: () => void;
    /** Error message if token save failed, null if saved OK */
    tokenSaveError: string | null;
    /** Whether there's a buffered AI response the user can reveal */
    hasAiDetails: boolean;
    /** Reveal the buffered AI response */
    revealAi: () => void;
    /** Detected project .env path (display hint only — not written by wizard) */
    deviceAuthEnvPath: string | null;
    /** Whether Hexara AI is available (after auth) */
    aiAvailable: boolean;
    /** Whether inline AI prompt is showing (for non-text steps) */
    showAiPrompt: boolean;
    /** Toggle AI prompt */
    toggleAiPrompt: () => void;
    /** Chat history for current step (Q&A pairs) */
    chatHistory: { question: string; answer: string }[];
    /** Current question being answered by Hexara */
    currentAiQuestion: string | null;
    /** Provider summary data for provider-summary step */
    providerSummary: ProviderSummaryData | null;
    /** Pending AI confirmation for write actions */
    pendingConfirmation: PendingConfirmation | null;
    /** Approve or reject a pending AI confirmation */
    confirmAi: (approved: boolean) => Promise<void>;
    /** Tool calls made during current AI response */
    aiToolCalls: AiToolCall[];
    /** True during the 8s choreography delay after submitAnswer — hides step content */
    transitioning: boolean;
}

export interface ProviderSummaryData {
    gpuModel: string;
    vramGb: number;
    xcuScore: number;
    tflops: number;
    verified: boolean;
    verificationState: string;
    hostId: string;
    pricing: string;
    customRate?: string;
    costPerHour: number;
    admitted: boolean;
    runtimeRecommendation: string;
    reputationPoints: number;
    tier: string;
}

export interface PendingConfirmation {
    confirmationId: string;
    toolName: string;
    toolArgs: Record<string, unknown>;
}

export interface AiToolCall {
    name: string;
    input: Record<string, unknown>;
    output?: Record<string, unknown>;
}

// ── Helpers ──────────────────────────────────────────────────────────

/** Validate an API connection by hitting /healthz */
async function checkApi(baseUrl: string, token: string): Promise<CheckResult[]> {
    try {
        const url = new URL("/healthz", baseUrl);
        const headers: Record<string, string> = {};
        if (token) headers["Authorization"] = `Bearer ${token}`;
        const resp = await fetch(url.toString(), { signal: AbortSignal.timeout(10_000), headers });
        if (resp.ok) {
            return [{ name: "API Connection", ok: true, detail: `${url.origin} — healthy` }];
        }
        return [{ name: "API Connection", ok: false, detail: `HTTP ${resp.status}` }];
    } catch (err) {
        const msg = err instanceof Error ? err.message : "Connection failed";
        return [{ name: "API Connection", ok: false, detail: msg }];
    }
}

/** Detect GPUs via nvidia-smi (full info for providers) */
async function checkGpuBasic(): Promise<CheckResult[]> {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const exec = promisify(execFile);

    try {
        const { stdout } = await exec("nvidia-smi", [
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ], { timeout: 10_000 });

        const gpus = stdout.trim().split("\n").filter(Boolean);
        if (gpus.length === 0) {
            return [{ name: "GPU Detection", ok: false, detail: "No GPUs found" }];
        }
        return gpus.map((line, i) => ({
            name: `GPU ${i}`,
            ok: true,
            detail: line.trim(),
        }));
    } catch {
        return [{ name: "GPU Detection", ok: false, detail: "nvidia-smi not available" }];
    }
}

/** Save token to ~/.xcelsior/token.json (0o600 perms).
 *  Returns true on success, error message string on failure. */
async function saveToken(token: string): Promise<true | string> {
    try {
        const fs = await import("node:fs");
        const path = await import("node:path");
        const dir = path.dirname(TOKEN_FILE);
        fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
        fs.writeFileSync(TOKEN_FILE, JSON.stringify({ access_token: token }, null, 2), {
            mode: 0o600,
        });
        // Verify the write
        const written = fs.readFileSync(TOKEN_FILE, "utf-8");
        const parsed = JSON.parse(written);
        if (parsed.access_token !== token) return "Token file verification failed — written content does not match";
        return true;
    } catch (err) {
        return err instanceof Error ? err.message : "Failed to save token";
    }
}

/** Write auth credentials to the project's .env file (append if exists, create if not).
 *  Writes OAuth client credentials when provided, and API token as fallback.
 *  Returns true on success, error message string on failure. */
async function writeProjectEnv(
    envPath: string,
    token: string,
    oauthClientId?: string,
    oauthClientSecret?: string,
): Promise<true | string> {
    try {
        const fs = await import("node:fs");
        const pairs: Array<{ key: string; value: string }> = [];
        if (oauthClientId && oauthClientSecret) {
            pairs.push({ key: "XCELSIOR_OAUTH_CLIENT_ID", value: oauthClientId });
            pairs.push({ key: "XCELSIOR_OAUTH_CLIENT_SECRET", value: oauthClientSecret });
        }
        if (token) {
            pairs.push({ key: "XCELSIOR_API_TOKEN", value: token });
        }

        if (fs.existsSync(envPath)) {
            let content = fs.readFileSync(envPath, "utf-8");
            for (const { key, value } of pairs) {
                const regex = new RegExp(`^${key}=.*$`, "m");
                if (regex.test(content)) {
                    content = content.replace(regex, `${key}=${value}`);
                } else {
                    content = content.trimEnd() + "\n" + `${key}=${value}` + "\n";
                }
            }
            fs.writeFileSync(envPath, content);
        } else {
            const lines = pairs.map(({ key, value }) => `${key}=${value}`);
            fs.writeFileSync(envPath, `# Added by Xcelsior setup wizard\n${lines.join("\n")}\n`);
        }
        return true;
    } catch (err) {
        return err instanceof Error ? err.message : `Failed to write ${envPath}`;
    }
}

/** Save config to ~/.xcelsior/config.toml */
async function saveConfig(answers: Record<string, string | string[]>): Promise<void> {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const dir = path.dirname(CONFIG_FILE);
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });

    const lines = [
        `# Xcelsior configuration — generated by setup wizard`,
        `api_url = "${API_BASE_URL}"`,
    ];
    if (answers.mode) lines.push(`mode = "${answers.mode}"`);
    if (answers.workload) lines.push(`workload = "${answers.workload}"`);
    if (answers.pricing) lines.push(`pricing = "${answers.pricing}"`);
    if (answers["custom-rate"]) lines.push(`custom_rate = ${answers["custom-rate"]}`);
    if (answers["_host_id"]) lines.push(`host_id = "${answers["_host_id"]}"`);
    fs.writeFileSync(CONFIG_FILE, lines.join("\n") + "\n", { mode: 0o600 });

    // Write .env for worker agent (provider mode)
    // Prefer OAuth client credentials when available; keep API token as fallback.
    if ((answers.mode === "provide" || answers.mode === "both") && answers["_host_id"]) {
        const envFile = `${CONFIG_HOME}/.xcelsior/.env`;
        const envLines = [
            `# Xcelsior worker environment — generated by setup wizard`,
            `XCELSIOR_HOST_ID=${answers["_host_id"]}`,
            `XCELSIOR_SCHEDULER_URL=${API_BASE_URL}`,
        ];
        if (answers["oauth-client-id"] && answers["oauth-client-secret"]) {
            envLines.push(`XCELSIOR_OAUTH_CLIENT_ID=${answers["oauth-client-id"]}`);
            envLines.push(`XCELSIOR_OAUTH_CLIENT_SECRET=${answers["oauth-client-secret"]}`);
        }
        if (answers["api-key"]) {
            envLines.push(`XCELSIOR_API_TOKEN=${answers["api-key"]}`);
        }
        if (answers["custom-rate"]) envLines.push(`XCELSIOR_COST_PER_HOUR=${answers["custom-rate"]}`);
        fs.writeFileSync(envFile, envLines.join("\n") + "\n", { mode: 0o600 });
    }
}

/** Find the project root by walking up from cwd looking for .git, .env, or well-known markers */
function findProjectRoot(): string {
    const path = require("node:path");
    let dir = process.cwd();
    // Walk up until we find .git or hit filesystem root
    for (let i = 0; i < 10; i++) {
        if (existsSync(path.join(dir, ".git")) || existsSync(path.join(dir, ".env"))) return dir;
        const parent = path.dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    return process.cwd();
}

/** Detect project framework in the project root */
function detectFramework(): { name: string; envPath: string } | null {
    const path = require("node:path");
    const root = findProjectRoot();
    try {
        const pkgPath = path.join(root, "package.json");
        if (existsSync(pkgPath)) {
            const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
            const deps = { ...pkg.dependencies, ...pkg.devDependencies };
            if (deps["next"]) return { name: "Next.js", envPath: path.join(root, ".env.local") };
            if (deps["react"]) return { name: "React", envPath: path.join(root, ".env") };
            if (deps["vue"]) return { name: "Vue", envPath: path.join(root, ".env") };
            if (deps["svelte"] || deps["@sveltejs/kit"]) return { name: "SvelteKit", envPath: path.join(root, ".env") };
            return { name: "Node.js", envPath: path.join(root, ".env") };
        }
        if (existsSync(path.join(root, "requirements.txt")) || existsSync(path.join(root, "pyproject.toml"))) {
            return { name: "Python", envPath: path.join(root, ".env") };
        }
        if (existsSync(path.join(root, "Cargo.toml"))) return { name: "Rust", envPath: path.join(root, ".env") };
        if (existsSync(path.join(root, "go.mod"))) return { name: "Go", envPath: path.join(root, ".env") };
    } catch {
        // ignore
    }
    return null;
}

/** Open a URL in the default browser */
async function openBrowser(url: string): Promise<boolean> {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const exec = promisify(execFile);

    // Try platform-specific openers
    const cmds: [string, string[]][] = process.platform === "darwin"
        ? [["open", [url]]]
        : process.platform === "win32"
            ? [["cmd", ["/c", "start", url]]]
            : [["xdg-open", [url]], ["sensible-browser", [url]], ["x-www-browser", [url]]];

    for (const [cmd, args] of cmds) {
        try {
            await exec(cmd, args, { timeout: 5_000 });
            return true;
        } catch {
            continue;
        }
    }
    return false;
}

/**
 * Build a rich page_context string for the AI, including wizard state.
 * This gives the server-side AI full situational awareness.
 */
export function buildWizardContext(
    stepId: string,
    answers: Record<string, string | string[]>,
    checkResults: Record<string, AutoCheckResults>,
    providerSummary: ProviderSummaryData | null,
    gpuListings: MarketplaceListing[],
    browseError: string | null,
    earlyGpu?: GpuInfo | null,
    earlyBench?: BenchmarkResult | null,
    earlyNetwork?: NetworkBenchResult | null,
): string {
    const parts: string[] = [`cli-wizard:${stepId}`];

    // Mode
    if (answers.mode) parts.push(`mode=${answers.mode}`);

    // Provider context
    if (answers.mode === "provide" || answers.mode === "both") {
        if (answers.pricing) parts.push(`pricing=${answers.pricing}`);
        if (answers["custom-rate"]) parts.push(`custom_rate=$${answers["custom-rate"]}/hr`);
        if (answers["_rate"]) parts.push(`rate=${answers["_rate"]}`);
        if (answers["_host_id"]) parts.push(`host_id=${answers["_host_id"]}`);
        if (answers["_host_ip"]) parts.push(`host_ip=${answers["_host_ip"]}`);
        if (answers["_host_port"]) parts.push(`host_port=${answers["_host_port"]}`);

        // Check results summary — URL-encode values to avoid nested = truncation
        const failedChecks: string[] = [];
        for (const [stepKey, result] of Object.entries(checkResults)) {
            if (!result.allPassed) {
                const failures = result.items.filter((i) => !i.ok).map((i) => `${i.name}: ${i.detail}`);
                failedChecks.push(`${stepKey}=[${failures.join("; ")}]`);
            }
        }
        if (failedChecks.length > 0) parts.push(`failed_checks=${encodeURIComponent(`{${failedChecks.join(", ")}}`)}`);

        // GPU/benchmark data — use providerSummary if available, fall back to early refs
        if (providerSummary) {
            parts.push(`gpu=${providerSummary.gpuModel}`);
            parts.push(`vram=${providerSummary.vramGb}GB`);
            parts.push(`xcu=${providerSummary.xcuScore}`);
            parts.push(`tflops=${providerSummary.tflops}`);
            parts.push(`tier=${providerSummary.tier}`);
            parts.push(`verified=${providerSummary.verified}`);
        } else {
            // Early fallback from detection/benchmark refs (available before step 13)
            if (earlyGpu) {
                parts.push(`gpu=${earlyGpu.gpu_model}`);
                parts.push(`vram=${earlyGpu.total_vram_gb}GB`);
            }
            if (earlyBench) {
                parts.push(`tflops=${earlyBench.tflops}`);
                parts.push(`xcu=${earlyBench.xcu_score}`);
            }
        }

        // Network benchmark data
        if (earlyNetwork) {
            parts.push(`latency=${earlyNetwork.latency_avg_ms}ms`);
            parts.push(`jitter=${earlyNetwork.jitter_ms}ms`);
            parts.push(`throughput=${earlyNetwork.throughput_mbps}Mbps`);
        }
    }

    // Renter context
    if (answers.mode === "rent" || answers.mode === "both") {
        if (answers.workload) parts.push(`workload=${answers.workload}`);
        if (answers["gpu-preference"]) parts.push(`gpu_pref=${answers["gpu-preference"]}`);
        if (answers["gpu-pick"]) {
            const listing = gpuListings.find((l) => l.host_id === (answers["gpu-pick"] as string));
            if (listing) {
                parts.push(`picked_gpu=${listing.gpu_model}/${listing.vram_gb}GB/$${listing.price_per_hour}/hr`);
                parts.push(`rate=$${listing.price_per_hour}/hr`);
            }
        }
        if (answers["image-pick"]) parts.push(`image=${answers["image-pick"]}`);
        if (answers["_instance_id"]) parts.push(`instance_id=${answers["_instance_id"]}`);
        if (answers["_balance"]) parts.push(`balance=$${answers["_balance"]}`);
        if (browseError) parts.push(`browse_error=${browseError}`);
    }

    return parts.join(" | ");
}

/** Generate a memorable instance name — exported for testing */
export function generateInstanceName(): string {
    const adj = ["swift", "bright", "cosmic", "nova", "stellar", "quantum", "astral", "blazing"];
    const noun = ["forge", "nexus", "pulse", "flux", "spark", "core", "beam", "arc"];
    const pick = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
    return `${pick(adj)}-${pick(noun)}-${Math.floor(Math.random() * 1000)}`;
}

// ── Hook ─────────────────────────────────────────────────────────────

export function useWizardFlow(): UseWizardFlowReturn {
    const [stepIndex, setStepIndex] = useState(0);
    const stepIndexRef = useRef(0);
    const [answers, setAnswers] = useState<Record<string, string | string[]>>({ "_api_base_url": API_BASE_URL });
    const answersRef = useRef<Record<string, string | string[]>>({ "_api_base_url": API_BASE_URL });
    const [wizardState, setWizardState] = useState<WizardState>("idle");
    const [wizardMessage, setWizardMessage] = useState(WIZARD_STEPS[0].prompt);
    const [checkResults, setCheckResults] = useState<Record<string, AutoCheckResults>>({});
    const [aiResponse, setAiResponse] = useState<string | null>(null);
    const [aiStreaming, setAiStreaming] = useState(false);
    const lastAiContentRef = useRef<string | null>(null);
    const [isComplete, setIsComplete] = useState(false);
    const [transitioning, setTransitioning] = useState(false);
    const [validationError, setValidationError] = useState<string | null>(null);
    const [confirmError, setConfirmError] = useState<string | null>(null);
    const [showAiPrompt, setShowAiPrompt] = useState(false);

    // AI chat history — persists within current step, cleared on step advance
    const [chatHistory, setChatHistory] = useState<{ question: string; answer: string }[]>([]);
    const [currentAiQuestion, setCurrentAiQuestion] = useState<string | null>(null);

    // Device auth
    const [deviceAuth, setDeviceAuth] = useState<DeviceAuthState>({
        status: "loading",
        userCode: null,
        verificationUri: null,
        token: null,
        email: null,
        errorMessage: null,
    });
    const devicePollRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const browserTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Marketplace
    const [gpuListings, setGpuListings] = useState<MarketplaceListing[]>([]);
    const [browseError, setBrowseError] = useState<string | null>(null);
    const [gpuOptions, setGpuOptions] = useState<{ label: string; value: string }[]>([]);
    const [imageOptions, setImageOptions] = useState<{ label: string; value: string }[]>([]);

    // Instance
    const [instanceInfo, setInstanceInfo] = useState<InstanceInfo | null>(null);

    // Payment
    const [paymentGate, setPaymentGate] = useState({
        balance: 0, required: 0, polling: false, billingUrl: `${API_BASE_URL}/dashboard/billing`,
    });
    const walletPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Auto-check retry
    const [checkCanRetry, setCheckCanRetry] = useState(false);
    const [checkAwaitContinue, setCheckAwaitContinue] = useState(false);
    const activeCheckRef = useRef<{ checkId: string; stepId: string } | null>(null);
    const lastAdvanceRef = useRef<number>(0);

    // Device auth — detected .env path for display
    const [deviceAuthEnvPath, setDeviceAuthEnvPath] = useState<string | null>(null);
    // Token save error — null means saved OK
    const [tokenSaveError, setTokenSaveError] = useState<string | null>(null);

    // AI conversation tracking
    const conversationIdRef = useRef<string | null>(null);
    const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
    const [aiToolCalls, setAiToolCalls] = useState<AiToolCall[]>([]);

    // Provider flow state
    const gpuInfoRef = useRef<GpuInfo | null>(null);
    const benchResultRef = useRef<BenchmarkResult | null>(null);
    const networkResultRef = useRef<NetworkBenchResult | null>(null);
    const versionChecksRef = useRef<VersionCheck[]>([]);
    const [providerSummary, setProviderSummary] = useState<ProviderSummaryData | null>(null);

    const step = WIZARD_STEPS[stepIndex];
    const apiToken = (answers["api-key"] as string) || "";
    const aiAvailable = !!apiToken;

    // Compute launch summary for confirm-launch step
    const launchSummary: string[] = [];
    if (step.id === "confirm-launch") {
        const pickedHost = answers["gpu-pick"] as string;
        const listing = gpuListings.find((l) => l.host_id === pickedHost);
        const image = answers["image-pick"] as string;
        if (listing) {
            launchSummary.push(`GPU: ${listing.gpu_model} · ${listing.vram_gb} GB`);
            launchSummary.push(`Host: ${listing.owner}`);
            launchSummary.push(`Rate: $${listing.price_per_hour.toFixed(2)}/hr CAD`);
        }
        if (image) {
            const tpl = IMAGE_TEMPLATES.find((t) => t.value === image);
            launchSummary.push(`Image: ${tpl?.label ?? image}`);
        }
    }

    // ── Device auth flow ─────────────────────────────────────────────

    const stopDevicePoll = useCallback(() => {
        if (browserTimeoutRef.current) {
            clearTimeout(browserTimeoutRef.current);
            browserTimeoutRef.current = null;
        }
        if (devicePollRef.current) {
            clearInterval(devicePollRef.current);
            devicePollRef.current = null;
        }
    }, []);

    /** Open browser immediately — cancels the 15s countdown timer */
    const openBrowserNow = useCallback(() => {
        if (browserTimeoutRef.current) {
            clearTimeout(browserTimeoutRef.current);
            browserTimeoutRef.current = null;
        }
        if (deviceAuth.verificationUri) {
            openBrowser(deviceAuth.verificationUri);
        }
    }, [deviceAuth.verificationUri]);

    const startDeviceAuth = useCallback(async () => {
        stopDevicePoll();
        setDeviceAuth({ status: "loading", userCode: null, verificationUri: null, token: null, email: null, errorMessage: null });
        setWizardState("thinking");

        try {
            const result = await requestDeviceCode(API_BASE_URL);
            setDeviceAuth({
                status: "waiting",
                userCode: result.user_code,
                verificationUri: result.verification_uri,
                token: null,
                email: null,
                errorMessage: null,
            });
            setWizardState("waiting");
            setWizardMessage("Enter the code shown below in your browser...");

            // Delay browser open by 15 seconds so user can see the code
            browserTimeoutRef.current = setTimeout(() => {
                openBrowser(result.verification_uri);
            }, 15_000);

            // Start polling
            let devicePollInFlight = false;
            const pollStartTime = Date.now();
            devicePollRef.current = setInterval(async () => {
                if (devicePollInFlight) return; // prevent overlapping poll iterations
                // Check expiry
                if (Date.now() - pollStartTime > DEVICE_CODE_EXPIRY_MS) {
                    stopDevicePoll();
                    setDeviceAuth((prev) => ({ ...prev, status: "error", errorMessage: "Device code expired — please retry" }));
                    setWizardState("error");
                    setWizardMessage("Device code expired — press Enter to retry");
                    return;
                }
                devicePollInFlight = true;
                try {
                    const tokenResult = await pollDeviceToken(API_BASE_URL, result.device_code);
                    if (tokenResult) {
                        stopDevicePoll();
                        const sessionToken = tokenResult.access_token;

                        // Generate a proper API key so it appears in dashboard Settings
                        let apiKey = sessionToken;
                        try {
                            const keyResult = await generateApiKey(API_BASE_URL, sessionToken, "CLI Wizard");
                            apiKey = keyResult.key;
                        } catch {
                            // Fall back to session token if key generation fails
                        }

                        // Save API key
                        const saveResult = await saveToken(apiKey);

                        // Get user profile
                        let email: string | null = null;
                        let customerId: string | null = null;
                        try {
                            const profile = await getMe(API_BASE_URL, sessionToken);
                            email = profile.email;
                            customerId = profile.customer_id;
                        } catch {
                            // profile fetch is best-effort
                        }

                        setDeviceAuth({
                            status: "authorized",
                            userCode: result.user_code,
                            verificationUri: result.verification_uri,
                            token: apiKey,
                            email,
                            errorMessage: null,
                        });

                        // Store in answers
                        const updated: Record<string, string | string[]> = {
                            ...answersRef.current,
                            "api-key": apiKey,
                            "device-auth": "authorized",
                        };
                        if (customerId) updated["_customer_id"] = customerId;
                        if (email) updated["_email"] = email;
                        answersRef.current = updated;
                        setAnswers(updated);

                        if (saveResult === true) {
                            setTokenSaveError(null);
                            setWizardState("excited");

                            // Write to project .env if detected
                            const fw = detectFramework();
                            if (fw) {
                                const envResult = await writeProjectEnv(fw.envPath, apiKey);
                                const displayPath = fw.envPath.replace(CONFIG_HOME, "~");
                                if (envResult === true) {
                                    setDeviceAuthEnvPath(displayPath);
                                    setWizardMessage(`Key saved to ~/.xcelsior/token.json and ${displayPath} — press Enter to continue`);
                                } else {
                                    setDeviceAuthEnvPath(null);
                                    setWizardMessage(`Key saved to ~/.xcelsior/token.json (could not write ${displayPath}: ${envResult}) — press Enter to continue`);
                                }
                            } else {
                                setDeviceAuthEnvPath(null);
                                setWizardMessage("Key saved to ~/.xcelsior/token.json — press Enter to continue");
                            }
                        } else {
                            setTokenSaveError(saveResult);
                            setWizardState("error");
                            setDeviceAuthEnvPath(null);
                            setWizardMessage(`Token save failed: ${saveResult}`);
                        }
                    }
                } catch (err) {
                    // Don't stop polling on transient errors — only on fatal ones
                    const msg = err instanceof Error ? err.message : "Auth failed";
                    // "expired_token" or "access_denied" are fatal
                    if (msg.includes("expired") || msg.includes("denied")) {
                        stopDevicePoll();
                        setDeviceAuth((prev) => ({ ...prev, status: "error", errorMessage: msg }));
                        setWizardState("error");
                        setWizardMessage("Authentication failed — press Enter to retry");
                    }
                    // Otherwise keep polling — user might not have entered code yet
                } finally {
                    devicePollInFlight = false;
                }
            }, DEVICE_POLL_MS);
        } catch (err) {
            const msg = err instanceof Error ? err.message : "Auth init failed";
            setDeviceAuth({ status: "error", userCode: null, verificationUri: null, token: null, email: null, errorMessage: msg });
            setWizardState("error");
            setWizardMessage("Authentication failed — press Enter to retry or m for manual");
        }
    }, [stopDevicePoll]);

    const switchToManualAuth = useCallback(() => {
        stopDevicePoll();
        setDeviceAuth((prev) => ({ ...prev, status: "manual" }));
        setWizardState("idle");
        setWizardMessage("Paste your API token below:");
    }, [stopDevicePoll]);

    const retryDeviceAuth = useCallback(() => {
        startDeviceAuth();
    }, [startDeviceAuth]);

    const submitManualToken = useCallback(async (token: string) => {
        const updated = { ...answersRef.current, "api-key": token, "device-auth": "manual" };
        answersRef.current = updated;
        setAnswers(updated);

        // Save token and verify
        const saveResult = await saveToken(token);

        setDeviceAuth((prev) => ({
            ...prev,
            status: "authorized",
            token,
        }));

        if (saveResult === true) {
            setTokenSaveError(null);
            setWizardState("success");

            // Write to project .env if detected
            const fw = detectFramework();
            if (fw) {
                const envResult = await writeProjectEnv(fw.envPath, token);
                if (envResult === true) {
                    setDeviceAuthEnvPath(fw.envPath);
                    setWizardMessage(`Token saved to ~/.xcelsior/token.json and ${fw.envPath} — press Enter to continue`);
                } else {
                    setDeviceAuthEnvPath(null);
                    setWizardMessage(`Token saved to ~/.xcelsior/token.json (could not write ${fw.envPath}: ${envResult}) — press Enter to continue`);
                }
            } else {
                setDeviceAuthEnvPath(null);
                setWizardMessage("Token saved to ~/.xcelsior/token.json — press Enter to continue");
            }
        } else {
            setTokenSaveError(saveResult);
            setWizardState("error");
            setDeviceAuthEnvPath(null);
            setWizardMessage(`Token save failed: ${saveResult}`);
        }
    }, []);

    // ── Marketplace browsing ─────────────────────────────────────────

    const browseGpus = useCallback(async (currentAnswers: Record<string, string | string[]>) => {
        setWizardState("thinking");
        setWizardMessage("Searching the marketplace...");
        setBrowseError(null);

        const token = currentAnswers["api-key"] as string;
        const workload = currentAnswers.workload as string;
        const preference = currentAnswers["gpu-preference"] as string;

        const filters: MarketplaceFilters = { limit: 20 };
        if (preference === "cheapest") filters.sort_by = "price";
        else if (preference === "best") filters.sort_by = "vram";
        else filters.sort_by = "score";
        if (workload === "training") filters.min_vram = 24;

        try {
            const result = await searchMarketplace(API_BASE_URL, token, filters);
            if (result.listings.length === 0) {
                setBrowseError("No GPUs available right now. Hexara suggests checking back shortly.");
                setWizardState("error");
                setWizardMessage("No GPUs found — press Enter to retry");
                return;
            }

            setGpuListings(result.listings);
            setGpuOptions(result.listings.map((l) => ({
                label: `${l.gpu_model} · ${l.vram_gb} GB · $${l.price_per_hour.toFixed(2)}/hr · ${l.owner}`,
                value: l.host_id,
            })));

            setWizardState("success");
            setWizardMessage(`Found ${result.listings.length} GPU(s)! Pick one:`);

            // Auto-advance to gpu-pick
            const updated = { ...currentAnswers, "browse-gpus": "done" };
            answersRef.current = updated;
            setAnswers(updated);
            setTimeout(() => {
                advanceToNext(updated);
            }, CHOREOGRAPHY_DELAY_MS);
        } catch (err) {
            const msg = err instanceof Error ? err.message : "Search failed";
            // Friendly error messages for common failures
            let friendlyMsg: string;
            if (msg.includes("401") || msg.includes("Unauthorized")) {
                friendlyMsg = "Authentication expired or invalid. Try restarting the wizard to re-authenticate.";
            } else if (msg.includes("403") || msg.includes("Forbidden")) {
                friendlyMsg = "Your account doesn't have marketplace access yet. Contact support.";
            } else if (msg.includes("ECONNREFUSED") || msg.includes("ENOTFOUND") || msg.includes("Connection")) {
                friendlyMsg = "Can't reach the marketplace — check your internet connection.";
            } else if (msg.includes("timeout") || msg.includes("ETIMEDOUT")) {
                friendlyMsg = "Marketplace request timed out — try again in a moment.";
            } else {
                friendlyMsg = `Marketplace error: ${msg}`;
            }
            setBrowseError(friendlyMsg);
            setWizardState("error");
            setWizardMessage("Marketplace unavailable — press Enter to retry");
        }
    }, []);

    // ── Wallet check ─────────────────────────────────────────────────

    const checkWallet = useCallback(async (currentAnswers: Record<string, string | string[]>): Promise<CheckResult[]> => {
        const token = currentAnswers["api-key"] as string;
        const customerId = currentAnswers["_customer_id"] as string;

        if (!customerId) {
            // Try to get customer ID
            try {
                const profile = await getMe(API_BASE_URL, token);
                const updated = { ...currentAnswers, "_customer_id": profile.customer_id };
                answersRef.current = updated;
                setAnswers(updated);
                return checkWalletInner(token, profile.customer_id, currentAnswers);
            } catch (err) {
                // Mark insufficient so payment-gate shows and user can retry after fixing auth
                const updated = { ...currentAnswers, "_wallet_insufficient": "true" };
                answersRef.current = updated;
                setAnswers(updated);
                return [{ name: "Wallet", ok: false, detail: "Could not fetch profile" }];
            }
        }

        return checkWalletInner(token, customerId, currentAnswers);
    }, []);

    const checkWalletInner = useCallback(async (
        token: string,
        customerId: string,
        currentAnswers: Record<string, string | string[]>,
    ): Promise<CheckResult[]> => {
        try {
            // Try claiming free credits first (idempotent, best-effort)
            let creditResult = { already_claimed: true, amount: 0 };
            try {
                creditResult = await claimFreeCredits(API_BASE_URL, token, customerId);
            } catch {
                // credit claim is best-effort — don't block wallet check
            }

            const wallet = await getWallet(API_BASE_URL, token, customerId);
            const balance = wallet.balance_cad;

            // Determine required rate
            const pickedHost = currentAnswers["gpu-pick"] as string;
            const listing = gpuListings.find((l) => l.host_id === pickedHost);
            const required = listing?.price_per_hour ?? 0;

            if (balance < required) {
                // Mark insufficient — payment-gate step will show
                const updated = { ...currentAnswers, "_wallet_insufficient": "true" };
                answersRef.current = updated;
                setAnswers(updated);
                setPaymentGate({
                    balance,
                    required,
                    polling: false,
                    billingUrl: `${API_BASE_URL}/dashboard/billing`,
                });
            } else {
                const updated = { ...currentAnswers, "_wallet_insufficient": "false" };
                answersRef.current = updated;
                setAnswers(updated);
            }

            const detail = creditResult.already_claimed
                ? `$${balance.toFixed(2)} CAD`
                : `$${balance.toFixed(2)} CAD (includes $10 welcome bonus!)`;

            return [{ name: "Wallet Balance", ok: balance >= required, detail }];
        } catch (err) {
            // Wallet check failed — mark insufficient so payment-gate shows
            const updated = { ...currentAnswers, "_wallet_insufficient": "true" };
            answersRef.current = updated;
            setAnswers(updated);
            return [{ name: "Wallet", ok: false, detail: err instanceof Error ? err.message : "Failed to check wallet" }];
        }
    }, [gpuListings]);

    // ── Instance launch ──────────────────────────────────────────────

    const launchGpuInstance = useCallback(async (currentAnswers: Record<string, string | string[]>): Promise<CheckResult[]> => {
        const token = currentAnswers["api-key"] as string;
        const hostId = currentAnswers["gpu-pick"] as string;
        const image = currentAnswers["image-pick"] as string;
        const listing = gpuListings.find((l) => l.host_id === hostId);

        try {
            const instance = await launchInstance(API_BASE_URL, token, {
                name: generateInstanceName(),
                host_id: hostId,
                image: image || "nvidia/cuda:12.4.1-devel-ubuntu22.04",
                interactive: true,
                vram_needed_gb: listing?.vram_gb ?? 0,
            });

            setInstanceInfo(instance);

            // Try to open dashboard
            const dashUrl = `${API_BASE_URL}/dashboard/instances/${instance.job_id}`;
            openBrowser(dashUrl).catch(() => { }); // best-effort

            return [{
                name: "Instance",
                ok: true,
                detail: `${instance.job_id} — ${instance.status}`,
            }];
        } catch (err) {
            return [{
                name: "Instance",
                ok: false,
                detail: err instanceof Error ? err.message : "Launch failed",
            }];
        }
    }, [gpuListings]);

    // ── Auto-check runner ────────────────────────────────────────────

    const runCheck = useCallback(async (
        checkId: string,
        currentAnswers: Record<string, string | string[]>,
    ): Promise<CheckResult[]> => {
        switch (checkId) {
            case "docker":
                return checkDocker();
            case "api":
                return checkApi(API_BASE_URL, currentAnswers["api-key"] as string || "");
            case "gpu": {
                // Full GPU detection — store result for provider flow
                const gpuFull = await detectGpuFull();
                if (gpuFull) {
                    gpuInfoRef.current = gpuFull;
                    return [{
                        name: "GPU Detection",
                        ok: true,
                        detail: `${gpuFull.gpu_model} · ${gpuFull.total_vram_gb} GB · Driver ${gpuFull.driver_version}`,
                    }];
                }
                // Fallback to basic detection — flag so benchmark skips gracefully
                const updated = { ...currentAnswers, "_gpu_basic_only": "true" };
                answersRef.current = updated;
                setAnswers(updated);
                return checkGpuBasic();
            }
            case "versions": {
                const results = await checkVersions();
                versionChecksRef.current = results;
                return results.map((v) => ({
                    name: v.component,
                    ok: v.passed,
                    detail: v.version
                        ? `v${v.version}${v.passed ? "" : ` — needs ≥${v.minimum}`}`
                        : `not found — needs ≥${v.minimum}`,
                }));
            }
            case "benchmark": {
                // Skip benchmark if only basic GPU detection passed (no detailed nvidia-smi data)
                if (currentAnswers["_gpu_basic_only"] === "true") {
                    return [{ name: "Benchmark", ok: true, detail: "Skipped — detailed GPU data unavailable (nvidia-smi query failed)" }];
                }
                const bench = await runComputeBenchmark();
                if (!bench || bench.error) {
                    const errorDetail = bench?.error === "no_torch" ? "PyTorch not installed"
                        : bench?.error === "no_cuda" ? "CUDA not available"
                            : bench?.error || "Failed — is Python 3 with PyTorch + CUDA installed?";
                    return [{ name: "Benchmark", ok: false, detail: errorDetail }];
                }
                benchResultRef.current = bench;
                const thermalMeasured = bench.gpu_temp_celsius > 0;
                return [
                    { name: "FP16 Matmul", ok: bench.tflops > 0, detail: `${bench.tflops} TFLOPS · XCU score: ${bench.xcu_score}` },
                    { name: "PCIe Bandwidth", ok: bench.pcie_bandwidth_gbps >= 8, detail: `${bench.pcie_bandwidth_gbps} GB/s (H2D: ${bench.pcie_h2d_gbps}, D2H: ${bench.pcie_d2h_gbps})` },
                    { name: "Thermal Stability", ok: !thermalMeasured || bench.gpu_temp_celsius <= 90, detail: thermalMeasured ? `Peak ${bench.gpu_temp_celsius}°C · Avg ${bench.gpu_temp_avg_celsius}°C (${bench.gpu_temp_samples} samples)` : "Temperature sensor unavailable — skipped" },
                ];
            }
            case "network": {
                const net = await runNetworkBenchmark(API_BASE_URL);
                networkResultRef.current = net;
                return [
                    { name: "Latency", ok: net.latency_avg_ms > 0, detail: `${net.latency_avg_ms}ms avg (${net.latency_min_ms}–${net.latency_max_ms}ms)` },
                    { name: "Jitter", ok: net.jitter_ms <= 50, detail: `${net.jitter_ms}ms` },
                    { name: "Packet Loss", ok: net.packet_loss_pct <= 2, detail: `${net.packet_loss_pct}%` },
                    { name: "Throughput", ok: net.throughput_mbps >= 100, detail: `${net.throughput_mbps} Mbps` },
                ];
            }
            case "verify": {
                const gpu = gpuInfoRef.current;
                const bench = benchResultRef.current;
                const net = networkResultRef.current;
                if (!gpu || !bench) {
                    return [{ name: "Verification", ok: false, detail: "Missing GPU or benchmark data — please retry previous steps" }];
                }
                // Use zeroed network data if network bench was skipped
                const netData = net || { latency_avg_ms: 0, latency_min_ms: 0, latency_max_ms: 0, jitter_ms: 0, packet_loss_pct: 0, throughput_mbps: 0 };
                const report = buildVerificationReport(gpu, bench, netData, versionChecksRef.current);
                // Submit to server
                const token = currentAnswers["api-key"] as string;
                const hostId = (currentAnswers["_host_id"] as string) || `host-${Date.now()}`;
                try {
                    const result = await reportVerification(API_BASE_URL, token, hostId, report as unknown as Record<string, unknown>);
                    // Store verification state
                    const updated = { ...currentAnswers, "_verification_state": result.state, "_verification_score": String(result.score) };
                    answersRef.current = updated;
                    setAnswers(updated);
                } catch {
                    // Server verification is best-effort during wizard
                }
                return report.checks.map((c) => ({
                    name: c.name,
                    ok: c.passed,
                    detail: c.detail,
                }));
            }
            case "host-register": {
                const gpu = gpuInfoRef.current;
                const bench = benchResultRef.current;
                if (!gpu) {
                    return [{ name: "Host Registration", ok: false, detail: "No GPU detected — please retry GPU detection" }];
                }
                const token = currentAnswers["api-key"] as string;
                const pricing = currentAnswers.pricing as string;
                const customRate = currentAnswers["custom-rate"] as string;

                // Compute cost based on marketplace data when possible
                let costPerHour = 0.20;
                if (pricing === "custom" && customRate) {
                    costPerHour = parseFloat(customRate);
                } else {
                    // Look up market rates for this GPU model
                    try {
                        const market = await searchMarketplace(API_BASE_URL, token, { gpu_model: gpu.gpu_model, limit: 20 });
                        if (market.listings.length > 0) {
                            const avgPrice = market.listings.reduce((sum, l) => sum + l.price_per_hour, 0) / market.listings.length;
                            costPerHour = pricing === "competitive" ? avgPrice * 0.85 : avgPrice;
                        } else {
                            costPerHour = pricing === "competitive" ? 0.15 : 0.20;
                        }
                    } catch {
                        // Fallback to defaults if marketplace unavailable
                        costPerHour = pricing === "competitive" ? 0.15 : 0.20;
                    }
                    // Round to 2 decimal places
                    costPerHour = Math.round(costPerHour * 100) / 100;
                }

                const hostId = `host-${Date.now()}`;
                const versions: Record<string, string> = {};
                for (const v of versionChecksRef.current) {
                    if (v.version) versions[v.component] = v.version;
                }

                try {
                    const host = await registerHost(API_BASE_URL, token, {
                        host_id: hostId,
                        ip: "auto",
                        gpu_model: gpu.gpu_model,
                        total_vram_gb: gpu.total_vram_gb,
                        free_vram_gb: gpu.free_vram_gb,
                        cost_per_hour: costPerHour,
                        versions,
                    });

                    // Store host ID and cost
                    const updated = { ...currentAnswers, "_host_id": host.host_id || hostId, "_host_cost_per_hour": String(costPerHour) };
                    answersRef.current = updated;
                    setAnswers(updated);

                    // Report benchmark if available
                    if (bench && bench.tflops > 0) {
                        try {
                            await reportBenchmark(
                                API_BASE_URL, token, host.host_id || hostId,
                                gpu.gpu_model, bench.xcu_score, bench.tflops,
                                { pcie_bandwidth_gbps: bench.pcie_bandwidth_gbps, gpu_temp_celsius: bench.gpu_temp_celsius },
                            );
                        } catch {
                            // benchmark report is best-effort
                        }
                    }

                    return [{ name: "Host Registration", ok: true, detail: `Registered as ${host.host_id || hostId} · ${gpu.gpu_model} · $${costPerHour.toFixed(2)}/hr` }];
                } catch (err) {
                    return [{ name: "Host Registration", ok: false, detail: err instanceof Error ? err.message : "Registration failed" }];
                }
            }
            case "admission": {
                const token = currentAnswers["api-key"] as string;
                const hostId = currentAnswers["_host_id"] as string;
                if (!hostId) {
                    return [{ name: "Admission", ok: false, detail: "Host not registered yet" }];
                }
                const versions: Record<string, string> = {};
                for (const v of versionChecksRef.current) {
                    if (v.version) versions[v.component] = v.version;
                }
                try {
                    const result = await reportVersions(API_BASE_URL, token, hostId, versions);
                    const admitted = result.admitted;
                    const runtime = (result.details as Record<string, string>)?.recommended_runtime || "runc";

                    // Build provider summary
                    const gpu = gpuInfoRef.current;
                    const bench = benchResultRef.current;
                    const verState = currentAnswers["_verification_state"] as string || "unknown";
                    const pricing = currentAnswers.pricing as string || "recommended";
                    const customRateVal = currentAnswers["custom-rate"] as string;

                    setProviderSummary({
                        gpuModel: gpu?.gpu_model ?? "Unknown",
                        vramGb: gpu?.total_vram_gb ?? 0,
                        xcuScore: bench?.xcu_score ?? 0,
                        tflops: bench?.tflops ?? 0,
                        verified: verState === "verified",
                        verificationState: verState,
                        hostId,
                        pricing,
                        customRate: customRateVal,
                        costPerHour: parseFloat(currentAnswers["_host_cost_per_hour"] as string || "0.20"),
                        admitted,
                        runtimeRecommendation: runtime,
                        reputationPoints: (result.details as Record<string, number>)?.reputation_points ?? 0,
                        tier: (result.details as Record<string, string>)?.tier ?? "Unranked",
                    });

                    return [
                        { name: "Admission", ok: admitted, detail: admitted ? "Admitted to the network" : "Not admitted — version requirements not met" },
                        { name: "Runtime", ok: true, detail: `Recommended: ${runtime}` },
                    ];
                } catch (err) {
                    return [{ name: "Admission", ok: false, detail: err instanceof Error ? err.message : "Admission check failed" }];
                }
            }
            case "wallet":
                return checkWallet(currentAnswers);
            case "launch":
                return launchGpuInstance(currentAnswers);
            case "network-setup": {
                try {
                    const { setupNetworking } = await import("./provider-checks.js");
                    const result = await setupNetworking();
                    // Store the detected IP for later use
                    currentAnswers["_host_ip"] = result.ip;
                    currentAnswers["_network_method"] = result.method;
                    return [
                        { name: "Mesh Network", ok: result.method !== "none", detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "Mesh Network", ok: false, detail: err instanceof Error ? err.message : "Network setup failed" }];
                }
            }
            case "worker-install": {
                try {
                    const { installWorkerAgent } = await import("./provider-checks.js");
                    const token = currentAnswers["api-key"] as string;
                    const hostId = currentAnswers["_host_id"] as string;
                    const hostIp = currentAnswers["_host_ip"] as string || "";
                    const result = await installWorkerAgent(API_BASE_URL, token, hostId, hostIp);
                    return [
                        { name: "Worker Agent", ok: result.installed, detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "Worker Agent", ok: false, detail: err instanceof Error ? err.message : "Worker install failed" }];
                }
            }
            case "ssh-key-setup": {
                try {
                    const { setupSshKeys } = await import("./provider-checks.js");
                    const token = currentAnswers["api-key"] as string;
                    const result = await setupSshKeys(API_BASE_URL, token);
                    return [
                        { name: "SSH Keys", ok: result.keyFound, detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "SSH Keys", ok: false, detail: err instanceof Error ? err.message : "SSH key setup failed" }];
                }
            }
            default:
                return [{ name: checkId, ok: false, detail: "Unknown check" }];
        }
    }, [checkWallet, launchGpuInstance]);

    // ── Step advancement ─────────────────────────────────────────────

    const advanceToNext = useCallback(
        async (currentAnswers: Record<string, string | string[]>) => {
            // Debounce rapid Enter presses — prevent double-advancing
            const now = Date.now();
            if (now - lastAdvanceRef.current < ADVANCE_DEBOUNCE_MS) return;
            lastAdvanceRef.current = now;

            setTransitioning(false);
            setValidationError(null);
            setConfirmError(null);
            setCheckCanRetry(false);
            setCheckAwaitContinue(false);
            setShowAiPrompt(false);
            setChatHistory([]);
            setCurrentAiQuestion(null);

            const next = getNextStep(stepIndexRef.current, currentAnswers);
            if (next === -1 || WIZARD_STEPS[next].type === "done") {
                // Find the done step
                const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                if (doneIdx >= 0) {
                    stepIndexRef.current = doneIdx;
                    setStepIndex(doneIdx);
                    setWizardState("finishing");
                    setWizardMessage(WIZARD_STEPS[doneIdx].prompt);

                    // Detect project framework
                    detectFramework();

                    // Save config
                    saveConfig(currentAnswers).catch((err) => {
                        setWizardMessage((prev) => `${prev}\n⚠ Could not save config: ${err instanceof Error ? err.message : "unknown error"}`);
                    });
                }
                setIsComplete(true);
                return;
            }

            const nextStep = WIZARD_STEPS[next];
            stepIndexRef.current = next;
            setStepIndex(next);

            // Brief pause so the user can read the step prompt before init kicks in
            const needsInit = nextStep.type === "device-auth"
                || nextStep.type === "auto-check"
                || nextStep.type === "auto-fetch"
                || nextStep.type === "payment-gate";
            if (needsInit) {
                setWizardMessage(nextStep.prompt);
                await new Promise((r) => setTimeout(r, CHOREOGRAPHY_DELAY_MS));
            } else {
                setWizardMessage(nextStep.prompt);
            }

            // ── Handle step type-specific init ───────────────────────
            if (nextStep.type === "device-auth") {
                startDeviceAuth();
                return;
            }

            if (nextStep.type === "auto-fetch") {
                // Browse GPUs
                browseGpus(currentAnswers);
                return;
            }

            if (nextStep.type === "payment-gate") {
                // Start polling wallet
                setWizardState("waiting");
                setPaymentGate((prev) => ({ ...prev, polling: true }));

                // Open billing page
                openBrowser(paymentGate.billingUrl).catch(() => { });

                let walletPollFailures = 0;
                walletPollRef.current = setInterval(async () => {
                    const customerId = currentAnswers["_customer_id"] as string;
                    if (!customerId) return;
                    try {
                        const wallet = await getWallet(API_BASE_URL, currentAnswers["api-key"] as string, customerId);
                        walletPollFailures = 0; // reset on success
                        setPaymentGate((prev) => ({ ...prev, balance: wallet.balance_cad }));
                        const listing = gpuListings.find((l) => l.host_id === (currentAnswers["gpu-pick"] as string));
                        if (wallet.balance_cad >= (listing?.price_per_hour ?? 0)) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            const updated = { ...currentAnswers, "_wallet_insufficient": "false", "payment-gate": "funded" };
                            answersRef.current = updated;
                            setAnswers(updated);
                            setWizardState("excited");
                            setWizardMessage("Wallet funded! Proceeding...");
                            setTimeout(() => advanceToNext(updated), CHOREOGRAPHY_DELAY_MS);
                        }
                    } catch {
                        walletPollFailures++;
                        if (walletPollFailures >= 10) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            setWizardState("error");
                            setWizardMessage("Wallet polling failed repeatedly — press s to skip");
                        }
                    }
                }, WALLET_POLL_MS);
                return;
            }

            if (nextStep.id === "gpu-pick") {
                // Populate options from listings
                setWizardState("excited");
                return;
            }

            if (nextStep.id === "image-pick") {
                // Populate image options based on workload
                const workload = (currentAnswers.workload as string) || "other";
                const defaultImage = WORKLOAD_IMAGE_MAP[workload] ?? WORKLOAD_IMAGE_MAP.other;
                const opts = IMAGE_TEMPLATES.map((t) => ({
                    label: t.value === defaultImage ? `${t.label} ← recommended` : t.label,
                    value: t.value,
                }));
                setImageOptions(opts);
                setWizardState("idle");
                return;
            }

            if (nextStep.type === "auto-check" && nextStep.checkId) {
                setWizardState("thinking");

                // Special messages for long-running provider checks
                if (nextStep.checkId === "benchmark") {
                    setWizardMessage("Running GPU benchmarks — this takes about 60 seconds...");
                } else if (nextStep.checkId === "verify") {
                    setWizardMessage("Running 7-point hardware verification...");
                }

                activeCheckRef.current = { checkId: nextStep.checkId, stepId: nextStep.id };

                runCheck(nextStep.checkId, currentAnswers).then((results) => {
                    const allPassed = results.every((r) => r.ok);
                    setCheckResults((prev) => ({
                        ...prev,
                        [nextStep.id]: { items: results, allPassed },
                    }));

                    if (allPassed) {
                        // Use "excited" (dance) for big milestones, "success" (eureka) for routine
                        const isMilestone = nextStep.checkId === "launch" || nextStep.checkId === "verify" || nextStep.checkId === "host-register" || nextStep.checkId === "docker";
                        setWizardState(isMilestone ? "excited" : "success");
                        const successMsg = nextStep.checkId === "launch" ? "Instance launched!"
                            : nextStep.checkId === "benchmark" ? "Benchmarks complete!"
                                : nextStep.checkId === "verify" ? "Hardware verified!"
                                    : nextStep.checkId === "host-register" ? "Host registered on the marketplace!"
                                        : nextStep.checkId === "docker" ? "Docker environment ready!"
                                            : "All checks passed!";
                        setWizardMessage(successMsg);

                        // Wait for Enter to continue
                        setCheckAwaitContinue(true);
                    } else {
                        const failCount = results.filter((r) => !r.ok).length;
                        const failDetails = results
                            .filter((r) => !r.ok)
                            .map((r) => `${r.name}: ${r.detail}`)
                            .join("; ");
                        setWizardState("error");
                        setWizardMessage(
                            apiToken
                                ? `${failCount} check(s) failed — Hexara is looking into it...`
                                : `${failCount} check(s) failed — retry or skip`,
                        );
                        setCheckCanRetry(true);

                        // Auto-trigger AI analysis for check failures if authenticated
                        // (skip api-check — can't reach AI if API is down)
                        if (apiToken && nextStep.checkId && nextStep.checkId !== "api") {
                            const pageCtx = buildWizardContext(
                                nextStep.id, currentAnswers, {
                                ...checkResults,
                                [nextStep.id]: { items: results, allPassed: false },
                            }, providerSummary, gpuListings, browseError,
                                gpuInfoRef.current, benchResultRef.current, networkResultRef.current,
                            );
                            const config: ApiClientConfig = {
                                baseUrl: API_BASE_URL,
                                apiKey: apiToken,
                                pageContext: pageCtx,
                            };
                            // Buffer tokens — show spinner, then reveal complete result
                            setAiStreaming(true);
                            setAiResponse(null);  // Keep panel hidden during analysis
                            setCurrentAiQuestion(null);
                            setWizardState("thinking");
                            setWizardMessage(`Hexara is analyzing ${failCount} issue(s)...`);
                            void (async () => {
                                let explanation = "";
                                try {
                                    for await (const event of streamChat(config,
                                        `The following checks failed during provider setup: ${failDetails}. ` +
                                        `Diagnose each failure and give the exact commands to fix it.`,
                                        conversationIdRef.current ?? undefined,
                                    )) {
                                        if (event.type === "meta" && event.conversation_id) {
                                            conversationIdRef.current = event.conversation_id;
                                        } else if (event.type === "token") {
                                            explanation += event.content ?? "";
                                        } else if (event.type === "tool_call" && event.name) {
                                            setWizardMessage(`Using ${event.name}...`);
                                        } else if (event.type === "tool_result") {
                                            setWizardMessage(`Hexara is analyzing ${failCount} issue(s)...`);
                                        }
                                    }
                                    // Stream complete — reveal full analysis
                                    if (explanation) {
                                        setAiResponse(explanation);
                                        setWizardState("error");
                                        setWizardMessage(`${failCount} issue(s) found — see analysis below`);
                                    }
                                } catch (err) {
                                    const msg = err instanceof Error ? err.message : "unknown";
                                    setAiResponse(explanation || `Analysis failed: ${msg}`);
                                    setWizardState("error");
                                    setWizardMessage(`${failCount} check(s) failed`);
                                } finally {
                                    setAiStreaming(false);
                                }
                            })();
                        }
                    }
                }).catch(() => {
                    setWizardState("error");
                    setWizardMessage("Check failed unexpectedly — retry");
                    setCheckCanRetry(true);
                });
                return;
            }

            if (nextStep.type === "confirm") {
                // Show excited state for provider summary
                setWizardState(nextStep.id === "provider-summary" ? "success" : "idle");
                return;
            }

            // Default — set appropriate state
            setWizardState("idle");
        },
        [startDeviceAuth, browseGpus, runCheck, gpuListings, paymentGate.billingUrl],
    );

    // ── Submit answer ────────────────────────────────────────────────

    const submitAnswer = useCallback(
        (value: string | string[]) => {
            const currentStep = WIZARD_STEPS[stepIndexRef.current];

            // Auto-fetch retry — re-trigger browse instead of advancing
            if (currentStep.type === "auto-fetch" && value === "retry") {
                browseGpus(answersRef.current);
                return;
            }

            // Confirm step validation — only y/n allowed
            if (currentStep.type === "confirm") {
                if (value !== "yes" && value !== "no") {
                    setConfirmError("Press y to confirm or n to cancel");
                    setWizardState("error");
                    setTimeout(() => setWizardState("idle"), CHOREOGRAPHY_DELAY_MS);
                    return;
                }
                setConfirmError(null);
                if (value === "no") {
                    // On cancel, skip the rest of the renter launch flow
                    if (currentStep.id === "confirm-launch") {
                        const updated = { ...answersRef.current, [currentStep.id]: "cancelled" };
                        answersRef.current = updated;
                        setAnswers(updated);
                        // Jump to done without saving config (user cancelled)
                        const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                        if (doneIdx >= 0) {
                            stepIndexRef.current = doneIdx;
                            setStepIndex(doneIdx);
                            setWizardState("success");
                            setWizardMessage(WIZARD_STEPS[doneIdx].prompt);
                            setIsComplete(true);
                        }
                        return;
                    }
                    // On provider flow cancel, jump to done without saving config
                    if (currentStep.id === "provider-summary" || currentStep.id === "confirm-setup") {
                        const updated = { ...answersRef.current, [currentStep.id]: "cancelled" };
                        answersRef.current = updated;
                        setAnswers(updated);
                        const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                        if (doneIdx >= 0) {
                            stepIndexRef.current = doneIdx;
                            setStepIndex(doneIdx);
                            setWizardState("success");
                            setWizardMessage("Setup cancelled. Run the wizard again when you're ready.");
                            setIsComplete(true);
                        }
                        return;
                    }
                }
            }

            // Text step validation
            if (currentStep.type === "text" && currentStep.validate && typeof value === "string") {
                const error = currentStep.validate(value);
                if (error) {
                    setValidationError(error);
                    setWizardState("error");
                    setWizardMessage(error);
                    setTimeout(() => {
                        setWizardState("idle");
                        setWizardMessage(currentStep.prompt);
                    }, CHOREOGRAPHY_DELAY_MS);
                    return;
                }
                setValidationError(null);
            }

            // Choreography: show a transition message, pause, then advance
            const stepMessages: Record<string, string> = {
                "mode": "Great choice! Let's get you set up...",
                "pricing": "Got it! Setting your rate...",
                "custom-rate": "Rate locked in!",
                "workload": "Great pick! Finding the best GPUs for you...",
                "gpu-preference": "Noted! Searching available options...",
                "gpu-pick": "Excellent choice!",
                "image-pick": "Environment selected!",
                "confirm-launch": "Launching your instance...",
                "confirm-setup": "Saving your configuration...",
                "provider-summary": "Onward!",
            };
            const transitionMsg = stepMessages[currentStep.id];
            setTransitioning(true);
            if (transitionMsg) {
                setWizardState("excited");
                setWizardMessage(transitionMsg);
            }

            const updated = { ...answersRef.current, [currentStep.id]: value };
            answersRef.current = updated;
            setAnswers(updated);

            setTimeout(() => advanceToNext(updated), CHOREOGRAPHY_DELAY_MS);
        },
        [advanceToNext],
    );

    // ── Check retry/skip ─────────────────────────────────────────────

    const retryCheck = useCallback(() => {
        if (!activeCheckRef.current) return;
        setCheckCanRetry(false);
        setCheckAwaitContinue(false);
        const { checkId, stepId } = activeCheckRef.current;
        setWizardState("thinking");
        setWizardMessage("Retrying...");

        runCheck(checkId, answersRef.current).then((results) => {
            const allPassed = results.every((r) => r.ok);
            setCheckResults((prev) => ({
                ...prev,
                [stepId]: { items: results, allPassed },
            }));

            if (allPassed) {
                setWizardState("success");
                setWizardMessage("All checks passed!");
                setCheckAwaitContinue(true);
            } else {
                const failCount = results.filter((r) => !r.ok).length;
                const failDetails = results
                    .filter((r) => !r.ok)
                    .map((r) => `${r.name}: ${r.detail}`)
                    .join("; ");
                setWizardState("error");
                setWizardMessage(
                    apiToken
                        ? `${failCount} check(s) still failing — Hexara is re-analyzing...`
                        : `${failCount} check(s) failed`,
                );
                setCheckCanRetry(true);

                // Auto-trigger AI re-analysis on retry failure
                if (apiToken) {
                    const pageCtx = buildWizardContext(
                        stepId, answersRef.current, {
                        ...checkResults,
                        [stepId]: { items: results, allPassed: false },
                    }, providerSummary, gpuListings, browseError,
                        gpuInfoRef.current, benchResultRef.current, networkResultRef.current,
                    );
                    const config: ApiClientConfig = {
                        baseUrl: API_BASE_URL,
                        apiKey: apiToken,
                        pageContext: pageCtx,
                    };
                    setAiStreaming(true);
                    setAiResponse(null);
                    setCurrentAiQuestion(null);
                    setWizardState("thinking");
                    setWizardMessage(`Hexara is re-analyzing ${failCount} issue(s)...`);
                    void (async () => {
                        let explanation = "";
                        try {
                            for await (const event of streamChat(config,
                                `Retry attempt: these checks are still failing: ${failDetails}. ` +
                                `The user already tried fixing them. Dig deeper — suggest alternative solutions.`,
                                conversationIdRef.current ?? undefined,
                            )) {
                                if (event.type === "meta" && event.conversation_id) {
                                    conversationIdRef.current = event.conversation_id;
                                } else if (event.type === "token") {
                                    explanation += event.content ?? "";
                                } else if (event.type === "tool_call" && event.name) {
                                    setWizardMessage(`Using ${event.name}...`);
                                } else if (event.type === "tool_result") {
                                    setWizardMessage(`Hexara is re-analyzing ${failCount} issue(s)...`);
                                }
                            }
                            if (explanation) {
                                setAiResponse(explanation);
                                setWizardState("error");
                                setWizardMessage(`${failCount} issue(s) persist — see analysis below`);
                            }
                        } catch (err) {
                            const msg = err instanceof Error ? err.message : "unknown";
                            setAiResponse(explanation || `Re-analysis failed: ${msg}`);
                            setWizardState("error");
                            setWizardMessage(`${failCount} check(s) failed`);
                        } finally {
                            setAiStreaming(false);
                        }
                    })();
                }
            }
        }).catch(() => {
            setWizardState("error");
            setWizardMessage("Check failed unexpectedly — retry");
            setCheckCanRetry(true);
        });
    }, [runCheck, advanceToNext, apiToken, checkResults, providerSummary, gpuListings, browseError]);

    const skipCheck = useCallback(() => {
        if (!activeCheckRef.current) return;
        const { stepId } = activeCheckRef.current;
        const currentStep = WIZARD_STEPS[stepIndexRef.current];
        if (currentStep.checkRequired) return; // can't skip required checks

        setCheckCanRetry(false);
        const updated = { ...answersRef.current, [stepId]: "skipped" };
        answersRef.current = updated;
        setAnswers(updated);
        advanceToNext(updated);
    }, [advanceToNext]);

    const continueFromCheck = useCallback(() => {
        if (!activeCheckRef.current || !checkAwaitContinue) return;
        const { stepId } = activeCheckRef.current;
        setCheckAwaitContinue(false);
        const updated = { ...answersRef.current, [stepId]: "passed" };
        answersRef.current = updated;
        setAnswers(updated);
        advanceToNext(updated);
    }, [advanceToNext, checkAwaitContinue]);

    // ── Device-auth continue (Enter after authorized) ────────────────

    const continueFromAuth = useCallback(() => {
        if (deviceAuth.status !== "authorized") return;
        advanceToNext(answersRef.current);
    }, [advanceToNext, deviceAuth.status]);

    // ── Payment skip ─────────────────────────────────────────────────

    const skipPayment = useCallback(() => {
        if (walletPollRef.current) {
            clearInterval(walletPollRef.current);
            walletPollRef.current = null;
        }
        const updated = { ...answersRef.current, "payment-gate": "skipped", "_wallet_insufficient": "false" };
        answersRef.current = updated;
        setAnswers(updated);
        setWizardState("idle");
        advanceToNext(updated);
    }, [advanceToNext]);

    // ── AI escape hatch ──────────────────────────────────────────────

    const askAi = useCallback(
        async (question: string) => {
            if (!apiToken) return;

            const pageContext = buildWizardContext(
                step.id, answersRef.current, checkResults, providerSummary, gpuListings, browseError,
                gpuInfoRef.current, benchResultRef.current, networkResultRef.current,
            );

            const config: ApiClientConfig = {
                baseUrl: API_BASE_URL,
                apiKey: apiToken,
                pageContext,
            };

            setAiStreaming(true);
            setAiResponse(null);  // Keep panel hidden — reveal when complete
            lastAiContentRef.current = null;
            setCurrentAiQuestion(question);
            setWizardState("thinking");
            setWizardMessage("Hexara is thinking...");
            setShowAiPrompt(false);
            setPendingConfirmation(null);
            setAiToolCalls([]);

            let content = "";
            const toolCalls: AiToolCall[] = [];
            let hadConfirmation = false;

            try {
                for await (const event of streamChat(config, question, conversationIdRef.current ?? undefined)) {
                    switch (event.type) {
                        case "meta":
                            if (event.conversation_id) {
                                conversationIdRef.current = event.conversation_id;
                            }
                            break;

                        case "token":
                            content += event.content ?? "";
                            // Tokens buffered — not revealed until stream completes
                            break;

                        case "tool_call":
                            if (event.name) {
                                const call: AiToolCall = { name: event.name, input: event.input ?? {} };
                                toolCalls.push(call);
                                setAiToolCalls([...toolCalls]);
                                setWizardMessage(`Using ${event.name}...`);
                            }
                            break;

                        case "tool_result":
                            if (event.name) {
                                const existing = toolCalls.find((tc) => tc.name === event.name && !tc.output);
                                if (existing) existing.output = event.output ?? {};
                                setAiToolCalls([...toolCalls]);
                                setWizardMessage("Hexara is thinking...");
                            }
                            break;

                        case "confirmation_required":
                            if (event.confirmation_id && event.tool_name) {
                                hadConfirmation = true;
                                // Reveal buffered content so far for confirmation context
                                if (content) setAiResponse(content);
                                setPendingConfirmation({
                                    confirmationId: event.confirmation_id,
                                    toolName: event.tool_name,
                                    toolArgs: event.tool_args ?? {},
                                });
                                setWizardState("idle");
                                setWizardMessage(`Hexara wants to run: ${event.tool_name} — press y/n`);
                            }
                            break;

                        case "error":
                            content += content ? `\n\nError: ${event.message}` : `Error: ${event.message}`;
                            break;

                        case "done":
                            break;
                    }
                }

                // Stream complete — signal outcome, keep response hidden but buffered
                if (!hadConfirmation) {
                    lastAiContentRef.current = content || null;
                    if (content) {
                        setWizardState("success");
                        setWizardMessage("Done — press d to see details");
                    } else {
                        setWizardState("idle");
                        setWizardMessage(step.prompt);
                    }
                }
            } catch (err) {
                lastAiContentRef.current = content || null;
                setWizardState("error");
                setWizardMessage(content ? "AI error — press d to see details" : "AI unavailable — continue with the wizard steps.");
            } finally {
                setAiStreaming(false);
            }
        },
        [apiToken, step, checkResults, providerSummary, gpuListings, browseError],
    );

    const confirmAi = useCallback(async (approved: boolean) => {
        if (!pendingConfirmation || !apiToken) return;

        const config: ApiClientConfig = {
            baseUrl: API_BASE_URL,
            apiKey: apiToken,
            pageContext: `cli-wizard:${step.id}`,
        };

        setWizardState("thinking");
        setWizardMessage(approved ? "Executing..." : "Cancelled.");
        setAiStreaming(true);

        let content = aiResponse ?? "";
        try {
            for await (const event of confirmAction(config, pendingConfirmation.confirmationId, approved)) {
                if (event.type === "token") {
                    content += event.content ?? "";
                    setAiResponse(content);
                } else if (event.type === "error") {
                    content += `\n\nError: ${event.message}`;
                    setAiResponse(content);
                }
            }
            setWizardState("idle");
            setWizardMessage(step.prompt);
        } catch (err) {
            content += `\n\nConfirmation error: ${err instanceof Error ? err.message : "unknown"}`;
            setAiResponse(content);
            setWizardState("error");
        } finally {
            setPendingConfirmation(null);
            setAiStreaming(false);
        }
    }, [pendingConfirmation, apiToken, step, aiResponse]);

    const dismissAi = useCallback(() => {
        // Save current Q&A to chat history before dismissing
        if (aiResponse && currentAiQuestion) {
            setChatHistory((prev) => [...prev, { question: currentAiQuestion, answer: aiResponse }]);
        }
        setAiResponse(null);
        lastAiContentRef.current = null;
        setCurrentAiQuestion(null);
        setWizardState("idle");
        setWizardMessage(step.prompt);
    }, [step, aiResponse, currentAiQuestion]);

    const toggleAiPrompt = useCallback(() => {
        if (!aiAvailable) return;
        setShowAiPrompt((prev) => !prev);
    }, [aiAvailable]);

    // Cleanup polls on unmount
    useEffect(() => {
        return () => {
            if (browserTimeoutRef.current) clearTimeout(browserTimeoutRef.current);
            if (devicePollRef.current) clearInterval(devicePollRef.current);
            if (walletPollRef.current) clearInterval(walletPollRef.current);
        };
    }, []);

    return {
        step,
        stepIndex,
        answers,
        wizardState,
        wizardMessage,
        checkResults,
        aiResponse,
        aiStreaming,
        submitAnswer,
        askAi,
        dismissAi,
        isComplete,
        deviceAuth,
        switchToManualAuth,
        retryDeviceAuth,
        openBrowserNow,
        submitManualToken,
        gpuListings,
        gpuOptions,
        imageOptions,
        instanceInfo,
        validationError,
        confirmError,
        launchSummary,
        paymentGate,
        skipPayment,
        browseError,
        checkCanRetry,
        checkAwaitContinue,
        retryCheck,
        skipCheck,
        continueFromCheck,
        continueFromAuth,
        tokenSaveError,
        deviceAuthEnvPath,
        hasAiDetails: lastAiContentRef.current !== null && aiResponse === null,
        revealAi: useCallback(() => {
            if (lastAiContentRef.current) {
                setAiResponse(lastAiContentRef.current);
                setWizardState("idle");
                setWizardMessage("Hexara's response");
            }
        }, []),
        aiAvailable,
        showAiPrompt,
        toggleAiPrompt,
        chatHistory,
        currentAiQuestion,
        providerSummary,
        pendingConfirmation,
        confirmAi,
        aiToolCalls,
        transitioning,
    };
}
