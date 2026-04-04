// useWizardFlow — Hook that drives the structured wizard step-by-step.
// Manages step progression, answer collection, auto-checks, device auth,
// marketplace browsing, payment gating, instance launch, and AI escape hatch.

import { useState, useCallback, useRef, useEffect } from "react";
import { WIZARD_STEPS, getNextStep, type WizardStep, IMAGE_TEMPLATES, WORKLOAD_IMAGE_MAP } from "./wizard-flow.js";
import {
    streamChat, type ApiClientConfig,
    requestDeviceCode, pollDeviceToken, type DeviceCodeResult,
    getMe, searchMarketplace, type MarketplaceListing, type MarketplaceFilters,
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
const TOKEN_PATH = "~/.xcelsior/token.json";
const CONFIG_HOME = process.env["HOME"] ?? "/tmp";
const TOKEN_FILE = `${CONFIG_HOME}/.xcelsior/token.json`;
const CONFIG_FILE = `${CONFIG_HOME}/.xcelsior/config.toml`;
const DEVICE_POLL_MS = 5_000;
const WALLET_POLL_MS = 5_000;
const CHOREOGRAPHY_DELAY_MS = 800;

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
    /** Retry failed auto-check */
    retryCheck: () => void;
    /** Skip failed auto-check */
    skipCheck: () => void;
    /** Whether Hexara AI is available (after auth) */
    aiAvailable: boolean;
    /** Whether inline AI prompt is showing (for non-text steps) */
    showAiPrompt: boolean;
    /** Toggle AI prompt */
    toggleAiPrompt: () => void;
    /** Detected project framework */
    detectedFramework: string | null;
    /** Chat history for current step (Q&A pairs) */
    chatHistory: { question: string; answer: string }[];
    /** Current question being answered by Hexara */
    currentAiQuestion: string | null;
    /** Provider summary data for provider-summary step */
    providerSummary: ProviderSummaryData | null;
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
    admitted: boolean;
    runtimeRecommendation: string;
    reputationPoints: number;
    tier: string;
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

/** Save token to ~/.xcelsior/token.json (0o600 perms) */
async function saveToken(token: string): Promise<void> {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const dir = path.dirname(TOKEN_FILE);
    fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
    fs.writeFileSync(TOKEN_FILE, JSON.stringify({ access_token: token }, null, 2), {
        mode: 0o600,
    });
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
    if ((answers.mode === "provide" || answers.mode === "both") && answers["_host_id"]) {
        const envFile = `${CONFIG_HOME}/.xcelsior/.env`;
        const envLines = [
            `# Xcelsior worker environment — generated by setup wizard`,
            `XCELSIOR_HOST_ID=${answers["_host_id"]}`,
            `XCELSIOR_SCHEDULER_URL=${API_BASE_URL}`,
            `XCELSIOR_API_TOKEN=${answers["api-key"] || ""}`,
        ];
        if (answers["custom-rate"]) envLines.push(`XCELSIOR_COST_PER_HOUR=${answers["custom-rate"]}`);
        fs.writeFileSync(envFile, envLines.join("\n") + "\n", { mode: 0o600 });
    }
}

/** Detect project framework in cwd */
function detectFramework(): { name: string; envPath: string } | null {
    try {
        const fs = require("node:fs");
        if (fs.existsSync("package.json")) {
            const pkg = JSON.parse(fs.readFileSync("package.json", "utf-8"));
            const deps = { ...pkg.dependencies, ...pkg.devDependencies };
            if (deps["next"]) return { name: "Next.js", envPath: ".env.local" };
            if (deps["react"]) return { name: "React", envPath: ".env" };
            if (deps["vue"]) return { name: "Vue", envPath: ".env" };
            if (deps["svelte"] || deps["@sveltejs/kit"]) return { name: "SvelteKit", envPath: ".env" };
            return { name: "Node.js", envPath: ".env" };
        }
        if (fs.existsSync("requirements.txt") || fs.existsSync("pyproject.toml")) {
            return { name: "Python", envPath: ".env" };
        }
        if (fs.existsSync("Cargo.toml")) return { name: "Rust", envPath: ".env" };
        if (fs.existsSync("go.mod")) return { name: "Go", envPath: ".env" };
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

/** Generate a memorable instance name */
function generateInstanceName(): string {
    const adj = ["swift", "bright", "cosmic", "nova", "stellar", "quantum", "astral", "blazing"];
    const noun = ["forge", "nexus", "pulse", "flux", "spark", "core", "beam", "arc"];
    const pick = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
    return `${pick(adj)}-${pick(noun)}-${Math.floor(Math.random() * 1000)}`;
}

// ── Hook ─────────────────────────────────────────────────────────────

export function useWizardFlow(): UseWizardFlowReturn {
    const [stepIndex, setStepIndex] = useState(0);
    const stepIndexRef = useRef(0);
    const [answers, setAnswers] = useState<Record<string, string | string[]>>({});
    const answersRef = useRef<Record<string, string | string[]>>({});
    const [wizardState, setWizardState] = useState<WizardState>("idle");
    const [wizardMessage, setWizardMessage] = useState(WIZARD_STEPS[0].prompt);
    const [checkResults, setCheckResults] = useState<Record<string, AutoCheckResults>>({});
    const [aiResponse, setAiResponse] = useState<string | null>(null);
    const [aiStreaming, setAiStreaming] = useState(false);
    const [isComplete, setIsComplete] = useState(false);
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
    const activeCheckRef = useRef<{ checkId: string; stepId: string } | null>(null);

    // Project detection
    const [detectedFramework, setDetectedFramework] = useState<string | null>(null);

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
        if (devicePollRef.current) {
            clearInterval(devicePollRef.current);
            devicePollRef.current = null;
        }
    }, []);

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

            // Delay browser open by 10 seconds so user can see the code
            setTimeout(() => {
                openBrowser(result.verification_uri);
            }, 10_000);

            // Start polling
            devicePollRef.current = setInterval(async () => {
                try {
                    const tokenResult = await pollDeviceToken(API_BASE_URL, result.device_code);
                    if (tokenResult) {
                        stopDevicePoll();
                        const token = tokenResult.access_token;

                        // Save token
                        await saveToken(token);

                        // Get user profile
                        let email: string | null = null;
                        let customerId: string | null = null;
                        try {
                            const profile = await getMe(API_BASE_URL, token);
                            email = profile.email;
                            customerId = profile.customer_id;
                        } catch {
                            // profile fetch is best-effort
                        }

                        setDeviceAuth({
                            status: "authorized",
                            userCode: result.user_code,
                            verificationUri: result.verification_uri,
                            token,
                            email,
                            errorMessage: null,
                        });

                        // Store in answers
                        const updated: Record<string, string | string[]> = {
                            ...answersRef.current,
                            "api-key": token,
                            "device-auth": "authorized",
                        };
                        if (customerId) updated["_customer_id"] = customerId;
                        if (email) updated["_email"] = email;
                        answersRef.current = updated;
                        setAnswers(updated);

                        setWizardState("success");
                        setWizardMessage("Authenticated! Moving on...");

                        // Auto-advance after showing success
                        setTimeout(() => {
                            advanceToNext(updated);
                        }, CHOREOGRAPHY_DELAY_MS * 2);
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

    const submitManualToken = useCallback((token: string) => {
        const updated = { ...answersRef.current, "api-key": token, "device-auth": "manual" };
        answersRef.current = updated;
        setAnswers(updated);

        // Save token
        saveToken(token);

        setDeviceAuth((prev) => ({
            ...prev,
            status: "authorized",
            token,
        }));
        setWizardState("success");
        setWizardMessage("Token saved! Verifying connection...");

        setTimeout(() => {
            advanceToNext(updated);
        }, CHOREOGRAPHY_DELAY_MS);
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
            // Try claiming free credits first (idempotent)
            const creditResult = await claimFreeCredits(API_BASE_URL, token, customerId);

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
            return [{ name: "Wallet", ok: false, detail: err instanceof Error ? err.message : "Failed" }];
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
                // Fallback to basic detection
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
                const bench = await runComputeBenchmark();
                if (!bench) {
                    return [{ name: "Benchmark", ok: false, detail: "Failed — is Python 3 with PyTorch + CUDA installed?" }];
                }
                if (bench.error) {
                    return [{ name: "Benchmark", ok: false, detail: bench.error === "no_torch" ? "PyTorch not installed" : bench.error === "no_cuda" ? "CUDA not available" : bench.error }];
                }
                benchResultRef.current = bench;
                return [
                    { name: "FP16 Matmul", ok: bench.tflops > 0, detail: `${bench.tflops} TFLOPS · XCU score: ${bench.xcu_score}` },
                    { name: "PCIe Bandwidth", ok: bench.pcie_bandwidth_gbps >= 8, detail: `${bench.pcie_bandwidth_gbps} GB/s (H2D: ${bench.pcie_h2d_gbps}, D2H: ${bench.pcie_d2h_gbps})` },
                    { name: "Thermal Stability", ok: bench.gpu_temp_celsius > 0 && bench.gpu_temp_celsius <= 90, detail: `Peak ${bench.gpu_temp_celsius}°C · Avg ${bench.gpu_temp_avg_celsius}°C (${bench.gpu_temp_samples} samples)` },
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
                if (!gpu || !bench || !net) {
                    return [{ name: "Verification", ok: false, detail: "Missing GPU, benchmark, or network data — please retry previous steps" }];
                }
                const report = buildVerificationReport(gpu, bench, net, versionChecksRef.current);
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
                let costPerHour = 0.20;
                if (pricing === "competitive") costPerHour = 0.15;
                else if (pricing === "custom" && customRate) costPerHour = parseFloat(customRate);

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

                    // Store host ID
                    const updated = { ...currentAnswers, "_host_id": host.host_id || hostId };
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
                        admitted,
                        runtimeRecommendation: runtime,
                        reputationPoints: 125, // EMAIL(50) + HARDWARE_AUDIT(75)
                        tier: "Bronze",
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
            default:
                return [{ name: checkId, ok: false, detail: "Unknown check" }];
        }
    }, [checkWallet, launchGpuInstance]);

    // ── Step advancement ─────────────────────────────────────────────

    const advanceToNext = useCallback(
        (currentAnswers: Record<string, string | string[]>) => {
            setValidationError(null);
            setConfirmError(null);
            setCheckCanRetry(false);
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
                    setWizardState("success");
                    setWizardMessage(WIZARD_STEPS[doneIdx].prompt);

                    // Detect project framework
                    const fw = detectFramework();
                    if (fw) setDetectedFramework(fw.name);

                    // Save config
                    saveConfig(currentAnswers).catch(() => { });
                }
                setIsComplete(true);
                return;
            }

            const nextStep = WIZARD_STEPS[next];
            stepIndexRef.current = next;
            setStepIndex(next);
            setWizardMessage(nextStep.prompt);

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

                walletPollRef.current = setInterval(async () => {
                    const customerId = currentAnswers["_customer_id"] as string;
                    if (!customerId) return;
                    try {
                        const wallet = await getWallet(API_BASE_URL, currentAnswers["api-key"] as string, customerId);
                        setPaymentGate((prev) => ({ ...prev, balance: wallet.balance_cad }));
                        const listing = gpuListings.find((l) => l.host_id === (currentAnswers["gpu-pick"] as string));
                        if (wallet.balance_cad >= (listing?.price_per_hour ?? 0)) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            const updated = { ...currentAnswers, "_wallet_insufficient": "false", "payment-gate": "funded" };
                            answersRef.current = updated;
                            setAnswers(updated);
                            setWizardState("success");
                            setWizardMessage("Wallet funded! Proceeding...");
                            setTimeout(() => advanceToNext(updated), CHOREOGRAPHY_DELAY_MS);
                        }
                    } catch {
                        // ignore poll errors
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
                        setWizardState("success");
                        const successMsg = nextStep.checkId === "launch" ? "Instance launched!"
                            : nextStep.checkId === "benchmark" ? "Benchmarks complete!"
                            : nextStep.checkId === "verify" ? "Hardware verified!"
                            : nextStep.checkId === "host-register" ? "Host registered on the marketplace!"
                            : "All checks passed!";
                        setWizardMessage(successMsg);

                        // Auto-advance after delay
                        setTimeout(() => {
                            const updated = { ...currentAnswers, [nextStep.id]: "passed" };
                            answersRef.current = updated;
                            setAnswers(updated);
                            advanceToNext(updated);
                        }, nextStep.checkId === "launch" ? CHOREOGRAPHY_DELAY_MS * 2 : CHOREOGRAPHY_DELAY_MS);
                    } else {
                        setWizardState("error");
                        setWizardMessage(`${results.filter((r) => !r.ok).length} check(s) failed`);
                        setCheckCanRetry(true);
                    }
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
                        // Jump to done
                        const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                        if (doneIdx >= 0) {
                            stepIndexRef.current = doneIdx;
                            setStepIndex(doneIdx);
                            setWizardState("success");
                            setWizardMessage(WIZARD_STEPS[doneIdx].prompt);
                            setIsComplete(true);
                            saveConfig(updated).catch(() => { });
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

            // Choreography delay after mode select
            const delay = currentStep.id === "mode" ? CHOREOGRAPHY_DELAY_MS : 0;
            if (currentStep.id === "mode") {
                setWizardState("excited");
                setWizardMessage("Great choice! Let's get you set up...");
            }

            const updated = { ...answersRef.current, [currentStep.id]: value };
            answersRef.current = updated;
            setAnswers(updated);

            if (delay > 0) {
                setTimeout(() => advanceToNext(updated), delay);
            } else {
                advanceToNext(updated);
            }
        },
        [advanceToNext],
    );

    // ── Check retry/skip ─────────────────────────────────────────────

    const retryCheck = useCallback(() => {
        if (!activeCheckRef.current) return;
        setCheckCanRetry(false);
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
                setTimeout(() => {
                    const updated = { ...answersRef.current, [stepId]: "passed" };
                    answersRef.current = updated;
                    setAnswers(updated);
                    advanceToNext(updated);
                }, CHOREOGRAPHY_DELAY_MS);
            } else {
                setWizardState("error");
                setWizardMessage(`${results.filter((r) => !r.ok).length} check(s) failed`);
                setCheckCanRetry(true);
            }
        });
    }, [runCheck, advanceToNext]);

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

            const config: ApiClientConfig = {
                baseUrl: API_BASE_URL,
                apiKey: apiToken,
                pageContext: `cli-wizard:${step.id}`,
            };

            setAiStreaming(true);
            setAiResponse("");
            setCurrentAiQuestion(question);
            setWizardState("thinking");
            setWizardMessage("Hexara is thinking...");
            setShowAiPrompt(false);

            let content = "";
            try {
                for await (const event of streamChat(config, question)) {
                    if (event.type === "token") {
                        content += event.content ?? "";
                        setAiResponse(content);
                    } else if (event.type === "error") {
                        content = `Error: ${event.message}`;
                        setAiResponse(content);
                    }
                }
                setWizardState("idle");
                setWizardMessage(step.prompt);
            } catch (err) {
                content = `Connection error: ${err instanceof Error ? err.message : "unknown"}`;
                setAiResponse(content);
                setWizardState("error");
                setWizardMessage("AI unavailable — continue with the wizard steps.");
            } finally {
                setAiStreaming(false);
            }
        },
        [apiToken, step],
    );

    const dismissAi = useCallback(() => {
        // Save current Q&A to chat history before dismissing
        if (aiResponse && currentAiQuestion) {
            setChatHistory((prev) => [...prev, { question: currentAiQuestion, answer: aiResponse }]);
        }
        setAiResponse(null);
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
        retryCheck,
        skipCheck,
        aiAvailable,
        showAiPrompt,
        toggleAiPrompt,
        detectedFramework,
        chatHistory,
        currentAiQuestion,
        providerSummary,
    };
}
