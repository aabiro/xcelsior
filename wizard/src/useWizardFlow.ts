// useWizardFlow — Hook that drives the structured wizard step-by-step.
// Manages step progression, answer collection, auto-checks, device auth,
// marketplace browsing, payment gating, instance launch, and AI escape hatch.

import { useState, useCallback, useRef, useEffect } from "react";
import * as path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { openUrl as openBrowser } from "./open-url.js";
import { configDirectory, updateEnvFile, writePrivateFile } from "./config-files.js";
import { WIZARD_STEPS, getNextStep, STATIC_STEP_HELP, type WizardStep, IMAGE_TEMPLATES, WORKLOAD_IMAGE_MAP } from "./wizard-flow.js";
import {
    streamChat, confirmAction, type ApiClientConfig,
    requestDeviceCode, pollDeviceToken,
    getMe, createOAuthClient, searchMarketplace, type MarketplaceListing, type MarketplaceFilters,
    getWallet, claimFreeCredits,
    launchInstance, type InstanceInfo,
    registerHost, reportVersions, reportBenchmark, reportVerification,
} from "./api-client.js";
import type { WizardState } from "../sprites/wizard/wizard-sprite.js";
import { checkDocker, type CheckResult } from "./checks.js";
import {
    checkVersions, detectGpuFull, runComputeBenchmark,
    runNetworkBenchmark, buildVerificationReport,
    benchmarkUnavailableResults, buildLocalVerificationResults,
    CHECK_REMEDIATION,
    type GpuInfo, type BenchmarkResult, type NetworkBenchResult,
    type VersionCheck,
} from "./provider-checks.js";
import {
    clearWizardCheckpoint,
    hydrateWizardCheckpoint,
    loadWizardCheckpoint,
    saveWizardCheckpoint,
    tokenFilePath,
    type WizardCheckpoint,
} from "./wizard-state.js";
import { sanitizeContextValue, validateApiBaseUrl, validateApiToken } from "./wizard-guards.js";
import { fetchServiceStatus, aiHealthyFromReport, type StatusReport } from "./preflight.js";
import { labelForStep } from "./task-model.js";
import { summarizeFailure } from "./messaging.js";
import { fetchMarketplaceStats, type MarketplaceStats } from "./marketplace-stats.js";
import { detectEnvironment } from "./environment.js";
import {
    checkSdkProject,
    checkSdkPackage,
    writeSdkEnvSnippet,
    checkSdkApi,
    buildSdkStarterSnippet,
} from "./sdk-checks.js";

// ── Config ───────────────────────────────────────────────────────────

const API_BASE_URL = process.env["XCELSIOR_API_URL"] || "https://xcelsior.ca";
const DEFAULT_DEVICE_POLL_MS = 5_000;
const DEVICE_CODE_EXPIRY_MS = 15 * 60 * 1000;
const WALLET_POLL_MS = 5_000;
const CHOREOGRAPHY_DELAY_MS = 2_000;
const ADVANCE_DEBOUNCE_MS = 300;
const CHECKPOINT_DEBOUNCE_MS = 400;
const AI_ANALYSIS_COOLDOWN_MS = 30_000;
const DEVICE_AUTH_STEP_INDEX = WIZARD_STEPS.findIndex((s) => s.id === "device-auth");

if (validateApiBaseUrl(API_BASE_URL)) {
    console.warn(`[wizard] XCELSIOR_API_URL may be invalid: ${API_BASE_URL}`);
}

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
    /** True during the choreography delay after submitAnswer — hides step content */
    transitioning: boolean;
    /** Resume metadata when a prior checkpoint was found */
    resumeInfo: { resumed: boolean; needsReauth: boolean; expired: boolean };
    /** Persist current progress (also runs automatically on step changes) */
    flushCheckpoint: () => void;
    /** Preflight gate phase — "passed" means the flow is visible */
    gatePhase: GatePhase;
    /** Aggregated service-health report backing the gate */
    serviceStatus: StatusReport | null;
    /** Proceed past a degraded gate (Enter) */
    proceedFromGate: () => void;
    /** Re-run the preflight health check */
    recheckGate: () => void;
    /** Continue past a blocked gate in best-effort mode */
    continueAnywayFromGate: () => void;
    /** The resumed step's human label (for the resume notice) */
    resumedStepLabel: string | null;
    /** Live, line-by-line log of the current check's items as they resolve */
    checkProgress: string[];
    /** Live marketplace snapshot for the Learn pane (null until fetched/offline) */
    marketplaceStats: MarketplaceStats | null;
}

export type GatePhase = "checking" | "ready" | "blocked" | "passed";

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
    spotEnabled?: boolean;
    spotMinCents?: number;
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

/** Save only after the current authentication attempt has been verified. */
function saveToken(token: string): void {
    writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }, null, 2));
}

/** Required writes finish before the flow can report completion. */
function saveConfig(answers: Record<string, string | string[]>): void {
    const lines = [
        "# Xcelsior configuration — generated by setup wizard",
        `api_url = ${JSON.stringify(API_BASE_URL)}`,
    ];
    for (const key of ["mode", "workload", "pricing"] as const) {
        if (answers[key]) lines.push(`${key} = ${JSON.stringify(answers[key])}`);
    }
    if (answers["custom-rate"]) lines.push(`custom_rate = ${Number(answers["custom-rate"])}`);
    if (answers["_host_id"]) lines.push(`host_id = ${JSON.stringify(answers["_host_id"])}`);

    if ((answers.mode === "provide" || answers.mode === "both") && answers["_host_id"]) {
        const values: Record<string, string> = {
            XCELSIOR_HOST_ID: String(answers["_host_id"]),
            XCELSIOR_SCHEDULER_URL: API_BASE_URL,
        };
        if (answers["oauth-client-id"] && answers["oauth-client-secret"]) {
            values.XCELSIOR_OAUTH_CLIENT_ID = String(answers["oauth-client-id"]);
            values.XCELSIOR_OAUTH_CLIENT_SECRET = String(answers["oauth-client-secret"]);
        }
        if (answers["api-key"]) values.XCELSIOR_API_TOKEN = String(answers["api-key"]);
        if (answers["custom-rate"]) values.XCELSIOR_COST_PER_HOUR = String(answers["custom-rate"]);
        updateEnvFile(path.join(configDirectory(), ".env"), values);
    }
    writePrivateFile(path.join(configDirectory(), "config.toml"), lines.join("\n") + "\n");
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
    const parts: string[] = [`cli-wizard:${sanitizeContextValue(stepId, 64)}`];

    const push = (key: string, value: string) => {
        parts.push(`${key}=${sanitizeContextValue(value)}`);
    };

    // Mode
    if (answers.mode) push("mode", String(answers.mode));

    // Provider context
    if (answers.mode === "provide" || answers.mode === "both") {
        if (answers.pricing) push("pricing", String(answers.pricing));
        if (answers["custom-rate"]) push("custom_rate", `$${answers["custom-rate"]}/hr`);
        if (answers["_rate"]) push("rate", String(answers["_rate"]));
        if (answers["_host_id"]) push("host_id", String(answers["_host_id"]));
        if (answers["_host_ip"]) push("host_ip", String(answers["_host_ip"]));
        if (answers["_host_port"]) push("host_port", String(answers["_host_port"]));

        // Check results summary — URL-encode values to avoid nested = truncation
        const failedChecks: string[] = [];
        for (const [stepKey, result] of Object.entries(checkResults)) {
            if (!result.allPassed) {
                const failures = result.items.filter((i) => !i.ok).map((i) => `${i.name}: ${i.detail}`);
                failedChecks.push(`${stepKey}=[${failures.join("; ")}]`);
            }
        }
        if (failedChecks.length > 0) {
            push("failed_checks", encodeURIComponent(sanitizeContextValue(`{${failedChecks.join(", ")}}`, 400)));
        }

        // GPU/benchmark data — use providerSummary if available, fall back to early refs
        if (providerSummary) {
            push("gpu", providerSummary.gpuModel);
            push("vram", `${providerSummary.vramGb}GB`);
            push("xcu", String(providerSummary.xcuScore));
            push("tflops", String(providerSummary.tflops));
            push("tier", providerSummary.tier);
            push("verified", String(providerSummary.verified));
        } else {
            // Early fallback from detection/benchmark refs (available before step 13)
            if (earlyGpu) {
                push("gpu", earlyGpu.gpu_model);
                push("vram", `${earlyGpu.total_vram_gb}GB`);
            }
            if (earlyBench) {
                push("tflops", String(earlyBench.tflops));
                push("xcu", String(earlyBench.xcu_score));
            }
        }

        // Network benchmark data
        if (earlyNetwork) {
            push("latency", `${earlyNetwork.latency_avg_ms}ms`);
            push("jitter", `${earlyNetwork.jitter_ms}ms`);
            push("throughput", `${earlyNetwork.throughput_mbps}Mbps`);
        }
    }

    // Renter context
    if (answers.mode === "rent" || answers.mode === "both") {
        if (answers.workload) push("workload", String(answers.workload));
        if (answers["gpu-preference"]) push("gpu_pref", String(answers["gpu-preference"]));
        if (answers["gpu-pick"]) {
            const listing = gpuListings.find((l) => l.host_id === (answers["gpu-pick"] as string));
            if (listing) {
                push("picked_gpu", `${listing.gpu_model}/${listing.vram_gb}GB/$${listing.price_per_hour}/hr`);
                push("rate", `$${listing.price_per_hour}/hr`);
            }
        }
        if (answers["image-pick"]) push("image", String(answers["image-pick"]));
        if (answers["_instance_id"]) push("instance_id", String(answers["_instance_id"]));
        if (answers["_balance"]) push("balance", `$${answers["_balance"]}`);
        if (browseError) push("browse_error", browseError);
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

function shouldProvisionWizardOAuthClient(
    answers: Record<string, string | string[]>,
): boolean {
    return typeof answers.mode === "string" && answers.mode.length > 0;
}

function wizardOAuthScopes(mode: string): string[] {
    if (mode === "provide" || mode === "both") {
        return ["api", "hosts:read", "hosts:write"];
    }
    if (mode === "sdk") {
        return ["api", "instances:read", "instances:write", "billing:read", "marketplace:read"];
    }
    return ["api", "instances:read", "instances:write", "billing:read"];
}

function buildWorkerOAuthClientName(
    answers: Record<string, string | string[]>,
): string {
    const label = String(
        answers["_host_id"]
        || answers["_email"]
        || answers["_customer_id"]
        || answers.mode
        || "worker",
    )
        .replace(/[^a-zA-Z0-9._-]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 32);
    const suffix = Date.now().toString(36);
    return `CLI Wizard ${label || "session"} ${suffix}`;
}

// ── Hook ─────────────────────────────────────────────────────────────

export function useWizardFlow(): UseWizardFlowReturn {
    const [hydrated] = useState(() => hydrateWizardCheckpoint(
        loadWizardCheckpoint(),
        WIZARD_STEPS.length,
        DEVICE_AUTH_STEP_INDEX >= 0 ? DEVICE_AUTH_STEP_INDEX : 0,
    ));
    const initialCheckpoint = hydrated?.expired ? null : hydrated?.checkpoint ?? null;
    const initialStepIndex = initialCheckpoint?.stepIndex ?? 0;
    const initialAnswers = initialCheckpoint?.answers ?? { "_api_base_url": API_BASE_URL };
    const initialStep = WIZARD_STEPS[initialStepIndex] ?? WIZARD_STEPS[0];

    const [stepIndex, setStepIndex] = useState(initialStepIndex);
    const stepIndexRef = useRef(initialStepIndex);
    const [answers, setAnswers] = useState<Record<string, string | string[]>>(initialAnswers);
    const answersRef = useRef<Record<string, string | string[]>>(initialAnswers);
    const completedStepIdsRef = useRef<string[]>(initialCheckpoint?.completedStepIds ?? []);
    const [resumeInfo, setResumeInfo] = useState({
        resumed: hydrated?.resumed ?? false,
        needsReauth: hydrated?.needsReauth ?? false,
        expired: hydrated?.expired ?? false,
    });
    const [wizardState, setWizardState] = useState<WizardState>("idle");
    const [wizardMessage, setWizardMessage] = useState(() => {
        if (hydrated?.expired) {
            return "Previous wizard session expired — starting fresh.";
        }
        if (hydrated?.resumed) {
            return hydrated.needsReauth
                ? "Welcome back — please sign in again to continue."
                : `Resuming at: ${initialStep.prompt}`;
        }
        return WIZARD_STEPS[0].prompt;
    });
    const [checkResults, setCheckResults] = useState<Record<string, AutoCheckResults>>({});
    const [aiResponse, setAiResponse] = useState<string | null>(null);
    const [aiStreaming, setAiStreaming] = useState(false);
    const lastAiContentRef = useRef<string | null>(null);
    const [isComplete, setIsComplete] = useState(false);
    const completedRef = useRef(false);
    const mountedRef = useRef(true);
    const transitionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
    const devicePollRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const browserTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const authGenRef = useRef(0);
    const manualAuthInFlightRef = useRef(false);

    // Marketplace
    const [gpuListings, setGpuListings] = useState<MarketplaceListing[]>(initialCheckpoint?.runtime?.listings ?? []);
    const [browseError, setBrowseError] = useState<string | null>(null);
    const [gpuOptions, setGpuOptions] = useState<{ label: string; value: string }[]>(() =>
        (initialCheckpoint?.runtime?.listings ?? []).map((listing) => ({
            label: `${listing.gpu_model} · ${listing.vram_gb} GB · $${listing.price_per_hour.toFixed(2)}/hr · ${listing.owner}`,
            value: listing.host_id,
        })));
    const [imageOptions, setImageOptions] = useState<{ label: string; value: string }[]>([]);

    // Instance
    const [instanceInfo, setInstanceInfo] = useState<InstanceInfo | null>(initialCheckpoint?.runtime?.instance ?? null);

    // Payment
    const [paymentGate, setPaymentGate] = useState({
        balance: 0, required: 0, polling: false, billingUrl: `${API_BASE_URL}/dashboard/billing`,
    });
    const walletPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Auto-check retry
    const [checkCanRetry, setCheckCanRetry] = useState(false);
    const [checkAwaitContinue, setCheckAwaitContinue] = useState(false);
    // Live, line-by-line check progress (Part E — "Tail logs" / granular status).
    const [checkProgress, setCheckProgress] = useState<string[]>([]);
    // Live marketplace snapshot powering the Learn pane charts (best-effort).
    const [marketplaceStats, setMarketplaceStats] = useState<MarketplaceStats | null>(null);
    const checkRunningRef = useRef(false);
    const browseRunningRef = useRef(false);
    const activeCheckRef = useRef<{ checkId: string; stepId: string } | null>(null);
    const lastAdvanceRef = useRef<number>(0);
    const checkpointTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const flowGenRef = useRef(0);
    const gpuListingsRef = useRef<MarketplaceListing[]>(initialCheckpoint?.runtime?.listings ?? []);
    const lastAutoAnalysisRef = useRef<{ key: string; at: number } | null>(null);

    // Device auth — detected .env path for display
    const [deviceAuthEnvPath, setDeviceAuthEnvPath] = useState<string | null>(null);
    // Token save error — null means saved OK
    const [tokenSaveError, setTokenSaveError] = useState<string | null>(null);

    // AI conversation tracking
    const conversationIdRef = useRef<string | null>(initialCheckpoint?.conversationId ?? null);

    const scheduleCheckpoint = useCallback((override?: Partial<WizardCheckpoint>) => {
        if (checkpointTimerRef.current) clearTimeout(checkpointTimerRef.current);
        if (completedRef.current) return;
        checkpointTimerRef.current = setTimeout(() => {
            if (completedRef.current) return;
            saveWizardCheckpoint({
                runtime: runtimeSnapshot(),
                stepIndex: stepIndexRef.current,
                answers: answersRef.current,
                conversationId: conversationIdRef.current ?? undefined,
                completedStepIds: [...completedStepIdsRef.current],
                savedAt: new Date().toISOString(),
                ...override,
            });
        }, CHECKPOINT_DEBOUNCE_MS);
    }, []);

    const flushCheckpoint = useCallback(() => {
        if (checkpointTimerRef.current) {
            clearTimeout(checkpointTimerRef.current);
            checkpointTimerRef.current = null;
        }
        if (completedRef.current) return;
        saveWizardCheckpoint({
                runtime: runtimeSnapshot(),
            stepIndex: stepIndexRef.current,
            answers: answersRef.current,
            conversationId: conversationIdRef.current ?? undefined,
            completedStepIds: [...completedStepIdsRef.current],
            savedAt: new Date().toISOString(),
        });
    }, [isComplete]);

    const shouldRunAutoAnalysis = useCallback((stepId: string, failDetails: string) => {
        const key = `${stepId}:${failDetails}`;
        const now = Date.now();
        const last = lastAutoAnalysisRef.current;
        if (last && last.key === key && now - last.at < AI_ANALYSIS_COOLDOWN_MS) {
            return false;
        }
        lastAutoAnalysisRef.current = { key, at: now };
        return true;
    }, []);

    const isFlowStale = useCallback((gen: number) => gen !== flowGenRef.current, []);

    const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
    const [aiToolCalls, setAiToolCalls] = useState<AiToolCall[]>([]);

    // Provider flow state
    const gpuInfoRef = useRef<GpuInfo | null>(initialCheckpoint?.runtime?.gpu ?? null);
    const benchResultRef = useRef<BenchmarkResult | null>(initialCheckpoint?.runtime?.benchmark ?? null);
    const networkResultRef = useRef<NetworkBenchResult | null>(initialCheckpoint?.runtime?.network ?? null);
    const versionChecksRef = useRef<VersionCheck[]>(initialCheckpoint?.runtime?.versions ?? []);
    const [providerSummary, setProviderSummary] = useState<ProviderSummaryData | null>(initialCheckpoint?.runtime?.providerSummary ?? null);
    const providerSummaryRef = useRef(providerSummary);
    providerSummaryRef.current = providerSummary;
    const instanceInfoRef = useRef(instanceInfo);
    instanceInfoRef.current = instanceInfo;
    const runtimeSnapshot = () => ({
        gpu: gpuInfoRef.current, benchmark: benchResultRef.current, network: networkResultRef.current,
        versions: versionChecksRef.current, listings: gpuListingsRef.current,
        providerSummary: providerSummaryRef.current, instance: instanceInfoRef.current,
    });

    // ── Preflight service-health gate (Part A) ───────────────────────
    const [gatePhase, setGatePhase] = useState<GatePhase>("checking");
    const [serviceStatus, setServiceStatus] = useState<StatusReport | null>(null);
    // AI health is determined by the gate's report; default true until known so
    // the "?" hint isn't suppressed during the brief check.
    const [aiServiceHealthy, setAiServiceHealthy] = useState(true);
    const gateGenRef = useRef(0);

    const step = WIZARD_STEPS[stepIndex];
    const apiToken = (answers["api-key"] as string) || "";
    // aiAvailable reflects *real* AI health (gate report), not just token presence (A — hardening).
    const aiAvailable = !!apiToken && aiServiceHealthy;

    // Compute launch summary for confirm-launch step
    const launchSummary: string[] = [];
    if (step.id === "confirm-launch") {
        const pickedHost = answers["gpu-pick"] as string;
        const listing = gpuListingsRef.current.find((l) => l.host_id === pickedHost);
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

    // ── Preflight gate runner ────────────────────────────────────────

    const runPreflight = useCallback(async () => {
        const gen = ++gateGenRef.current;
        setGatePhase("checking");
        setWizardState("thinking");
        setWizardMessage("Checking Xcelsior service health…");
        const token = (answersRef.current["api-key"] as string) || undefined;
        const report = await fetchServiceStatus(API_BASE_URL, token);
        if (gen !== gateGenRef.current) return; // superseded by a newer check
        setServiceStatus(report);
        setAiServiceHealthy(aiHealthyFromReport(report));
        if (report.verdict === "operational") {
            // Healthy — fall straight into the flow, no panel.
            setGatePhase("passed");
            setWizardState("idle");
            setWizardMessage(WIZARD_STEPS[stepIndexRef.current]?.prompt ?? WIZARD_STEPS[0].prompt);
        } else if (report.verdict === "blocked") {
            setGatePhase("blocked");
            setWizardState("error");
            setWizardMessage("A required service is down — see details below");
        } else {
            setGatePhase("ready");
            setWizardState("waiting");
            setWizardMessage("Some services are degraded — press Enter to continue anyway");
        }
    }, []);

    const proceedFromGate = useCallback(() => {
        setGatePhase("passed");
        setWizardState("idle");
        setWizardMessage(WIZARD_STEPS[stepIndexRef.current]?.prompt ?? WIZARD_STEPS[0].prompt);
    }, []);

    const continueAnywayFromGate = useCallback(() => {
        setGatePhase("passed");
        setWizardState("idle");
        setWizardMessage(WIZARD_STEPS[stepIndexRef.current]?.prompt ?? WIZARD_STEPS[0].prompt);
    }, []);

    const recheckGate = useCallback(() => {
        void runPreflight();
    }, [runPreflight]);

    // Run the preflight check once on mount.
    useEffect(() => {
        void runPreflight();
        // eslint-disable-next-line react-hooks/exhaustive-deps -- run once on mount
    }, []);

    // Once authenticated, fetch a live marketplace snapshot for the Learn pane
    // charts (best-effort; the pane falls back to concept cards if this is null).
    useEffect(() => {
        if (!apiToken || marketplaceStats) return;
        let cancelled = false;
        void fetchMarketplaceStats(API_BASE_URL, apiToken).then((s) => {
            if (!cancelled && s) setMarketplaceStats(s);
        });
        return () => { cancelled = true; };
    }, [apiToken, marketplaceStats]);

    // ── Device auth flow ─────────────────────────────────────────────

    const clearDeviceTimers = useCallback(() => {
        if (browserTimeoutRef.current) {
            clearTimeout(browserTimeoutRef.current);
            browserTimeoutRef.current = null;
        }
        if (devicePollRef.current) {
            clearTimeout(devicePollRef.current as unknown as ReturnType<typeof setTimeout>);
            devicePollRef.current = null;
        }
    }, []);

    const stopDevicePoll = useCallback(() => {
        authGenRef.current += 1;
        manualAuthInFlightRef.current = false;
        clearDeviceTimers();
    }, [clearDeviceTimers]);

    const acceptAuthentication = useCallback(async (token: string, source: "authorized" | "manual", gen: number) => {
        const profile = await getMe(API_BASE_URL, token);
        if (gen !== authGenRef.current) return;
        if (!profile.user_id) throw new Error("The server returned an incomplete account profile");
        try {
            // This atomic write is synchronous: no superseding auth attempt can
            // interleave between the generation check and credential storage.
            saveToken(token);
        } catch (err) {
            const message = err instanceof Error ? err.message : "Could not save credential";
            setTokenSaveError(message);
            throw new Error(`Credential could not be saved: ${message}`);
        }
        const updated = {
            ...answersRef.current,
            "api-key": token,
            "device-auth": source,
            "_session-token": token,
            "_customer_id": profile.customer_id || "",
            "_email": profile.email || "",
        };
        answersRef.current = updated;
        setAnswers(updated);
        setDeviceAuth((prev) => ({ ...prev, status: "authorized", token, email: profile.email, errorMessage: null }));
        setTokenSaveError(null);
        setDeviceAuthEnvPath(null);
        setWizardState("excited");
        setWizardMessage(`Credential saved to ${tokenFilePath()} — press Enter to continue`);
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

    const ensureWorkerOAuthClient = useCallback(async (
        currentAnswers: Record<string, string | string[]>,
        sessionToken?: string,
    ): Promise<Record<string, string | string[]>> => {
        if (!shouldProvisionWizardOAuthClient(currentAnswers)) return currentAnswers;
        if (currentAnswers["oauth-client-id"] && currentAnswers["oauth-client-secret"]) return currentAnswers;

        const userGrant = String(sessionToken || currentAnswers["_session-token"] || currentAnswers["api-key"] || "").trim();
        if (!userGrant) throw new Error("Sign in before creating worker credentials");
        const gen = flowGenRef.current;
        const origin = stepIndexRef.current;

        const mode = String(currentAnswers.mode || "rent");
        try {
            // The general progress checkpoint deliberately strips secrets. Keep
            // application credentials in a separate private file so retries and
            // resumes reuse this application's client instead of creating more.
            const identity = JSON.stringify([API_BASE_URL, currentAnswers["_email"],
                currentAnswers["_customer_id"], mode, mode === "sdk" ? process.cwd() : currentAnswers["_host_id"]]);
            const credentialsFile = path.join(configDirectory(), "clients", `${createHash("sha256").update(identity).digest("hex")}.json`);
            const saved = existsSync(credentialsFile)
                ? JSON.parse(readFileSync(credentialsFile, "utf8")) as { client_id?: string; client_secret?: string }
                : null;
            if (saved && (!saved.client_id || !saved.client_secret)) throw new Error(`Incomplete credentials in ${credentialsFile}`);
            const client = saved || await createOAuthClient(API_BASE_URL, userGrant, {
                client_name: buildWorkerOAuthClientName(currentAnswers),
                client_type: "confidential",
                redirect_uris: [],
                grant_types: ["client_credentials"],
                scopes: wizardOAuthScopes(mode),
            });
            if (!saved) writePrivateFile(credentialsFile, JSON.stringify(client));
            if (!mountedRef.current || completedRef.current || gen !== flowGenRef.current || origin !== stepIndexRef.current) {
                throw new Error("Credential setup was cancelled");
            }
            const updated = {
                ...answersRef.current,
                "oauth-client-id": client.client_id!,
                "oauth-client-secret": client.client_secret!,
            };
            answersRef.current = updated;
            setAnswers(updated);
            return updated;
        } catch (err) {
            throw new Error(`Could not create application credentials: ${err instanceof Error ? err.message : "unknown error"}`);
        }
    }, []);

    const startDeviceAuth = useCallback(async () => {
        stopDevicePoll();
        const gen = authGenRef.current;
        const isCurrent = () => gen === authGenRef.current;
        setValidationError(null);
        setTokenSaveError(null);
        setDeviceAuth({ status: "loading", userCode: null, verificationUri: null, token: null, email: null, errorMessage: null });
        setWizardState("thinking");

        const fail = (message: string) => {
            if (!isCurrent()) return;
            clearDeviceTimers();
            setDeviceAuth((prev) => ({ ...prev, status: "error", errorMessage: message }));
            setWizardState("error");
            setWizardMessage(`${message} — press Enter to retry or m for manual`);
        };
        try {
            const result = await requestDeviceCode(API_BASE_URL);
            if (!isCurrent()) return;
            setDeviceAuth({
                status: "waiting", userCode: result.user_code,
                verificationUri: result.verification_uri, token: null, email: null, errorMessage: null,
            });
            setWizardState("waiting");
            setWizardMessage("Enter the code shown below in your browser...");
            browserTimeoutRef.current = setTimeout(() => {
                if (isCurrent()) openBrowser(result.verification_uri);
            }, 15_000);

            let devicePollInFlight = false;
            const pollStartTime = Date.now();
            let pollIntervalMs = Math.max((result.interval ?? 5) * 1000, DEFAULT_DEVICE_POLL_MS);
            const schedulePoll = (delayMs: number) => {
                if (!isCurrent()) return;
                if (devicePollRef.current) clearTimeout(devicePollRef.current);
                devicePollRef.current = setTimeout(() => { void pollOnce(); }, delayMs);
            };
            const pollOnce = async () => {
                if (!isCurrent() || devicePollInFlight) return;
                if (Date.now() - pollStartTime >= (result.expires_in * 1000 || DEVICE_CODE_EXPIRY_MS)) {
                    fail("Device code expired");
                    return;
                }
                devicePollInFlight = true;
                let tokenReceived = false;
                try {
                    const pollResult = await pollDeviceToken(API_BASE_URL, result.device_code);
                    if (!isCurrent()) return;
                    if (pollResult.status === "authorized") {
                        tokenReceived = true;
                        clearDeviceTimers();
                        await acceptAuthentication(pollResult.token.access_token, "authorized", gen);
                    } else if (pollResult.status === "slow_down") {
                        pollIntervalMs = Math.min(pollIntervalMs + 5000, 60_000);
                        schedulePoll(pollIntervalMs);
                    } else if (pollResult.status === "expired") {
                        fail("Device code expired");
                    } else if (pollResult.status === "error") {
                        fail(pollResult.message);
                    } else {
                        schedulePoll(pollIntervalMs);
                    }
                } catch (err) {
                    if (!isCurrent()) return;
                    const msg = err instanceof Error ? err.message : "Authentication failed";
                    // Once authorized the device grant may be consumed. Profile
                    // or save failures require a fresh attempt, not more polling.
                    if (tokenReceived || msg.includes("expired") || msg.includes("denied")) fail(msg);
                    else schedulePoll(pollIntervalMs);
                } finally {
                    devicePollInFlight = false;
                }
            };
            schedulePoll(pollIntervalMs);
        } catch (err) {
            fail(err instanceof Error ? err.message : "Authentication failed");
        }
    }, [stopDevicePoll, clearDeviceTimers, acceptAuthentication]);

    const switchToManualAuth = useCallback(() => {
        stopDevicePoll();
        setValidationError(null);
        setTokenSaveError(null);
        setDeviceAuth({ status: "manual", userCode: null, verificationUri: null, token: null, email: null, errorMessage: null });
        setWizardState("idle");
        setWizardMessage("Paste your API key or sign-in token below:");
    }, [stopDevicePoll]);

    const retryDeviceAuth = useCallback(() => {
        void startDeviceAuth();
    }, [startDeviceAuth]);

    const submitManualToken = useCallback(async (token: string) => {
        if (manualAuthInFlightRef.current) return;
        stopDevicePoll();
        const gen = authGenRef.current;
        const trimmed = token.trim();
        const tokenError = validateApiToken(trimmed);
        if (tokenError) {
            setValidationError(tokenError);
            setWizardState("error");
            setWizardMessage(tokenError);
            return;
        }
        manualAuthInFlightRef.current = true;
        setValidationError(null);
        setTokenSaveError(null);
        setWizardState("thinking");
        setWizardMessage("Verifying credential...");
        try {
            await acceptAuthentication(trimmed, "manual", gen);
        } catch (err) {
            if (gen !== authGenRef.current) return;
            const msg = err instanceof Error ? err.message : "Invalid credential";
            setDeviceAuth((prev) => ({ ...prev, status: "manual", errorMessage: msg }));
            setValidationError(msg);
            setWizardState("error");
            setWizardMessage(`Sign-in failed — ${msg}`);
        } finally {
            if (gen === authGenRef.current) manualAuthInFlightRef.current = false;
        }
    }, [stopDevicePoll, acceptAuthentication]);

    // ── Marketplace browsing ─────────────────────────────────────────

    const browseGpus = useCallback(async (currentAnswers: Record<string, string | string[]>) => {
        if (browseRunningRef.current) return;
        browseRunningRef.current = true;
        const gen = flowGenRef.current;
        const origin = stepIndexRef.current;
        const isCurrent = () => mountedRef.current && !completedRef.current
            && gen === flowGenRef.current && origin === stepIndexRef.current;
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
            if (!isCurrent()) return;
            if (result.listings.length === 0) {
                setBrowseError("No GPUs available right now. Hexara suggests checking back shortly.");
                setWizardState("error");
                setWizardMessage("No GPUs found — press Enter to retry");
                return;
            }

            gpuListingsRef.current = result.listings;
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
            transitionTimerRef.current = setTimeout(() => {
                transitionTimerRef.current = null;
                if (isCurrent()) void advanceToNext(updated);
            }, CHOREOGRAPHY_DELAY_MS);
        } catch (err) {
            if (!isCurrent()) return;
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
        } finally {
            browseRunningRef.current = false;
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
            const listing = gpuListingsRef.current.find((l) => l.host_id === pickedHost);
            if (!listing || !Number.isFinite(listing.price_per_hour) || listing.price_per_hour < 0) {
                throw new Error("The selected GPU price is unavailable. Return to marketplace selection before launching.");
            }
            const required = listing.price_per_hour;

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
                : `$${balance.toFixed(2)} CAD (includes $${creditResult.amount.toFixed(2)} welcome credit)`;

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
        const listing = gpuListingsRef.current.find((l) => l.host_id === hostId);

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
        const gen = flowGenRef.current;
        const origin = stepIndexRef.current;
        const isCurrent = () => mountedRef.current && !completedRef.current
            && gen === flowGenRef.current && origin === stepIndexRef.current;
        const assertCurrent = () => {
            if (!isCurrent()) throw new Error("This check was cancelled");
        };
        const checked = async <T,>(operation: Promise<T>): Promise<T> => {
            const result = await operation;
            assertCurrent();
            return result;
        };
        // Never let a previous step append progress or mutate current answers.
        assertCurrent();
        setCheckProgress([]);
        const streamItem = (name: string, ok: boolean, detail: string) =>
            isCurrent() && setCheckProgress((prev) => [...prev, `${ok ? "✓" : "✗"} ${name}: ${detail}`].slice(-30));
        // Phase markers for long single-process checks (benchmark/network).
        const streamPhase = (msg: string) =>
            isCurrent() && setCheckProgress((prev) => [...prev, `⟳ ${msg}`].slice(-30));

        switch (checkId) {
            case "docker":
                return checkDocker((r) => streamItem(r.name, r.ok, r.detail));
            case "api":
                return checkApi(API_BASE_URL, currentAnswers["api-key"] as string || "");
            case "gpu": {
                // Full GPU detection — store result for provider flow
                const gpuFull = await checked(detectGpuFull());
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
                const results = await checked(checkVersions((v) =>
                    streamItem(v.component, v.passed, v.version ? `v${v.version}` : `not found — needs ≥${v.minimum}`),
                ));
                versionChecksRef.current = results;
                return results.map((v) => ({
                    name: v.component,
                    ok: v.passed,
                    detail: v.version
                        ? `v${v.version}${v.passed ? "" : ` — needs ≥${v.minimum}`}`
                        : `not found — needs ≥${v.minimum}`,
                    ...(v.passed ? {} : { remediation: CHECK_REMEDIATION[v.component] }),
                }));
            }
            case "benchmark": {
                // Skip benchmark if only basic GPU detection passed (no detailed nvidia-smi data)
                if (currentAnswers["_gpu_basic_only"] === "true") {
                    return benchmarkUnavailableResults(
                        "Failed — detailed GPU data unavailable (nvidia-smi query failed)",
                    );
                }
                const bench = await checked(runComputeBenchmark(streamPhase));
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
                const net = await checked(runNetworkBenchmark(API_BASE_URL, streamPhase, currentAnswers["api-key"] as string));
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
                if (!net) {
                    return [{ name: "Verification", ok: false, detail: "Missing network measurement — please retry the network test" }];
                }
                const report = buildVerificationReport(gpu, bench, net, versionChecksRef.current);
                // The host is not registered yet, so only local readiness can be
                // checked here. The installed worker submits the server report.
                return buildLocalVerificationResults(report);
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

                let costPerHour: number;
                if (pricing === "custom" && customRate) {
                    costPerHour = Number(customRate);
                } else {
                    try {
                        const market = await checked(searchMarketplace(API_BASE_URL, token, { gpu_model: gpu.gpu_model, limit: 20 }));
                        const rates = market.listings.map((listing) => listing.price_per_hour)
                            .filter((rate) => Number.isFinite(rate) && rate > 0);
                        if (!rates.length) throw new Error("No comparable marketplace rates are available");
                        const average = rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
                        costPerHour = Math.round(average * (pricing === "competitive" ? 0.85 : 1) * 100) / 100;
                    } catch (err) {
                        assertCurrent();
                        return [{ name: "Host Registration", ok: false,
                            detail: `Cannot determine your rate: ${err instanceof Error ? err.message : "marketplace unavailable"}. Retry when rates are available.` }];
                    }
                }
                if (!Number.isFinite(costPerHour) || costPerHour <= 0) {
                    return [{ name: "Host Registration", ok: false, detail: "A positive hourly rate is required" }];
                }
                const hostIp = String(currentAnswers["_host_ip"] || "");
                if (!hostIp) return [{ name: "Host Registration", ok: false, detail: "Complete network setup before registration" }];
                // Persist the intended ID before the remote mutation. An ambiguous
                // timeout and a resumed attempt update the same host.
                const hostId = String(currentAnswers["_host_id"] || `host-${randomUUID()}`);
                const registrationAnswers = { ...answersRef.current, "_host_id": hostId, "_host_cost_per_hour": String(costPerHour) };
                answersRef.current = registrationAnswers;
                setAnswers(registrationAnswers);
                flushCheckpoint();
                const versions: Record<string, string> = {};
                for (const v of versionChecksRef.current) {
                    if (v.version) versions[v.component] = v.version;
                }

                const spotEnabled = currentAnswers["spot-enabled"] !== "no";
                const spotMinRaw = currentAnswers["spot-min-cents"] as string | undefined;
                const spotMinCents = spotEnabled && spotMinRaw
                    ? parseInt(spotMinRaw, 10)
                    : 0;

                try {
                    assertCurrent();
                    const host = await checked(registerHost(API_BASE_URL, token, {
                        host_id: hostId,
                        ip: hostIp,
                        gpu_model: gpu.gpu_model,
                        total_vram_gb: gpu.total_vram_gb,
                        free_vram_gb: gpu.free_vram_gb,
                        cost_per_hour: costPerHour,
                        versions,
                        spot_enabled: spotEnabled,
                        spot_min_cents: spotMinCents,
                    }));

                    // Store host ID and cost
                    const updated = { ...answersRef.current, "_host_id": host.host_id || hostId, "_host_cost_per_hour": String(costPerHour) };
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

                    return [{
                        name: "Host Registration",
                        ok: true,
                        detail: `Registered as ${host.host_id || hostId} · pending worker verification · not yet listed`,
                    }];
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
                    const result = await checked(reportVersions(API_BASE_URL, token, hostId, versions));
                    const compatible = result.compatible === true;
                    const admitted = result.admitted === true;
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
                        spotEnabled: currentAnswers["spot-enabled"] !== "no",
                        spotMinCents: currentAnswers["spot-enabled"] !== "no"
                            ? parseInt(String(currentAnswers["spot-min-cents"] || "0"), 10) || 0
                            : 0,
                    });

                    return [
                        {
                            name: "Version Compatibility",
                            ok: compatible,
                            detail: compatible
                                ? "Required component versions are compatible"
                                : "Required component versions need remediation",
                        },
                        {
                            name: "Admission Boundary",
                            ok: !admitted && result.admission_applied === false,
                            detail: !admitted
                                ? "Pending authoritative worker verification — not admitted or listed"
                                : "Unexpected admission from a self-reported compatibility check",
                        },
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
                    const { setupNetworking } = await checked(import("./provider-checks.js"));
                    const result = await checked(setupNetworking());
                    const updated = {
                        ...currentAnswers,
                        "_host_ip": result.ip,
                        "_network_method": result.method,
                    };
                    answersRef.current = updated;
                    setAnswers(updated);
                    return [
                        { name: "Mesh Network", ok: result.method !== "none", detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "Mesh Network", ok: false, detail: err instanceof Error ? err.message : "Network setup failed" }];
                }
            }
            case "worker-install": {
                try {
                    const { installWorkerAgent } = await checked(import("./provider-checks.js"));
                    const answersWithWorkerAuth = await checked(ensureWorkerOAuthClient(currentAnswers));
                    const token = answersWithWorkerAuth["api-key"] as string;
                    const hostId = answersWithWorkerAuth["_host_id"] as string;
                    const hostIp = answersWithWorkerAuth["_host_ip"] as string || "";
                    const oauthClientId = answersWithWorkerAuth["oauth-client-id"] as string | undefined;
                    const oauthClientSecret = answersWithWorkerAuth["oauth-client-secret"] as string | undefined;
                    const result = await installWorkerAgent(API_BASE_URL, token, hostId, hostIp, oauthClientId, oauthClientSecret);
                    return [
                        { name: "Worker Agent", ok: result.installed, detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "Worker Agent", ok: false, detail: err instanceof Error ? err.message : "Worker install failed" }];
                }
            }
            case "ssh-key-setup": {
                try {
                    const { setupSshKeys } = await checked(import("./provider-checks.js"));
                    const token = currentAnswers["api-key"] as string;
                    const result = await checked(setupSshKeys(API_BASE_URL, token));
                    return [
                        { name: "SSH Keys", ok: result.keyFound, detail: result.detail },
                    ];
                } catch (err) {
                    return [{ name: "SSH Keys", ok: false, detail: err instanceof Error ? err.message : "SSH key setup failed" }];
                }
            }
            case "sdk-detect": {
                const results = checkSdkProject();
                for (const r of results) streamItem(r.name, r.ok, r.detail);
                return results;
            }
            case "sdk-install": {
                const results = await checked(checkSdkPackage());
                for (const r of results) streamItem(r.name, r.ok, r.detail);
                return results;
            }
            case "sdk-credentials": {
                const token = currentAnswers["api-key"] as string;
                const baseUrl = (currentAnswers["_api_base_url"] as string) || API_BASE_URL;
                const withOAuth = token.startsWith("xcel_ai_")
                    ? currentAnswers
                    : await checked(ensureWorkerOAuthClient(currentAnswers, token));
                const oauthId = withOAuth["oauth-client-id"] as string | undefined;
                const oauthSecret = withOAuth["oauth-client-secret"] as string | undefined;
                const envPath = writeSdkEnvSnippet(baseUrl, token, oauthId, oauthSecret);
                const fw = detectEnvironment();
                const snippet = buildSdkStarterSnippet(fw.framework, baseUrl);
                const updated = {
                    ...withOAuth,
                    "_sdk_env_path": envPath,
                    "_sdk_snippet": snippet,
                };
                answersRef.current = updated;
                setAnswers(updated);
                streamItem("OAuth client", true, oauthId ? oauthId : "Using the supplied credential");
                streamItem("Environment", true, envPath);
                return [
                    {
                        name: "OAuth client",
                        ok: true,
                        detail: oauthId ? `Configured ${oauthId} with automatic token renewal` : "Using the supplied durable API key",
                    },
                    { name: "Environment", ok: true, detail: envPath },
                ];
            }
            case "sdk-verify": {
                const token = currentAnswers["api-key"] as string;
                const baseUrl = (currentAnswers["_api_base_url"] as string) || API_BASE_URL;
                const credentials = token.startsWith("xcel_ai_")
                    ? currentAnswers
                    : await checked(ensureWorkerOAuthClient(currentAnswers, token));
                const results = await checked(checkSdkApi(baseUrl, token,
                    credentials["oauth-client-id"] as string | undefined,
                    credentials["oauth-client-secret"] as string | undefined));
                for (const r of results) streamItem(r.name, r.ok, r.detail);
                return results;
            }
            default:
                return [{ name: checkId, ok: false, detail: "Unknown check" }];
        }
    }, [checkWallet, ensureWorkerOAuthClient, launchGpuInstance]);

    // ── Step advancement ─────────────────────────────────────────────

    const advanceToNext = useCallback(
        async (currentAnswers: Record<string, string | string[]>, resumeAt?: number) => {
            // Debounce rapid Enter presses — prevent double-advancing
            if (completedRef.current || !mountedRef.current) return;
            const now = Date.now();
            if (resumeAt === undefined && now - lastAdvanceRef.current < ADVANCE_DEBOUNCE_MS) return;
            lastAdvanceRef.current = now;

            setTransitioning(false);
            setValidationError(null);
            setConfirmError(null);
            setCheckCanRetry(false);
            setCheckAwaitContinue(false);
            setShowAiPrompt(false);
            setChatHistory([]);
            setCurrentAiQuestion(null);
            setAiStreaming(false);
            setAiResponse(null);

            const currentStep = WIZARD_STEPS[stepIndexRef.current];
            if (currentStep?.type === "device-auth") stopDevicePoll();
            if (resumeAt === undefined && currentStep && !completedStepIdsRef.current.includes(currentStep.id)) {
                completedStepIdsRef.current.push(currentStep.id);
            }

            const next = resumeAt ?? getNextStep(stepIndexRef.current, currentAnswers);
            if (next === -1 || WIZARD_STEPS[next].type === "done") {
                try {
                    saveConfig(currentAnswers);
                    if (checkpointTimerRef.current) clearTimeout(checkpointTimerRef.current);
                    clearWizardCheckpoint();
                } catch (err) {
                    setWizardState("error");
                    setWizardMessage(`Could not save configuration: ${err instanceof Error ? err.message : "unknown error"}. Fix the destination and continue this step to retry.`);
                    if (currentStep?.type === "auto-check") setCheckAwaitContinue(true);
                    return;
                }
                completedRef.current = true;
                stopDevicePoll();
                const doneIdx = WIZARD_STEPS.findIndex((s) => s.type === "done");
                stepIndexRef.current = doneIdx;
                setStepIndex(doneIdx);
                setWizardState("finishing");
                setWizardMessage(WIZARD_STEPS[doneIdx].prompt);
                setIsComplete(true);
                return;
            }

            const nextStep = WIZARD_STEPS[next];
            const entryGen = ++flowGenRef.current;
            const isEntryCurrent = () => mountedRef.current && !completedRef.current
                && !isFlowStale(entryGen) && stepIndexRef.current === next;
            activeCheckRef.current = null;
            checkRunningRef.current = nextStep.type === "auto-check";
            stepIndexRef.current = next;
            setStepIndex(next);
            scheduleCheckpoint({
                stepIndex: next,
                answers: currentAnswers,
                completedStepIds: [...completedStepIdsRef.current],
            });

            // Brief pause so the user can read the step prompt before init kicks in
            const needsInit = nextStep.type === "device-auth"
                || nextStep.type === "auto-check"
                || nextStep.type === "auto-fetch"
                || nextStep.type === "payment-gate";
            if (needsInit && resumeAt === undefined) {
                setWizardMessage(nextStep.prompt);
                await new Promise((r) => setTimeout(r, CHOREOGRAPHY_DELAY_MS));
            } else {
                setWizardMessage(nextStep.prompt);
            }
            if (!isEntryCurrent()) return;

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
                let walletPollInFlight = false;
                const pollGen = ++flowGenRef.current;
                walletPollRef.current = setInterval(async () => {
                    if (isFlowStale(pollGen)) {
                        if (walletPollRef.current) clearInterval(walletPollRef.current);
                        walletPollRef.current = null;
                        return;
                    }
                    if (walletPollInFlight) return;
                    const liveAnswers = answersRef.current;
                    const customerId = liveAnswers["_customer_id"] as string;
                    if (!customerId) return;
                    walletPollInFlight = true;
                    try {
                        const wallet = await getWallet(API_BASE_URL, liveAnswers["api-key"] as string, customerId);
                        if (isFlowStale(pollGen)) return;
                        walletPollFailures = 0;
                        setPaymentGate((prev) => ({ ...prev, balance: wallet.balance_cad }));
                        const listing = gpuListingsRef.current.find(
                            (l) => l.host_id === (liveAnswers["gpu-pick"] as string),
                        );
                        if (!listing || !Number.isFinite(listing.price_per_hour) || listing.price_per_hour < 0) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            setPaymentGate((prev) => ({ ...prev, polling: false }));
                            setWizardState("error");
                            setWizardMessage("The selected GPU price is unavailable. Skip this launch and select a GPU again.");
                            return;
                        }
                        if (wallet.balance_cad >= listing.price_per_hour) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            const updated = {
                                ...liveAnswers,
                                "_wallet_insufficient": "false",
                                "payment-gate": "funded",
                            };
                            answersRef.current = updated;
                            setAnswers(updated);
                            setWizardState("excited");
                            setWizardMessage("Wallet funded! Proceeding...");
                            setPaymentGate((prev) => ({ ...prev, polling: false }));
                            transitionTimerRef.current = setTimeout(() => {
                                transitionTimerRef.current = null;
                                if (!isFlowStale(pollGen)) void advanceToNext(updated);
                            }, CHOREOGRAPHY_DELAY_MS);
                        }
                    } catch {
                        if (isFlowStale(pollGen)) return;
                        walletPollFailures++;
                        if (walletPollFailures >= 10) {
                            if (walletPollRef.current) clearInterval(walletPollRef.current);
                            walletPollRef.current = null;
                            setWizardState("error");
                            setPaymentGate((prev) => ({ ...prev, polling: false }));
                            setWizardMessage("Wallet polling failed repeatedly — press s to skip");
                        }
                    } finally {
                        walletPollInFlight = false;
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
                    if (!isEntryCurrent()) return;
                    const allPassed = results.every((r) => r.ok);
                    setCheckResults((prev) => ({
                        ...prev,
                        [nextStep.id]: { items: results, allPassed },
                    }));

                    if (allPassed) {
                        // Use "excited" (dance) for big milestones, "success" (eureka) for routine
                        // Local checks and a pending registration are progress,
                        // not a verified outcome, so they get the brief acknowledgement.
                        const isMilestone = nextStep.checkId === "launch" || nextStep.checkId === "docker"
                            || nextStep.checkId === "sdk-verify" || nextStep.checkId === "sdk-credentials";
                        setWizardState(isMilestone ? "excited" : "success");
                        const successMsg = nextStep.checkId === "launch" ? "Instance launched!"
                            : nextStep.checkId === "benchmark" ? "Benchmarks complete!"
                                : nextStep.checkId === "verify" ? "Local hardware checks passed!"
                                    : nextStep.checkId === "host-register" ? "Host registered as pending verification."
                                        : nextStep.checkId === "admission" ? "Compatibility recorded; admission remains pending."
                                        : nextStep.checkId === "docker" ? "Docker environment ready!"
                                            : nextStep.checkId === "sdk-detect" ? "Project detected!"
                                                : nextStep.checkId === "sdk-install" ? "SDK package ready!"
                                                    : nextStep.checkId === "sdk-credentials" ? "Credentials saved!"
                                                        : nextStep.checkId === "sdk-verify" ? "API connection verified!"
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
                        const failSummary = summarizeFailure(results);
                        setWizardMessage(apiToken ? `${failSummary} — Hexara is on it` : failSummary);
                        setCheckCanRetry(true);

                        // Auto-trigger AI analysis for check failures if authenticated
                        // (skip api-check — can't reach AI if API is down)
                        if (
                            apiToken
                            && nextStep.checkId
                            && nextStep.checkId !== "api"
                            && shouldRunAutoAnalysis(nextStep.id, failDetails)
                        ) {
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
                                        if (!isEntryCurrent()) return;
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
                                    if (!isEntryCurrent()) return;
                                    // Stream complete — reveal full analysis
                                    if (explanation) {
                                        setAiResponse(explanation);
                                        setWizardState("error");
                                        setWizardMessage(`${failCount} issue(s) found — see analysis below`);
                                    }
                                } catch (err) {
                                    if (!isEntryCurrent()) return;
                                    const msg = err instanceof Error ? err.message : "unknown";
                                    setAiResponse(explanation || `Analysis failed: ${msg}`);
                                    setWizardState("error");
                                    setWizardMessage(`${failCount} check(s) failed`);
                                } finally {
                                    if (isEntryCurrent()) setAiStreaming(false);
                                }
                            })();
                        }
                    }
                }).catch(() => {
                    if (!isEntryCurrent()) return;
                    setWizardState("error");
                    setWizardMessage("Check failed unexpectedly — retry");
                    setCheckCanRetry(true);
                }).finally(() => {
                    if (isEntryCurrent()) checkRunningRef.current = false;
                });
                return;
            }

            if (nextStep.type === "confirm") {
                if (nextStep.id === "provider-summary") {
                    setWizardState("success");
                } else if (nextStep.id === "sdk-snippet") {
                    setWizardState("excited");
                    setWizardMessage("Your SDK integration is ready — copy the starter code below!");
                } else {
                    setWizardState("idle");
                }
                return;
            }

            // Default — set appropriate state
            setWizardState("idle");
        },
        [startDeviceAuth, stopDevicePoll, browseGpus, runCheck, gpuListings, paymentGate.billingUrl, scheduleCheckpoint, isFlowStale, shouldRunAutoAnalysis],
    );

    // ── Submit answer ────────────────────────────────────────────────

    const submitAnswer = useCallback(
        (value: string | string[]) => {
            if (completedRef.current || transitionTimerRef.current) return;
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
                            setWizardState("idle");
                            setWizardMessage("Launch cancelled. No instance was launched.");
                            completedRef.current = true;
                            if (checkpointTimerRef.current) clearTimeout(checkpointTimerRef.current);
                            clearWizardCheckpoint();
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
                            setWizardState("idle");
                            setWizardMessage("Setup cancelled. Run the wizard again when you're ready.");
                            completedRef.current = true;
                            if (checkpointTimerRef.current) clearTimeout(checkpointTimerRef.current);
                            clearWizardCheckpoint();
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
                "mode": value === "sdk"
                    ? "SDK track — let's wire your app to Xcelsior..."
                    : "Great choice! Let's get you set up...",
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
            scheduleCheckpoint({ answers: updated });

            transitionTimerRef.current = setTimeout(() => {
                transitionTimerRef.current = null;
                void advanceToNext(answersRef.current);
            }, CHOREOGRAPHY_DELAY_MS);
        },
        [advanceToNext],
    );

    // ── Check retry/skip ─────────────────────────────────────────────

    const retryCheck = useCallback(() => {
        if (!activeCheckRef.current || checkRunningRef.current || completedRef.current) return;
        // Reuse the same entry path and lifecycle guards as the first attempt.
        void advanceToNext(answersRef.current, stepIndexRef.current);
    }, [advanceToNext]);

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
        if (deviceAuth.status !== "authorized" || tokenSaveError) return;
        stopDevicePoll();
        advanceToNext(answersRef.current);
    }, [advanceToNext, deviceAuth.status, tokenSaveError, stopDevicePoll]);

    // ── Payment skip ─────────────────────────────────────────────────

    const skipPayment = useCallback(() => {
        flowGenRef.current += 1;
        if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
        transitionTimerRef.current = null;
        if (walletPollRef.current) {
            clearInterval(walletPollRef.current);
            walletPollRef.current = null;
        }
        const updated = { ...answersRef.current, "payment-gate": "skipped", "want-launch": "no" };
        answersRef.current = updated;
        setAnswers(updated);
        setPaymentGate((prev) => ({ ...prev, polling: false }));
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
                const staticHelp = STATIC_STEP_HELP[step.id];
                const helpHint = staticHelp?.length ? ` ${staticHelp[0]}` : "";
                setWizardMessage(
                    content
                        ? `AI error — press d to see details.${helpHint}`
                        : `AI unavailable — continue with the wizard steps.${helpHint}`,
                );
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

    // Restored steps need the same initialization as normal navigation. Wait
    // for preflight before starting network requests, processes or wallet polls.
    const resumeInitializedRef = useRef(false);
    useEffect(() => {
        if (gatePhase !== "passed" || resumeInitializedRef.current) return;
        resumeInitializedRef.current = true;
        if (resumeInfo.expired) setResumeInfo((prev) => ({ ...prev, expired: false }));
        if (initialCheckpoint) void advanceToNext(answersRef.current, stepIndexRef.current);
    }, [gatePhase, initialCheckpoint, advanceToNext, resumeInfo.expired]);

    // Cleanup polls and persist progress on unmount
    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            flowGenRef.current += 1;
            gateGenRef.current += 1;
            stopDevicePoll();
            if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current);
            if (walletPollRef.current) clearInterval(walletPollRef.current);
            if (checkpointTimerRef.current) clearTimeout(checkpointTimerRef.current);
            if (!completedRef.current) {
                saveWizardCheckpoint({
                runtime: runtimeSnapshot(),
                    stepIndex: stepIndexRef.current,
                    answers: answersRef.current,
                    conversationId: conversationIdRef.current ?? undefined,
                    completedStepIds: [...completedStepIdsRef.current],
                    savedAt: new Date().toISOString(),
                });
            }
        };
    }, [stopDevicePoll]);

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
        resumeInfo,
        flushCheckpoint,
        gatePhase,
        serviceStatus,
        proceedFromGate,
        recheckGate,
        continueAnywayFromGate,
        resumedStepLabel: resumeInfo.resumed ? labelForStep(initialStep) : null,
        checkProgress,
        marketplaceStats,
    };
}
