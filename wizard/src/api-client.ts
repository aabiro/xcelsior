// API client for the Xcelsior wizard.
// Connects to AI chat SSE, device auth, marketplace, billing, and instance endpoints.

import http from "node:http";
import https from "node:https";
import { parseSseBuffer } from "./sse-parser.js";
import { redactSecrets } from "./wizard-state.js";

export interface SSEEvent {
    type: "meta" | "token" | "tool_call" | "tool_result" | "confirmation_required" | "done" | "error";
    content?: string;
    conversation_id?: string;
    name?: string;
    input?: Record<string, unknown>;
    output?: Record<string, unknown>;
    confirmation_id?: string;
    tool_name?: string;
    tool_args?: Record<string, unknown>;
    message?: string;
}

export interface ApiClientConfig {
    baseUrl: string;
    apiKey: string;
    pageContext?: string;
}

export class WizardError extends Error {
    code: string;
    remediation?: string;
    url?: string;

    constructor(message: string, opts: { code: string; remediation?: string; url?: string }) {
        super(message);
        this.name = "WizardError";
        this.code = opts.code;
        this.remediation = opts.remediation;
        this.url = opts.url;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

function transport(url: URL) {
    return url.protocol === "https:" ? https : http;
}

const RETRYABLE_STATUS = new Set([408, 500, 502, 503, 504]);
const IDEMPOTENT_METHODS = new Set(["GET", "HEAD", "PUT"]);

function retryDelay(attempt: number, retryAfterSec?: number): number {
    if (retryAfterSec && retryAfterSec > 0) return Math.min(retryAfterSec * 1000, 120_000);
    const cap = Math.min(30_000, 1000 * 2 ** attempt);
    return Math.floor(Math.random() * cap);
}

function mapHttpError(status: number, detail: string, baseUrl: string): WizardError {
    if (status === 401) {
        return new WizardError(detail || "Unauthorized", {
            code: "auth_required",
            remediation: "Re-authenticate with device flow (Enter) or paste a fresh OAuth token (m).",
        });
    }
    if (status === 402) {
        return new WizardError(detail || "Payment required", {
            code: "payment_required",
            remediation: "Add funds or resolve billing suspension in the dashboard.",
            url: `${baseUrl.replace(/\/api$/, "")}/dashboard/billing`,
        });
    }
    if (status === 403) {
        return new WizardError(detail || "Forbidden", {
            code: "forbidden",
            remediation: "Check account permissions or contact support.",
        });
    }
    if (status === 404) {
        return new WizardError(detail || "Not found", {
            code: "not_found",
            remediation: "Verify the resource exists and your account has access.",
        });
    }
    return new WizardError(detail || `HTTP ${status}`, {
        code: "http_error",
        remediation: "Retry in a moment. If this persists, check connectivity to the API.",
    });
}

async function sleep(ms: number): Promise<void> {
    await new Promise((r) => setTimeout(r, ms));
}

async function jsonRequestOnce<T>(
    method: string,
    baseUrl: string,
    path: string,
    body?: Record<string, unknown>,
    token?: string,
    timeoutMs = 30_000,
): Promise<{ status: number; data: T; retryAfterSec?: number }> {
    const url = new URL(path, baseUrl);
    const t = transport(url);

    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const payload = body ? JSON.stringify(body) : undefined;

    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
        const req = t.request(
            {
                hostname: url.hostname,
                port: url.port || (url.protocol === "https:" ? 443 : 80),
                path: url.pathname + url.search,
                method,
                headers,
                timeout: timeoutMs,
            },
            resolve,
        );
        req.on("error", reject);
        req.on("timeout", () => { req.destroy(new Error("Request timed out")); });
        if (payload) req.write(payload);
        req.end();
    });

    const chunks: Buffer[] = [];
    for await (const chunk of response) chunks.push(chunk as Buffer);
    const text = Buffer.concat(chunks).toString();
    let data: T;
    try {
        data = text ? JSON.parse(text) as T : {} as T;
    } catch {
        throw new WizardError(
            `Invalid JSON response (HTTP ${response.statusCode ?? "?"}): ${text.slice(0, 200)}`,
            { code: "bad_json", remediation: "The API returned a non-JSON body. Retry or check API status." },
        );
    }
    const retryAfter = response.headers["retry-after"];
    const retryAfterSec = retryAfter ? Number(Array.isArray(retryAfter) ? retryAfter[0] : retryAfter) : undefined;
    return { status: response.statusCode ?? 500, data, retryAfterSec: Number.isFinite(retryAfterSec) ? retryAfterSec : undefined };
}

async function jsonRequest<T>(
    method: string,
    baseUrl: string,
    path: string,
    body?: Record<string, unknown>,
    token?: string,
    opts?: { maxRetries?: number; timeoutMs?: number },
): Promise<{ status: number; data: T }> {
    const maxRetries = opts?.maxRetries ?? 4;
    const retriable = IDEMPOTENT_METHODS.has(method.toUpperCase())
        || path.includes("/api/auth/token")
        || path.includes("/api/auth/device");
    let lastErr: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
            const result = await jsonRequestOnce<T>(
                method, baseUrl, path, body, token, opts?.timeoutMs ?? 30_000,
            );
            if (retriable && RETRYABLE_STATUS.has(result.status) && attempt < maxRetries) {
                await sleep(retryDelay(attempt, result.retryAfterSec));
                continue;
            }
            return { status: result.status, data: result.data };
        } catch (err) {
            lastErr = err instanceof Error ? err : new Error(String(err));
            if (!retriable || attempt >= maxRetries) break;
            await sleep(retryDelay(attempt));
        }
    }
    throw new WizardError(redactSecrets(lastErr?.message ?? "Request failed"), {
        code: "network_error",
        remediation: "Check network connectivity and retry.",
    });
}

async function* readSseResponse(response: http.IncomingMessage): AsyncGenerator<SSEEvent> {
    let buffer = "";
    for await (const chunk of response) {
        buffer += chunk.toString();
        const { frames, remainder } = parseSseBuffer(buffer);
        buffer = remainder;
        for (const frame of frames) {
            const jsonStr = frame.data.trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try {
                yield JSON.parse(jsonStr) as SSEEvent;
            } catch {
                // skip malformed frame
            }
        }
    }
    if (buffer.trim()) {
        const { frames } = parseSseBuffer(buffer + "\n\n");
        for (const frame of frames) {
            const jsonStr = frame.data.trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try { yield JSON.parse(jsonStr) as SSEEvent; } catch { /* skip */ }
        }
    }
}

// ── AI Chat ──────────────────────────────────────────────────────────

export async function* streamChat(
    config: ApiClientConfig,
    message: string,
    conversationId?: string,
): AsyncGenerator<SSEEvent> {
    const body = JSON.stringify({
        message,
        conversation_id: conversationId ?? null,
        page_context: config.pageContext ?? "cli-wizard",
    });

    const url = new URL("/api/ai/chat", config.baseUrl);
    const t = transport(url);

    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
        const req = t.request(
            {
                hostname: url.hostname,
                port: url.port || (url.protocol === "https:" ? 443 : 80),
                path: url.pathname,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${config.apiKey}`,
                    "Accept": "text/event-stream",
                },
                timeout: 120_000,
            },
            resolve,
        );
        req.on("error", reject);
        req.on("timeout", () => req.destroy(new Error("Stream request timed out")));
        req.write(body);
        req.end();
    });

    if (response.statusCode !== 200) {
        const chunks: Buffer[] = [];
        for await (const chunk of response) chunks.push(chunk as Buffer);
        const text = Buffer.concat(chunks).toString();
        yield { type: "error", message: `HTTP ${response.statusCode}: ${text}` };
        return;
    }

    yield* readSseResponse(response);
}

export async function* confirmAction(
    config: ApiClientConfig,
    confirmationId: string,
    approved: boolean,
): AsyncGenerator<SSEEvent> {
    const body = JSON.stringify({ confirmation_id: confirmationId, approved });
    const url = new URL("/api/ai/confirm", config.baseUrl);
    const t = transport(url);

    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
        const req = t.request(
            {
                hostname: url.hostname,
                port: url.port || (url.protocol === "https:" ? 443 : 80),
                path: url.pathname,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${config.apiKey}`,
                    "Accept": "text/event-stream",
                },
                timeout: 120_000,
            },
            resolve,
        );
        req.on("error", reject);
        req.on("timeout", () => req.destroy(new Error("Confirm request timed out")));
        req.write(body);
        req.end();
    });

    if (response.statusCode !== 200) {
        const chunks: Buffer[] = [];
        for await (const chunk of response) chunks.push(chunk as Buffer);
        const text = Buffer.concat(chunks).toString();
        yield { type: "error", message: `HTTP ${response.statusCode}: ${text}` };
        return;
    }

    yield* readSseResponse(response);
}

// ── Device Auth (RFC 8628) ───────────────────────────────────────────

export interface DeviceCodeResult {
    device_code: string;
    user_code: string;
    verification_uri: string;
    expires_in: number;
    interval: number;
}

export interface DeviceTokenResult {
    access_token: string;
    token_type: string;
    expires_in: number;
}

export type DevicePollResult =
    | { status: "authorized"; token: DeviceTokenResult }
    | { status: "pending" }
    | { status: "slow_down"; intervalSec?: number }
    | { status: "expired" }
    | { status: "error"; message: string };

export async function requestDeviceCode(baseUrl: string): Promise<DeviceCodeResult> {
    const { status, data } = await jsonRequest<DeviceCodeResult>(
        "POST", baseUrl, "/api/auth/device",
    );
    if (status !== 200) {
        throw mapHttpError(status, `Device auth init failed: HTTP ${status}`, baseUrl);
    }
    return data;
}

export async function pollDeviceToken(
    baseUrl: string,
    deviceCode: string,
): Promise<DevicePollResult> {
    const { status, data } = await jsonRequest<DeviceTokenResult & { detail?: string; error?: string }>(
        "POST", baseUrl, "/api/auth/token",
        { device_code: deviceCode, grant_type: "urn:ietf:params:oauth:grant-type:device_code" },
    );

    if (status === 200) return { status: "authorized", token: data };
    if (status === 428) return { status: "pending" };
    if (status === 429) return { status: "slow_down" };
    if (status === 410) return { status: "expired" };

    const detail = (data as { detail?: string }).detail
        ?? (data as { error?: string }).error
        ?? "unknown error";
    if (detail.includes("slow_down")) return { status: "slow_down" };
    if (detail.includes("expired")) return { status: "expired" };
    return { status: "error", message: detail };
}

// ── OAuth client (replaces deprecated API key generation — A4) ───────

export interface OAuthClientCreateParams {
    client_name: string;
    client_type?: "public" | "confidential";
    redirect_uris?: string[];
    grant_types?: string[];
    scopes?: string[];
    is_first_party?: boolean;
}

export interface OAuthClientCredentials {
    client_id: string;
    client_secret: string;
    client_name: string;
    client_type: string;
    grant_types: string[];
    scopes: string[];
}

/** @deprecated API keys are permanently disabled — wizard uses OAuth session + scoped worker clients */
export async function generateApiKey(
    _baseUrl: string,
    _sessionToken: string,
    _name: string = "CLI Wizard",
): Promise<{ key: string; name: string }> {
    throw new WizardError("API keys are disabled — use OAuth device/session tokens", {
        code: "api_keys_disabled",
        remediation: "Continue with the session token from device auth; worker credentials use OAuth client_credentials.",
    });
}

// ── User Profile ─────────────────────────────────────────────────────

export interface UserProfile {
    user_id: string;
    email: string;
    name: string;
    role: string;
    customer_id: string;
    country: string;
    province: string;
}

export async function getMe(baseUrl: string, token: string): Promise<UserProfile> {
    const { status, data } = await jsonRequest<{ ok: boolean; user: UserProfile }>(
        "GET", baseUrl, "/api/auth/me", undefined, token,
    );
    if (status !== 200 || !data.user) {
        throw mapHttpError(status, `Failed to get user profile: HTTP ${status}`, baseUrl);
    }
    return data.user;
}

export async function createOAuthClient(
    baseUrl: string,
    token: string,
    params: OAuthClientCreateParams,
): Promise<OAuthClientCredentials> {
    const { status, data } = await jsonRequest<{
        ok: boolean;
        client: OAuthClientCredentials;
    }>(
        "POST",
        baseUrl,
        "/api/oauth/clients",
        {
            client_name: params.client_name,
            client_type: params.client_type ?? "confidential",
            redirect_uris: params.redirect_uris ?? [],
            grant_types: params.grant_types ?? ["client_credentials"],
            scopes: params.scopes ?? ["api"],
            is_first_party: params.is_first_party ?? false,
        },
        token,
    );
    if (status !== 200 || !data.client?.client_id || !data.client?.client_secret) {
        throw mapHttpError(status, `OAuth client creation failed: HTTP ${status}`, baseUrl);
    }
    return data.client;
}

// ── Marketplace ──────────────────────────────────────────────────────

export interface MarketplaceListing {
    host_id: string;
    gpu_model: string;
    vram_gb: number;
    price_per_hour: number;
    owner: string;
    active: boolean;
    total_jobs: number;
    total_earned: number;
    description: string;
}

export interface MarketplaceSearchResult {
    ok: boolean;
    total: number;
    listings: MarketplaceListing[];
}

export interface MarketplaceFilters {
    gpu_model?: string;
    min_vram?: number;
    max_price?: number;
    sort_by?: "price" | "vram" | "reputation" | "score";
    limit?: number;
}

export async function searchMarketplace(
    baseUrl: string,
    token: string,
    filters?: MarketplaceFilters,
): Promise<MarketplaceSearchResult> {
    const params = new URLSearchParams();
    if (filters?.gpu_model) params.set("gpu_model", filters.gpu_model);
    if (filters?.min_vram) params.set("min_vram", String(filters.min_vram));
    if (filters?.max_price) params.set("max_price", String(filters.max_price));
    if (filters?.sort_by) params.set("sort_by", filters.sort_by);
    if (filters?.limit) params.set("limit", String(filters.limit));

    const qs = params.toString();
    const path = `/marketplace/search${qs ? `?${qs}` : ""}`;

    const { status, data } = await jsonRequest<MarketplaceSearchResult>(
        "GET", baseUrl, path, undefined, token,
    );
    if (status !== 200) {
        throw mapHttpError(status, `Marketplace search failed: HTTP ${status}`, baseUrl);
    }
    return data;
}

// ── Billing / Wallet ─────────────────────────────────────────────────

export interface Wallet {
    customer_id: string;
    balance_cad: number;
    total_deposited_cad: number;
    total_spent_cad: number;
    status: string;
}

export async function getWallet(
    baseUrl: string,
    token: string,
    customerId: string,
): Promise<Wallet> {
    const { status, data } = await jsonRequest<{ ok: boolean; wallet: Wallet }>(
        "GET", baseUrl, `/api/billing/wallet/${encodeURIComponent(customerId)}`,
        undefined, token,
    );
    if (status !== 200) {
        throw mapHttpError(status, `Wallet fetch failed: HTTP ${status}`, baseUrl);
    }
    return data.wallet;
}

export async function claimFreeCredits(
    baseUrl: string,
    token: string,
    customerId: string,
): Promise<{ ok: boolean; amount: number; already_claimed: boolean }> {
    const { status, data } = await jsonRequest<{ ok: boolean; amount: number; already_claimed?: boolean }>(
        "POST", baseUrl, `/api/billing/free-credits/${encodeURIComponent(customerId)}`,
        undefined, token,
    );
    if (status === 200) return { ok: true, amount: data.amount ?? 10, already_claimed: false };
    if (status === 409) return { ok: true, amount: 0, already_claimed: true };
    throw mapHttpError(status, `Free credits claim failed: HTTP ${status}`, baseUrl);
}

export async function checkFreeCreditStatus(
    baseUrl: string,
    token: string,
    customerId: string,
): Promise<boolean> {
    const { status, data } = await jsonRequest<{ claimed: boolean }>(
        "GET", baseUrl, `/api/billing/free-credits/${encodeURIComponent(customerId)}/status`,
        undefined, token,
    );
    if (status !== 200) return false;
    return data.claimed;
}

// ── Instance Management ──────────────────────────────────────────────

export interface LaunchParams {
    name: string;
    host_id: string;
    image: string;
    interactive?: boolean;
    num_gpus?: number;
    vram_needed_gb?: number;
}

export interface InstanceInfo {
    job_id: string;
    name: string;
    status: string;
    host_id: string;
    host_ip?: string;
    host_gpu?: string;
    host_vram_gb?: number;
    ssh_port?: number;
    container_id?: string;
    submitted_by?: string;
}

export async function launchInstance(
    baseUrl: string,
    token: string,
    params: LaunchParams,
): Promise<InstanceInfo> {
    const { status, data } = await jsonRequest<{ ok: boolean; instance: InstanceInfo }>(
        "POST", baseUrl, "/instance",
        {
            name: params.name,
            host_id: params.host_id,
            image: params.image,
            interactive: params.interactive ?? true,
            num_gpus: params.num_gpus ?? 1,
            vram_needed_gb: params.vram_needed_gb ?? 0,
        },
        token,
    );
    if (status !== 200 || !data.instance) {
        const detail = (data as Record<string, unknown>)?.detail;
        if (status === 402) {
            const msg = typeof detail === "string" && /suspend/i.test(detail)
                ? "Wallet suspended — visit the dashboard to resolve"
                : `Insufficient balance — add funds at ${baseUrl.replace(/\/api$/, "")}/dashboard/billing`;
            throw new WizardError(msg, {
                code: "payment_required",
                remediation: "Add funds in the billing dashboard.",
                url: `${baseUrl.replace(/\/api$/, "")}/dashboard/billing`,
            });
        }
        throw mapHttpError(status, typeof detail === "string" ? detail : `Instance launch failed: HTTP ${status}`, baseUrl);
    }
    return data.instance;
}

export async function getInstance(
    baseUrl: string,
    token: string,
    jobId: string,
): Promise<InstanceInfo> {
    const { status, data } = await jsonRequest<{ ok: boolean; instance: InstanceInfo }>(
        "GET", baseUrl, `/instance/${encodeURIComponent(jobId)}`,
        undefined, token,
    );
    if (status !== 200 || !data.instance) {
        throw mapHttpError(status, `Instance fetch failed: HTTP ${status}`, baseUrl);
    }
    return data.instance;
}

// ── Provider Host Registration ───────────────────────────────────────

export interface HostRegistration {
    host_id: string;
    ip: string;
    gpu_model: string;
    total_vram_gb: number;
    free_vram_gb: number;
    cost_per_hour: number;
    country?: string;
    province?: string;
    versions?: Record<string, string>;
    spot_enabled?: boolean;
    spot_gpu_slots?: number;
    spot_min_cents?: number;
}

export interface HostInfo {
    host_id: string;
    gpu_model: string;
    total_vram_gb: number;
    cost_per_hour: number;
    status: string;
    ip: string;
}

export async function registerHost(
    baseUrl: string,
    token: string,
    host: HostRegistration,
): Promise<HostInfo> {
    const { status, data } = await jsonRequest<{ ok: boolean; host: HostInfo }>(
        "PUT", baseUrl, "/host", host as unknown as Record<string, unknown>, token,
    );
    if (status !== 200 || !data.host) {
        throw mapHttpError(status, `Host registration failed: HTTP ${status}`, baseUrl);
    }
    return data.host;
}

export async function getHostStatus(
    baseUrl: string,
    token: string,
    hostId: string,
): Promise<HostInfo> {
    const { status, data } = await jsonRequest<{ ok: boolean; host: HostInfo }>(
        "GET", baseUrl, `/host/${encodeURIComponent(hostId)}`,
        undefined, token,
    );
    if (status !== 200 || !data.host) {
        throw mapHttpError(status, `Host status fetch failed: HTTP ${status}`, baseUrl);
    }
    return data.host;
}

// ── Provider Agent Reports ───────────────────────────────────────────

export interface VersionReportResult {
    ok: boolean;
    admitted: boolean;
    details: Record<string, unknown>;
}

export async function reportVersions(
    baseUrl: string,
    token: string,
    hostId: string,
    versions: Record<string, string>,
): Promise<VersionReportResult> {
    const { status, data } = await jsonRequest<VersionReportResult>(
        "POST", baseUrl, "/agent/versions",
        { host_id: hostId, versions }, token,
    );
    if (status !== 200) {
        throw mapHttpError(status, `Version report failed: HTTP ${status}`, baseUrl);
    }
    return data;
}

export interface BenchmarkReportResult {
    ok: boolean;
    xcu: number;
}

export async function reportBenchmark(
    baseUrl: string,
    token: string,
    hostId: string,
    gpuModel: string,
    score: number,
    tflops: number,
    details?: Record<string, unknown>,
): Promise<BenchmarkReportResult> {
    const { status, data } = await jsonRequest<BenchmarkReportResult>(
        "POST", baseUrl, "/agent/benchmark",
        { host_id: hostId, gpu_model: gpuModel, score, tflops, details }, token,
    );
    if (status !== 200) {
        throw mapHttpError(status, `Benchmark report failed: HTTP ${status}`, baseUrl);
    }
    return data;
}

export interface VerifyResult {
    ok: boolean;
    host_id: string;
    state: string;
    score: number;
    checks: Record<string, unknown>;
    gpu_fingerprint: string;
}

export async function reportVerification(
    baseUrl: string,
    token: string,
    hostId: string,
    report: Record<string, unknown>,
): Promise<VerifyResult> {
    const { status, data } = await jsonRequest<VerifyResult>(
        "POST", baseUrl, "/agent/verify",
        { host_id: hostId, report }, token,
    );
    if (status !== 200) {
        throw mapHttpError(status, `Verification report failed: HTTP ${status}`, baseUrl);
    }
    return data;
}