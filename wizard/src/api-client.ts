// API client for the Xcelsior wizard.
// Connects to AI chat SSE, device auth, marketplace, billing, and instance endpoints.

import http from "node:http";
import https from "node:https";

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
    baseUrl: string;       // e.g. "https://xcelsior.ca" or "http://localhost:9500"
    apiKey: string;        // Bearer token for auth
    pageContext?: string;  // optional context hint
}

// ── Helpers ──────────────────────────────────────────────────────────

function transport(url: URL) {
    return url.protocol === "https:" ? https : http;
}

async function jsonRequest<T>(
    method: string,
    baseUrl: string,
    path: string,
    body?: Record<string, unknown>,
    token?: string,
): Promise<{ status: number; data: T }> {
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
                timeout: 30_000,
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
        throw new Error(`Invalid JSON response (HTTP ${response.statusCode ?? "?"}): ${text.slice(0, 200)}`);
    }
    return { status: response.statusCode ?? 500, data };
}

// ── AI Chat ──────────────────────────────────────────────────────────

/**
 * Stream a chat message to the Hexara AI assistant.
 * Yields parsed SSE events as they arrive.
 */
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

    // Parse SSE stream
    let buffer = "";
    for await (const chunk of response) {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try {
                const event = JSON.parse(jsonStr) as SSEEvent;
                yield event;
            } catch {
                // skip malformed SSE lines
            }
        }
    }
    // Process remaining buffer after stream ends
    if (buffer.startsWith("data: ")) {
        const jsonStr = buffer.slice(6).trim();
        if (jsonStr && jsonStr !== "[DONE]") {
            try { yield JSON.parse(jsonStr) as SSEEvent; } catch { /* skip */ }
        }
    }
}

/**
 * Confirm or reject a pending write action.
 */
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

    // Guard against non-200 responses
    if (response.statusCode !== 200) {
        const chunks: Buffer[] = [];
        for await (const chunk of response) chunks.push(chunk as Buffer);
        const text = Buffer.concat(chunks).toString();
        yield { type: "error", message: `HTTP ${response.statusCode}: ${text}` };
        return;
    }

    let buffer = "";
    for await (const chunk of response) {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try {
                yield JSON.parse(jsonStr) as SSEEvent;
            } catch {
                // skip
            }
        }
    }
    // Process remaining buffer after stream ends
    if (buffer.startsWith("data: ")) {
        const jsonStr = buffer.slice(6).trim();
        if (jsonStr && jsonStr !== "[DONE]") {
            try { yield JSON.parse(jsonStr) as SSEEvent; } catch { /* skip */ }
        }
    }
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

/** Initiate device authorization flow. */
export async function requestDeviceCode(baseUrl: string): Promise<DeviceCodeResult> {
    const { status, data } = await jsonRequest<DeviceCodeResult>(
        "POST", baseUrl, "/api/auth/device",
    );
    if (status !== 200) {
        throw new Error(`Device auth init failed: HTTP ${status}`);
    }
    return data;
}

/**
 * Poll for device token. Returns the token result or null if still pending.
 * Throws on expired or invalid codes.
 */
export async function pollDeviceToken(
    baseUrl: string,
    deviceCode: string,
): Promise<DeviceTokenResult | null> {
    const { status, data } = await jsonRequest<DeviceTokenResult & { detail?: string }>(
        "POST", baseUrl, "/api/auth/token",
        { device_code: deviceCode, grant_type: "urn:ietf:params:oauth:grant-type:device_code" },
    );

    if (status === 200) return data;
    if (status === 428) return null; // authorization_pending

    const detail = (data as { detail?: string }).detail ?? "unknown error";
    throw new Error(detail);
}

// ── API Key Generation ───────────────────────────────────────────────

/** Generate a proper API key that appears in the user's dashboard Settings. */
export async function generateApiKey(
    baseUrl: string,
    sessionToken: string,
    name: string = "CLI Wizard",
): Promise<{ key: string; name: string }> {
    const { status, data } = await jsonRequest<{ ok: boolean; key: string; name: string }>(
        "POST", baseUrl, "/api/keys/generate",
        { name, scope: "full-access" },
        sessionToken,
    );
    if (status !== 200 || !data.ok) throw new Error("Failed to generate API key");
    return { key: data.key, name: data.name };
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

/** Get current user profile. */
export async function getMe(
    baseUrl: string,
    token: string,
): Promise<UserProfile> {
    const { status, data } = await jsonRequest<{ ok: boolean; user: UserProfile }>(
        "GET", baseUrl, "/api/auth/me", undefined, token,
    );
    if (status !== 200 || !data.user) {
        throw new Error(`Failed to get user profile: HTTP ${status}`);
    }
    return data.user;
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

/** Search the GPU marketplace. */
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
        throw new Error(`Marketplace search failed: HTTP ${status}`);
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

/** Get wallet for a customer. */
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
        throw new Error(`Wallet fetch failed: HTTP ${status}`);
    }
    return data.wallet;
}

/** Claim one-time free credits ($10 CAD). Idempotent. */
export async function claimFreeCredits(
    baseUrl: string,
    token: string,
    customerId: string,
): Promise<{ ok: boolean; amount: number; already_claimed: boolean }> {
    const { status, data } = await jsonRequest<{ ok: boolean; amount: number; already_claimed?: boolean }>(
        "POST", baseUrl, `/api/billing/free-credits/${encodeURIComponent(customerId)}`,
        undefined, token,
    );
    // Both 200 (claimed) and 409 (already claimed) are acceptable
    if (status === 200) return { ok: true, amount: data.amount ?? 10, already_claimed: false };
    if (status === 409) return { ok: true, amount: 0, already_claimed: true };
    throw new Error(`Free credits claim failed: HTTP ${status}`);
}

/** Check if free credits were already claimed. */
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

/** Launch a new instance. */
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
        throw new Error(`Instance launch failed: HTTP ${status}`);
    }
    return data.instance;
}

/** Get instance details (enriched with connection info). */
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
        throw new Error(`Instance fetch failed: HTTP ${status}`);
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
}

export interface HostInfo {
    host_id: string;
    gpu_model: string;
    total_vram_gb: number;
    cost_per_hour: number;
    status: string;
    ip: string;
}

/** Register or update a host. */
export async function registerHost(
    baseUrl: string,
    token: string,
    host: HostRegistration,
): Promise<HostInfo> {
    const { status, data } = await jsonRequest<{ ok: boolean; host: HostInfo }>(
        "PUT", baseUrl, "/host", host as unknown as Record<string, unknown>, token,
    );
    if (status !== 200 || !data.host) {
        throw new Error(`Host registration failed: HTTP ${status}`);
    }
    return data.host;
}

/** Get host status. */
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
        throw new Error(`Host status fetch failed: HTTP ${status}`);
    }
    return data.host;
}

// ── Provider Agent Reports ───────────────────────────────────────────

export interface VersionReportResult {
    ok: boolean;
    admitted: boolean;
    details: Record<string, unknown>;
}

/** Report component versions for admission check. */
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
        throw new Error(`Version report failed: HTTP ${status}`);
    }
    return data;
}

export interface BenchmarkReportResult {
    ok: boolean;
    xcu: number;
}

/** Report benchmark scores. */
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
        throw new Error(`Benchmark report failed: HTTP ${status}`);
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

/** Submit verification report. */
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
        throw new Error(`Verification report failed: HTTP ${status}`);
    }
    return data;
}
