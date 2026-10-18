// preflight.ts — Service-health gate data source (Part A).
//
// Primary source is the backend aggregator GET /api/status, which probes every
// subsystem and returns { ok, verdict, services[] }. When that endpoint is
// unavailable (older control plane), we fall back to a client-side probe of
// /healthz so the gate can still render a verdict.
//
// Pure helpers (classifyVerdict, parseStatusResponse, isServiceOk) are exported
// for unit tests; fetchServiceStatus does the I/O and never throws.

export type ServiceState = "operational" | "degraded" | "down" | "unknown";
export type Verdict = "operational" | "degraded" | "blocked";

export interface ServiceStatus {
    name: string;
    state: ServiceState;
    detail: string;
    required: boolean;
}

export interface StatusReport {
    verdict: Verdict;
    services: ServiceStatus[];
    /** True when the report came from a client-side fallback probe, not /api/status. */
    fallback: boolean;
    /** Set when even the fallback probe failed (network down). */
    error?: string;
}

const VALID_STATES: ReadonlySet<string> = new Set(["operational", "degraded", "down", "unknown"]);

/** Derive the overall verdict from a service list (mirrors the backend rule). */
export function classifyVerdict(services: ServiceStatus[]): Verdict {
    const requiredDown = services.some((s) => s.required && s.state === "down");
    if (requiredDown) return "blocked";
    const anyProblem = services.some((s) => s.state === "down" || s.state === "degraded");
    return anyProblem ? "degraded" : "operational";
}

/** Normalize one raw service entry into a typed ServiceStatus. */
function normalizeService(raw: unknown): ServiceStatus | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const name = typeof r.name === "string" ? r.name : null;
    if (!name) return null;
    const stateRaw = typeof r.state === "string" ? r.state : "unknown";
    const state = (VALID_STATES.has(stateRaw) ? stateRaw : "unknown") as ServiceState;
    return {
        name,
        state,
        detail: typeof r.detail === "string" ? r.detail : "",
        required: r.required === true,
    };
}

/**
 * Parse and validate a raw /api/status JSON body. Returns null if the shape is
 * unrecognizable so callers can fall back. The verdict is always recomputed
 * from the services to stay consistent even if the server's field drifts.
 */
export function parseStatusResponse(raw: unknown): Omit<StatusReport, "fallback"> | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    if (!Array.isArray(r.services)) return null;
    const services = r.services
        .map(normalizeService)
        .filter((s): s is ServiceStatus => s !== null);
    if (services.length === 0) return null;
    return { verdict: classifyVerdict(services), services };
}

/** Find a service by (case-insensitive) name prefix — e.g. "AI" matches "AI (Hexara)". */
export function findService(report: StatusReport | null, namePrefix: string): ServiceStatus | undefined {
    if (!report) return undefined;
    const needle = namePrefix.toLowerCase();
    return report.services.find((s) => s.name.toLowerCase().startsWith(needle));
}

/** Whether the AI assistant is healthy per the report. Defaults to true when unknown. */
export function aiHealthyFromReport(report: StatusReport | null): boolean {
    const ai = findService(report, "ai");
    if (!ai) return true; // no signal → don't disable AI on a missing field
    return ai.state === "operational";
}

async function fetchJson(
    url: string,
    token: string | undefined,
    timeoutMs: number,
): Promise<{ status: number; body: unknown }> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), headers });
    let body: unknown = null;
    try {
        body = await resp.json();
    } catch {
        body = null;
    }
    return { status: resp.status, body };
}

/** Client-side fallback: probe /healthz so we at least know if the API is up. */
async function fallbackProbe(baseUrl: string, token: string | undefined, timeoutMs: number): Promise<StatusReport> {
    try {
        const url = new URL("/healthz", baseUrl).toString();
        const { status, body } = await fetchJson(url, token, timeoutMs);
        const b = (body ?? {}) as Record<string, unknown>;
        const apiUp = status >= 200 && status < 500; // any HTTP response means the API answered
        const dbState: ServiceState =
            typeof b.database === "string"
                ? (b.database === "connected" ? "operational" : "down")
                : (status >= 200 && status < 300 ? "operational" : "down");
        const services: ServiceStatus[] = [
            {
                name: "API",
                state: apiUp ? "operational" : "down",
                detail: apiUp ? `reachable (HTTP ${status})` : `HTTP ${status}`,
                required: true,
            },
            { name: "Database", state: dbState, detail: dbState === "operational" ? "connected" : "unreachable", required: true },
        ];
        return { verdict: classifyVerdict(services), services, fallback: true };
    } catch (err) {
        const msg = err instanceof Error ? err.message : "connection failed";
        const services: ServiceStatus[] = [
            { name: "API", state: "down", detail: msg, required: true },
        ];
        return { verdict: "blocked", services, fallback: true, error: msg };
    }
}

/**
 * Fetch aggregated service health for the preflight gate. Never throws — always
 * resolves to a StatusReport (verdict "blocked" with an error when even the
 * fallback probe fails).
 */
export async function fetchServiceStatus(
    baseUrl: string,
    token?: string,
    opts: { timeoutMs?: number } = {},
): Promise<StatusReport> {
    const timeoutMs = opts.timeoutMs ?? 8_000;
    try {
        const url = new URL("/api/status", baseUrl).toString();
        const { status, body } = await fetchJson(url, token, timeoutMs);
        if (status === 404 || status === 405) {
            // Endpoint not present on this control plane — fall back.
            return fallbackProbe(baseUrl, token, timeoutMs);
        }
        const parsed = parseStatusResponse(body);
        if (parsed) {
            return { ...parsed, fallback: false };
        }
        // Got a response but couldn't parse it — fall back to a basic probe.
        return fallbackProbe(baseUrl, token, timeoutMs);
    } catch {
        return fallbackProbe(baseUrl, token, timeoutMs);
    }
}
