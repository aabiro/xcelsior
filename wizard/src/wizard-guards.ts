/** Input validation and sanitization for the CLI wizard. */

export const MAX_CONTEXT_VALUE_LEN = 500;
export const MIN_TOKEN_LEN = 8;
export const MAX_TOKEN_LEN = 8192;

const SECRET_ANSWER_KEYS = new Set([
    "api-key",
    "oauth-client-secret",
    "_session-token",
]);

/** Keys stripped from checkpoint files — rehydrated from token.json when possible. */
export const CHECKPOINT_SECRET_KEYS = SECRET_ANSWER_KEYS;

export function clampStepIndex(index: number, stepCount: number): number {
    if (!Number.isFinite(index) || stepCount <= 0) return 0;
    return Math.max(0, Math.min(Math.floor(index), stepCount - 1));
}

export function sanitizeContextValue(value: string, maxLen = MAX_CONTEXT_VALUE_LEN): string {
    const trimmed = value.replace(/[\r\n]+/g, " ").trim();
    if (trimmed.length <= maxLen) return trimmed;
    return `${trimmed.slice(0, maxLen - 1)}…`;
}

export function validateApiToken(token: string): string | null {
    const trimmed = token.trim();
    if (!trimmed) return "Token cannot be empty";
    if (trimmed.length < MIN_TOKEN_LEN) return "Token looks too short";
    if (trimmed.length > MAX_TOKEN_LEN) return "Token looks too long";
    if (/[\r\n]/.test(trimmed)) return "Token cannot contain newlines";
    if (trimmed.startsWith("oauth_")) {
        return "That is an OAuth client ID (oauth_…), not an access token — paste an xoa_… session token from device sign-in";
    }
    if (!trimmed.startsWith("xoa_")) {
        return "Access tokens start with xoa_ — sign in via device flow (Enter) or copy from Dashboard → Settings";
    }
    return null;
}

export function validateApiBaseUrl(raw: string): string | null {
    const trimmed = raw.trim();
    if (!trimmed) return "API URL is required";
    try {
        const url = new URL(trimmed);
        if (url.protocol !== "http:" && url.protocol !== "https:") {
            return "API URL must use http or https";
        }
        if (!url.hostname) return "API URL must include a hostname";
        return null;
    } catch {
        return "API URL is not valid";
    }
}

export function stripSecretsFromAnswers(
    answers: Record<string, string | string[]>,
): Record<string, string | string[]> {
    const out: Record<string, string | string[]> = {};
    let hadAuth = false;
    for (const [key, value] of Object.entries(answers)) {
        if (SECRET_ANSWER_KEYS.has(key)) {
            hadAuth = true;
            continue;
        }
        out[key] = value;
    }
    if (hadAuth || answers["device-auth"]) {
        out["_checkpoint_had_auth"] = "true";
    }
    return out;
}

export function mergeTokenIntoAnswers(
    answers: Record<string, string | string[]>,
    token: string | null,
): Record<string, string | string[]> {
    if (!token || answers["api-key"]) return answers;
    return {
        ...answers,
        "api-key": token,
        "device-auth": answers["device-auth"] ?? "resumed",
    };
}