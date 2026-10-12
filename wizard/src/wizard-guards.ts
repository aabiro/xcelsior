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
    if (/\s/.test(trimmed)) return "Credentials cannot contain whitespace";
    if (trimmed.startsWith("oauth_")) {
        return "That is an OAuth client ID (oauth_…), not an access token — paste an xoa_… session token from device sign-in";
    }
    if (!trimmed.startsWith("xoa_") && !trimmed.startsWith("xcel_ai_")) {
        return "Use an xcel_ai_ API key from Dashboard → MCP → CLI, or an xoa_ token from device sign-in";
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

/** Answers that hold the sign-in or a server-side credential, not a path choice. */
const IDENTITY_ANSWER_KEYS = new Set([
    ...SECRET_ANSWER_KEYS,
    "device-auth",
    "oauth-client-id",
]);

/**
 * The answers to carry back to the path choice when the user presses "b" on a
 * failed check or the sign-in screen. A resumed run used to have no way out of
 * a check the chosen path cannot pass (the SDK path in a Python repo): Enter
 * retried, and running the wizard again resumed the same step.
 *
 * Drops every path choice so its questions are asked again. Keeps the sign-in
 * and every internal `_` observation: `_host_id` and `_launch_*` are what stop
 * a later attempt registering a second host or launching a second instance.
 */
export function answersForNewPath(
    answers: Record<string, string | string[]>,
): Record<string, string | string[]> {
    const out: Record<string, string | string[]> = {};
    for (const [key, value] of Object.entries(answers)) {
        if (key.startsWith("_") || IDENTITY_ANSWER_KEYS.has(key)) out[key] = value;
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