/** Checkpoint wizard progress for resume-after-crash (A5). */

import {
    chmodSync,
    existsSync,
    mkdirSync,
    readFileSync,
    renameSync,
    unlinkSync,
    writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { clampStepIndex, mergeTokenIntoAnswers, stripSecretsFromAnswers } from "./wizard-guards.js";

export const CHECKPOINT_VERSION = 1;
export const CHECKPOINT_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

function configHome(): string {
    return process.env["HOME"] ?? "/tmp";
}

export function wizardStateFile(): string {
    return `${configHome()}/.xcelsior/wizard-state.json`;
}

export function tokenFilePath(): string {
    return `${configHome()}/.xcelsior/token.json`;
}

export interface WizardCheckpoint {
    version?: number;
    stepIndex: number;
    answers: Record<string, string | string[]>;
    conversationId?: string;
    completedStepIds: string[];
    savedAt: string;
}

export interface HydratedCheckpoint {
    checkpoint: WizardCheckpoint;
    resumed: boolean;
    needsReauth: boolean;
    expired: boolean;
}

export function readLocalAccessToken(): string | null {
    const file = tokenFilePath();
    if (!existsSync(file)) return null;
    try {
        const parsed = JSON.parse(readFileSync(file, "utf8")) as { access_token?: string };
        const token = parsed.access_token?.trim();
        return token || null;
    } catch {
        return null;
    }
}

function parseSavedAt(savedAt: string): number | null {
    const ms = Date.parse(savedAt);
    return Number.isFinite(ms) ? ms : null;
}

export function isCheckpointExpired(savedAt: string, now = Date.now()): boolean {
    const ms = parseSavedAt(savedAt);
    if (ms === null) return true;
    return now - ms > CHECKPOINT_MAX_AGE_MS;
}

export function validateCheckpoint(
    raw: unknown,
    stepCount: number,
): WizardCheckpoint | null {
    if (!raw || typeof raw !== "object") return null;
    const cp = raw as WizardCheckpoint;
    if (typeof cp.stepIndex !== "number" || !cp.answers || typeof cp.answers !== "object") {
        return null;
    }
    if (!Array.isArray(cp.completedStepIds)) return null;
    if (typeof cp.savedAt !== "string" || parseSavedAt(cp.savedAt) === null) return null;
    if (cp.version !== undefined && cp.version !== CHECKPOINT_VERSION) return null;

    return {
        version: CHECKPOINT_VERSION,
        stepIndex: clampStepIndex(cp.stepIndex, stepCount),
        answers: cp.answers,
        conversationId: typeof cp.conversationId === "string" ? cp.conversationId : undefined,
        completedStepIds: cp.completedStepIds.filter((id) => typeof id === "string"),
        savedAt: cp.savedAt,
    };
}

export function loadWizardCheckpoint(): WizardCheckpoint | null {
    const file = wizardStateFile();
    if (!existsSync(file)) return null;
    try {
        const raw = JSON.parse(readFileSync(file, "utf8"));
        return validateCheckpoint(raw, Number.MAX_SAFE_INTEGER);
    } catch {
        return null;
    }
}

export function hydrateWizardCheckpoint(
    checkpoint: WizardCheckpoint | null,
    stepCount: number,
    deviceAuthStepIndex: number,
): HydratedCheckpoint | null {
    if (!checkpoint) return null;

    const expired = isCheckpointExpired(checkpoint.savedAt);
    const validated = validateCheckpoint(checkpoint, stepCount);
    if (!validated) {
        clearWizardCheckpoint();
        return null;
    }

    let answers = mergeTokenIntoAnswers(validated.answers, readLocalAccessToken());
    let stepIndex = validated.stepIndex;
    let needsReauth = false;

    const pastAuth = stepIndex > deviceAuthStepIndex;
    const hasToken = typeof answers["api-key"] === "string" && answers["api-key"].length > 0;

    if (pastAuth && !hasToken) {
        needsReauth = true;
        stepIndex = deviceAuthStepIndex;
        answers = {
            ...answers,
            "device-auth": "",
            "api-key": "",
        };
        delete answers["_checkpoint_had_auth"];
    }

    if (expired) {
        clearWizardCheckpoint();
        return {
            checkpoint: {
                ...validated,
                stepIndex,
                answers,
            },
            resumed: false,
            needsReauth,
            expired: true,
        };
    }

    return {
        checkpoint: {
            ...validated,
            stepIndex,
            answers,
        },
        resumed: true,
        needsReauth,
        expired: false,
    };
}

export function saveWizardCheckpoint(checkpoint: WizardCheckpoint): void {
    const file = wizardStateFile();
    const dir = dirname(file);
    mkdirSync(dir, { recursive: true, mode: 0o700 });

    const payload: WizardCheckpoint = {
        version: CHECKPOINT_VERSION,
        stepIndex: checkpoint.stepIndex,
        answers: stripSecretsFromAnswers(checkpoint.answers),
        conversationId: checkpoint.conversationId,
        completedStepIds: checkpoint.completedStepIds,
        savedAt: checkpoint.savedAt,
    };

    const tmp = `${file}.tmp`;
    writeFileSync(tmp, JSON.stringify(payload, null, 2), { mode: 0o600 });
    renameSync(tmp, file);
    try {
        chmodSync(file, 0o600);
    } catch {
        // best-effort on platforms that ignore mode
    }
}

export function clearWizardCheckpoint(): void {
    const file = wizardStateFile();
    if (existsSync(file)) {
        unlinkSync(file);
    }
    const tmp = `${file}.tmp`;
    if (existsSync(tmp)) {
        unlinkSync(tmp);
    }
}

export function redactSecrets(text: string): string {
    return text
        .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer [REDACTED]")
        .replace(/xc-[A-Za-z0-9]{8,}/g, "xc-[REDACTED]")
        .replace(/client_secret["']?\s*[:=]\s*["']?[^"'\s]+/gi, "client_secret=[REDACTED]")
        .replace(/XCELSIOR_API_TOKEN=[^\s]+/gi, "XCELSIOR_API_TOKEN=[REDACTED]")
        .replace(/XCELSIOR_OAUTH_CLIENT_SECRET=[^\s]+/gi, "XCELSIOR_OAUTH_CLIENT_SECRET=[REDACTED]");
}

export function maskToken(token: string): string {
    if (token.length <= 8) return "••••••••";
    return `${token.slice(0, 4)}••••${token.slice(-4)}`;
}