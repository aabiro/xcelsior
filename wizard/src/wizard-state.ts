/** Checkpoint wizard progress for resume-after-crash (A5). */

import { chmodSync, existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

function configHome(): string {
    return process.env["HOME"] ?? "/tmp";
}

export function wizardStateFile(): string {
    return `${configHome()}/.xcelsior/wizard-state.json`;
}

export interface WizardCheckpoint {
    stepIndex: number;
    answers: Record<string, string | string[]>;
    conversationId?: string;
    completedStepIds: string[];
    savedAt: string;
}

export function loadWizardCheckpoint(): WizardCheckpoint | null {
    const file = wizardStateFile();
    if (!existsSync(file)) return null;
    try {
        const raw = readFileSync(file, "utf8");
        const parsed = JSON.parse(raw) as WizardCheckpoint;
        if (typeof parsed.stepIndex !== "number" || !parsed.answers) return null;
        return parsed;
    } catch {
        return null;
    }
}

export function saveWizardCheckpoint(checkpoint: WizardCheckpoint): void {
    const file = wizardStateFile();
    mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
    writeFileSync(file, JSON.stringify(checkpoint, null, 2), { mode: 0o600 });
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
}

export function redactSecrets(text: string): string {
    return text
        .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, "Bearer [REDACTED]")
        .replace(/xc-[A-Za-z0-9]{8,}/g, "xc-[REDACTED]")
        .replace(/client_secret["']?\s*[:=]\s*["']?[^"'\s]+/gi, "client_secret=[REDACTED]");
}

export function maskToken(token: string): string {
    if (token.length <= 8) return "••••••••";
    return `${token.slice(0, 4)}••••${token.slice(-4)}`;
}