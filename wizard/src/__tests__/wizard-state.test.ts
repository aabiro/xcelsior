import { describe, expect, it, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
    clearWizardCheckpoint,
    loadWizardCheckpoint,
    maskToken,
    redactSecrets,
    saveWizardCheckpoint,
    wizardStateFile,
} from "../wizard-state.js";

describe("wizard-state", () => {
    let tmpHome: string;

    beforeEach(() => {
        tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), "wiz-state-"));
        process.env.HOME = tmpHome;
    });

    afterEach(() => {
        clearWizardCheckpoint();
    });

    it("saves and loads checkpoint at mode 600", () => {
        saveWizardCheckpoint({
            stepIndex: 3,
            answers: { mode: "provide" },
            completedStepIds: ["mode", "auth"],
            savedAt: new Date().toISOString(),
        });
        const loaded = loadWizardCheckpoint();
        expect(loaded?.stepIndex).toBe(3);
        expect(loaded?.answers.mode).toBe("provide");
        const stat = fs.statSync(wizardStateFile());
        expect(stat.mode & 0o777).toBe(0o600);
    });

    it("redacts secrets and masks tokens", () => {
        expect(redactSecrets("Bearer secret-token-123")).toContain("[REDACTED]");
        expect(maskToken("abcdefghijklmnop")).toMatch(/^abcd••••mnop$/);
    });
});