import { describe, expect, it } from "vitest";
import fs from "node:fs";
import {
    loadWizardCheckpoint,
    maskToken,
    redactSecrets,
    saveWizardCheckpoint,
    wizardStateFile,
} from "../wizard-state.js";

describe("wizard-state", () => {
    it("saves and loads checkpoint at mode 600 without secrets", () => {
        saveWizardCheckpoint({
            stepIndex: 3,
            answers: { mode: "provide", "api-key": "must-not-persist" },
            completedStepIds: ["mode", "auth"],
            savedAt: new Date().toISOString(),
        });
        const loaded = loadWizardCheckpoint();
        expect(loaded?.stepIndex).toBe(3);
        expect(loaded?.answers.mode).toBe("provide");
        expect(loaded?.answers["api-key"]).toBeUndefined();
        const stat = fs.statSync(wizardStateFile());
        expect(stat.mode & 0o777).toBe(0o600);
    });

    it("redacts secrets and masks tokens", () => {
        expect(redactSecrets("Bearer secret-token-123")).toContain("[REDACTED]");
        expect(maskToken("abcdefghijklmnop")).toMatch(/^abcd••••mnop$/);
    });
});
