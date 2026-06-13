import { describe, expect, it, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
    CHECKPOINT_VERSION,
    clearWizardCheckpoint,
    hydrateWizardCheckpoint,
    isCheckpointExpired,
    readLocalAccessToken,
    saveWizardCheckpoint,
    validateCheckpoint,
    wizardStateFile,
    tokenFilePath,
} from "../wizard-state.js";
import {
    sanitizeContextValue,
    stripSecretsFromAnswers,
    validateApiBaseUrl,
    validateApiToken,
    mergeTokenIntoAnswers,
} from "../wizard-guards.js";
import { buildWizardContext } from "../useWizardFlow.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

describe("wizard-guards", () => {
    it("validates API tokens", () => {
        expect(validateApiToken("")).not.toBeNull();
        expect(validateApiToken("short")).not.toBeNull();
        expect(validateApiToken("xoa_validtoken12345678")).toBeNull();
        expect(validateApiToken("oauth_clientid123456")).not.toBeNull();
        expect(validateApiToken("valid-token-12345678")).not.toBeNull();
    });

    it("validates API base URLs", () => {
        expect(validateApiBaseUrl("https://xcelsior.ca")).toBeNull();
        expect(validateApiBaseUrl("ftp://bad")).not.toBeNull();
        expect(validateApiBaseUrl("not-a-url")).not.toBeNull();
    });

    it("strips secrets from checkpoint answers", () => {
        const stripped = stripSecretsFromAnswers({
            mode: "provide",
            "api-key": "secret",
            "oauth-client-secret": "sec",
            "_session-token": "tok",
        });
        expect(stripped["api-key"]).toBeUndefined();
        expect(stripped["_checkpoint_had_auth"]).toBe("true");
        expect(stripped.mode).toBe("provide");
    });

    it("merges token from local file into answers", () => {
        const merged = mergeTokenIntoAnswers({ mode: "rent" }, "file-token-12345678");
        expect(merged["api-key"]).toBe("file-token-12345678");
        expect(merged["device-auth"]).toBe("resumed");
    });

    it("caps context values", () => {
        const long = "x".repeat(600);
        expect(sanitizeContextValue(long).length).toBeLessThanOrEqual(500);
    });
});

describe("wizard checkpoint hardening", () => {
    let tmpHome: string;
    const deviceAuthIndex = WIZARD_STEPS.findIndex((s) => s.id === "device-auth");

    beforeEach(() => {
        tmpHome = fs.mkdtempSync(path.join(os.tmpdir(), "wiz-hard-"));
        process.env.HOME = tmpHome;
    });

    afterEach(() => {
        clearWizardCheckpoint();
        const tokenFile = tokenFilePath();
        if (fs.existsSync(tokenFile)) fs.unlinkSync(tokenFile);
    });

    it("rejects malformed checkpoints", () => {
        expect(validateCheckpoint(null, 10)).toBeNull();
        expect(validateCheckpoint({ stepIndex: "bad" }, 10)).toBeNull();
    });

    it("does not persist secrets to disk", () => {
        saveWizardCheckpoint({
            stepIndex: 4,
            answers: {
                mode: "provide",
                "api-key": "super-secret-token-value",
                "oauth-client-secret": "oauth-sec",
            },
            completedStepIds: ["mode"],
            savedAt: new Date().toISOString(),
        });
        const raw = JSON.parse(fs.readFileSync(wizardStateFile(), "utf8"));
        expect(raw.version).toBe(CHECKPOINT_VERSION);
        expect(raw.answers["api-key"]).toBeUndefined();
        expect(raw.answers["oauth-client-secret"]).toBeUndefined();
        expect(raw.answers._checkpoint_had_auth).toBe("true");
    });

    it("hydrates token from token.json when checkpoint omits secrets", () => {
        fs.mkdirSync(path.dirname(tokenFilePath()), { recursive: true });
        fs.writeFileSync(
            tokenFilePath(),
            JSON.stringify({ access_token: "restored-token-abcdef12" }),
            { mode: 0o600 },
        );

        const hydrated = hydrateWizardCheckpoint(
            {
                stepIndex: deviceAuthIndex + 2,
                answers: { mode: "rent", _checkpoint_had_auth: "true" },
                completedStepIds: ["mode", "device-auth"],
                savedAt: new Date().toISOString(),
            },
            WIZARD_STEPS.length,
            deviceAuthIndex,
        );

        expect(hydrated?.resumed).toBe(true);
        expect(hydrated?.needsReauth).toBe(false);
        expect(hydrated?.checkpoint.answers["api-key"]).toBe("restored-token-abcdef12");
    });

    it("forces re-auth when past device-auth without token", () => {
        const hydrated = hydrateWizardCheckpoint(
            {
                stepIndex: deviceAuthIndex + 3,
                answers: { mode: "rent", _checkpoint_had_auth: "true" },
                completedStepIds: ["mode"],
                savedAt: new Date().toISOString(),
            },
            WIZARD_STEPS.length,
            deviceAuthIndex,
        );

        expect(hydrated?.needsReauth).toBe(true);
        expect(hydrated?.checkpoint.stepIndex).toBe(deviceAuthIndex);
        expect(hydrated?.checkpoint.answers["api-key"]).toBe("");
    });

    it("expires checkpoints older than max age", () => {
        const old = new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString();
        expect(isCheckpointExpired(old)).toBe(true);
        const hydrated = hydrateWizardCheckpoint(
            {
                stepIndex: 2,
                answers: { mode: "rent" },
                completedStepIds: [],
                savedAt: old,
            },
            WIZARD_STEPS.length,
            deviceAuthIndex,
        );
        expect(hydrated?.expired).toBe(true);
        expect(hydrated?.resumed).toBe(false);
        expect(fs.existsSync(wizardStateFile())).toBe(false);
    });

    it("reads local access token", () => {
        fs.mkdirSync(path.dirname(tokenFilePath()), { recursive: true });
        fs.writeFileSync(tokenFilePath(), JSON.stringify({ access_token: "abc12345" }));
        expect(readLocalAccessToken()).toBe("abc12345");
    });
});

describe("buildWizardContext hardening", () => {
    it("truncates oversized context segments", () => {
        const huge = "z".repeat(800);
        const ctx = buildWizardContext("benchmark", { mode: "provide", pricing: huge }, {}, null, [], null);
        expect(ctx.length).toBeLessThan(1200);
        expect(ctx).toContain("cli-wizard:benchmark");
    });
});