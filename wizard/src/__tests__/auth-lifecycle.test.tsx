import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { render } from "ink-testing-library";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { configDirectory, writePrivateFile } from "../config-files.js";
import { saveWizardCheckpoint, tokenFilePath, wizardStateFile } from "../wizard-state.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

const api = vi.hoisted(() => ({
    requestDeviceCode: vi.fn(), pollDeviceToken: vi.fn(), getMe: vi.fn(),
    searchMarketplace: vi.fn(), getWallet: vi.fn(), claimFreeCredits: vi.fn(), launchInstance: vi.fn(),
}));
const sdk = vi.hoisted(() => ({ checkSdkPackage: vi.fn() }));
vi.mock("../sdk-checks.js", async (original) => ({
    ...await original<typeof import("../sdk-checks.js")>(), ...sdk,
}));
vi.mock("../api-client.js", async (original) => ({
    ...await original<typeof import("../api-client.js")>(), ...api,
}));
vi.mock("../preflight.js", () => ({
    fetchServiceStatus: vi.fn(async () => ({ verdict: "operational", services: [] })),
    aiHealthyFromReport: () => false,
}));
vi.mock("../marketplace-stats.js", () => ({ fetchMarketplaceStats: vi.fn(async () => null) }));

const token = "xoa_test_credential_123456789";
const profile = { user_id: "user-1", email: "wizard@example.test", customer_id: "customer-1" };
const deviceCode = {
    device_code: "device-1", user_code: "ABCD-EFGH", verification_uri: "https://example.test/device",
    expires_in: 900, interval: 5,
};
const authorized = { status: "authorized", token: { access_token: token } };
let flow: UseWizardFlowReturn;

function Probe() {
    flow = useWizardFlow();
    return null;
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (error: Error) => void;
    const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

async function settle() {
    for (let i = 0; i < 3; i += 1) await new Promise<void>((resolve) => setImmediate(resolve));
}

async function start(stepId = "device-auth", answers: Record<string, string> = { mode: "rent" }) {
    saveWizardCheckpoint({
        stepIndex: WIZARD_STEPS.findIndex((step) => step.id === stepId),
        answers, completedStepIds: ["mode"], savedAt: new Date().toISOString(),
    });
    const app = render(<Probe />);
    await settle();
    return app;
}

beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    api.requestDeviceCode.mockReset().mockResolvedValue(deviceCode);
    api.pollDeviceToken.mockReset().mockResolvedValue({ status: "pending" });
    api.getMe.mockReset().mockResolvedValue(profile);
    api.searchMarketplace.mockReset().mockResolvedValue({ listings: [{
        host_id: "host-1", gpu_model: "RTX 4090", vram_gb: 24, price_per_hour: 3,
        owner: "Test provider", active: true,
    }] });
    api.getWallet.mockReset().mockResolvedValue({ balance_cad: 1 });
    api.claimFreeCredits.mockReset().mockResolvedValue({ already_claimed: true, amount: 0 });
    api.launchInstance.mockReset();
    sdk.checkSdkPackage.mockReset().mockResolvedValue([{ name: "SDK", ok: true, detail: "ready" }]);
});
afterEach(() => vi.restoreAllMocks());

describe("authentication through the flow controller", () => {
    it("discards a device-code response received after switching to manual", async () => {
        const request = deferred<typeof deviceCode>();
        api.requestDeviceCode.mockReturnValueOnce(request.promise);
        await start();
        flow.switchToManualAuth();
        request.resolve(deviceCode);
        await settle();
        await vi.advanceTimersByTimeAsync(20_000);
        expect(flow.deviceAuth.status).toBe("manual");
        expect(api.pollDeviceToken).not.toHaveBeenCalled();
        expect(existsSync(tokenFilePath())).toBe(false);
    });

    it("discards an authorized poll received after switching methods", async () => {
        const poll = deferred<typeof authorized>();
        api.pollDeviceToken.mockReturnValueOnce(poll.promise);
        await start();
        await vi.advanceTimersByTimeAsync(5_000);
        flow.switchToManualAuth();
        poll.resolve(authorized);
        await settle();
        expect(api.getMe).not.toHaveBeenCalled();
        expect(flow.deviceAuth.status).toBe("manual");
        expect(existsSync(tokenFilePath())).toBe(false);
    });

    it("does not save a device credential whose profile could not be verified", async () => {
        api.pollDeviceToken.mockResolvedValueOnce(authorized);
        api.getMe.mockRejectedValueOnce(new Error("Account verification unavailable"));
        await start();
        await vi.advanceTimersByTimeAsync(5_000);
        await settle();
        expect(flow.deviceAuth.status).toBe("error");
        expect(flow.deviceAuth.errorMessage).toContain("Account verification unavailable");
        expect(existsSync(tokenFilePath())).toBe(false);
        await vi.advanceTimersByTimeAsync(10_000);
        expect(api.pollDeviceToken).toHaveBeenCalledTimes(1);
    });

    it("does not let stale manual verification replace a newer device attempt", async () => {
        const pendingProfile = deferred<typeof profile>();
        api.getMe.mockReturnValueOnce(pendingProfile.promise);
        await start();
        flow.switchToManualAuth();
        flow.submitManualToken(token);
        flow.retryDeviceAuth();
        pendingProfile.resolve(profile);
        await settle();
        expect(flow.deviceAuth.status).toBe("waiting");
        expect(existsSync(tokenFilePath())).toBe(false);
    });

    it("ignores duplicate manual submissions and writes only the private token file", async () => {
        const project = join(configDirectory(), "project");
        mkdirSync(project);
        writeFileSync(join(project, "package.json"), JSON.stringify({ dependencies: { next: "16" } }));
        writeFileSync(join(project, ".env.local"), "EXISTING_SETTING=keep\n");
        vi.spyOn(process, "cwd").mockReturnValue(project);
        const pendingProfile = deferred<typeof profile>();
        api.getMe.mockReturnValueOnce(pendingProfile.promise);
        await start();
        flow.switchToManualAuth();
        flow.submitManualToken(token);
        flow.submitManualToken(token);
        pendingProfile.resolve(profile);
        await settle();
        expect(api.getMe).toHaveBeenCalledTimes(1);
        expect(flow.deviceAuth.status).toBe("authorized");
        expect(JSON.parse(readFileSync(tokenFilePath(), "utf8")).access_token).toBe(token);
        expect(readFileSync(join(project, ".env.local"), "utf8")).toBe("EXISTING_SETTING=keep\n");
        expect(flow.answers["_customer_id"]).toBe(profile.customer_id);
    });

    it("blocks progression when credential storage fails", async () => {
        mkdirSync(tokenFilePath());
        await start();
        flow.switchToManualAuth();
        flow.submitManualToken(token);
        await settle();
        expect(flow.deviceAuth.status).toBe("manual");
        expect(flow.tokenSaveError).toBeTruthy();
        flow.continueFromAuth();
        await settle();
        expect(flow.step.id).toBe("device-auth");
        expect(flow.answers["api-key"]).toBeUndefined();
    });

    it("does not persist authentication after unmount", async () => {
        const pendingProfile = deferred<typeof profile>();
        api.getMe.mockReturnValueOnce(pendingProfile.promise);
        const app = await start();
        flow.switchToManualAuth();
        flow.submitManualToken(token);
        app.unmount();
        await settle();
        pendingProfile.resolve(profile);
        await settle();
        expect(existsSync(tokenFilePath())).toBe(false);
    });
});

describe("completion and cancellation", () => {
    it("restarts a restored check and deduplicates retry input", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        sdk.checkSdkPackage.mockResolvedValueOnce([{ name: "SDK", ok: false, detail: "Registry unavailable" }]);
        await start("sdk-install", { mode: "sdk" });
        expect(sdk.checkSdkPackage).toHaveBeenCalledTimes(1);
        expect(flow.checkCanRetry).toBe(true);
        const retry = deferred<{ name: string; ok: boolean; detail: string }[]>();
        sdk.checkSdkPackage.mockReturnValueOnce(retry.promise);
        flow.retryCheck();
        flow.retryCheck();
        await settle();
        expect(sdk.checkSdkPackage).toHaveBeenCalledTimes(2);
        retry.resolve([{ name: "SDK", ok: true, detail: "installed" }]);
        await settle();
        expect(flow.checkAwaitContinue).toBe(true);
        expect(flow.step.id).toBe("sdk-install");
    });

    it("restarts a restored marketplace lookup without duplicate requests", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        await start("browse-gpus", { mode: "rent", "want-launch": "yes" });
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(api.searchMarketplace).toHaveBeenCalledTimes(1);
        expect(flow.step.id).toBe("gpu-pick");
        expect(flow.gpuOptions[0].value).toBe("host-1");
    });

    it("preserves marketplace observations across unmount and resume", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        const app = await start("browse-gpus", { mode: "rent", "want-launch": "yes" });
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.step.id).toBe("gpu-pick");
        app.unmount();
        await settle();
        render(<Probe />);
        await settle();
        expect(flow.step.id).toBe("gpu-pick");
        expect(flow.gpuOptions[0].value).toBe("host-1");
        expect(flow.gpuListings[0].price_per_hour).toBe(3);
    });

    it("discards a marketplace result after exit", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        const search = deferred<{ listings: never[] }>();
        api.searchMarketplace.mockReturnValueOnce(search.promise);
        const app = await start("browse-gpus", { mode: "rent", "want-launch": "yes" });
        app.unmount();
        await settle();
        search.resolve({ listings: [] });
        await settle();
        await vi.advanceTimersByTimeAsync(5_000);
        const checkpoint = JSON.parse(readFileSync(wizardStateFile(), "utf8"));
        expect(checkpoint.answers["browse-gpus"]).toBeUndefined();
    });

    it("uses the selected GPU price and does not launch when payment is skipped", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        await start("gpu-preference", { mode: "rent", "want-launch": "yes", workload: "other", "_customer_id": profile.customer_id });
        flow.submitAnswer("cheapest");
        await vi.advanceTimersByTimeAsync(6_000);
        await settle();
        expect(flow.step.id).toBe("gpu-pick");
        flow.submitAnswer("host-1");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        flow.submitAnswer("nvidia/cuda:12.4.1-devel-ubuntu22.04");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        flow.submitAnswer("yes");
        await vi.advanceTimersByTimeAsync(4_000);
        await settle();
        expect(flow.step.id).toBe("wallet-check");
        expect(flow.paymentGate.required).toBe(3);
        expect(flow.answers["_wallet_insufficient"]).toBe("true");
        flow.skipCheck();
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.step.id).toBe("payment-gate");
        const wallet = deferred<{ balance_cad: number }>();
        api.getWallet.mockReturnValueOnce(wallet.promise);
        await vi.advanceTimersByTimeAsync(5_000);
        flow.skipPayment();
        wallet.resolve({ balance_cad: 100 });
        await settle();
        await vi.advanceTimersByTimeAsync(5_000);
        expect(flow.isComplete).toBe(true);
        expect(flow.answers["want-launch"]).toBe("no");
        expect(api.launchInstance).not.toHaveBeenCalled();
    });

    it("does not start delayed step initialization after exit", async () => {
        const app = render(<Probe />);
        await settle();
        flow.submitAnswer("rent");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.step.id).toBe("device-auth");
        app.unmount();
        await settle();
        await vi.advanceTimersByTimeAsync(5_000);
        expect(api.requestDeviceCode).not.toHaveBeenCalled();
    });

    it("starts fresh instead of restoring an expired checkpoint", async () => {
        saveWizardCheckpoint({
            stepIndex: WIZARD_STEPS.findIndex((step) => step.id === "want-launch"),
            answers: { mode: "rent" }, completedStepIds: ["mode"],
            savedAt: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
        });
        render(<Probe />);
        await settle();
        expect(flow.step.id).toBe("mode");
        expect(flow.answers.mode).toBeUndefined();
        expect(api.requestDeviceCode).not.toHaveBeenCalled();
    });

    it("keeps a failed save retryable and never recreates a completed checkpoint", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        const configFile = join(configDirectory(), "config.toml");
        mkdirSync(configFile);
        const app = await start("want-launch");
        flow.submitAnswer("no");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.isComplete).toBe(false);
        expect(flow.step.id).toBe("want-launch");
        expect(flow.wizardMessage).toContain("Could not save configuration");
        rmSync(configFile, { recursive: true });
        flow.submitAnswer("no");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.isComplete).toBe(true);
        expect(readFileSync(configFile, "utf8")).toContain('mode = "rent"');
        app.unmount();
        await settle();
        await vi.advanceTimersByTimeAsync(1_000);
        expect(existsSync(wizardStateFile())).toBe(false);
    });

    it("does not let repeated input change an answer during its transition", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        await start("want-launch");
        flow.submitAnswer("no");
        flow.submitAnswer("yes");
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.answers["want-launch"]).toBe("no");
        expect(flow.isComplete).toBe(true);
    });

    it("reports launch cancellation honestly and clears its checkpoint", async () => {
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
        const app = await start("confirm-launch");
        flow.submitAnswer("no");
        await settle();
        expect(flow.isComplete).toBe(true);
        expect(flow.wizardMessage).toBe("Launch cancelled. No instance was launched.");
        app.unmount();
        await settle();
        expect(existsSync(wizardStateFile())).toBe(false);
        expect(existsSync(join(configDirectory(), "config.toml"))).toBe(false);
    });
});
