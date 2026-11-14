import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { render } from "ink-testing-library";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { writePrivateFile } from "../config-files.js";
import { saveWizardCheckpoint, tokenFilePath } from "../wizard-state.js";
import { WIZARD_STEPS } from "../wizard-flow.js";
import { launchPaymentError } from "../api-client.js";

// A launch costs money, so a retry must never start a second instance when
// the first attempt's response was lost after the server created it.
const api = vi.hoisted(() => ({ launchInstance: vi.fn(), listInstances: vi.fn(), getMe: vi.fn() }));
vi.mock("../api-client.js", async (original) => ({
    ...await original<typeof import("../api-client.js")>(), ...api,
}));
vi.mock("../preflight.js", () => ({
    fetchServiceStatus: vi.fn(async () => ({ verdict: "operational", services: [] })),
    aiHealthyFromReport: () => false,
}));
vi.mock("../marketplace-stats.js", () => ({ fetchMarketplaceStats: vi.fn(async () => null) }));
vi.mock("../open-url.js", () => ({ openUrl: vi.fn(async () => undefined) }));

const token = "xoa_renter_credential_123456789";
let flow: UseWizardFlowReturn;
function Probe() {
    flow = useWizardFlow();
    return null;
}
async function settle() {
    for (let i = 0; i < 4; i += 1) await new Promise<void>((resolve) => setImmediate(resolve));
}
async function startAtLaunch() {
    saveWizardCheckpoint({
        stepIndex: WIZARD_STEPS.findIndex((step) => step.id === "launch-instance"),
        answers: {
            mode: "rent", "api-key": token, "want-launch": "yes", workload: "other",
            "gpu-pick": "RTX 4090|24", "image-pick": "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        },
        completedStepIds: ["mode", "device-auth", "want-launch", "gpu-pick", "image-pick", "confirm-launch", "wallet-check"],
        savedAt: new Date().toISOString(),
    });
    render(<Probe />);
    await settle();
    await vi.advanceTimersByTimeAsync(2_000);
    await settle();
}
const launched = (name: string) => ({ job_id: "job-1", name, status: "queued", host_id: "host-1", submitted_at: Date.now() / 1000 });

beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    api.launchInstance.mockReset();
    api.listInstances.mockReset().mockResolvedValue([]);
    api.getMe.mockReset().mockResolvedValue({ user_id: "u", email: "renter@example.test", customer_id: "c" });
    writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
});
afterEach(() => vi.restoreAllMocks());

describe("launching from the wizard", () => {
    it("adopts the instance a lost response already created instead of launching again", async () => {
        api.launchInstance.mockRejectedValueOnce(new Error("Request timed out"));
        await startAtLaunch();
        expect(flow.step.id).toBe("launch-instance");
        expect(flow.checkCanRetry).toBe(true);
        const name = api.launchInstance.mock.calls[0][2].name;
        api.listInstances.mockResolvedValue([
            { job_id: "old", name, status: "terminated", host_id: "host-1", submitted_at: Date.now() / 1000 - 90 * 86_400 },
            launched(name),
        ]);
        flow.retryCheck();
        await settle();
        expect(api.launchInstance).toHaveBeenCalledTimes(1);
        expect(flow.checkResults["launch-instance"].allPassed).toBe(true);
        expect(flow.instanceInfo?.job_id).toBe("job-1");
    });

    it("does not replay a timed-out POST when the first instance has not appeared yet", async () => {
        api.launchInstance.mockRejectedValueOnce(new Error("Request timed out"));
        await startAtLaunch();
        const name = api.launchInstance.mock.calls[0][2].name;
        api.launchInstance.mockResolvedValueOnce(launched(name));
        flow.retryCheck();
        await settle();
        expect(api.listInstances).toHaveBeenCalledTimes(1);
        expect(api.launchInstance).toHaveBeenCalledTimes(1);
        expect(flow.checkResults["launch-instance"].allPassed).toBe(false);
        expect(flow.checkResults["launch-instance"].items[0].detail).toContain("Nothing new was launched");
        // The original POST can finish after that empty read. A queued job
        // may not have an assigned host yet; it must still be recovered.
        api.listInstances.mockResolvedValue([{ ...launched(name), host_id: null }]);
        flow.retryCheck();
        await settle();
        expect(api.launchInstance).toHaveBeenCalledTimes(1);
        expect(flow.instanceInfo?.job_id).toBe("job-1");
    });

    it("launches by GPU model and lets the scheduler pick the machine", async () => {
        api.launchInstance.mockImplementationOnce(async (_url: string, _token: string, params: { name: string }) => launched(params.name));
        await startAtLaunch();
        const params = api.launchInstance.mock.calls[0][2];
        expect(params).toMatchObject({ gpu_model: "RTX 4090", vram_needed_gb: 24 });
        expect(params.host_id).toBeUndefined();
        expect(flow.checkResults["launch-instance"].allPassed).toBe(true);
    });

    it("says to top up, then retry, when the server reports a shortfall", async () => {
        api.launchInstance.mockRejectedValueOnce(launchPaymentError(
            "Insufficient available wallet balance for launch (~$0.35 CAD hold required; $0.10 CAD available after holds)",
            "https://xcelsior.ca/dashboard/billing",
        ));
        await startAtLaunch();
        const [item] = flow.checkResults["launch-instance"].items;
        expect(item.topUp).toBe(true);
        expect(item.detail).toContain("$0.10 CAD available");
        expect(item.remediation).toMatch(/top up/i);
        expect(item.remediation).toMatch(/retry/i);
        expect(item.url).toBe("https://xcelsior.ca/dashboard/billing");
    });

    it("does not call a failed fund hold a low balance", async () => {
        // The same 402 the server sends when it cannot place a hold at all.
        api.launchInstance.mockRejectedValueOnce(launchPaymentError(
            "Unable to reserve wallet funds for launch", "https://xcelsior.ca/dashboard/billing",
        ));
        await startAtLaunch();
        const [item] = flow.checkResults["launch-instance"].items;
        expect(item.topUp).toBe(false);
        expect(item.detail).toBe("Unable to reserve wallet funds for launch");
        expect(`${item.detail} ${item.remediation}`).not.toMatch(/insufficient|top up/i);
    });

    it("refuses to launch a pick that does not name a GPU (an old host-id checkpoint)", async () => {
        saveWizardCheckpoint({
            stepIndex: WIZARD_STEPS.findIndex((step) => step.id === "launch-instance"),
            answers: { mode: "rent", "api-key": token, "want-launch": "yes", "gpu-pick": "host-1" },
            completedStepIds: ["mode"], savedAt: new Date().toISOString(),
        });
        render(<Probe />);
        await settle();
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(api.launchInstance).not.toHaveBeenCalled();
        expect(flow.checkResults["launch-instance"].items[0].remediation).toMatch(/pick a GPU/);
    });

    it("launches nothing when it cannot tell whether the first attempt went through", async () => {
        api.launchInstance.mockRejectedValueOnce(new Error("Request timed out"));
        await startAtLaunch();
        api.listInstances.mockRejectedValueOnce(new Error("API unreachable"));
        flow.retryCheck();
        await settle();
        expect(api.launchInstance).toHaveBeenCalledTimes(1);
        const [item] = flow.checkResults["launch-instance"].items;
        expect(item.ok).toBe(false);
        expect(item.detail).toContain("Nothing new was launched");
    });
});
