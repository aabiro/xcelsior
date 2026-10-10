import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { render } from "ink-testing-library";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { writePrivateFile } from "../config-files.js";
import { saveWizardCheckpoint, tokenFilePath } from "../wizard-state.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

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
            "gpu-pick": "host-1", "image-pick": "nvidia/cuda:12.4.1-devel-ubuntu22.04",
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

    it("launches again under the same name when the first attempt created nothing", async () => {
        api.launchInstance.mockRejectedValueOnce(new Error("Request timed out"));
        await startAtLaunch();
        const name = api.launchInstance.mock.calls[0][2].name;
        api.launchInstance.mockResolvedValueOnce(launched(name));
        flow.retryCheck();
        await settle();
        expect(api.listInstances).toHaveBeenCalledTimes(1);
        expect(api.launchInstance).toHaveBeenCalledTimes(2);
        expect(api.launchInstance.mock.calls[1][2].name).toBe(name);
        expect(flow.checkResults["launch-instance"].allPassed).toBe(true);
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
