import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { render } from "ink-testing-library";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { DeviceAuthStep } from "../steps.js";
import { readableDetail } from "../api-client.js";
import { writePrivateFile } from "../config-files.js";
import { saveWizardCheckpoint, tokenFilePath } from "../wizard-state.js";
import { DEVICE_BROWSER_DELAY_MS, WIZARD_STEPS } from "../wizard-flow.js";

// Found running 0.1.5 for real: the browser took 15s to open and landed on a
// developer placeholder, there was no way back from the sign-in screen, and a
// resumed run's expired token made Hexara answer with raw 401 JSON.
const api = vi.hoisted(() => ({
    requestDeviceCode: vi.fn(), pollDeviceToken: vi.fn(), getMe: vi.fn(), streamChat: vi.fn(),
}));
const opened = vi.hoisted(() => vi.fn(async () => true));
vi.mock("../api-client.js", async (original) => ({
    ...await original<typeof import("../api-client.js")>(), ...api,
}));
vi.mock("../open-url.js", () => ({ openUrl: opened }));
vi.mock("../preflight.js", () => ({
    fetchServiceStatus: vi.fn(async () => ({ verdict: "operational", services: [] })),
    aiHealthyFromReport: () => true,
}));
vi.mock("../marketplace-stats.js", () => ({ fetchMarketplaceStats: vi.fn(async () => null) }));

const deviceCode = {
    device_code: "device-1", user_code: "ABCD-EFGH",
    verification_uri: "https://xcelsior.example/device",
    verification_uri_complete: "https://xcelsior.example/device?user_code=ABCD-EFGH",
    expires_in: 900, interval: 5,
};
let flow: UseWizardFlowReturn;
function Probe() {
    flow = useWizardFlow();
    return null;
}
async function settle() {
    for (let i = 0; i < 4; i += 1) await new Promise<void>((resolve) => setImmediate(resolve));
}
async function startAt(stepId: string, answers: Record<string, string>) {
    saveWizardCheckpoint({
        stepIndex: WIZARD_STEPS.findIndex((step) => step.id === stepId),
        answers, completedStepIds: ["mode"], savedAt: new Date().toISOString(),
    });
    render(<Probe />);
    await settle();
    await vi.advanceTimersByTimeAsync(100);
    await settle();
}

beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    opened.mockClear();
    api.requestDeviceCode.mockReset().mockResolvedValue(deviceCode);
    api.pollDeviceToken.mockReset().mockResolvedValue({ status: "pending" });
    api.getMe.mockReset();
    api.streamChat.mockReset();
});
afterEach(() => vi.restoreAllMocks());

describe("signing in from the wizard", () => {
    it("opens the code-filled link after a short pause, not fifteen seconds", async () => {
        await startAt("device-auth", { mode: "rent" });
        expect(flow.deviceAuth.status).toBe("waiting");
        await vi.advanceTimersByTimeAsync(DEVICE_BROWSER_DELAY_MS - 500);
        expect(opened).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(1_000);
        expect(opened).toHaveBeenCalledWith(deviceCode.verification_uri_complete);
        expect(DEVICE_BROWSER_DELAY_MS).toBeLessThanOrEqual(5_000);
    });

    it("goes back to the path choice from the sign-in screen and stops waiting", async () => {
        await startAt("device-auth", { mode: "rent" });
        expect(flow.step.id).toBe("device-auth");
        flow.changePath();
        await settle();
        expect(flow.step.id).toBe("mode");
        expect(flow.answers.mode).toBeUndefined();
        const polls = api.pollDeviceToken.mock.calls.length;
        await vi.advanceTimersByTimeAsync(60_000);
        expect(api.pollDeviceToken.mock.calls.length).toBe(polls);
        expect(opened).not.toHaveBeenCalled();
    });
});

describe("Hexara with an expired saved sign-in", () => {
    it("drops the dead token and says so in words, not raw JSON", async () => {
        api.streamChat.mockImplementation(async function* () {
            yield { type: "error", code: "auth_required", message: "Your sign-in has expired — Sign in again to continue." };
        });
        // As on a real machine: the token survives in token.json from an
        // earlier run (checkpoints never store it) and has since expired.
        writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: "xoa_expired_credential_123456" }));
        await startAt("sdk-detect", { mode: "sdk", "device-auth": "authorized" });
        expect(flow.aiAvailable).toBe(true);

        await flow.askAi("hi");
        await settle();
        flow.revealAi();
        await settle();

        expect(flow.answers["api-key"]).toBeUndefined();
        expect(flow.aiAvailable).toBe(false);
        expect(flow.aiResponse).toContain("sign-in has expired");
        expect(flow.aiResponse).not.toContain("{");
    });
});

describe("readableDetail", () => {
    it("pulls the message out of the API's error envelope", () => {
        expect(readableDetail('{"ok":false,"error":{"code":"unauthorized","message":"Unauthorized"}}')).toBe("Unauthorized");
        expect(readableDetail('{"detail":"Not found"}')).toBe("Not found");
        expect(readableDetail("plain text")).toBe("plain text");
    });
});

describe("DeviceAuthStep", () => {
    const props = { userCode: "ABCD-EFGH", verificationUri: "https://xcelsior.example/device" };

    it("shows the code spaced out in a box and counts down from the short delay", () => {
        const { lastFrame } = render(<DeviceAuthStep {...props} status="waiting" onBack={() => {}} />);
        const out = lastFrame() ?? "";
        expect(out).toContain("A B C D - E F G H");
        expect(out).toContain(`${Math.round(DEVICE_BROWSER_DELAY_MS / 1000)}s`);
        expect(out).toContain("b to go back");
    });

    it("goes back on b while waiting or after an error, and not once signed in", async () => {
        for (const [status, expected] of [["waiting", 1], ["error", 1], ["authorized", 0]] as const) {
            const onBack = vi.fn();
            const { stdin, unmount } = render(<DeviceAuthStep {...props} status={status} onBack={onBack} />);
            await settle();
            stdin.write("b");
            await settle();
            expect(onBack).toHaveBeenCalledTimes(expected);
            unmount();
        }
    });
});
