import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { render } from "ink-testing-library";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { AutoCheckStep } from "../steps.js";
import { configDirectory, writePrivateFile } from "../config-files.js";
import { loadWizardCheckpoint, saveWizardCheckpoint, tokenFilePath } from "../wizard-state.js";
import { answersForNewPath } from "../wizard-guards.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

// A resumed run had no way out of a check its path cannot pass: the SDK path
// started in a Python repo failed "Detect project framework", Enter retried,
// and running the wizard again resumed the same failing step.
vi.mock("../preflight.js", () => ({
    fetchServiceStatus: vi.fn(async () => ({ verdict: "operational", services: [] })),
    aiHealthyFromReport: () => false,
}));
vi.mock("../marketplace-stats.js", () => ({ fetchMarketplaceStats: vi.fn(async () => null) }));
vi.mock("../open-url.js", () => ({ openUrl: vi.fn(async () => undefined) }));

const token = "xoa_sdk_credential_123456789";
let flow: UseWizardFlowReturn;
function Probe() {
    flow = useWizardFlow();
    return null;
}
async function settle() {
    for (let i = 0; i < 4; i += 1) await new Promise<void>((resolve) => setImmediate(resolve));
}

beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
});
afterEach(() => vi.restoreAllMocks());

describe("choosing a different path from a failed check", () => {
    it("takes a resumed SDK run in a Python repo back to the path choice, still signed in", async () => {
        const pythonRepo = configDirectory();
        writeFileSync(join(pythonRepo, "pyproject.toml"), "[project]\nname='example'\n");
        vi.spyOn(process, "cwd").mockReturnValue(pythonRepo);
        saveWizardCheckpoint({
            stepIndex: WIZARD_STEPS.findIndex((step) => step.id === "sdk-detect"),
            answers: { mode: "sdk", "device-auth": "authorized", _host_id: "host-kept" },
            completedStepIds: ["mode"],
            savedAt: new Date().toISOString(),
        });

        render(<Probe />);
        await settle();
        await vi.advanceTimersByTimeAsync(2_000);
        await settle();
        expect(flow.step.id).toBe("sdk-detect");
        expect(flow.checkCanRetry).toBe(true);
        expect(flow.resumedStepLabel).not.toBeNull();

        flow.changePath();
        await settle();
        expect(flow.step.id).toBe("mode");
        expect(flow.answers.mode).toBeUndefined();
        expect(flow.answers["api-key"]).toBe(token);
        expect(flow.answers._host_id).toBe("host-kept");
        expect(flow.resumedStepLabel).toBeNull();

        // Running the wizard again must not resume into the failed step.
        await vi.advanceTimersByTimeAsync(5_000);
        expect(loadWizardCheckpoint()?.stepIndex).toBe(0);
    });
});

describe("answersForNewPath", () => {
    it("drops path choices and keeps the sign-in and what prevents duplicates", () => {
        const kept = answersForNewPath({
            mode: "provide", workload: "training", "gpu-pick": "host-1",
            "api-key": token, "device-auth": "authorized", "oauth-client-id": "client", "oauth-client-secret": "secret",
            _host_id: "host-1", _launch_name: "run-1", _api_base_url: "https://xcelsior.ca",
        });
        expect(kept).toEqual({
            "api-key": token, "device-auth": "authorized", "oauth-client-id": "client", "oauth-client-secret": "secret",
            _host_id: "host-1", _launch_name: "run-1", _api_base_url: "https://xcelsior.ca",
        });
    });
});

describe("AutoCheckStep", () => {
    const failed = { items: [{ name: "Project", ok: false, detail: "This is a Python project" }], allPassed: false };

    it("offers b on a failed required check and calls back when pressed", async () => {
        const onChangePath = vi.fn();
        const { lastFrame, stdin } = render(
            <AutoCheckStep results={failed} canRetry required onChangePath={onChangePath} />,
        );
        expect(lastFrame()).toContain("b to go back and choose a different path");
        await settle();
        stdin.write("b");
        await settle();
        expect(onChangePath).toHaveBeenCalledTimes(1);
    });

    it("does not offer b where there is nowhere to go", () => {
        const { lastFrame } = render(<AutoCheckStep results={failed} canRetry required />);
        expect(lastFrame()).not.toContain("different path");
    });
});
