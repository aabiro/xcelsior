import React from "react";
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";
import { render } from "ink-testing-library";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { useWizardFlow, type UseWizardFlowReturn } from "../useWizardFlow.js";
import { configDirectory, writePrivateFile } from "../config-files.js";
import { saveWizardCheckpoint, tokenFilePath } from "../wizard-state.js";
import { WIZARD_STEPS } from "../wizard-flow.js";

// Drives the real controller through the provider track. Only the machine
// (GPU, Docker, network) and the HTTP boundary are replaced.
const api = vi.hoisted(() => ({
    getMe: vi.fn(), searchMarketplace: vi.fn(), createOAuthClient: vi.fn(),
    registerHost: vi.fn(), reportVersions: vi.fn(), reportBenchmark: vi.fn(),
}));
const machine = vi.hoisted(() => ({
    detectGpuFull: vi.fn(), checkVersions: vi.fn(), setupNetworking: vi.fn(),
    runComputeBenchmark: vi.fn(), runNetworkBenchmark: vi.fn(), installWorkerAgent: vi.fn(),
}));
const docker = vi.hoisted(() => ({ checkDocker: vi.fn() }));
vi.mock("../api-client.js", async (original) => ({
    ...await original<typeof import("../api-client.js")>(), ...api,
}));
vi.mock("../provider-checks.js", async (original) => ({
    ...await original<typeof import("../provider-checks.js")>(), ...machine,
}));
vi.mock("../checks.js", async (original) => ({
    ...await original<typeof import("../checks.js")>(), ...docker,
}));
vi.mock("../preflight.js", () => ({
    fetchServiceStatus: vi.fn(async () => ({ verdict: "operational", services: [] })),
    aiHealthyFromReport: () => false,
}));
vi.mock("../marketplace-stats.js", () => ({ fetchMarketplaceStats: vi.fn(async () => null) }));

const token = "xoa_provider_credential_123456789";
const gpu = {
    gpu_model: "RTX 4090", total_vram_gb: 24, free_vram_gb: 23, driver_version: "550.54",
    serial: "1324021000001", uuid: "GPU-1", pci_bus_id: "00000000:01:00.0", compute_capability: "8.9",
};
const versions = [
    { component: "runc", version: "1.1.12", minimum: "1.1.12", passed: true },
    { component: "nvidia_toolkit", version: "1.17.8", minimum: "1.17.8", passed: true },
    { component: "nvidia_driver", version: "550.54.14", minimum: "550.0.0", passed: true },
    { component: "docker", version: "24.0.7", minimum: "24.0.0", passed: true },
];
const bench = {
    tflops: 82.6, xcu_score: 850, pcie_bandwidth_gbps: 24, pcie_h2d_gbps: 24, pcie_d2h_gbps: 23,
    gpu_temp_celsius: 71, gpu_temp_avg_celsius: 66, gpu_temp_samples: 12,
    gpu_model: "RTX 4090", total_vram_gb: 24, compute_capability: "8.9", cuda_version: "12.4",
    driver_version: "550.54", elapsed_s: 61,
};
const network = {
    latency_avg_ms: 18, latency_min_ms: 15, latency_max_ms: 24, jitter_ms: 3,
    packet_loss_pct: 0, throughput_mbps: 480,
};

let flow: UseWizardFlowReturn;
function Probe() {
    flow = useWizardFlow();
    return null;
}

async function settle() {
    for (let i = 0; i < 4; i += 1) await new Promise<void>((resolve) => setImmediate(resolve));
}

/** Answer each prompt the way a provider would until `stopAt` is reached. */
async function drive(answers: Record<string, string>, stopAt: string) {
    const visited: string[] = [];
    for (let turn = 0; turn < 80; turn += 1) {
        await vi.advanceTimersByTimeAsync(2_500);
        await settle();
        const step = flow.step;
        if (visited[visited.length - 1] !== step.id) visited.push(step.id);
        if (step.id === stopAt || flow.isComplete) return visited;
        if (step.type === "auto-check") {
            if (flow.checkCanRetry) {
                const failed = flow.checkResults[step.id]?.items.filter((item) => !item.ok) ?? [];
                throw new Error(`${step.id} failed: ${failed.map((item) => `${item.name}: ${item.detail}`).join("; ")}`);
            }
            if (flow.checkAwaitContinue) flow.continueFromCheck();
            continue;
        }
        if (step.id in answers) flow.submitAnswer(answers[step.id]);
        else if (step.type === "confirm") flow.submitAnswer("yes");
    }
    throw new Error(`Journey did not reach ${stopAt}; visited ${visited.join(" → ")}`);
}

beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
    api.getMe.mockReset().mockResolvedValue({ user_id: "user-1", email: "provider@example.test", customer_id: "customer-1" });
    api.searchMarketplace.mockReset().mockResolvedValue({ listings: [
        { host_id: "peer-1", gpu_model: "RTX 4090", vram_gb: 24, price_per_hour: 0.8, owner: "peer", active: true },
        { host_id: "peer-2", gpu_model: "RTX 4090", vram_gb: 24, price_per_hour: 1.0, owner: "peer", active: true },
    ] });
    api.createOAuthClient.mockReset().mockResolvedValue({ client_id: "oauth_worker", client_secret: "secret_worker" });
    api.registerHost.mockReset().mockImplementation(async (_url: string, _token: string, body: { host_id: string }) => ({ host_id: body.host_id }));
    api.reportVersions.mockReset().mockResolvedValue({
        compatible: true, admitted: false, admission_applied: false, details: { recommended_runtime: "runc" },
    });
    api.reportBenchmark.mockReset().mockResolvedValue({ ok: true, xcu: 850 });
    docker.checkDocker.mockReset().mockResolvedValue([{ name: "Docker", ok: true, detail: "24.0.7" }]);
    machine.detectGpuFull.mockReset().mockResolvedValue(gpu);
    machine.checkVersions.mockReset().mockResolvedValue(versions);
    machine.setupNetworking.mockReset().mockResolvedValue({ method: "headscale", ip: "100.64.0.9", detail: "Mesh connected" });
    machine.runComputeBenchmark.mockReset().mockResolvedValue(bench);
    machine.runNetworkBenchmark.mockReset().mockResolvedValue(network);
    machine.installWorkerAgent.mockReset().mockResolvedValue({
        installed: true, detail: "Worker service active; fresh scheduler heartbeat confirmed. Admission is still pending.",
    });
    writePrivateFile(tokenFilePath(), JSON.stringify({ access_token: token }));
    saveWizardCheckpoint({
        stepIndex: WIZARD_STEPS.findIndex((step) => step.id === "gpu-detect"),
        answers: { mode: "provide", "api-key": token, "_email": "provider@example.test", "_customer_id": "customer-1" },
        completedStepIds: ["mode", "docker-check", "device-auth", "api-check"],
        savedAt: new Date().toISOString(),
    });
});
afterEach(() => vi.restoreAllMocks());

describe("provider journey through the flow controller", () => {
    it("passes local verification before the host exists, then registers, installs and saves", async () => {
        render(<Probe />);
        await settle();
        const visited = await drive({ pricing: "recommended", "spot-enabled": "no" }, "done");

        expect(visited).toEqual(expect.arrayContaining([
            "gpu-detect", "version-check", "network-setup", "benchmark", "network-bench",
            "verification", "pricing", "spot-enabled", "host-register", "admission-gate",
            "provider-summary", "worker-install",
        ]));
        // Local verification runs before registration, so it reports local
        // readiness only; it must not carry a server verdict that cannot exist yet.
        const verification = flow.checkResults["verification"];
        expect(verification.allPassed).toBe(true);
        expect(verification.items.map((item) => item.name)).not.toContain("Server Verification");

        expect(api.registerHost).toHaveBeenCalledTimes(1);
        const registered = api.registerHost.mock.calls[0][2];
        expect(registered.ip).toBe("100.64.0.9");
        expect(registered.cost_per_hour).toBe(0.9);
        expect(registered.spot_enabled).toBe(false);
        expect(machine.installWorkerAgent).toHaveBeenCalledTimes(1);
        expect(machine.installWorkerAgent.mock.calls[0][2]).toBe(registered.host_id);
        expect(flow.isComplete).toBe(true);
        expect(readFileSync(join(configDirectory(), "config.toml"), "utf8")).toContain('mode = "provide"');
    });

    it("stops at the network test instead of verifying with invented numbers", async () => {
        machine.runNetworkBenchmark.mockResolvedValue({ ...network, packet_loss_pct: 100, throughput_mbps: 0, latency_avg_ms: 0 });
        render(<Probe />);
        await settle();
        await expect(drive({ pricing: "recommended", "spot-enabled": "no" }, "verification"))
            .rejects.toThrow(/network-bench failed/);
        flow.skipCheck();
        await settle();
        expect(flow.step.id).toBe("network-bench");
        expect(api.registerHost).not.toHaveBeenCalled();
    });

    it("does not report success when the worker never sends a heartbeat", async () => {
        machine.installWorkerAgent.mockResolvedValue({ installed: false, detail: "Service installed, but no fresh worker heartbeat was confirmed." });
        render(<Probe />);
        await settle();
        await expect(drive({ pricing: "recommended", "spot-enabled": "no" }, "done"))
            .rejects.toThrow(/worker-install failed: Worker Agent: Service installed, but no fresh worker heartbeat/);
        expect(flow.isComplete).toBe(false);
    });
});
