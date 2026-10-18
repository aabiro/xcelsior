import { describe, expect, it, vi } from "vitest";
import { execFile, execSync } from "node:child_process";
import { checkGpuBasic } from "../gpu-detect.js";
import { networkSetupCheck, setupNetworking } from "../provider-checks.js";
import { summarizeFailure } from "../messaging.js";
import {
    HOST_FAIL_GPU,
    HOST_FAIL_HOST_IP,
    HOST_FAIL_NVIDIA_RUNTIME,
    HOST_FAIL_NVIDIA_SMI,
} from "../host-requirements.js";

vi.mock("node:child_process", () => ({
    execFile: vi.fn(),
    execSync: vi.fn(() => {
        throw new Error("command missing");
    }),
}));

function finish(stdout: string) {
    vi.mocked(execFile).mockImplementation(((...args: unknown[]) => {
        const callback = args[args.length - 1] as (err: null, result: { stdout: string; stderr: string }) => void;
        callback(null, { stdout, stderr: "" });
        return undefined as never;
    }) as typeof execFile);
}

function failWith(code: string) {
    vi.mocked(execFile).mockImplementation(((...args: unknown[]) => {
        const callback = args[args.length - 1] as (err: NodeJS.ErrnoException) => void;
        const error = new Error(code) as NodeJS.ErrnoException;
        error.code = code;
        callback(error);
        return undefined as never;
    }) as typeof execFile);
}

describe("checkGpuBasic", () => {
    it("says the driver is missing when nvidia-smi cannot be started", async () => {
        failWith("ENOENT");
        const [result] = await checkGpuBasic();
        expect(result.ok).toBe(false);
        expect(result.detail).toBe(HOST_FAIL_NVIDIA_SMI);
        expect(result.remediation).toContain("Install a supported NVIDIA driver");
        expect(result.detail).not.toContain("too old");
    });

    it("says no usable GPU when nvidia-smi returns an empty list", async () => {
        finish("\n");
        const [result] = await checkGpuBasic();
        expect(result.ok).toBe(false);
        expect(result.detail).toBe(HOST_FAIL_GPU);
        expect(result.remediation).toContain("Seat a supported NVIDIA GPU");
    });
});

describe("setupNetworking with no address", () => {
    it("uses the shared host-IP sentence and keeps Tailscale optional", async () => {
        vi.mocked(execSync).mockImplementation(() => {
            throw new Error("missing");
        });
        const result = await setupNetworking();
        expect(result.method).toBe("none");
        expect(result.detail).toBe(HOST_FAIL_HOST_IP);
        const row = networkSetupCheck(result);
        expect(row.ok).toBe(false);
        expect(row.detail).toBe(HOST_FAIL_HOST_IP);
        expect(row.remediation).toContain("Tailscale is optional");
        expect(row.remediation).toContain("api.ipify.org");
        const summary = summarizeFailure([row]);
        expect(summary).toContain("No host IP could be detected");
        expect(summary).not.toContain("Network quality is below target");
    });
});

describe("summarizeFailure uses the check's own cause", () => {
    it("does not call a missing nvidia-smi an old driver or a missing runtime a stopped daemon", () => {
        const summary = summarizeFailure([
            { name: "NVIDIA Driver", ok: false, detail: HOST_FAIL_NVIDIA_SMI, remediation: HOST_FAIL_NVIDIA_SMI },
            { name: "Docker", ok: false, detail: HOST_FAIL_NVIDIA_RUNTIME, remediation: HOST_FAIL_NVIDIA_RUNTIME },
        ]);
        expect(summary).toContain("nvidia-smi is not installed");
        expect(summary).toContain("NVIDIA container runtime is not installed");
        expect(summary).not.toContain("too old");
        expect(summary).not.toContain("isn't ready");
    });
});
