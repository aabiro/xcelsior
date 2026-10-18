import { describe, expect, it, vi, afterEach } from "vitest";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { configDirectory } from "../config-files.js";
import { checkSdkApi, checkSdkPackage, checkSdkProject } from "../sdk-checks.js";
import { installWorkerAgent } from "../provider-checks.js";
import { explainInstallFailure, HOST_FAIL_LINUX_DARWIN, SDK_FAIL_CREDENTIALS, SDK_FAIL_INSTALL, SDK_FAIL_NO_PACKAGE, SDK_FAIL_PYTHON } from "../host-requirements.js";

vi.mock("node:child_process", () => ({
    execFile: vi.fn((_file: string, _args: string[], _opts: unknown, callback?: (err: Error) => void) => {
        const done = typeof _opts === "function" ? _opts as (err: Error) => void : callback;
        const error = new Error("npm ERR! 404 not found\n    at install (/usr/lib/node_modules/npm/index.js:1:1)\nError: Command failed");
        done?.(error);
    }),
}));

afterEach(() => vi.restoreAllMocks());

describe("worker install unhappy path", () => {
    it("refuses macOS with the same Linux next step as the installer", async () => {
        const original = Object.getOwnPropertyDescriptor(process, "platform");
        Object.defineProperty(process, "platform", { value: "darwin" });
        try {
            const result = await installWorkerAgent("https://example.test", "xcel_ai_key", "host_1", "203.0.113.8");
            expect(result.installed).toBe(false);
            expect(result.detail).toBe(HOST_FAIL_LINUX_DARWIN);
            expect(result.detail).toContain("Use a Linux host");
            expect(result.detail).not.toMatch(/at Object|Command failed/);
        } finally {
            if (original) Object.defineProperty(process, "platform", original);
        }
    });

    it("surfaces the installer sentence instead of a stack trace", () => {
        const text = explainInstallFailure(
            "\u001b[0;31m[✗]\u001b[0m nvidia-smi is not installed, so no NVIDIA driver was found. Install a supported NVIDIA driver (550 or newer), then run nvidia-smi once to confirm a GPU is listed: https://www.nvidia.com/Download/index.aspx\n",
            "Error: Command failed: bash -c source install.sh\n    at ChildProcess.<anonymous>",
            "Command failed",
        );
        expect(text).toContain("nvidia-smi is not installed");
        expect(text).toContain("Install a supported NVIDIA driver");
        expect(text).not.toContain("at ChildProcess");
    });
});

describe("SDK unhappy path", () => {
    it("names a Python project and the next step", () => {
        const root = configDirectory();
        writeFileSync(join(root, "pyproject.toml"), "[project]\nname='example'\n");
        vi.spyOn(process, "cwd").mockReturnValue(root);
        const [result] = checkSdkProject();
        expect(result.ok).toBe(false);
        expect(result.detail).toBe(SDK_FAIL_PYTHON);
        expect(result.remediation).toContain("press b");
        expect(result.detail).not.toMatch(/at Object/);
    });

    it("names a missing package.json and where to go next", () => {
        const root = configDirectory();
        vi.spyOn(process, "cwd").mockReturnValue(root);
        const [result] = checkSdkProject();
        expect(result.detail).toBe(SDK_FAIL_NO_PACKAGE);
        expect(result.remediation).toContain("package.json");
        expect(result.remediation).toContain("press b");
    });

    it("does not use a package-manager dump as the only install failure", async () => {
        const root = join(configDirectory(), "sdk-app");
        mkdirSync(root, { recursive: true });
        writeFileSync(join(root, "package.json"), JSON.stringify({ name: "app" }));
        vi.spyOn(process, "cwd").mockReturnValue(root);
        const [result] = await checkSdkPackage();
        expect(result.ok).toBe(false);
        expect(result.detail).toBe(SDK_FAIL_INSTALL);
        expect(result.remediation).toContain("retry this step");
        expect(result.detail).not.toContain("npm ERR!");
        expect(result.remediation).not.toContain("at install");
    });

    it("names setup before credentials", async () => {
        const [result] = await checkSdkApi("https://example.test", "");
        expect(result.ok).toBe(false);
        expect(result.detail).toBe(SDK_FAIL_CREDENTIALS);
        expect(result.remediation).toContain("API key");
        expect(result.detail).not.toMatch(/xcel_ai_|xoa_/);
    });
});
