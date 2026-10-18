// Docker environment checks — executed during wizard preflight.
// Uses child_process to probe Docker, Docker Compose, and NVIDIA runtime.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import {
    HOST_FAIL_DOCKER_DOWN,
    HOST_FAIL_DOCKER_MISSING,
    HOST_FAIL_GPU,
    HOST_FAIL_NVIDIA_RUNTIME,
    HOST_FAIL_NVIDIA_SMI,
} from "./host-requirements.js";

const exec = promisify(execFile);

export interface CheckResult {
    name: string;
    ok: boolean;
    detail: string;
    /** Actionable fix shown beneath a failed check (A9 — surface remediation). */
    remediation?: string;
    /** Optional link the user can open to resolve the failure. */
    url?: string;
}

async function run(cmd: string, args: string[]): Promise<{ stdout: string; ok: boolean }> {
    try {
        const { stdout } = await exec(cmd, args, { timeout: 10_000 });
        return { stdout: stdout.trim(), ok: true };
    } catch {
        return { stdout: "", ok: false };
    }
}

/** Parse a semver-like version string into [major, minor, patch] */
function parseVersion(raw: string): [number, number, number] {
    const m = raw.match(/(\d+)\.(\d+)\.(\d+)/);
    if (!m) return [0, 0, 0];
    return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function versionGte(
    actual: [number, number, number],
    min: [number, number, number],
): boolean {
    for (let i = 0; i < 3; i++) {
        if (actual[i] > min[i]) return true;
        if (actual[i] < min[i]) return false;
    }
    return true; // equal
}

export async function checkDocker(onItem?: (r: CheckResult) => void): Promise<CheckResult[]> {
    const results: CheckResult[] = [];
    // Stream each item to the caller as it resolves (Part E — line-by-line status).
    const push = (r: CheckResult) => { results.push(r); onItem?.(r); };

    const dockerCli = await run("bash", ["-lc", "command -v docker"]);
    if (!dockerCli.ok) {
        push({ name: "Docker", ok: false, detail: HOST_FAIL_DOCKER_MISSING, remediation: HOST_FAIL_DOCKER_MISSING });
    } else {
        const docker = await run("docker", ["info", "--format", "{{.ServerVersion}}"]);
        if (!docker.ok) {
            push({ name: "Docker", ok: false, detail: HOST_FAIL_DOCKER_DOWN, remediation: HOST_FAIL_DOCKER_DOWN });
        } else {
            const runtime = await run("docker", ["info", "--format", "{{json .Runtimes}}"]);
            if (!(runtime.ok && /nvidia/i.test(runtime.stdout))) {
                push({ name: "Docker", ok: false, detail: HOST_FAIL_NVIDIA_RUNTIME, remediation: HOST_FAIL_NVIDIA_RUNTIME });
            } else {
                const ver = parseVersion(docker.stdout);
                const minVer: [number, number, number] = [24, 0, 0];
                const ok = versionGte(ver, minVer);
                push({
                    name: "Docker",
                    ok,
                    detail: ok ? `v${docker.stdout}` : `v${docker.stdout} — needs ≥24.0.0`,
                    ...(ok ? {} : { remediation: "Upgrade Docker Engine to 24 or newer and restart it: https://docs.docker.com/engine/install/" }),
                });
            }
        }
    }

    // 2. Docker Compose
    const compose = await run("docker", ["compose", "version", "--short"]);
    if (compose.ok) {
        push({ name: "Docker Compose", ok: true, detail: `v${compose.stdout}` });
    } else {
        push({ name: "Docker Compose", ok: false, detail: "not found" });
    }

    // 3. NVIDIA Container Toolkit (nvidia-smi)
    const nvidiaCli = await run("bash", ["-lc", "command -v nvidia-smi"]);
    const nvidia = nvidiaCli.ok
        ? await run("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader"])
        : { stdout: "", ok: false };
    if (nvidia.ok && nvidia.stdout.trim()) {
        const driver = nvidia.stdout.split("\n")[0];
        push({ name: "NVIDIA Driver", ok: true, detail: driver });
    } else if (!nvidiaCli.ok) {
        push({ name: "NVIDIA Driver", ok: false, detail: HOST_FAIL_NVIDIA_SMI, remediation: HOST_FAIL_NVIDIA_SMI });
    } else {
        push({ name: "NVIDIA Driver", ok: false, detail: HOST_FAIL_GPU, remediation: HOST_FAIL_GPU });
    }
    // 4. runc version (gVisor compatibility)
    const runc = await run("runc", ["--version"]);
    if (runc.ok) {
        const ver = parseVersion(runc.stdout);
        const minVer: [number, number, number] = [1, 1, 12];
        const ok = versionGte(ver, minVer);
        push({
            name: "runc",
            ok,
            detail: ok
                ? `v${ver.join(".")}`
                : `v${ver.join(".")} — needs ≥1.1.12`,
            ...(ok ? {} : { remediation: "Upgrade runc to ≥1.1.12 via your distro packages or containerd" }),
        });
    } else {
        push({ name: "runc", ok: false, detail: "not found", remediation: "Install runc ≥1.1.12 via your distro packages or containerd" });
    }

    return results;
}
