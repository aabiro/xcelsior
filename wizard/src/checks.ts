// Docker environment checks — executed during wizard preflight.
// Uses child_process to probe Docker, Docker Compose, and NVIDIA runtime.

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const exec = promisify(execFile);

export interface CheckResult {
    name: string;
    ok: boolean;
    detail: string;
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

export async function checkDocker(): Promise<CheckResult[]> {
    const results: CheckResult[] = [];

    // 1. Docker daemon
    const docker = await run("docker", ["info", "--format", "{{.ServerVersion}}"]);
    if (docker.ok) {
        const ver = parseVersion(docker.stdout);
        const minVer: [number, number, number] = [20, 10, 0];
        const ok = versionGte(ver, minVer);
        results.push({
            name: "Docker",
            ok,
            detail: ok ? `v${docker.stdout}` : `v${docker.stdout} — needs ≥20.10.0`,
        });
    } else {
        results.push({ name: "Docker", ok: false, detail: "not found or daemon not running" });
    }

    // 2. Docker Compose
    const compose = await run("docker", ["compose", "version", "--short"]);
    if (compose.ok) {
        results.push({ name: "Docker Compose", ok: true, detail: `v${compose.stdout}` });
    } else {
        results.push({ name: "Docker Compose", ok: false, detail: "not found" });
    }

    // 3. NVIDIA Container Toolkit (nvidia-smi)
    const nvidia = await run("nvidia-smi", ["--query-gpu=driver_version", "--format=csv,noheader"]);
    if (nvidia.ok) {
        const driver = nvidia.stdout.split("\n")[0];
        results.push({ name: "NVIDIA Driver", ok: true, detail: `v${driver}` });
    } else {
        results.push({ name: "NVIDIA Driver", ok: false, detail: "nvidia-smi not found or no GPU" });
    }

    // 4. runc version (gVisor compatibility)
    const runc = await run("runc", ["--version"]);
    if (runc.ok) {
        const ver = parseVersion(runc.stdout);
        const minVer: [number, number, number] = [1, 1, 12];
        const ok = versionGte(ver, minVer);
        results.push({
            name: "runc",
            ok,
            detail: ok
                ? `v${ver.join(".")}`
                : `v${ver.join(".")} — needs ≥1.1.12`,
        });
    } else {
        results.push({ name: "runc", ok: false, detail: "not found" });
    }

    return results;
}
