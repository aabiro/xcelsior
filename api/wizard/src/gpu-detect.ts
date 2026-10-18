// Provide-GPUs detection when the full nvidia-smi query is unavailable.
// A missing binary and an empty GPU list are different failures.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { CheckResult } from "./checks.js";
import { HOST_FAIL_GPU, HOST_FAIL_NVIDIA_SMI } from "./host-requirements.js";

const exec = promisify(execFile);

export async function checkGpuBasic(): Promise<CheckResult[]> {
    try {
        const { stdout } = await exec("nvidia-smi", [
            "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ], { timeout: 10_000 });
        const gpus = stdout.trim().split("\n").filter(Boolean);
        if (gpus.length === 0) {
            return [{ name: "GPU Detection", ok: false, detail: HOST_FAIL_GPU, remediation: HOST_FAIL_GPU }];
        }
        return gpus.map((line, i) => ({
            name: `GPU ${i}`,
            ok: true,
            detail: line.trim(),
        }));
    } catch (err) {
        const missing = !!err && typeof err === "object" && "code" in err && (err as { code?: string }).code === "ENOENT";
        const sentence = missing ? HOST_FAIL_NVIDIA_SMI : HOST_FAIL_GPU;
        return [{ name: "GPU Detection", ok: false, detail: sentence, remediation: sentence }];
    }
}
