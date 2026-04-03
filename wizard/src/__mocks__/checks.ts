// Mock for checks.ts — used in tests to avoid real Docker/nvidia-smi calls.
// Returns configurable check results.

import type { CheckResult } from "../checks.js";

export type MockCheckPreset = "all-pass" | "all-fail" | "partial";

const PRESETS: Record<MockCheckPreset, CheckResult[]> = {
    "all-pass": [
        { name: "Docker", ok: true, detail: "v27.3.1" },
        { name: "Docker Compose", ok: true, detail: "v2.30.0" },
        { name: "NVIDIA Driver", ok: true, detail: "v560.35.03" },
        { name: "runc", ok: true, detail: "v1.2.1" },
    ],
    "all-fail": [
        { name: "Docker", ok: false, detail: "not found or daemon not running" },
        { name: "Docker Compose", ok: false, detail: "not found" },
        { name: "NVIDIA Driver", ok: false, detail: "nvidia-smi not found or no GPU" },
        { name: "runc", ok: false, detail: "not found" },
    ],
    "partial": [
        { name: "Docker", ok: true, detail: "v27.3.1" },
        { name: "Docker Compose", ok: true, detail: "v2.30.0" },
        { name: "NVIDIA Driver", ok: false, detail: "nvidia-smi not found or no GPU" },
        { name: "runc", ok: true, detail: "v1.2.1" },
    ],
};

let _preset: MockCheckPreset = "all-pass";
let _customResults: CheckResult[] | null = null;

/** Set which mock preset `checkDocker()` returns */
export function setMockPreset(preset: MockCheckPreset) {
    _preset = preset;
    _customResults = null;
}

/** Override with fully custom check results */
export function setMockResults(results: CheckResult[]) {
    _customResults = results;
}

/** Reset to default */
export function resetMock() {
    _preset = "all-pass";
    _customResults = null;
}

/** Mock checkDocker — resolves immediately with preset results */
export async function checkDocker(): Promise<CheckResult[]> {
    return _customResults ?? [...PRESETS[_preset]];
}
