import { afterEach, beforeEach, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cleanup } from "ink-testing-library";

let configDir: string;

beforeEach(() => {
    configDir = mkdtempSync(join(tmpdir(), "xcelsior-wizard-test-"));
    vi.stubEnv("XCELSIOR_CONFIG_DIR", configDir);
    vi.stubEnv("XCELSIOR_NO_BROWSER", "1");
    vi.stubEnv("XCELSIOR_NO_SPRITE", "1");
});

afterEach(async () => {
    // Unmount while the test directory still exists, including exit checkpoints.
    cleanup();
    vi.useRealTimers();
    await new Promise<void>((resolve) => setImmediate(resolve));
    rmSync(configDir, { recursive: true, force: true });
    vi.unstubAllEnvs();
});
