import { defineConfig, devices } from "@playwright/test";

const PORT = 3100;
const REMOTE_BASE_URL = process.env.XCELSIOR_PWA_SMOKE_BASE_URL?.trim();
const BASE_URL = REMOTE_BASE_URL || `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  timeout: 30_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    ...devices["Desktop Chrome"],
    baseURL: BASE_URL,
    serviceWorkers: "allow",
  },
  webServer: REMOTE_BASE_URL
    ? undefined
    : {
        command: `HOSTNAME=127.0.0.1 PORT=${PORT} node ./scripts/start-standalone-for-smoke.mjs`,
        port: PORT,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
