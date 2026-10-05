import { defineConfig, devices } from "@playwright/test";

/**
 * Examines the real production site. No mocks, no local server.
 *
 * Kept apart from `playwright.config.ts` on purpose: every spec in `e2e/`
 * fulfils API routes with fixtures, which proves rendering and nothing about
 * production. Pointing that suite at prod would report green on a broken
 * backend; pointing this one at a local build would examine the wrong thing.
 *
 *   XCELSIOR_PROD_BASE_URL=https://xcelsior.ca npx playwright test -c playwright.prod.config.ts
 */
// The installed Chrome rather than a Playwright-managed download: this examines
// the site as a real browser renders it, and needs no `playwright install`.
const browser = process.env.XCELSIOR_PLAYWRIGHT_EXECUTABLE_PATH?.trim()
  ? { launchOptions: { executablePath: process.env.XCELSIOR_PLAYWRIGHT_EXECUTABLE_PATH.trim() } }
  : { channel: "chrome" as const };

export default defineConfig({
  testDir: "./e2e-prod",
  outputDir: "./test-results-prod",
  fullyParallel: false,
  workers: 2,
  retries: 0,
  timeout: 60_000,
  reporter: [["list"], ["json", { outputFile: "test-results-prod/report.json" }]],
  use: {
    baseURL: process.env.XCELSIOR_PROD_BASE_URL?.trim() || "https://xcelsior.ca",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], ...browser } },
    { name: "mobile", use: { ...devices["Pixel 7"], ...browser } },
  ],
});
