import { expect, test, type BrowserContext, type Route } from "@playwright/test";

const ENDPOINT_ID = "sep-smoke-1";
const AUTH_COOKIE = "xcelsior_session";

const mockUser = {
  user_id: "smoke-user",
  email: "smoke@xcelsior.ca",
  name: "Smoke User",
  role: "user",
  customer_id: "cust-smoke",
  team_can_write_instances: true,
  team_can_manage_billing: true,
};

const mockEndpoint = {
  endpoint_id: ENDPOINT_ID,
  owner_id: "cust-smoke",
  name: "smoke-llama",
  model_id: "meta-llama/Llama-3.1-8B-Instruct",
  model_name: "meta-llama/Llama-3.1-8B-Instruct",
  model_ref: "meta-llama/Llama-3.1-8B-Instruct",
  gpu_type: "RTX 4090",
  region: "ca-east",
  docker_image: "xcelsior/serverless-vllm:12.4",
  mode: "preset",
  status: "active",
  min_workers: 0,
  max_workers: 2,
  total_requests: 0,
  total_cost_cad: 0,
  openai_base_url: `/v1/serverless/${ENDPOINT_ID}/openai/v1`,
  created_at: 1_700_000_000,
  updated_at: 1_700_000_100,
};

function jsonResponse(body: unknown, status = 200) {
  return {
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  };
}

/** Bypass Next middleware (proxy.ts) + mock API before standalone rewrites hit prod. */
async function installAuthenticatedSmokeContext(ctx: BrowserContext, baseURL: string) {
  const { hostname } = new URL(baseURL);
  await ctx.addCookies([
    {
      name: AUTH_COOKIE,
      value: "smoke-session",
      domain: hostname,
      path: "/",
    },
  ]);
}

async function installServerlessRouteMocks(ctx: BrowserContext, baseURL: string) {
  const apiGlob = `${baseURL.replace(/\/$/, "")}/api/**`;
  await ctx.route(apiGlob, async (route: Route) => {
    const request = route.request();
    const url = request.url();
    const method = request.method();

    if (url.includes("/api/auth/me")) {
      return route.fulfill(jsonResponse({ ok: true, user: mockUser }));
    }
    if (url.includes("/api/auth/refresh")) {
      return route.fulfill(jsonResponse({ ok: true, access_token: "smoke", expires_in: 3600 }));
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}/keys`) && method === "POST") {
      return route.fulfill(
        jsonResponse({
          ok: true,
          key: { key_id: "key-1", name: "smoke", key_prefix: "xcel_", api_key: "xcel_secret_once" },
        }),
      );
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}/keys`)) {
      return route.fulfill(jsonResponse({ ok: true, keys: [] }));
    }
    if (url.includes("/api/v2/serverless/endpoints") && method === "GET") {
      if (url.includes(ENDPOINT_ID)) {
        return route.fulfill(jsonResponse({ ok: true, endpoint: mockEndpoint }));
      }
      return route.fulfill(jsonResponse({ ok: true, endpoints: [mockEndpoint] }));
    }
    if (url.includes("/api/v2/serverless/endpoints") && method === "POST") {
      return route.fulfill(jsonResponse({ ok: true, endpoint: mockEndpoint }));
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}/metrics`)) {
      return route.fulfill(
        jsonResponse({
          ok: true,
          metrics: {
            endpoint_id: ENDPOINT_ID,
            total_requests: 0,
            total_cost_cad: 0,
            active_workers: 0,
            idle_workers: 0,
            busy_workers: 0,
          },
        }),
      );
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}/workers`)) {
      return route.fulfill(jsonResponse({ ok: true, workers: [] }));
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}/jobs`)) {
      return route.fulfill(jsonResponse({ ok: true, jobs: [] }));
    }
    if (url.includes(`/api/v2/serverless/endpoints/${ENDPOINT_ID}`)) {
      return route.fulfill(jsonResponse({ ok: true, endpoint: mockEndpoint }));
    }
    if (url.includes("/api/v2/gpu/available")) {
      return route.fulfill(
        jsonResponse({
          ok: true,
          gpus: [
            {
              gpu_model: "RTX 4090",
              vram_gb: 24,
              region: "ca-east",
              province: "QC",
              count_available: 4,
              price_per_hour_cad: 1.2,
            },
          ],
        }),
      );
    }
    if (url.includes("/api/stream")) {
      return route.fulfill({ status: 204, body: "" });
    }

    // Prevent apiFetch 401 → login redirect for any other dashboard probe.
    return route.fulfill(jsonResponse({ ok: true }));
  });
}

test.describe("serverless dashboard", () => {
  test("redirects unauthenticated users to login", async ({ browser }) => {
    const context = await browser.newContext({ serviceWorkers: "block" });
    const page = await context.newPage();
    try {
      await page.goto("/dashboard/inference", { waitUntil: "domcontentloaded" });
      await expect(page).toHaveURL(/\/login\?redirect=%2Fdashboard%2Finference/, { timeout: 15_000 });
    } finally {
      await context.close();
    }
  });

  test("full smoke — list, Deploy Studio, detail tabs", async ({ browser, baseURL }) => {
    test.setTimeout(90_000);
    const origin = baseURL ?? "http://127.0.0.1:3100";
    const context = await browser.newContext({ serviceWorkers: "block" });
    await installAuthenticatedSmokeContext(context, origin);
    await installServerlessRouteMocks(context, origin);
    await context.addInitScript(() => {
      class MockEventSource extends EventTarget {
        readonly url: string;
        readonly readyState = 1;
        onopen: ((ev: Event) => void) | null = null;
        onerror: ((ev: Event) => void) | null = null;
        onmessage: ((ev: MessageEvent) => void) | null = null;
        constructor(url: string | URL) {
          super();
          this.url = String(url);
          queueMicrotask(() => {
            const ev = new Event("open");
            this.onopen?.(ev);
            this.dispatchEvent(ev);
          });
        }
        close() {}
        addEventListener() {}
        removeEventListener() {}
      }
      window.EventSource = MockEventSource as unknown as typeof EventSource;
    });
    const page = await context.newPage();

    try {
      const authResponse = page.waitForResponse(
        (res) => res.url().includes("/api/auth/me") && res.ok(),
      );
      await page.goto("/dashboard/inference", { waitUntil: "domcontentloaded" });
      await expect(page).toHaveURL(/\/dashboard\/inference/, { timeout: 15_000 });
      await authResponse;
      await expect(
        page.getByRole("link", { name: /Deploy Studio|dash\.serverless\.open_studio/i }).first(),
      ).toBeVisible({ timeout: 15_000 });
      await expect(page.getByText("smoke-llama")).toBeVisible({ timeout: 15_000 });

      await page.getByRole("link", { name: /Deploy Studio|dash\.serverless\.open_studio/i }).first().click();
      await expect(page).toHaveURL(/\/dashboard\/inference\/new/);

      const continueBtn = page.getByRole("button", { name: /Continue|Continuer|dash\.serverless\.continue/i });
      await continueBtn.click();
      await continueBtn.click();
      await page.locator("select").first().selectOption({ index: 1 });
      await continueBtn.click();
      await continueBtn.click();
      await continueBtn.click();
      await page.getByRole("button", { name: /Deploy endpoint|Déployer|dash\.serverless\.deploy/i }).click();

      await expect(page).toHaveURL(new RegExp(`/dashboard/inference/${ENDPOINT_ID}$`));
      await page.getByRole("button", { name: /API keys|Clés API|dash\.serverless\.tab_keys/i }).click();
      await page.getByRole("button", { name: /Create key|Créer une clé|dash\.serverless\.key_create/i }).click();
      await expect(page.getByText("xcel_secret_once")).toBeVisible();
    } finally {
      await context.close();
    }
  });
});