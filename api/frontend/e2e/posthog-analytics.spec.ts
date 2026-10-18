import { expect, test, type Request } from "@playwright/test";
import { gunzipSync } from "node:zlib";
import LZString from "lz-string";

const LIVE_INGESTION = process.env.XCELSIOR_POSTHOG_SMOKE_LIVE === "1";
const LIFECYCLE_EVENTS = ["$pageview", "$pageleave"] as const;

function decodePostHogPayload(url: URL, body: Buffer | null): string {
  if (!body) return "";
  const compression = url.searchParams.get("compression");
  if (compression === "gzip" || compression === "gzip-js") {
    return gunzipSync(body).toString("utf8");
  }

  const text = body.toString("utf8");
  const encodedData = new URLSearchParams(text).get("data");
  if (compression === "lz64") {
    // URLSearchParams follows form semantics and turns '+' into spaces; put
    // them back before decoding the Base64 transport.
    return (
      LZString.decompressFromBase64((encodedData ?? text).replace(/ /g, "+")) ??
      ""
    );
  }
  return encodedData ?? text;
}

test("production bundle sends page lifecycle events through /ingest", async ({
  browser,
}) => {
  const context = await browser.newContext({
    serviceWorkers: "block",
    userAgent:
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
      "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
  });
  const captured = new Set<string>();
  const acknowledged = new Set<string>();
  const requestEvents = new Map<Request, Set<string>>();
  let ingestRequestCount = 0;
  let ingestPostCount = 0;
  const postDiagnostics: Array<{ path: string; bytes: number }> = [];

  // PostHog intentionally suppresses automated browsers. This smoke opts back
  // in while retaining the normal browser SDK and production bundle.
  await context.addInitScript(() => {
    Object.defineProperty(Navigator.prototype, "webdriver", {
      configurable: true,
      get: () => false,
    });
  });

  context.on("response", (response) => {
    if (!response.ok()) return;
    for (const event of requestEvents.get(response.request()) ?? []) {
      acknowledged.add(event);
    }
  });

  await context.route("**/ingest/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const pathname = url.pathname;
    const body = request.postDataBuffer();
    const payload = decodePostHogPayload(url, body);
    ingestRequestCount += 1;
    if (request.method() === "POST") {
      ingestPostCount += 1;
      postDiagnostics.push({
        path: pathname.replace(/phc_[^/]+/g, "<project-token>"),
        bytes: body?.length ?? 0,
      });
    }

    const eventsInRequest = new Set<string>();
    for (const event of LIFECYCLE_EVENTS) {
      if (payload.includes(event)) {
        captured.add(event);
        eventsInRequest.add(event);
      }
    }
    if (eventsInRequest.size) requestEvents.set(request, eventsInRequest);

    // Empty supportedCompression keeps payloads inspectable without changing
    // the application's PostHog configuration.
    if (pathname.endsWith("/config.js")) {
      return route.fulfill({
        status: 200,
        contentType: "application/javascript",
        body: "",
      });
    }
    if (pathname.endsWith("/config")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ supportedCompression: [] }),
      });
    }
    if (pathname.includes("/flags/")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ flags: {} }),
      });
    }

    if (
      LIVE_INGESTION &&
      LIFECYCLE_EVENTS.some((event) => payload.includes(event))
    ) {
      return route.continue();
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "{}",
    });
  });

  const page = await context.newPage();
  try {
    await page.goto("/", { waitUntil: "networkidle" });
    await page.waitForTimeout(4_000);
    expect(
      captured.has("$pageview"),
      `observed ${ingestRequestCount} /ingest requests and ` +
        `${ingestPostCount} POSTs: ${JSON.stringify(postDiagnostics)}`,
    ).toBe(true);
    await expect.poll(() => acknowledged.has("$pageview")).toBe(true);

    // Exercise the same lifecycle signal browsers emit when leaving a page.
    // Dispatching it explicitly avoids BFCache/navigation timing races in CI.
    await page.evaluate(() => {
      window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: false }));
    });
    await page.waitForTimeout(4_000);
    expect(
      captured.has("$pageleave"),
      `observed ${ingestRequestCount} /ingest requests and ` +
        `${ingestPostCount} POSTs: ${JSON.stringify(postDiagnostics)}`,
    ).toBe(true);
  } finally {
    await context.close();
  }
});
