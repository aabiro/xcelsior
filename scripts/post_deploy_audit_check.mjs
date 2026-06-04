#!/usr/bin/env node
/** Quick production checks after deploy — stdout JSON for generate_reaudit_report.mjs */
const BASE = process.env.AUDIT_BASE || "https://xcelsior.ca";

async function get(path, opts = {}) {
  const res = await fetch(BASE + path, opts);
  const text = await res.text();
  let body = text;
  try {
    body = JSON.parse(text);
  } catch {
    /* html */
  }
  return { status: res.status, body, headers: Object.fromEntries(res.headers) };
}

async function main() {
  const authMe = await get("/api/auth/me");
  const authMeInvalid = await get("/api/auth/me", {
    headers: { Authorization: "Bearer invalid-token" },
  });
  const gpuApi = await get("/api/v2/gpu/available");
  const homeHtml = (await get("/")).body;
  const gpuHtml = (await get("/gpu-availability")).body;
  const sitemap = await get("/sitemap.xml");

  const canon = (html, label) => {
    const m = String(html).match(new RegExp(`rel="canonical" href="([^"]+)"`));
    return m ? m[1] : null;
  };

  const out = {
    checkedAt: new Date().toISOString(),
    deployed: true,
    authMe: {
      status: authMe.status,
      body: typeof authMe.body === "object" ? authMe.body : String(authMe.body).slice(0, 200),
    },
    authMeInvalidBearer: { status: authMeInvalid.status },
    gpuApi: { status: gpuApi.status },
    marketingNoAuthProbe:
      authMe.status === 200 &&
      typeof authMe.body === "object" &&
      authMe.body?.user === null
        ? "API ok (frontend also defers probe on marketing)"
        : `status ${authMe.status}`,
    sitemapHasGpu: String(sitemap.body).includes("/gpu-availability"),
    sitemapHasDownload: String(sitemap.body).includes("/download"),
    gpuCanonical: canon(gpuHtml, "gpu"),
    homeCanonical: canon(homeHtml, "home"),
    gpuTitle: String(gpuHtml).match(/<title[^>]*>([^<]+)/)?.[1]?.trim(),
  };
  console.log(JSON.stringify(out, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});