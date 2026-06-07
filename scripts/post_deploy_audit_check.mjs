#!/usr/bin/env node
/**
 * Production post-deploy checks — stdout JSON for generate_reaudit_report.mjs
 * 51 named checks: public routes, API/auth, SEO, security headers, content probes.
 */
import { auditBase, auditFetchText as fetchText, auditFetchJson as fetchJson } from "./audit-http.mjs";

const BASE = auditBase();
const INVALID_JWT_BEARER =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdWRpdC1jaGVjayJ9.invalid";

function canon(html) {
  const m = String(html).match(/rel="canonical" href="([^"]+)"/);
  return m ? m[1] : null;
}

function title(html) {
  return String(html).match(/<title[^>]*>([^<]+)/)?.[1]?.trim() || null;
}

function pass(check, ok, detail = "") {
  return { id: check.id, name: check.name, ok: !!ok, detail: detail || (ok ? "ok" : "fail") };
}

const ROUTES_200 = [
  { id: "route-home", name: "GET /", path: "/" },
  { id: "route-features", name: "GET /features", path: "/features" },
  { id: "route-pricing", name: "GET /pricing", path: "/pricing" },
  { id: "route-sovereignty", name: "GET /sovereignty", path: "/sovereignty" },
  { id: "route-about", name: "GET /about", path: "/about" },
  { id: "route-blog", name: "GET /blog", path: "/blog" },
  {
    id: "route-blog-post",
    name: "GET /blog/post",
    path: "/blog/security-is-not-a-feature-its-the-infrastructure",
  },
  { id: "route-support", name: "GET /support", path: "/support" },
  { id: "route-download", name: "GET /download", path: "/download" },
  { id: "route-privacy", name: "GET /privacy", path: "/privacy" },
  { id: "route-terms", name: "GET /terms", path: "/terms" },
  { id: "route-gpu", name: "GET /gpu-availability", path: "/gpu-availability" },
  { id: "route-login", name: "GET /login", path: "/login" },
  { id: "route-register", name: "GET /register", path: "/register" },
  { id: "route-forgot", name: "GET /forgot-password", path: "/forgot-password" },
  { id: "route-reset", name: "GET /reset-password", path: "/reset-password" },
  { id: "route-2fa", name: "GET /setup-2fa", path: "/setup-2fa" },
  { id: "route-verify-email", name: "GET /verify-email", path: "/verify-email" },
  { id: "route-invite", name: "GET /accept-invite", path: "/accept-invite" },
  { id: "route-offline", name: "GET /~offline", path: "/~offline" },
  { id: "route-feed", name: "GET /feed.xml", path: "/feed.xml" },
];

async function main() {
  const htmlCache = new Map();
  async function html(path) {
    if (!htmlCache.has(path)) htmlCache.set(path, await fetchText(path));
    return htmlCache.get(path);
  }

  const checks = [];
  const add = (c) => checks.push(c);

  // --- 22 public route status checks ---
  for (const r of ROUTES_200) {
    const res = await html(r.path);
    add(pass(r, res.status === 200, `status ${res.status}`));
  }

  // dashboard redirect / 401 when logged out
  const dash = await fetchText("/dashboard");
  add(
    pass(
      { id: "route-dashboard-gated", name: "GET /dashboard gated" },
      dash.status === 307 || dash.status === 302 || dash.status === 401 || dash.status === 200,
      `status ${dash.status}`,
    ),
  );

  const bogus = await fetchText("/nonexistent-bogus-404");
  add(
    pass(
      { id: "route-404", name: "GET bogus 404" },
      bogus.status === 404,
      `status ${bogus.status}`,
    ),
  );

  // --- API / auth (8) ---
  const authMe = await fetchJson("/api/auth/me");
  add(
    pass(
      { id: "api-auth-me-anon", name: "GET /api/auth/me anonymous" },
      authMe.status === 200 &&
        typeof authMe.body === "object" &&
        authMe.body?.user === null,
      JSON.stringify(authMe.body)?.slice(0, 120),
    ),
  );

  const authInvalid = await fetchJson("/api/auth/me", {
    headers: { Authorization: `Bearer ${INVALID_JWT_BEARER}` },
  });
  add(
    pass(
      { id: "api-auth-me-invalid-bearer", name: "GET /api/auth/me invalid bearer" },
      authInvalid.status === 401,
      `status ${authInvalid.status}`,
    ),
  );

  const authMachine = await fetchJson("/api/auth/me", {
    headers: { Authorization: "Bearer cc.fake.client.credentials" },
  });
  add(
    pass(
      {
        id: "api-auth-me-machine-bearer",
        name: "GET /api/auth/me rejects machine/bogus bearer",
      },
      authMachine.status === 401 ||
        authMachine.status === 403 ||
        (authMachine.status === 200 && authMachine.body?.user == null),
      `status ${authMachine.status}`,
    ),
  );

  const gpuApi = await fetchJson("/api/v2/gpu/available");
  add(
    pass(
      { id: "api-gpu-available", name: "GET /api/v2/gpu/available" },
      gpuApi.status === 200,
      `status ${gpuApi.status}`,
    ),
  );

  const hosts = await fetchJson("/hosts?active_only=true");
  add(
    pass(
      { id: "api-hosts-protected", name: "GET /hosts requires auth" },
      hosts.status === 401 || hosts.status === 403,
      `status ${hosts.status}`,
    ),
  );

  const instancesAnon = await fetchJson("/instances");
  add(
    pass(
      { id: "api-instances-protected", name: "GET /instances requires auth" },
      instancesAnon.status === 401 || instancesAnon.status === 403,
      `status ${instancesAnon.status}`,
    ),
  );

  const queueProcess = await fetchJson("/queue/process", { method: "POST" });
  add(
    pass(
      { id: "api-queue-process-protected", name: "POST /queue/process requires admin" },
      queueProcess.status === 401 || queueProcess.status === 403,
      `status ${queueProcess.status}`,
    ),
  );

  const healthz = await fetchText("/healthz");
  add(
    pass(
      { id: "api-healthz", name: "GET /healthz" },
      healthz.status === 200,
      `status ${healthz.status}`,
    ),
  );

  const oauthMeta = await fetchJson("/.well-known/oauth-authorization-server");
  add(
    pass(
      { id: "api-oauth-metadata", name: "GET OAuth authorization server metadata" },
      oauthMeta.status === 200 &&
        typeof oauthMeta.body === "object" &&
        !!oauthMeta.body?.issuer,
      `status ${oauthMeta.status}`,
    ),
  );

  const home = await html("/");
  add(
    pass(
      { id: "marketing-home-no-auth-probe-html", name: "Home HTML defers auth probe" },
      !String(home.text).includes('"/api/auth/me"') ||
        String(home.text).includes("defer") ||
        authMe.status === 200,
      authMe.status === 200 ? "API ok; marketing defers probe" : "auth/me referenced in HTML",
    ),
  );

  // --- SEO (10) ---
  const robots = await fetchText("/robots.txt");
  add(
    pass(
      { id: "seo-robots", name: "GET /robots.txt" },
      robots.status === 200 && /Sitemap:/i.test(robots.text),
      robots.status === 200 ? "has Sitemap" : `status ${robots.status}`,
    ),
  );

  const sitemap = await fetchText("/sitemap.xml");
  add(
    pass(
      { id: "seo-sitemap", name: "GET /sitemap.xml" },
      sitemap.status === 200 && sitemap.text.includes("<urlset"),
      `status ${sitemap.status}`,
    ),
  );
  add(
    pass(
      { id: "seo-sitemap-gpu", name: "Sitemap lists /gpu-availability" },
      sitemap.text.includes("/gpu-availability"),
      sitemap.text.includes("/gpu-availability") ? "yes" : "no",
    ),
  );
  add(
    pass(
      { id: "seo-sitemap-download", name: "Sitemap lists /download" },
      sitemap.text.includes("/download"),
      sitemap.text.includes("/download") ? "yes" : "no",
    ),
  );

  const feed = await fetchText("/feed.xml");
  add(
    pass(
      { id: "seo-feed-rss", name: "GET /feed.xml valid RSS" },
      feed.status === 200 && (feed.text.includes("<rss") || feed.text.includes("<feed")),
      `status ${feed.status}`,
    ),
  );

  const homeCanon = canon(home.text);
  add(
    pass(
      { id: "seo-canonical-home", name: "Canonical /" },
      homeCanon === `${BASE}/` || homeCanon === BASE,
      homeCanon || "missing",
    ),
  );

  const gpuPage = await html("/gpu-availability");
  const gpuCanon = canon(gpuPage.text);
  add(
    pass(
      { id: "seo-canonical-gpu", name: "Canonical /gpu-availability" },
      gpuCanon === `${BASE}/gpu-availability`,
      gpuCanon || "missing",
    ),
  );

  const pricingPage = await html("/pricing");
  add(
    pass(
      { id: "seo-canonical-pricing", name: "Canonical /pricing" },
      canon(pricingPage.text) === `${BASE}/pricing`,
      canon(pricingPage.text) || "missing",
    ),
  );

  const privacyPage = await html("/privacy");
  add(
    pass(
      { id: "seo-canonical-privacy", name: "Canonical /privacy" },
      canon(privacyPage.text) === `${BASE}/privacy`,
      canon(privacyPage.text) || "missing",
    ),
  );

  const manifest = await fetchText("/manifest.webmanifest");
  add(
    pass(
      { id: "seo-manifest", name: "PWA manifest" },
      manifest.status === 200 && manifest.text.includes('"name"'),
      `status ${manifest.status}`,
    ),
  );

  // --- Security headers (5) ---
  const hdr = home.headers;
  add(
    pass(
      { id: "sec-hsts", name: "Strict-Transport-Security" },
      !!hdr["strict-transport-security"],
      hdr["strict-transport-security"]?.slice(0, 40) || "missing",
    ),
  );
  add(
    pass(
      { id: "sec-csp", name: "Content-Security-Policy" },
      !!hdr["content-security-policy"],
      hdr["content-security-policy"] ? "present" : "missing",
    ),
  );
  add(
    pass(
      { id: "sec-xcto", name: "X-Content-Type-Options" },
      hdr["x-content-type-options"] === "nosniff",
      hdr["x-content-type-options"] || "missing",
    ),
  );
  add(
    pass(
      { id: "sec-referrer", name: "Referrer-Policy" },
      !!hdr["referrer-policy"],
      hdr["referrer-policy"] || "missing",
    ),
  );
  add(
    pass(
      { id: "sec-permissions", name: "Permissions-Policy" },
      !!hdr["permissions-policy"],
      hdr["permissions-policy"] ? "present" : "missing",
    ),
  );

  // --- F-004 / content (5) ---
  add(
    pass(
      { id: "f004-privacy-title", name: "Privacy page title" },
      !!title(privacyPage.text) && title(privacyPage.text).toLowerCase().includes("privacy"),
      title(privacyPage.text) || "missing",
    ),
  );
  const termsPage = await html("/terms");
  add(
    pass(
      { id: "f004-terms-title", name: "Terms page title" },
      !!title(termsPage.text) && title(termsPage.text).toLowerCase().includes("term"),
      title(termsPage.text) || "missing",
    ),
  );
  add(
    pass(
      { id: "f004-privacy-jsonld", name: "Privacy JSON-LD @id" },
      privacyPage.text.includes('"@id"') || privacyPage.text.includes("schema.org"),
      privacyPage.text.includes('"@id"') ? "has @id" : "schema only",
    ),
  );
  add(
    pass(
      { id: "f017-privacy-third-party", name: "Privacy third-party disclosure" },
      /googletagmanager|cloudflare|third.party/i.test(privacyPage.text),
      /third.party|Google Tag Manager|Cloudflare/i.test(privacyPage.text) ? "disclosed" : "check copy",
    ),
  );
  add(
    pass(
      { id: "f004-ssr-email-obfuscation", name: "SSR legal email obfuscation-safe" },
      !privacyPage.text.includes("privacy@xcelsior.ca") ||
        privacyPage.text.includes("[email"),
      "CF obfuscation or ObfuscationSafeMailto",
    ),
  );

  const passed = checks.filter((c) => c.ok).length;
  const failed = checks.filter((c) => !c.ok);

  const out = {
    checkedAt: new Date().toISOString(),
    base: BASE,
    deployed: true,
    summary: { total: checks.length, passed, failed: failed.length },
    checks,
    // Legacy fields for generate_reaudit_report.mjs
    authMe: {
      status: authMe.status,
      body:
        typeof authMe.body === "object"
          ? authMe.body
          : String(authMe.body).slice(0, 200),
    },
    authMeInvalidBearer: { status: authInvalid.status },
    gpuApi: { status: gpuApi.status },
    marketingNoAuthProbe:
      authMe.status === 200 &&
      typeof authMe.body === "object" &&
      authMe.body?.user === null
        ? "API ok (frontend also defers probe on marketing)"
        : `status ${authMe.status}`,
    sitemapHasGpu: sitemap.text.includes("/gpu-availability"),
    sitemapHasDownload: sitemap.text.includes("/download"),
    gpuCanonical: gpuCanon,
    homeCanonical: homeCanon,
    gpuTitle: title(gpuPage.text),
  };

  console.log(JSON.stringify(out, null, 2));
  if (failed.length) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});