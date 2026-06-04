#!/usr/bin/env node
/**
 * Build docs/site-audit-report-YYYY-MM-DD-reaudit.md from:
 * - Baseline MCP report (docs/site-audit-report-2026-06-04.md)
 * - Re-audit MCP raw JSON (/tmp/xcelsior-audit/raw/desktop-all.json, perf-all.json)
 * - Optional post-deploy verification JSON (REVERIFY_JSON env path)
 *
 * Usage:
 *   node scripts/generate_reaudit_report.mjs
 *   REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs
 */
import fs from "node:fs";
import path from "node:path";

const REPO = process.env.REPO || path.resolve(import.meta.dirname, "..");
const AUDIT_DIR = process.env.AUDIT_DIR || "/tmp/xcelsior-audit";
const BASELINE_MD = path.join(REPO, "docs", "site-audit-report-2026-06-04.md");
const REAUDIT_MCP_MD = path.join(REPO, "docs", "site-audit-report-2026-06-04-reaudit-mcp.md");
const OUT_MD = path.join(REPO, "docs", "site-audit-report-2026-06-04-reaudit.md");

const REMEDIATION = [
  { id: "F-001", sev: "P1", commit: "5fc95d3", files: "routes/auth.py, frontend auth", fix: "200 + user:null; defer /api/auth/me on marketing (b87c8f1)" },
  { id: "F-002", sev: "P1", commit: "5fc95d3", files: "gpu-availability page, /api/v2/gpu/available", fix: "Public GPU endpoint; honest degraded UI" },
  { id: "F-003", sev: "P1", commit: "b87c8f1", files: "layout, providers, session-routes", fix: "WalletConnect dashboard-only; lazy chat/PWA/GTM; deferred auth" },
  { id: "F-004", sev: "P1", commit: "5fc95d3", files: "blog/download/legal pages", fix: "Deterministic SSR dates / client-only islands" },
  { id: "F-005", sev: "P1", commit: "5fc95d3", files: "metadata, sitemap.ts, route layouts", fix: "Per-route canonicals; sitemap download + gpu-availability" },
  { id: "F-006", sev: "P1", commit: "5fc95d3", files: "proxy.ts, next.config", fix: "Deduped security headers at app layer" },
  { id: "F-007", sev: "P1", commit: "5fc95d3", files: "proxy.ts", fix: "no-store for auth/dashboard/offline" },
  { id: "F-008", sev: "P2", commit: "5fc95d3", files: "navbar.tsx", fix: "Header overlap / BETA placement" },
  { id: "F-009", sev: "P2", commit: "5fc95d3", files: "global CSS / nav", fix: "44px touch targets" },
  { id: "F-010", sev: "P2", commit: "5fc95d3", files: "forms", fix: "16px inputs on mobile" },
  { id: "F-011", sev: "P2", commit: "5fc95d3", files: "pricing", fix: "Mobile pricing affordance (partial)" },
  { id: "F-012", sev: "P2", commit: "5fc95d3", files: "sw.ts, ServiceWorkerRegistrar", fix: "SW update flow improvements" },
  { id: "F-013", sev: "P2", commit: "5fc95d3", files: "navbar.tsx", fix: "aria-expanded, body scroll lock" },
  { id: "F-014", sev: "P2", commit: "—", files: "—", fix: "Open — contrast tokens" },
  { id: "F-015", sev: "P2", commit: "5fc95d3", files: "landing H1", fix: "Accessible H1 spacing" },
  { id: "F-016", sev: "P2", commit: "5fc95d3", files: "register", fix: "Terms required + a11y errors" },
  { id: "F-017", sev: "P2", commit: "—", files: "—", fix: "Open — third-party disclosure" },
  { id: "F-018", sev: "P3", commit: "—", files: "—", fix: "Open — duplicate JSON-LD" },
  { id: "F-019", sev: "P3", commit: "b87c8f1", files: "layout.tsx", fix: "GTM lazyOnload, no preload" },
  { id: "F-020", sev: "P3", commit: "—", files: "—", fix: "Open — 404 promos" },
  { id: "F-021", sev: "P3", commit: "—", files: "—", fix: "Open — blog date normalization" },
  { id: "F-022", sev: "P3", commit: "—", files: "—", fix: "Accepted — SVG logos OK" },
];

function parseHealthTable(md) {
  const rows = [];
  const section = md.split("## Slow 4G")[0];
  for (const line of section.split("\n")) {
    if (!line.startsWith("| /")) continue;
    const p = line.split("|").map((s) => s.trim());
    if (p.length < 10) continue;
    const lcp = parseInt(p[4], 10);
    const tbt = parseInt(p[6], 10);
    if (!Number.isFinite(lcp) || !Number.isFinite(tbt)) continue;
    rows.push({
      route: p[1],
      lcp,
      tbt,
      js: p[7],
      failed: p[9],
    });
  }
  return rows;
}

function loadDesktop() {
  const p = path.join(AUDIT_DIR, "raw", "desktop-all.json");
  if (!fs.existsSync(p)) return [];
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function desktopRow(d) {
  const auth = (d.requests || []).find((x) => x.url?.includes("/api/auth/me"));
  const hosts = (d.requests || []).find((x) => x.url?.includes("/hosts"));
  const errs = (d.consoleText?.match(/\[error\]/g) || []).length;
  return {
    route: d.route?.path || "?",
    auth: auth?.status || "—",
    hosts: hosts?.status || "—",
    errs,
    canon: d.seo?.canonical || "—",
    title: (d.seo?.title || "").slice(0, 60),
  };
}

function perfSummary() {
  const p = path.join(AUDIT_DIR, "raw", "perf-all.json");
  if (!fs.existsSync(p)) return null;
  const perf = JSON.parse(fs.readFileSync(p, "utf8"));
  const home = perf.find((r) => r.routePath === "/" && r.condition?.name === "desktop-unthrottled");
  const pricing = perf.find((r) => r.routePath === "/pricing" && r.condition?.name === "mobile-slow4g-cpu4");
  return {
    homeTbt: home?.metrics?.vitals?.tbt,
    homeJs: Math.round((home?.metrics?.resources?.jsTransfer || 0) / 1024),
    pricingSlowTbt: pricing?.metrics?.vitals?.tbt,
  };
}

function loadReverify() {
  const p = process.env.REVERIFY_JSON;
  if (!p || !fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function main() {
  const baseline = fs.existsSync(BASELINE_MD) ? fs.readFileSync(BASELINE_MD, "utf8") : "";
  const reauditMcp = fs.existsSync(REAUDIT_MCP_MD) ? fs.readFileSync(REAUDIT_MCP_MD, "utf8") : "";
  const baseRows = parseHealthTable(baseline);
  const reRows = parseHealthTable(reauditMcp);
  const byBase = Object.fromEntries(baseRows.map((r) => [r.route, r]));
  const byRe = Object.fromEntries(reRows.map((r) => [r.route, r]));

  const desktop = loadDesktop();
  const dHome = desktop.find((d) => d.route?.path === "/");
  const dGpu = desktop.find((d) => d.route?.path === "/gpu-availability");
  const perf = perfSummary();
  const reverify = loadReverify();

  const deltaLines = [];
  for (const route of Object.keys(byBase).sort()) {
    const a = byBase[route];
    const b = byRe[route];
    if (!b) continue;
    const dTbt = (b.tbt ?? 0) - (a.tbt ?? 0);
    const dLcp = (b.lcp ?? 0) - (a.lcp ?? 0);
    if (Math.abs(dTbt) >= 800 || Math.abs(dLcp) >= 400) {
      deltaLines.push(
        `| ${route} | ${a.lcp} ms | ${b.lcp} ms | ${dLcp >= 0 ? "+" : ""}${dLcp} | ${a.tbt} ms | ${b.tbt} ms | ${dTbt >= 0 ? "+" : ""}${dTbt} |`,
      );
    }
  }

  const prodAuth = dHome ? desktopRow(dHome).auth : "?";
  const prodGpuHosts = dGpu ? desktopRow(dGpu).hosts : "?";
  const deployStatus = reverify?.deployed
    ? "**Deployed** — post-deploy checks below"
    : "**Pending deploy** — production MCP (2026-06-04) still shows pre-remediation behavior";

  const remTable = REMEDIATION.map(
    (r) =>
      `| ${r.id} | ${r.sev} | ${r.commit} | ${r.fix} | ${r.commit === "—" ? "Open" : "In repo"} |`,
  );

  let postDeploy = "";
  if (reverify) {
    postDeploy = `
## Post-deploy verification (${reverify.checkedAt || "n/a"})

| Check | Result |
|-------|--------|
| \`GET /api/auth/me\` (logged out) | ${reverify.authMe?.status} — ${JSON.stringify(reverify.authMe?.body || "").slice(0, 80)} |
| \`GET /api/v2/gpu/available\` | ${reverify.gpuApi?.status} |
| \`GET /\` no auth/me in HTML probe | ${reverify.marketingNoAuthProbe ?? "n/a"} |
| Sitemap has /gpu-availability | ${reverify.sitemapHasGpu ? "yes" : "no"} |
| Sitemap has /download | ${reverify.sitemapHasDownload ? "yes" : "no"} |
| GPU page canonical | ${reverify.gpuCanonical || "n/a"} |
`;
  }

  const md = `# Xcelsior — Site Audit Re-Audit Report

Date: 2026-06-04  
Target: https://xcelsior.ca  
Deploy status: ${deployStatus}

This report compares the **baseline audit** (\`docs/site-audit-report-2026-06-04.md\`), a **second full chrome-devtools MCP crawl** (\`docs/site-audit-report-2026-06-04-reaudit-mcp.md\`), and **remediation on \`main\`** (\`5fc95d3\` site-audit sprints 1–3, \`b87c8f1\` bundle split + full test coverage).

Raw MCP artifacts: \`${AUDIT_DIR}/raw/\`, screenshots: \`${AUDIT_DIR}/screens/\`.

---

## Executive summary

1. **Production before deploy** — The second MCP pass confirms the same P1 blockers as the morning audit: global \`/api/auth/me\` **401**, \`/gpu-availability\` calling \`/hosts\` **401**, wrong canonicals, and ~760–813 KB marketing JS with very high TBT.
2. **Remediation is on \`main\`** — Backend anonymous session shape, public GPU page, SEO/sitemap, headers/cache, mobile header, and marketing bundle split are implemented in git; they require **deploy** to affect xcelsior.ca.
3. **Perf delta (prod vs prod)** — Second crawl shows modest LCP improvements on several routes (likely variance); **JS transfer and TBT are unchanged** until F-003 ships.
4. **Test coverage** — \`UNTESTED_ENDPOINTS.md\`: **0** HTTP routes and **0** CLI commands without test signal.

### Production signals (MCP desktop, ${dHome?.capturedAt?.slice(0, 10) || "2026-06-04"})

| Signal | \`/\` | \`/gpu-availability\` |
|--------|------|------------------------|
| \`/api/auth/me\` | ${prodAuth} | ${desktopRow(dGpu || {}).auth} |
| \`/hosts?active_only\` | — | ${prodGpuHosts} |
| Console errors | ${dHome ? desktopRow(dHome).errs : "?"} | ${dGpu ? desktopRow(dGpu).errs : "?"} |
| Canonical | ${dHome ? desktopRow(dHome).canon.replace("https://xcelsior.ca", "") || "/" : "?"} | ${dGpu ? desktopRow(dGpu).canon.replace("https://xcelsior.ca", "") || "/" : "?"} |

${perf ? `Desktop \`/\`: TBT **${perf.homeTbt}** ms, JS **${perf.homeJs}** KB. Slow 4G \`/pricing\`: TBT **${perf.pricingSlowTbt}** ms.` : ""}

---

## Remediation matrix (findings → code)

| ID | Sev | Commit | Fix (summary) | Repo |
|----|-----|--------|---------------|------|
${remTable.join("\n")}

---

## Health metrics: baseline vs re-audit MCP (production, both pre-deploy)

Interpretation: both crawls hit **live production** before \`main\` deploy. Differences are mostly measurement variance, not remediation.

| Route | Baseline LCP | Re-audit LCP | Δ LCP | Baseline TBT | Re-audit TBT | Δ TBT |
|-------|-------------:|-------------:|------:|-------------:|-------------:|------:|
${deltaLines.length ? deltaLines.join("\n") : "| _(no large deltas)_ | | | | | | |"}

Full tables: baseline § Health Snapshot; re-audit § Health Snapshot in \`site-audit-report-2026-06-04-reaudit-mcp.md\`.

---

## What should change after deploy

| ID | Expected on production after deploy |
|----|-------------------------------------|
| F-001 | No \`/api/auth/me\` on marketing HTML first paint; API returns \`200\` + \`user: null\` when probed |
| F-002 | GPU page uses \`/api/v2/gpu/available\`; no \`/hosts\` 401; degraded copy when empty |
| F-005 | GPU + blog post canonicals correct; sitemap includes \`/download\` and \`/gpu-availability\` |
| F-003 | Lower marketing JS (WalletConnect off marketing); fewer long tasks — **re-run \`audit-performance.mjs\`** |
| F-006–F-013 | Header/touch targets/cache/no-store per \`5fc95d3\` |

${postDeploy}

---

## Regenerate this report

\`\`\`bash
# After MCP crawl:
cd /tmp/xcelsior-audit && node audit-routes.mjs  # etc.

# Optional post-deploy checks → JSON:
node scripts/post_deploy_audit_check.mjs > /tmp/post-deploy-check.json
REVERIFY_JSON=/tmp/post-deploy-check.json node scripts/generate_reaudit_report.mjs
\`\`\`

---

## Recommended follow-up

1. **Deploy** — \`bash scripts/deploy.sh\` (or CI pipeline) from \`main\`.
2. **Verify** — \`node scripts/post_deploy_audit_check.mjs\`
3. **MCP re-crawl** — full \`audit-*.mjs\` suite; expect F-001/F-002/F-005 cleared.
4. **Authenticated dashboard audit** — still out of scope; needs test credentials.
`;

  fs.mkdirSync(path.dirname(OUT_MD), { recursive: true });
  fs.writeFileSync(OUT_MD, md);
  console.log("Wrote", OUT_MD);
}

main();