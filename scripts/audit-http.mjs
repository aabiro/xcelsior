/**
 * HTTP fetch for production audits.
 * When AUDIT_ORIGIN_IP is set (e.g. 100.64.0.1 via Tailscale), connect to that IP
 * while preserving the public Host/SNI (xcelsior.ca) so origin nginx + TLS work.
 */
import http from "node:http";
import https from "node:https";

const TIMEOUT_MS = Number(process.env.AUDIT_FETCH_TIMEOUT_MS || 20000);

export function auditBase() {
  return (process.env.AUDIT_BASE || "https://xcelsior.ca").replace(/\/$/, "");
}

function originIp() {
  return process.env.AUDIT_ORIGIN_IP || "";
}

function request(path, opts = {}) {
  const base = auditBase();
  const url = new URL(path.startsWith("http") ? path : `${base}${path}`);
  const isHttps = url.protocol === "https:";
  const mod = isHttps ? https : http;
  const headers = { Host: url.hostname, ...(opts.headers || {}) };
  const requestOpts = {
    hostname: originIp() || url.hostname,
    port: url.port || (isHttps ? 443 : 80),
    path: url.pathname + url.search,
    method: opts.method || "GET",
    headers,
    servername: isHttps ? url.hostname : undefined,
    rejectUnauthorized: true,
  };

  return new Promise((resolve, reject) => {
    const req = mod.request(requestOpts, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        resolve({
          status: res.statusCode || 0,
          text,
          headers: res.headers,
        });
      });
    });
    req.on("error", reject);
    req.setTimeout(TIMEOUT_MS, () => req.destroy(new Error("timeout")));
    if (opts.body) req.write(opts.body);
    req.end();
  });
}

export async function auditFetch(path, opts = {}) {
  return request(path, opts);
}

export async function auditFetchText(path, opts = {}) {
  const res = await request(path, opts);
  return { status: res.status, text: res.text, headers: res.headers };
}

export async function auditFetchJson(path, opts = {}) {
  const res = await request(path, opts);
  let body = res.text;
  try {
    body = JSON.parse(res.text);
  } catch {
    /* html */
  }
  return { status: res.status, body, headers: res.headers };
}