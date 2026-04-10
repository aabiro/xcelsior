import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

const root = process.cwd();

function fail(message) {
  throw new Error(message);
}

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(root, relativePath), "utf8"));
}

const serviceWorkerPath = path.join(root, "public", "sw.js");
if (!existsSync(serviceWorkerPath)) {
  fail("Expected public/sw.js to exist after the production build.");
}

const serviceWorkerSource = readFileSync(serviceWorkerPath, "utf8");
if (!serviceWorkerSource.includes("/~offline")) {
  fail("Expected the generated service worker to reference the /~offline fallback.");
}
if (!serviceWorkerSource.includes("showNotification")) {
  fail("Expected the generated service worker to include desktop notification handling.");
}

const appPathsManifest = readJson(".next/server/app-paths-manifest.json");
if (!appPathsManifest["/manifest.webmanifest/route"]) {
  fail("Expected /manifest.webmanifest to be emitted by the App Router manifest route.");
}
if (!appPathsManifest["/~offline/page"]) {
  fail("Expected /~offline to be emitted as a built route.");
}

const prerenderManifest = readJson(".next/prerender-manifest.json");
if (!prerenderManifest.routes["/~offline"]) {
  fail("Expected /~offline to be prerendered for offline fallback.");
}

const routesManifest = readJson(".next/routes-manifest.json");
const swHeaders = routesManifest.headers.find((header) => header.source === "/sw.js");
if (!swHeaders) {
  fail("Expected explicit headers for /sw.js.");
}
if (!swHeaders.headers.some((header) => header.key === "Cache-Control" && header.value.includes("no-cache"))) {
  fail("Expected /sw.js to disable HTTP caching.");
}

const manifestHeaders = routesManifest.headers.find((header) => header.source === "/manifest.webmanifest");
if (!manifestHeaders) {
  fail("Expected explicit headers for /manifest.webmanifest.");
}
if (!manifestHeaders.headers.some((header) => header.key === "Cache-Control")) {
  fail("Expected /manifest.webmanifest to define Cache-Control headers.");
}

console.log("PWA build verification passed.");
