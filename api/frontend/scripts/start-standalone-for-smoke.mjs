import { cpSync, existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";

const root = process.cwd();
const standaloneRoot = path.join(root, ".next", "standalone");
const standaloneStaticRoot = path.join(standaloneRoot, ".next", "static");
const standalonePublicRoot = path.join(standaloneRoot, "public");

function copyRequiredAssets() {
  const staticSource = path.join(root, ".next", "static");
  const publicSource = path.join(root, "public");

  if (!existsSync(path.join(standaloneRoot, "server.js"))) {
    throw new Error("Missing .next/standalone/server.js. Run `npm run build` before the PWA smoke test.");
  }

  cpSync(staticSource, standaloneStaticRoot, { recursive: true, force: true });
  cpSync(publicSource, standalonePublicRoot, { recursive: true, force: true });
}

copyRequiredAssets();

const child = spawn("node", ["server.js"], {
  cwd: standaloneRoot,
  stdio: "inherit",
  env: {
    ...process.env,
    HOSTNAME: process.env.HOSTNAME ?? "127.0.0.1",
    PORT: process.env.PORT ?? "3100",
  },
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
