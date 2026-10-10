// sdk-checks.ts — Project detection and SDK integration checks (SDK wizard track).

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { updateEnvFile, writePrivateFile } from "./config-files.js";
import { validateApiToken } from "./wizard-guards.js";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { detectEnvironment } from "./environment.js";
import type { CheckResult } from "./checks.js";

const exec = promisify(execFile);

const SDK_PACKAGE = "@xcelsior-gpu/sdk";

export function findProjectRoot(start: string): string {
    let dir = start;
    for (let i = 0; i < 10; i++) {
        if (existsSync(join(dir, "package.json")) || existsSync(join(dir, "pyproject.toml"))) return dir;
        const parent = dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    return start;
}

export function checkSdkProject(): { name: string; ok: boolean; detail: string; remediation?: string }[] {
    const env = detectEnvironment();
    const root = findProjectRoot(process.cwd());
    const hasPkg = existsSync(join(root, "package.json"));
    const hasPy = existsSync(join(root, "pyproject.toml"));

    if (!hasPkg) {
        return [{
            name: "Project",
            ok: false,
            detail: hasPy ? "This track installs the TypeScript SDK; Python projects should use the REST API" : "No package.json found in this directory tree",
            remediation: "cd into your app root (where package.json lives) and re-run the wizard",
        }];
    }

    const framework = env.framework ?? "Node.js";
    return [
        { name: "Project root", ok: true, detail: root },
        { name: "Framework", ok: true, detail: `${framework} · ${env.platform}/${env.arch} · node ${env.node}` },
    ];
}

export function projectPackageManager(root: string): string {
    const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    const declared = typeof pkg.packageManager === "string" ? pkg.packageManager.split("@")[0] : undefined;
    if (declared && !["npm", "pnpm", "yarn", "bun"].includes(declared)) {
        throw new Error(`Unsupported package manager: ${declared}`);
    }
    if (declared) return declared;
    if (existsSync(join(root, "pnpm-lock.yaml"))) return "pnpm";
    if (existsSync(join(root, "yarn.lock"))) return "yarn";
    if (existsSync(join(root, "bun.lock")) || existsSync(join(root, "bun.lockb"))) return "bun";
    return "npm";
}

/**
 * The environment for a package manager run in the user's project.
 *
 * Launched through npx, the wizard inherits npm's resolved configuration as
 * npm_config_* variables. A child install treats those as command-line flags:
 * npm 12 rejects an inherited allow-scripts with EALLOWSCRIPTS, and an
 * inherited local_prefix points the install at npx's directory instead of the
 * project. The child must resolve configuration as if the user ran it there.
 */
export function packageManagerEnv(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
    return Object.fromEntries(Object.entries(env).filter(([name]) =>
        !/^npm_(config|package|lifecycle|execpath|node_execpath|command)(_|$)/i.test(name)));
}

export async function checkSdkPackage(): Promise<CheckResult[]> {
    const root = findProjectRoot(process.cwd());
    if (!existsSync(join(root, "package.json"))) {
        return [{ name: SDK_PACKAGE, ok: false, detail: "A Node.js project with package.json is required" }];
    }
    const needsDotenv = detectEnvironment(root).framework !== "Next.js";
    const verify = async () => {
        await exec(process.execPath, ["--input-type=module", "--eval",
            `const sdk = await import(${JSON.stringify(SDK_PACKAGE)}); if (typeof sdk.XcelsiorApiClient !== 'function') throw new Error('SDK client export missing');` +
            (needsDotenv ? " await import('dotenv/config');" : ""),
        ], { cwd: root, timeout: 15_000 });
    };
    try {
        await verify();
        return [{ name: SDK_PACKAGE, ok: true, detail: "Installed package loads successfully" }];
    } catch { /* Install missing or unusable dependencies, then verify again. */ }
    try {
        const manager = projectPackageManager(root);
        await exec(manager, [manager === "npm" ? "install" : "add", SDK_PACKAGE, ...(needsDotenv ? ["dotenv"] : [])], {
            cwd: root, timeout: 120_000, maxBuffer: 2 * 1024 * 1024, env: packageManagerEnv(),
        });
        await verify();
        return [{ name: SDK_PACKAGE, ok: true, detail: `Installed with ${manager}; client import verified` }];
    } catch (error) {
        return [{ name: SDK_PACKAGE, ok: false,
            detail: error instanceof Error ? error.message : "SDK installation failed",
            remediation: "Resolve the package-manager error above, then retry this step",
        }];
    }
}

export const SDK_CLIENT_FILE = "xcelsior-client.mjs";

/** Actual server-side integration. Short-lived OAuth tokens renew on demand. */
export function buildSdkClientModule(baseUrl: string, useOAuth: boolean, framework: string | null): string {
    const header = "// Generated by Xcelsior. Server-side only: never import into browser code.\n";
    const imports = (framework === "Next.js" ? "" : 'import "dotenv/config";\n') +
        `import { XcelsiorApiClient } from "${SDK_PACKAGE}";\n`;
    const configuration = `const baseUrl = process.env.XCELSIOR_API_URL || ${JSON.stringify(baseUrl)};\n`;
    const auth = useOAuth ? `let cached;
let pending;
async function authorization() {
  if (cached && Date.now() < cached.until) return cached.value;
  if (!pending) pending = (async () => {
    const response = await fetch(new URL("/oauth/token", baseUrl), {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "client_credentials",
        client_id: process.env.XCELSIOR_OAUTH_CLIENT_ID || "",
        client_secret: process.env.XCELSIOR_OAUTH_CLIENT_SECRET || "",
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new Error("Xcelsior OAuth failed: HTTP " + response.status);
    const result = await response.json();
    if (typeof result.access_token !== "string" || !result.access_token || !(result.expires_in > 0)) {
      throw new Error("Xcelsior returned an invalid OAuth token response");
    }
    const lifetime = Number(result.expires_in) * 1000;
    cached = { value: "Bearer " + result.access_token, until: Date.now() + lifetime - Math.min(30000, lifetime / 10) };
    return cached.value;
  })();
  try { return await pending; } finally { pending = undefined; }
}
` : `function authorization() {
  const key = process.env.XCELSIOR_API_TOKEN;
  if (!key || !key.startsWith("xcel_ai_")) throw new Error("Set XCELSIOR_API_TOKEN to your durable API key");
  return "Bearer " + key;
}
`;
    return header + imports + configuration + auth + `
export const client = new XcelsiorApiClient({
  environment: baseUrl,
  headers: { Authorization: authorization },
});
`;
}

export function writeSdkEnvSnippet(baseUrl: string, token: string, clientId?: string, clientSecret?: string): string {
    const root = findProjectRoot(process.cwd());
    const framework = detectEnvironment(root).framework;
    const envPath = join(root, framework === "Next.js" ? ".env.local" : ".env");
    const useOAuth = !!(clientId && clientSecret);
    if (!useOAuth && !token.startsWith("xcel_ai_")) throw new Error("SDK setup requires OAuth client credentials or a durable API key");
    const values: Record<string, string> = { XCELSIOR_API_URL: baseUrl };
    if (useOAuth) {
        values.XCELSIOR_OAUTH_CLIENT_ID = clientId!;
        values.XCELSIOR_OAUTH_CLIENT_SECRET = clientSecret!;
    } else {
        values.XCELSIOR_API_TOKEN = token;
    }
    const modulePath = join(root, SDK_CLIENT_FILE);
    if (existsSync(modulePath) && !readFileSync(modulePath, "utf8").startsWith("// Generated by Xcelsior.")) {
        throw new Error(`${modulePath} already contains your code; move it before retrying`);
    }
    // Never persist a browser sign-in token as a permanent application credential.
    updateEnvFile(envPath, values);
    writePrivateFile(modulePath, buildSdkClientModule(baseUrl, useOAuth, framework));
    return envPath;
}

export async function checkSdkApi(baseUrl: string, token: string, clientId?: string, clientSecret?: string): Promise<CheckResult[]> {
    if ((!clientId || !clientSecret) && validateApiToken(token || "")) {
        return [{ name: "SDK connection", ok: false, detail: "Complete authentication and credential setup first" }];
    }
    try {
        const root = findProjectRoot(process.cwd());
        await exec(process.execPath, ["--input-type=module", "--eval",
            `const { client } = await import('./${SDK_CLIENT_FILE}'); await client.instances.list();`,
        ], { cwd: root, timeout: 30_000, env: {
            ...process.env, XCELSIOR_API_URL: baseUrl, XCELSIOR_API_TOKEN: token,
            XCELSIOR_OAUTH_CLIENT_ID: clientId || "", XCELSIOR_OAUTH_CLIENT_SECRET: clientSecret || "",
        } });
        return [{ name: "SDK connection", ok: true, detail: "The installed SDK authenticated and listed instances" }];
    } catch (err) {
        return [{ name: "SDK connection", ok: false, detail: err instanceof Error ? err.message : "SDK request failed" }];
    }
}

export function buildSdkStarterSnippet(framework: string | null, _baseUrl: string): string {
    const note = framework === "Next.js"
        ? "// Use from a server route or server component. Next.js loads .env.local."
        : "// Run from your project root. The helper loads .env with dotenv.";
    return `${note}\nimport { client } from "./${SDK_CLIENT_FILE}";\n\nconst instances = await client.instances.list();`;
}
