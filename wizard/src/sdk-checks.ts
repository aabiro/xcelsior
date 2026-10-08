// sdk-checks.ts — Project detection and SDK integration checks (SDK wizard track).

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { updateEnvFile } from "./config-files.js";
import { validateApiToken } from "./wizard-guards.js";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { detectEnvironment } from "./environment.js";
import { getMe } from "./api-client.js";

const exec = promisify(execFile);

const SDK_PACKAGE = "@xcelsior-gpu/sdk";

function findProjectRoot(start: string): string {
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

    if (!hasPkg && !hasPy) {
        return [{
            name: "Project",
            ok: false,
            detail: "No package.json or pyproject.toml found in this directory tree",
            remediation: "cd into your app root (where package.json lives) and re-run the wizard",
        }];
    }

    const framework = env.framework ?? (hasPy ? "Python" : "Node.js");
    return [
        { name: "Project root", ok: true, detail: root },
        { name: "Framework", ok: true, detail: `${framework} · ${env.platform}/${env.arch} · node ${env.node}` },
    ];
}

export async function checkSdkPackage(): Promise<{ name: string; ok: boolean; detail: string; remediation?: string }[]> {
    const root = findProjectRoot(process.cwd());
    const pkgPath = join(root, "package.json");

    if (!existsSync(pkgPath)) {
        return [{
            name: SDK_PACKAGE,
            ok: false,
            detail: "Node.js project required for the TypeScript SDK",
            remediation: "Use a Node/TypeScript project, or call the REST API directly with your xoa_ token",
        }];
    }

    let deps: Record<string, string> = {};
    try {
        const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
        deps = { ...pkg.dependencies, ...pkg.devDependencies };
    } catch {
        return [{ name: "package.json", ok: false, detail: "Could not parse package.json" }];
    }

    if (deps[SDK_PACKAGE]) {
        return [{ name: SDK_PACKAGE, ok: true, detail: `Found in package.json (${deps[SDK_PACKAGE]})` }];
    }

    try {
        const { stdout } = await exec("npm", ["ls", SDK_PACKAGE, "--depth=0"], { cwd: root, timeout: 15_000 });
        if (stdout.includes(SDK_PACKAGE)) {
            return [{ name: SDK_PACKAGE, ok: true, detail: "Installed in node_modules" }];
        }
    } catch {
        // not installed
    }

    return [{
        name: SDK_PACKAGE,
        ok: false,
        detail: "Not installed yet",
        remediation: `Run: npm install ${SDK_PACKAGE}`,
    }];
}

export function writeSdkEnvSnippet(baseUrl: string, token: string, clientId?: string, clientSecret?: string): string {
    const root = findProjectRoot(process.cwd());
    const envPath = join(root, ".env.local");
    const values: Record<string, string> = {
        XCELSIOR_API_URL: baseUrl,
        XCELSIOR_API_TOKEN: token,
    };
    if (clientId && clientSecret) {
        values.XCELSIOR_OAUTH_CLIENT_ID = clientId;
        values.XCELSIOR_OAUTH_CLIENT_SECRET = clientSecret;
    }
    // Propagate failures so the wizard can offer Retry instead of claiming
    // credentials were saved to an unwritable project.
    updateEnvFile(envPath, values);
    return envPath;
}

export async function checkSdkApi(baseUrl: string, token: string): Promise<{ name: string; ok: boolean; detail: string }[]> {
    if (validateApiToken(token || "")) {
        return [{ name: "API token", ok: false, detail: "Missing API key or sign-in token — complete authentication first" }];
    }
    try {
        const profile = await getMe(baseUrl, token);
        return [
            { name: "API connection", ok: true, detail: `Authenticated as ${profile.email}` },
            { name: "Customer", ok: true, detail: profile.customer_id || profile.user_id },
        ];
    } catch (err) {
        return [{ name: "API connection", ok: false, detail: err instanceof Error ? err.message : "Request failed" }];
    }
}

export function buildSdkStarterSnippet(framework: string | null, baseUrl: string): string {
    const envVar = "process.env.XCELSIOR_API_TOKEN";
    if (framework === "Next.js") {
        return `import { XcelsiorApiClient } from "${SDK_PACKAGE}";\n\nconst client = new XcelsiorApiClient({\n  headers: { Authorization: \`Bearer \${${envVar}}\` },\n  environment: ${JSON.stringify(baseUrl)},\n});\n\nconst instances = await client.instances.list();`;
    }
    return `import { XcelsiorApiClient } from "${SDK_PACKAGE}";\n\nconst client = new XcelsiorApiClient({\n  headers: { Authorization: \`Bearer \${${envVar}}\` },\n  environment: ${JSON.stringify(baseUrl)},\n});\n\nconst me = await client.auth.me();`;
}