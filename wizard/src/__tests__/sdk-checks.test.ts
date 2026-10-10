import { describe, expect, it, vi, afterEach } from "vitest";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { configDirectory } from "../config-files.js";
import { checkSdkProject, projectPackageManager, writeSdkEnvSnippet, buildSdkClientModule, packageManagerEnv } from "../sdk-checks.js";

function project(dependencies: Record<string, string> = {}) {
    const root = join(configDirectory(), "project");
    mkdirSync(root);
    writeFileSync(join(root, "package.json"), JSON.stringify({ dependencies }));
    vi.spyOn(process, "cwd").mockReturnValue(root);
    return root;
}
afterEach(() => vi.restoreAllMocks());

describe("SDK integration configuration", () => {
    it("rejects a Python-only project before offering TypeScript installation", () => {
        const root = configDirectory();
        writeFileSync(join(root, "pyproject.toml"), "[project]\nname='example'\n");
        vi.spyOn(process, "cwd").mockReturnValue(root);
        expect(checkSdkProject()[0]).toMatchObject({ ok: false, detail: expect.stringContaining("Python") });
    });

    it.each([["pnpm-lock.yaml", "pnpm"], ["yarn.lock", "yarn"], ["bun.lock", "bun"]])("detects %s", (lockfile, expected) => {
        const root = project();
        writeFileSync(join(root, lockfile), "");
        expect(projectPackageManager(root)).toBe(expected);
    });

    it("honors the declared package manager", () => {
        const root = project();
        writeFileSync(join(root, "package.json"), JSON.stringify({ packageManager: "pnpm@10.0.0" }));
        writeFileSync(join(root, "package-lock.json"), "{}");
        expect(projectPackageManager(root)).toBe("pnpm");
    });

    it("preserves a Next.js environment and stores renewing client credentials without the sign-in token", () => {
        const root = project({ next: "16" });
        writeFileSync(join(root, ".env.local"), "# keep\nDATABASE_URL=existing\n");
        const file = writeSdkEnvSnippet("https://staging.example.test", "xoa_temporary", "client", "secret");
        expect(file).toBe(join(root, ".env.local"));
        const body = readFileSync(file, "utf8");
        expect(body).toContain("DATABASE_URL=existing");
        expect(body).not.toContain("xoa_temporary");
        expect(body).toContain('XCELSIOR_OAUTH_CLIENT_SECRET="secret"');
        expect(statSync(file).mode & 0o777).toBe(0o600);
        const module = readFileSync(join(root, "xcelsior-client.mjs"), "utf8");
        expect(module).toContain('new URL("/oauth/token", baseUrl)');
        expect(module).not.toContain('import "dotenv/config"');
    });

    it("loads a plain Node environment through dotenv and supports a durable API key", () => {
        const root = project();
        const file = writeSdkEnvSnippet("https://example.test", "xcel_ai_test_key");
        expect(file).toBe(join(root, ".env"));
        expect(readFileSync(join(root, "xcelsior-client.mjs"), "utf8")).toContain('import "dotenv/config"');
        expect(readFileSync(file, "utf8")).toContain("xcel_ai_test_key");
    });

    it("refuses to replace existing application code", () => {
        const root = project();
        const file = join(root, "xcelsior-client.mjs");
        writeFileSync(file, "// my own integration\n");
        expect(() => writeSdkEnvSnippet("https://example.test", "xcel_ai_test")).toThrow("already contains your code");
        expect(readFileSync(file, "utf8")).toBe("// my own integration\n");
    });

    it("does not accept an expiring token as permanent application configuration", () => {
        project();
        expect(() => writeSdkEnvSnippet("https://example.test", "xoa_temporary")).toThrow("OAuth client credentials or a durable API key");
    });

    it("keeps caller-provided URLs literal in generated JavaScript", () => {
        const url = 'https://example.test/";throw new Error("bad");';
        expect(buildSdkClientModule(url, true, null)).toContain(JSON.stringify(url));
    });
});

describe("package manager environment", () => {
    it("drops npm configuration inherited from npx so the project's own config applies", () => {
        const env = packageManagerEnv({
            PATH: "/usr/bin", HOME: "/home/user", XCELSIOR_CONFIG_DIR: "/tmp/x",
            npm_config_allow_scripts: "node-pty", npm_config_local_prefix: "/npx/cache",
            npm_package_name: "@xcelsior-gpu/wizard", npm_lifecycle_event: "dev", npm_execpath: "/npm-cli.js",
        });
        expect(env).toEqual({ PATH: "/usr/bin", HOME: "/home/user", XCELSIOR_CONFIG_DIR: "/tmp/x" });
    });
});
