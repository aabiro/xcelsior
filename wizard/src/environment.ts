// environment.ts — Fast, synchronous, best-effort local environment detection
// for the startup card. No subprocesses (so it never adds startup latency);
// the GPU/Docker probes happen later in the provider flow.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

export interface EnvInfo {
    /** Detected project framework/language, or null. */
    framework: string | null;
    platform: string;
    arch: string;
    node: string;
}

/** Walk up from cwd to find a project root marker. */
function findRoot(start: string): string {
    let dir = start;
    for (let i = 0; i < 10; i++) {
        if (existsSync(join(dir, ".git")) || existsSync(join(dir, "package.json"))) return dir;
        const parent = dirname(dir);
        if (parent === dir) break;
        dir = parent;
    }
    return start;
}

function detectFramework(root: string): string | null {
    try {
        const pkgPath = join(root, "package.json");
        if (existsSync(pkgPath)) {
            const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
            const deps = { ...pkg.dependencies, ...pkg.devDependencies } as Record<string, string>;
            if (deps["next"]) return "Next.js";
            if (deps["@sveltejs/kit"] || deps["svelte"]) return "SvelteKit";
            if (deps["vue"]) return "Vue";
            if (deps["react"]) return "React";
            return "Node.js";
        }
        if (existsSync(join(root, "pyproject.toml")) || existsSync(join(root, "requirements.txt"))) return "Python";
        if (existsSync(join(root, "Cargo.toml"))) return "Rust";
        if (existsSync(join(root, "go.mod"))) return "Go";
    } catch {
        // best-effort
    }
    return null;
}

export function detectEnvironment(cwd: string = process.cwd()): EnvInfo {
    let framework: string | null = null;
    try {
        framework = detectFramework(findRoot(cwd));
    } catch {
        framework = null;
    }
    return {
        framework,
        platform: process.platform,
        arch: process.arch,
        node: process.version,
    };
}

/** Human one-liner of the detected environment. */
export function describeEnvironment(env: EnvInfo): string {
    const parts = [env.framework ?? "project", `${env.platform}/${env.arch}`, `node ${env.node}`];
    return parts.join(" · ");
}
