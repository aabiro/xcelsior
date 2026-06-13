// agent-context.ts — Detect installed coding agents and (on confirm) drop a
// real Xcelsior CLI context file (Part G — the "Detected: Claude Code…/Keep the
// skills?" equivalent). We don't ship an MCP, so instead of installing one we
// teach whatever agent the user runs how to drive the Xcelsior CLI.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export interface DetectedAgent {
    id: string;
    name: string;
}

interface AgentProbe {
    id: string;
    name: string;
    /** Paths (relative to home) whose existence implies the agent is installed. */
    paths: string[];
}

const AGENT_PROBES: AgentProbe[] = [
    { id: "claude", name: "Claude Code", paths: [".claude", ".claude.json"] },
    { id: "cursor", name: "Cursor", paths: [".cursor"] },
    { id: "codex", name: "Codex", paths: [".codex"] },
    { id: "vscode", name: "VS Code", paths: [".vscode", ".config/Code", "Library/Application Support/Code"] },
    { id: "zed", name: "Zed", paths: [".config/zed", "Library/Application Support/Zed"] },
    { id: "windsurf", name: "Windsurf", paths: [".codeium/windsurf", ".windsurf"] },
];

/** Detect installed coding agents by probing well-known config paths. */
export function detectAgents(home: string = homedir()): DetectedAgent[] {
    return AGENT_PROBES
        .filter((a) => a.paths.some((p) => {
            try { return existsSync(join(home, p)); } catch { return false; }
        }))
        .map((a) => ({ id: a.id, name: a.name }));
}

export const XCELSIOR_CONTEXT_BEGIN = "<!-- BEGIN XCELSIOR CLI CONTEXT -->";
export const XCELSIOR_CONTEXT_END = "<!-- END XCELSIOR CLI CONTEXT -->";

/** The Xcelsior CLI context section (pure markdown). */
export function buildXcelsiorContext(baseUrl = "https://xcelsior.ca"): string {
    return [
        XCELSIOR_CONTEXT_BEGIN,
        "## Xcelsior GPU marketplace — CLI",
        "",
        "Xcelsior lets you rent GPUs from an open marketplace or earn by providing yours.",
        "",
        "Common commands:",
        "- `xcelsior setup` — interactive wizard (auth, GPU detect, benchmark, verify, register, launch).",
        "- `xcelsior marketplace` — browse available GPUs.",
        "- `xcelsior host-add` — register this machine as a provider.",
        "- `xcelsior worker install` — run the provider worker agent as a background service.",
        "- `xcelsior ai` — ask Hexara (the assistant) about your account, jobs, or hardware.",
        "",
        "Concepts: XCU score (normalized compute), spot/interruptible pricing, 7-point hardware",
        "verification, provider reputation tiers.",
        "",
        `Dashboard & docs: ${baseUrl}/dashboard`,
        XCELSIOR_CONTEXT_END,
        "",
    ].join("\n");
}

export interface WriteResult {
    path: string;
    action: "created" | "updated" | "unchanged";
}

/**
 * Write (or idempotently update) the Xcelsior context section in AGENTS.md at
 * `dir`. Non-destructive: an existing AGENTS.md keeps its other content; the
 * Xcelsior section is replaced between its markers. Returns the path + action.
 */
export function writeXcelsiorContext(dir: string, baseUrl = "https://xcelsior.ca"): WriteResult {
    const path = join(dir, "AGENTS.md");
    const section = buildXcelsiorContext(baseUrl);

    if (!existsSync(path)) {
        writeFileSync(path, `# Agent context\n\n${section}`, { mode: 0o644 });
        return { path, action: "created" };
    }

    const existing = readFileSync(path, "utf-8");
    const beginIdx = existing.indexOf(XCELSIOR_CONTEXT_BEGIN);
    const endIdx = existing.indexOf(XCELSIOR_CONTEXT_END);
    if (beginIdx >= 0 && endIdx > beginIdx) {
        const before = existing.slice(0, beginIdx);
        const after = existing.slice(endIdx + XCELSIOR_CONTEXT_END.length);
        const next = `${before}${section.trimEnd()}${after}`;
        if (next === existing) return { path, action: "unchanged" };
        writeFileSync(path, next, { mode: 0o644 });
        return { path, action: "updated" };
    }

    writeFileSync(path, `${existing.trimEnd()}\n\n${section}`, { mode: 0o644 });
    return { path, action: "updated" };
}

/** Short human summary of detected agents, e.g. "Claude Code, Cursor". */
export function describeAgents(agents: DetectedAgent[]): string {
    return agents.map((a) => a.name).join(", ");
}
