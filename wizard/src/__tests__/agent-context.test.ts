// Tests for agent-context.ts — agent detection + idempotent context writing.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
    detectAgents, buildXcelsiorContext, writeXcelsiorContext, describeAgents,
    XCELSIOR_CONTEXT_BEGIN, XCELSIOR_CONTEXT_END,
} from "../agent-context.js";

let dir: string;
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), "xcelsior-agent-")); });
afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

describe("detectAgents", () => {
    it("detects nothing in an empty home", () => {
        expect(detectAgents(dir)).toEqual([]);
    });

    it("detects Claude Code from ~/.claude", () => {
        mkdirSync(join(dir, ".claude"));
        const agents = detectAgents(dir);
        expect(agents.map((a) => a.id)).toContain("claude");
    });

    it("detects multiple agents", () => {
        mkdirSync(join(dir, ".claude"));
        mkdirSync(join(dir, ".cursor"));
        const ids = detectAgents(dir).map((a) => a.id);
        expect(ids).toContain("claude");
        expect(ids).toContain("cursor");
    });

    it("describeAgents joins names", () => {
        expect(describeAgents([{ id: "claude", name: "Claude Code" }, { id: "zed", name: "Zed" }]))
            .toBe("Claude Code, Zed");
    });
});

describe("buildXcelsiorContext", () => {
    it("includes the key CLI commands and markers", () => {
        const md = buildXcelsiorContext("https://x");
        expect(md).toContain(XCELSIOR_CONTEXT_BEGIN);
        expect(md).toContain(XCELSIOR_CONTEXT_END);
        expect(md).toContain("xcelsior setup");
        expect(md).toContain("xcelsior marketplace");
        expect(md).toContain("https://x/dashboard");
    });
});

describe("writeXcelsiorContext", () => {
    it("creates AGENTS.md when none exists", () => {
        const r = writeXcelsiorContext(dir);
        expect(r.action).toBe("created");
        expect(existsSync(r.path)).toBe(true);
        expect(readFileSync(r.path, "utf-8")).toContain(XCELSIOR_CONTEXT_BEGIN);
    });

    it("is idempotent — second write is unchanged", () => {
        writeXcelsiorContext(dir);
        const r2 = writeXcelsiorContext(dir);
        expect(r2.action).toBe("unchanged");
    });

    it("preserves existing AGENTS.md content and appends the section", () => {
        const path = join(dir, "AGENTS.md");
        writeFileSync(path, "# My project\n\nSome existing guidance.\n");
        const r = writeXcelsiorContext(dir);
        expect(r.action).toBe("updated");
        const content = readFileSync(path, "utf-8");
        expect(content).toContain("Some existing guidance.");
        expect(content).toContain(XCELSIOR_CONTEXT_BEGIN);
    });

    it("replaces only the Xcelsior section on update, keeping surrounding text", () => {
        const path = join(dir, "AGENTS.md");
        writeFileSync(path, `# Mine\n\nBefore.\n\n${XCELSIOR_CONTEXT_BEGIN}\nOLD\n${XCELSIOR_CONTEXT_END}\n\nAfter.\n`);
        writeXcelsiorContext(dir, "https://x");
        const content = readFileSync(path, "utf-8");
        expect(content).toContain("Before.");
        expect(content).toContain("After.");
        expect(content).not.toContain("OLD");
        expect(content).toContain("xcelsior setup");
    });
});
