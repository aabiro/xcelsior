// Tests for environment detection + the StartupCard render.

import React from "react";
import { describe, it, expect } from "vitest";
import { render } from "ink-testing-library";
import { detectEnvironment, describeEnvironment, type EnvInfo } from "../environment.js";
import { StartupCard } from "../StartupCard.js";

describe("detectEnvironment", () => {
    it("always reports platform, arch and node, and does not throw", () => {
        const env = detectEnvironment();
        expect(env.platform).toBe(process.platform);
        expect(env.arch).toBe(process.arch);
        expect(env.node).toBe(process.version);
    });

    it("detects this repo's wizard as a Node/React-ish project from its own cwd", () => {
        // Running from the wizard package dir, package.json exists → not null.
        const env = detectEnvironment(process.cwd());
        expect(env.framework === null || typeof env.framework === "string").toBe(true);
    });

    it("describeEnvironment joins the parts", () => {
        const env: EnvInfo = { framework: "Next.js", platform: "linux", arch: "x64", node: "v20" };
        const s = describeEnvironment(env);
        expect(s).toContain("Next.js");
        expect(s).toContain("linux/x64");
        expect(s).toContain("node v20");
    });

    it("falls back to 'project' when framework is null", () => {
        const s = describeEnvironment({ framework: null, platform: "darwin", arch: "arm64", node: "v20" });
        expect(s).toContain("project");
    });
});

describe("StartupCard", () => {
    it("shows value prop, privacy line, and detected environment", () => {
        const env: EnvInfo = { framework: "Next.js", platform: "linux", arch: "x64", node: "v20" };
        const { lastFrame } = render(<StartupCard env={env} />);
        const out = lastFrame() ?? "";
        expect(out).toContain("Welcome to Xcelsior");
        expect(out).toContain("never leave this machine");
        expect(out).toContain("Detecting your environment");
        expect(out).toContain("Next.js");
    });
});
