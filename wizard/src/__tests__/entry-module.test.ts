// Tests for the CLI entry guard: the wizard must render when launched through
// npx's node_modules/.bin symlink, not only when node is given the real file.

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, writeFileSync, symlinkSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { isEntryModule } from "../entry-module.js";

let dir: string;
let real: string;
let link: string;
let other: string;

beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "wizard-entry-"));
    real = join(dir, "index.js");
    link = join(dir, "xcelsior-wizard");
    other = join(dir, "other.js");
    writeFileSync(real, "");
    writeFileSync(other, "");
    symlinkSync(real, link);
});

afterAll(() => {
    rmSync(dir, { recursive: true, force: true });
});

describe("isEntryModule", () => {
    it("is the entry when node runs the file directly", () => {
        expect(isEntryModule(pathToFileURL(real).href, real)).toBe(true);
    });

    it("is the entry when npx runs it through a bin symlink", () => {
        expect(isEntryModule(pathToFileURL(real).href, link)).toBe(true);
    });

    it("is not the entry when another script was run, as when a test imports it", () => {
        expect(isEntryModule(pathToFileURL(real).href, other)).toBe(false);
    });

    it("is not the entry with no script path or a missing one", () => {
        expect(isEntryModule(pathToFileURL(real).href, undefined)).toBe(false);
        expect(isEntryModule(pathToFileURL(real).href, join(dir, "missing.js"))).toBe(false);
    });
});
