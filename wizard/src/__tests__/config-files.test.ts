import { describe, expect, it } from "vitest";
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { configDirectory, updateEnvFile, writePrivateFile } from "../config-files.js";

describe("private configuration writes", () => {
    it("preserves unrelated configuration and replaces duplicate credential definitions", () => {
        const file = join(configDirectory(), ".env.local");
        writeFileSync(file, '# keep this\nDATABASE_URL="untouched"\nexport XCELSIOR_API_TOKEN=old\nXCELSIOR_API_TOKEN=duplicate\n', { mode: 0o644 });
        updateEnvFile(file, { XCELSIOR_API_TOKEN: "new-token", XCELSIOR_API_URL: "https://example.test" });
        const body = readFileSync(file, "utf8");
        expect(body).toContain('# keep this\nDATABASE_URL="untouched"');
        expect(body.match(/XCELSIOR_API_TOKEN=/g)).toHaveLength(1);
        expect(body).toContain('XCELSIOR_API_TOKEN="new-token"');
        expect(statSync(file).mode & 0o777).toBe(0o600);
    });

    it("removes a temporary file if replacement fails", () => {
        const file = join(configDirectory(), "token.json");
        mkdirSync(file);
        expect(() => writePrivateFile(file, "secret")).toThrow();
        expect(readdirSync(configDirectory())).toEqual(["token.json"]);
    });
});
