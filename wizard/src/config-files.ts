/** Preserve the user's project configuration when installing credentials. */
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir } from "node:os";
import { randomUUID } from "node:crypto";

export function configDirectory(): string {
    const override = process.env.XCELSIOR_CONFIG_DIR?.trim();
    return override ? resolve(override) : join(homedir(), ".xcelsior");
}

/** Replace a private file atomically, including when an older file was public. */
export function writePrivateFile(file: string, body: string): void {
    mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
    const temp = `${file}.${randomUUID()}.tmp`;
    try {
        writeFileSync(temp, body, { mode: 0o600, flag: "wx" });
        chmodSync(temp, 0o600);
        renameSync(temp, file);
    } finally {
        if (existsSync(temp)) unlinkSync(temp);
    }
}

export function updateEnvFile(file: string, values: Record<string, string>): void {
    const remaining = new Map(Object.entries(values));
    const original = existsSync(file) ? readFileSync(file, "utf8") : "";
    const lines = original.split(/\r?\n/);
    const seen = new Set<string>();
    const updated = lines.flatMap((line) => {
        const key = line.match(/^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=/)?.[1];
        if (!key || !(key in values)) return [line];
        // Remove duplicate definitions of a credential instead of leaving an
        // older value later in the file to override the fresh one.
        if (seen.has(key)) return [];
        seen.add(key);
        remaining.delete(key);
        return [`${key}=${JSON.stringify(values[key])}`];
    });
    for (const [key, value] of remaining) updated.push(`${key}=${JSON.stringify(value)}`);
    const body = updated.join("\n").replace(/^\n/, "").replace(/\n*$/, "\n");
    writePrivateFile(file, body);
}
