/** Preserve the user's project configuration when installing credentials. */
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

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
    mkdirSync(dirname(file), { recursive: true, mode: 0o700 });
    const temp = `${file}.${randomUUID()}.tmp`;
    try {
        writeFileSync(temp, body, { mode: 0o600 });
        chmodSync(temp, 0o600);
        renameSync(temp, file);
    } finally {
        if (existsSync(temp)) unlinkSync(temp);
    }
}
