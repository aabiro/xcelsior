import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

/**
 * `package.json` carries an `overrides` block that forces transitive packages
 * to specific versions. Every entry was added to escape a security advisory.
 *
 * A pin like that has a shelf life nobody is reminded of. Eight of the nine
 * entries had fallen behind, and four were holding packages at versions with
 * *new* advisories — so a pin written to fix a vulnerability had become the
 * reason one persisted. `npm audit` reported 8 findings, 3 high.
 *
 * It went unnoticed because the workflow that runs `npm audit` could not start
 * at all (duplicate `if:` keys, fixed elsewhere in this branch), so the gate
 * had been dark for weeks.
 *
 * These checks are cheap and offline. They do not replace `npm audit` — which
 * needs the network and runs in CI — they catch the specific way this block
 * goes wrong: an entry that drifts out of the shape it was written in, or one
 * that outlives the dependency it was pinning.
 */

const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(resolve(here, "../../package.json"), "utf8"));
const lock = JSON.parse(readFileSync(resolve(here, "../../package-lock.json"), "utf8"));

describe("dependency overrides", () => {
  const overrides: Record<string, string> = pkg.overrides ?? {};

  it("exists — the block is load-bearing, not decorative", () => {
    expect(Object.keys(overrides).length).toBeGreaterThan(0);
  });

  it("pins exact versions, never ranges", () => {
    // A range defeats the purpose: `^4.13.1` re-admits anything the resolver
    // prefers, which is what the pin was added to prevent.
    for (const [name, version] of Object.entries(overrides)) {
      expect(version, `${name} is pinned to a range`).toMatch(/^\d+\.\d+\.\d+$/);
    }
  });

  it("pins only packages the tree actually installs", () => {
    // An override for a package no longer in the tree is dead weight that
    // reads as protection. It also hides the fact that whatever pulled it in
    // is gone.
    const installed = new Set<string>();
    for (const path of Object.keys(lock.packages ?? {})) {
      const marker = "node_modules/";
      const at = path.lastIndexOf(marker);
      if (at !== -1) installed.add(path.slice(at + marker.length));
    }
    const orphaned = Object.keys(overrides).filter((name) => !installed.has(name));
    expect(orphaned, `overrides for packages not in the lockfile: ${orphaned.join(", ")}`).toEqual([]);
  });

  it("is actually applied in the lockfile", () => {
    // The check that would have caught the stale pins being *ignored* rather
    // than merely old: an override present in package.json but not reflected
    // in the resolved tree is doing nothing at all.
    const resolved = new Map<string, Set<string>>();
    for (const [path, entry] of Object.entries<Record<string, unknown>>(lock.packages ?? {})) {
      const marker = "node_modules/";
      const at = path.lastIndexOf(marker);
      if (at === -1) continue;
      const name = path.slice(at + marker.length);
      const version = entry.version as string | undefined;
      if (!version) continue;
      if (!resolved.has(name)) resolved.set(name, new Set());
      resolved.get(name)!.add(version);
    }

    const violations: string[] = [];
    for (const [name, pinned] of Object.entries(overrides)) {
      const versions = resolved.get(name);
      if (!versions) continue; // covered by the orphan check above
      for (const v of versions) {
        if (v !== pinned) violations.push(`${name}: pinned ${pinned}, lockfile has ${v}`);
      }
    }
    expect(violations, violations.join("; ")).toEqual([]);
  });
});
