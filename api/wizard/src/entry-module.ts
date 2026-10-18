import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

/**
 * True when `moduleUrl` is the script node was asked to run, so the CLI
 * renders when launched but not when a test imports it.
 *
 * npx and global installs start the bin through a symlink
 * (node_modules/.bin/xcelsior-wizard), so argv[1] names the link while
 * import.meta.url names the real file. Comparing them as-is never matched:
 * `npx @xcelsior-gpu/wizard` exited 0 having drawn nothing, and install.sh
 * took that exit as a finished wizard. Compare real paths instead.
 */
export function isEntryModule(moduleUrl: string, entry: string | undefined = process.argv[1]): boolean {
  if (!entry) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(moduleUrl));
  } catch {
    return false;
  }
}
