import { readFileSync } from "node:fs";

/**
 * The version of the package this code ships in, read from its package.json so
 * no other file holds a copy.
 *
 * That is `@xcelsior-gpu/mcp-server` for the hosted server, and
 * `@xcelsior-gpu/mcp` when the stdio CLI copies `dist/` into its own package
 * (it carries `../assets` along for the icon, the same relative read).
 */
export const PACKAGE_VERSION: string = (
  JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string }
).version;
