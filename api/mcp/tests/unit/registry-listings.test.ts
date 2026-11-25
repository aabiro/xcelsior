import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";

/**
 * Two official-registry listings describe one server.
 *
 * `server.json` is `ca.xcelsior/gpu-cloud`, proven by a DNS record on
 * xcelsior.ca, so the listing is not tied to the monorepo root's README.
 * It also lists the npm package, whose `mcpName` points at it. The original
 * `io.github.aabiro/xcelsior` (`server.github.json`) is kept, remote only,
 * because the dashboard's VS Code install link names it as its gallery
 * identity.
 *
 * They must not drift: same version as package.json (what the live server
 * reports), same remote, same title and description.
 */

const read = (file: string) =>
  JSON.parse(readFileSync(new URL(`../../${file}`, import.meta.url), "utf8")) as Record<string, any>;

const pkg = read("package.json");
const dns = read("server.json");
const github = read("server.github.json");

describe("registry listings", () => {
  it("name the DNS and GitHub namespaces", () => {
    expect(dns.name).toBe("ca.xcelsior/gpu-cloud");
    expect(github.name).toBe("io.github.aabiro/xcelsior");
  });

  it.each([
    ["server.json", dns],
    ["server.github.json", github],
  ])("%s lists the version the server reports", (_file, listing) => {
    expect(listing.version).toBe(pkg.version);
  });

  it("agree on everything but the name and the npm package", () => {
    const { name: _a, packages: _b, ...dnsRest } = dns;
    const { name: _c, packages: _d, ...githubRest } = github;
    expect(dnsRest).toEqual(githubRest);
  });

  it("list the npm package only where its mcpName points", () => {
    const cli = JSON.parse(
      readFileSync(new URL("../../../mcp-cli/package.json", import.meta.url), "utf8"),
    ) as { name: string; mcpName: string };
    for (const listing of [dns, github]) {
      const listsIt = (listing.packages ?? []).some(
        (p: { identifier?: string }) => p.identifier === cli.name,
      );
      // The registry refuses a package whose mcpName names another server.
      expect(listsIt).toBe(cli.mcpName === listing.name);
    }
  });
});
