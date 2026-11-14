import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { proxy } from "@/proxy";

async function connectSrc(pathname: string): Promise<string> {
  const response = await proxy(new NextRequest(`https://xcelsior.ca${pathname}`));
  const csp = response.headers.get("content-security-policy") ?? "";
  return (
    csp
      .split(";")
      .map((part) => part.trim())
      .find((part) => part.startsWith("connect-src ")) ?? ""
  );
}

describe("site CSP lets Google Analytics send what it is allowed to send", () => {
  // Every public page logged "Fetch API cannot load https://www.google.com/g/collect":
  // with consent mode, GA4 pings www.google.com even before anyone opts in, and
  // collects through regional *.google-analytics.com / *.analytics.google.com hosts.
  it.each(["/", "/pricing", "/mcp"])("allows GA4's collection hosts on %s", async (path) => {
    const policy = await connectSrc(path);
    for (const host of [
      "https://www.google.com",
      "https://*.google-analytics.com",
      "https://*.analytics.google.com",
      "https://*.googletagmanager.com",
    ]) {
      expect(policy).toContain(host);
    }
  });
});
