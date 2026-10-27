import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { proxy } from "@/proxy";

async function cspFor(pathname: string): Promise<string> {
  const request = new NextRequest(`https://xcelsior.ca${pathname}`, {
    headers: { cookie: "xcelsior_session=token" },
  });
  const response = await proxy(request);
  return response.headers.get("content-security-policy") ?? "";
}

function directive(csp: string, name: string): string {
  return (
    csp
      .split(";")
      .map((part) => part.trim())
      .find((part) => part === name || part.startsWith(`${name} `)) ?? ""
  );
}

describe("dashboard CSP allows Stripe Connect embedded components", () => {
  it("permits connect-js.stripe.com in script, frame, and connect sources", async () => {
    const csp = await cspFor("/dashboard/earnings");

    expect(directive(csp, "script-src")).toContain("https://connect-js.stripe.com");
    expect(directive(csp, "frame-src")).toContain("https://connect-js.stripe.com");
    expect(directive(csp, "connect-src")).toContain("https://connect-js.stripe.com");
  });
});
