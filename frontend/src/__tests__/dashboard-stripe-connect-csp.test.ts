import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { proxy } from "@/proxy";

function cspFor(pathname: string): string {
  const request = new NextRequest(`https://xcelsior.ca${pathname}`, {
    headers: { cookie: "xcelsior_session=token" },
  });
  return proxy(request).headers.get("content-security-policy") ?? "";
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
  it("permits connect-js.stripe.com in script, frame, and connect sources", () => {
    const csp = cspFor("/dashboard/earnings");

    expect(directive(csp, "script-src")).toContain("https://connect-js.stripe.com");
    expect(directive(csp, "frame-src")).toContain("https://connect-js.stripe.com");
    expect(directive(csp, "connect-src")).toContain("https://connect-js.stripe.com");
  });
});
