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

  it("permits Stripe Elements, 3DS, Link, Radar hCaptcha, and Meta CAPIG endpoints", async () => {
    const csp = await cspFor("/dashboard/billing");

    for (const host of [
      "https://*.js.stripe.com",
      "https://hcaptcha.com",
      "https://*.hcaptcha.com",
    ]) {
      expect(directive(csp, "script-src")).toContain(host);
    }

    for (const host of [
      "https://*.js.stripe.com",
      "https://hooks.stripe.com",
      "https://m.stripe.network",
      "https://link.com",
      "https://*.link.com",
      "https://hcaptcha.com",
      "https://*.hcaptcha.com",
    ]) {
      expect(directive(csp, "frame-src")).toContain(host);
    }

    for (const host of [
      "https://hooks.stripe.com",
      "https://m.stripe.network",
      "https://r.stripe.com",
      "https://q.stripe.com",
      "https://merchant-ui-api.stripe.com",
      "https://*.link.com",
      "https://*.hcaptcha.com",
      "https://*.facebook.net",
      "https://*.facebook.com",
      "https://md-eecad2978f7a43f5b7838c919258e6de.ecs.us-east-2.on.aws",
      "https://cvw425zxroht.cloud.run",
    ]) {
      expect(directive(csp, "connect-src")).toContain(host);
    }
  });
});
