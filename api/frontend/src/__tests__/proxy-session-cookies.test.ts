import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { proxy } from "@/proxy";

// The access cookie expires after 15 minutes; the refresh cookie lasts 30
// days. A dashboard page load holding only the refresh cookie must reach the
// page, which refreshes on its first API call, not bounce to /login.

async function load(pathname: string, cookie?: string) {
  const request = new NextRequest(`https://xcelsior.ca${pathname}`, {
    headers: cookie ? { cookie } : {},
  });
  return proxy(request);
}

describe("proxy session check", () => {
  it("lets a visitor with only a refresh cookie reach the dashboard", async () => {
    const response = await load("/dashboard/instances", "xcelsior_refresh=rt");
    expect(response.headers.get("location")).toBeNull();
  });

  it("lets a visitor with an access cookie reach the dashboard", async () => {
    const response = await load("/dashboard", "xcelsior_session=at");
    expect(response.headers.get("location")).toBeNull();
  });

  it("sends a visitor with neither cookie to login, keeping where they were going", async () => {
    const response = await load("/dashboard/billing?tab=history");
    const location = new URL(response.headers.get("location") ?? "", "https://xcelsior.ca");
    expect(location.pathname).toBe("/login");
    expect(location.searchParams.get("redirect")).toBe("/dashboard/billing?tab=history");
  });
});
