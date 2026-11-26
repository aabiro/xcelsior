import { render as rtlRender, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "@/lib/locale";
import { StatusContent } from "@/app/(marketing)/status/content";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/status",
}));

function render(ui: ReactElement) {
  return rtlRender(ui, { wrapper: LocaleProvider });
}

function serveStatus(body: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok, status: ok ? 200 : 503, json: async () => body })),
  );
}

const services = [{ name: "API", state: "operational", detail: "", required: true }];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Status page preview features", () => {
  it("lists each preview feature by name with its live state", async () => {
    serveStatus({
      ok: true,
      verdict: "operational",
      services,
      features: [
        { key: "serverless", audience: "customers", state: "limited" },
        { key: "wise_payouts", audience: "providers", state: "off" },
      ],
    });
    render(<StatusContent />);
    const serverless = await waitFor(() => {
      const item = document.querySelector('[data-feature="serverless"]');
      expect(item).toBeTruthy();
      return item as HTMLElement;
    });
    expect(serverless.textContent).toContain("Serverless inference");
    expect(serverless.textContent).toContain("available to selected accounts");
    expect(serverless.textContent).toContain("for customers");
    const wise = document.querySelector('[data-feature="wise_payouts"]') as HTMLElement;
    expect(wise.textContent).toContain("Payouts through Wise");
    expect(wise.textContent).toContain("not yet available");
    expect(wise.textContent).toContain("for GPU providers");
    // Never a raw key on the page: every key the API sends has copy.
    expect(document.body.textContent).not.toMatch(/status\.feature_/);
  });

  it("hides the section when the API does not send the list", async () => {
    serveStatus({ ok: true, verdict: "operational", services });
    render(<StatusContent />);
    await screen.findByText(/All systems operational/);
    expect(screen.queryByText("Features in preview")).toBeNull();
  });

  it("does not claim anything about features when the status check fails", async () => {
    serveStatus({}, false);
    render(<StatusContent />);
    await screen.findByText(/Status check unreachable/);
    expect(screen.queryByText("Features in preview")).toBeNull();
  });
});
