import { render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "@/lib/locale";
import { HostRate, formatHourlyRate } from "@/components/hosts/host-rate";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/dashboard",
}));

function render(ui: ReactElement) {
  return rtlRender(ui, { wrapper: LocaleProvider });
}

describe("HostRate", () => {
  it("shows what renters pay and what the host keeps", () => {
    render(<HostRate host={{ renter_rate_per_hour: 0.12, host_earns_per_hour: 0.096, platform_cut_pct: 20 }} />);
    expect(screen.getByText("$0.12/hr")).toBeTruthy();
    expect(screen.getByText("you earn $0.096")).toBeTruthy();
    expect(screen.getByTestId("host-rate").getAttribute("title")).toContain("20% platform cut");
  });

  it("says when a GPU model has no listed rate instead of inventing one", () => {
    render(<HostRate host={{ renter_rate_per_hour: null, host_earns_per_hour: null, platform_cut_pct: 20 }} />);
    expect(screen.getByText("Not priced yet")).toBeTruthy();
    expect(screen.queryByText(/\/hr/)).toBeNull();
  });

  it("keeps the third decimal under a dollar", () => {
    expect(formatHourlyRate(0.12)).toBe("$0.12");
    expect(formatHourlyRate(0.096)).toBe("$0.096");
    expect(formatHourlyRate(2.5)).toBe("$2.50");
  });
});
