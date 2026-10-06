import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { SpotPriceRow } from "@/lib/api";

// framer-motion's in-view animation needs an IntersectionObserver, which jsdom
// does not have. The animation is not what this file checks.
vi.stubGlobal(
  "IntersectionObserver",
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  },
);
import { topSpotRates } from "@/app/(dashboard)/dashboard/analytics/spot-rates";
import { PlatformPulseOverview } from "@/app/(dashboard)/dashboard/analytics/analytics-empty-states";

function row(gpu_model: string, rate_cad: number): SpotPriceRow {
  return {
    gpu_model,
    rate_cad,
    spot_cents: Math.round(rate_cad * 100),
    on_demand_cad: rate_cad * 1.5,
    savings_pct: 33,
    supply: 2,
    demand: 1,
    provider_floor_cents: 10,
    recorded_at: 1_780_000_000,
  };
}

// What `/spot-prices` sends: the list of rows and the `{model: price}` map side
// by side, including the row with an empty model that production carries.
const response = {
  ok: true,
  prices: { "RTX 4090": 0.62, "RTX 3060": 0.18, "": 0.05 },
  spot_prices: [row("RTX 3060", 0.18), row("", 0.05), row("RTX 4090", 0.62)],
};

describe("analytics spot rates", () => {
  it("names each bar after its GPU and prices it in CAD", () => {
    expect(topSpotRates(response)).toEqual([
      { model: "RTX 4090", price: 0.62 },
      { model: "RTX 3060", price: 0.18 },
    ]);
  });

  it("drops rows the chart cannot label or price", () => {
    const res = { spot_prices: [row("  ", 1), { ...row("A100", 1), rate_cad: Number.NaN }, row("H100", 2)] };
    expect(topSpotRates(res)).toEqual([{ model: "H100", price: 2 }]);
  });

  it("survives a response without rows", () => {
    expect(topSpotRates(undefined)).toEqual([]);
    expect(topSpotRates({})).toEqual([]);
  });

  it("renders as rates, not as spend with a usage line", () => {
    render(
      <PlatformPulseOverview
        spotPrices={topSpotRates(response)}
        marketplaceStats={null}
        walletBalance={null}
        leaderboardCount={0}
        gpuModelsAvailable={2}
      />,
    );
    expect(screen.getByText("Spot rates")).toBeInTheDocument();
    expect(screen.getByText("RTX 4090")).toBeInTheDocument();
    expect(screen.getByText(/\$0\.62\/hr/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/NaN/);
    expect(document.body.textContent).not.toMatch(/Top GPU Models|By total spend|\d+ jobs|h GPU/);
  });
});
