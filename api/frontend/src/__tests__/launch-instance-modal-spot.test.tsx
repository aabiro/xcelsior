import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMocks = vi.hoisted(() => ({
  launchInstance: vi.fn(),
  fetchAvailableGPUs: vi.fn(),
  fetchPricingReference: vi.fn(),
  fetchImageTemplates: vi.fn(),
  fetchSpotPrices: vi.fn(),
  listAvailableVolumes: vi.fn(),
  fetchPricingRates: vi.fn(),
  fetchServiceTierCatalog: vi.fn(),
  fetchSpotFeatureStatus: vi.fn(),
  classifyLaunchError: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), warning: vi.fn() },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({
    user: {
      user_id: "u1",
      customer_id: "c1",
      billing_customer_id: "c1",
      team_id: "team1",
      team_name: "Test team",
      team_can_write_instances: true,
    },
  }),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
}));

vi.mock("@/components/InstallBanner", () => ({
  markInstanceLaunched: vi.fn(),
}));

import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";

describe("LaunchInstanceModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.fetchAvailableGPUs.mockResolvedValue({
      gpus: [{ gpu_model: "RTX 4090", vram_gb: 24, count_available: 2, price_per_hour_cad: 0.55 }],
    });
    apiMocks.fetchPricingReference.mockResolvedValue({
      reference: [{ gpu_model: "RTX 4090", on_demand_cad: 0.55, spot_cad: 0.22 }],
    });
    apiMocks.fetchImageTemplates.mockResolvedValue({ templates: [] });
    apiMocks.fetchServiceTierCatalog.mockImplementation(async (gpu_model: string, mode: string) => {
      const base = mode === "spot" ? 0.22 : 0.55;
      return {
        gpu_model, mode, vram_gb: 24, base_rate_micros: Math.round(base * 1_000_000), base_rate_per_gpu_cad: base,
        tiers: (["standard", "premium", "dedicated"] as const).map((tier, index) => ({
          tier, label: tier[0].toUpperCase() + tier.slice(1), priority: index + 1,
          stripe_price_id: `price_${tier}_${mode}`, stripe_product_id: "prod_gpu",
          unit_amount_micros: Math.round(base * [1, 1.6, 2.8][index] * 1_000_000),
          base_rate_micros: Math.round(base * 1_000_000),
          tier_addition_micros: Math.round(base * [0, 0.6, 1.8][index] * 1_000_000),
          rate_per_gpu_cad: base * [1, 1.6, 2.8][index], tier_addition_per_gpu_cad: base * [0, 0.6, 1.8][index],
        })),
        gpu_exclusive_addon: { stripe_price_id: "price_gpu_exclusive", unit_amount_micros: Math.round(base * 0.4 * 1_000_000), rate_per_gpu_cad: base * 0.4 },
        queue_bid_unit: {stripe_price_id: "price_queue_bid", unit_amount_micros: 10000},
        queue_bid_policy: {version:"tier-bid-aging-v1",aging_interval_seconds:900,aging_fraction_bps:1000,max_tier_promotion:1},
        physical_hosts: [{ host_id: "host-two-gpus", num_gpus: 2, region: "ca-east", available: true }],
      };
    });
    // The real shape of `/spot-prices`: `prices` is the lookup, `spot_prices`
    // is a **list**. This mock previously sent `spot_prices` as a Record — a
    // shape the route has never returned — so the suite validated the inverted
    // type instead of catching it, and the modal's `spot_prices || prices`
    // chain looked correct here while returning `undefined` in the browser.
    apiMocks.fetchSpotPrices.mockResolvedValue({
      ok: true,
      prices: { "RTX 4090": 0.22 },
      spot_prices: [
        {
          gpu_model: "RTX 4090",
          rate_cad: 0.22,
          spot_cents: 22,
          on_demand_cad: 0.55,
          savings_pct: 60,
          supply: 0,
          demand: 0,
          provider_floor_cents: 0,
          recorded_at: 0,
        },
      ],
    });
    apiMocks.fetchSpotFeatureStatus.mockResolvedValue({ enabled: true, message: null });
    apiMocks.listAvailableVolumes.mockResolvedValue({ volumes: [] });
    apiMocks.fetchPricingRates.mockResolvedValue({
      effective_rate_per_gpu: 0.22,
      total_per_hour: 0.22,
      tax_rate: 0.13,
      tax_description: "HST",
      tax_amount: 0.03,
      total_with_tax: 0.25,
      base_rate_cad: 0.22,
      priority_multiplier: 1,
      multi_gpu_discount: 0,
    });
    apiMocks.launchInstance.mockResolvedValue({
      instance: { job_id: "job-spot-1", status: "queued", docker_image: "nvcr.io/nvidia/pytorch:24.12-py3" },
    });
    apiMocks.classifyLaunchError.mockImplementation((err: Error) => ({ message: err.message }));
  });

  it(
    "shows spot rate with an optional bid and submits pricing_mode=spot",
    async () => {
      render(
        <LaunchInstanceModal
          open
          onClose={() => {}}
          initialGpuModel="RTX 4090"
          initialPricingMode="spot"
        />,
      );

      await waitFor(() => {
        expect(screen.getByText(/Interruptible spot instance/i)).toBeInTheDocument();
      });

      expect(screen.getByLabelText(/additional queue bid/i)).toHaveValue(0);
      expect(screen.queryByPlaceholderText(/bid/i)).not.toBeInTheDocument();

      expect(screen.getByPlaceholderText("docker.io/my-org/my-image:latest")).toHaveValue("nvcr.io/nvidia/pytorch:24.12-py3");

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /^continue$/i })).not.toBeDisabled();
      });

      fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));

      await waitFor(() => {
        expect(screen.getByText(/Spot rate of/i)).toBeInTheDocument();
      });


      fireEvent.click(screen.getByRole("button", { name: /launch instance/i }));

      await waitFor(() => {
        expect(apiMocks.launchInstance).toHaveBeenCalledWith(
          expect.objectContaining({
            pricing_mode: "spot",
            tier: "standard",
            service_tier_price_id: "price_standard_spot",
          }),
        );
      });

      const payload = apiMocks.launchInstance.mock.calls[0][0];
      expect(payload).not.toHaveProperty("max_bid");
      expect(payload).not.toHaveProperty("maxBid");
      expect(payload).not.toHaveProperty("priority");
    },
    15000,
  );

  it("updates the total immediately with GPU count and existing volume storage", async () => {
    apiMocks.listAvailableVolumes.mockResolvedValue({ volumes: [{
      volume_id: "vol-models", name: "Models", size_gb: 730,
      monthly_cost_cad: 21.9, price_per_gb_month_cad: 0.03,
    }] });
    render(<LaunchInstanceModal open onClose={() => {}} />);

    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    const quoteCount = apiMocks.fetchPricingRates.mock.calls.length;
    fireEvent.change(screen.getByLabelText("GPUs"), { target: { value: "2" } });
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$1.243/hr");
    expect(apiMocks.fetchPricingRates).toHaveBeenCalledTimes(quoteCount);

    fireEvent.click(screen.getByRole("button", { name: /Models/ }));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$1.2769/hr");
    expect(screen.getByText(/attaching adds no new fee/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$1.2769/hr");
    fireEvent.click(screen.getByRole("button", { name: /launch instance/i }));
    await waitFor(() => expect(apiMocks.launchInstance).toHaveBeenCalledWith(expect.objectContaining({
      num_gpus: 2, volume_ids: ["vol-models"], tier: "standard", service_tier_price_id: "price_standard_on_demand",
    })));
  });

  it("offers an interruption tradeoff and recomputes the total when pricing mode changes", async () => {
    render(<LaunchInstanceModal open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    expect(screen.getByText("Can your workload handle interruptions?")).toBeInTheDocument();
    expect(screen.getByText("Service Tier")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /critical/i })).not.toBeInTheDocument();
    expect(screen.queryByText("dash.team.banner_title")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^spot/i }));
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.2486/hr"));
    fireEvent.change(screen.getByLabelText("GPUs"), { target: { value: "3" } });
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.7458/hr");
  });

  it("shows unavailable storage prices rather than counting them as free", async () => {
    apiMocks.listAvailableVolumes.mockResolvedValue({ volumes: [{
      volume_id: "vol-legacy", name: "Legacy data", size_gb: 50,
    }] });
    render(<LaunchInstanceModal open onClose={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: /Legacy data/ }));
    expect(screen.getByText("Price unavailable")).toBeInTheDocument();
    expect(screen.getByTestId("launch-total")).toHaveTextContent("Unavailable");
  });

  it("adds a Stripe queue bid immediately for every GPU and pins its Price", async () => {
    render(<LaunchInstanceModal open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    const calls = apiMocks.fetchServiceTierCatalog.mock.calls.length;
    fireEvent.change(screen.getByLabelText(/additional queue bid/i), {target:{value:"0.20"}});
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.8475/hr");
    fireEvent.change(screen.getByLabelText("GPUs"), {target:{value:"2"}});
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$1.695/hr");
    fireEvent.click(screen.getByRole("button", {name:/^premium/i}));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$2.4408/hr");
    expect(apiMocks.fetchServiceTierCatalog.mock.calls).toHaveLength(calls);
    fireEvent.click(screen.getByRole("button", {name:"Continue"}));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$2.4408/hr");
    fireEvent.click(screen.getByRole("button",{name:/launch instance/i}));
    await waitFor(()=>expect(apiMocks.launchInstance).toHaveBeenCalledWith(expect.objectContaining({queue_bid_units:20,queue_bid_price_id:"price_queue_bid",tier:"premium",num_gpus:2})));
  });

  it("refuses fractional Stripe bid units and negative bids", async () => {
    render(<LaunchInstanceModal open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    for(const value of ["0.201", "-0.10"]){
      fireEvent.change(screen.getByLabelText(/additional queue bid/i), {target:{value}});
      expect(screen.getByTestId("launch-total")).toHaveTextContent("Unavailable");
      expect(screen.getByRole("button",{name:"Continue"})).toBeDisabled();
    }
  });

  it("adds the Stripe Premium price immediately and offers cheaper GPU-only exclusivity", async () => {
    render(<LaunchInstanceModal open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    const calls = apiMocks.fetchServiceTierCatalog.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: /^premium/i }));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.9944/hr");
    expect(apiMocks.fetchServiceTierCatalog).toHaveBeenCalledTimes(calls);
    fireEvent.click(screen.getByRole("button", { name: /^standard/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Exclusive allocated GPUs/i }));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.8701/hr");
  });

  it("Dedicated reserves and prices every GPU on a physical host without an extra GPU-only fee", async () => {
    render(<LaunchInstanceModal open onClose={() => {}} />);
    await waitFor(() => expect(screen.getByTestId("launch-total")).toHaveTextContent("$0.6215/hr"));
    fireEvent.click(screen.getByRole("checkbox", { name: /Exclusive allocated GPUs/i }));
    fireEvent.click(screen.getByRole("button", { name: /^dedicated/i }));
    expect(screen.getByTestId("launch-total")).toHaveTextContent("$3.4804/hr");
    expect(screen.getByLabelText("GPUs")).toHaveValue(2);
    expect(screen.getByLabelText("GPUs")).toBeDisabled();
    expect(screen.queryByRole("checkbox", { name: /Exclusive allocated GPUs/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^continue$/i }));
    fireEvent.click(screen.getByRole("button", { name: /launch instance/i }));
    await waitFor(() => expect(apiMocks.launchInstance).toHaveBeenCalledWith(expect.objectContaining({
      tier: "dedicated", num_gpus: 2, service_tier_price_id: "price_dedicated_on_demand", exclusive_gpu: false,
    })));
  });

  it("refuses to launch when Stripe-backed tier prices could not be loaded", async () => {
    apiMocks.fetchServiceTierCatalog.mockRejectedValue(new Error("unavailable"));
    render(<LaunchInstanceModal open onClose={() => {}} />);
    expect(await screen.findByText(/Unable to load service-tier prices/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^continue$/i })).toBeDisabled();
  });
});
