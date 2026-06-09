import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMocks = vi.hoisted(() => ({
  launchInstance: vi.fn(),
  fetchAvailableGPUs: vi.fn(),
  fetchPricingReference: vi.fn(),
  fetchProvinces: vi.fn(),
  fetchImageTemplates: vi.fn(),
  fetchSpotPrices: vi.fn(),
  listAvailableVolumes: vi.fn(),
  detectProvince: vi.fn(),
  fetchPricingRates: vi.fn(),
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

describe("LaunchInstanceModal spot flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.fetchAvailableGPUs.mockResolvedValue({
      gpus: [{ gpu_model: "RTX 4090", vram_gb: 24, count_available: 2, price_per_hour_cad: 0.55 }],
    });
    apiMocks.fetchPricingReference.mockResolvedValue({
      reference: [{ gpu_model: "RTX 4090", on_demand_cad: 0.55, spot_cad: 0.22 }],
    });
    apiMocks.fetchProvinces.mockResolvedValue({ provinces: { ON: { name: "Ontario", tax_rate: 0.13, tax_description: "HST" } } });
    apiMocks.fetchImageTemplates.mockResolvedValue({ templates: [] });
    apiMocks.fetchSpotPrices.mockResolvedValue({ spot_prices: { "RTX 4090": 0.22 } });
    apiMocks.fetchSpotFeatureStatus.mockResolvedValue({ enabled: true, message: null });
    apiMocks.listAvailableVolumes.mockResolvedValue({ volumes: [] });
    apiMocks.detectProvince.mockResolvedValue({ province: "ON" });
    apiMocks.fetchPricingRates.mockResolvedValue({
      effective_rate_per_gpu: 0.22,
      total_per_hour: 0.22,
      tax_rate: 0.13,
      tax_description: "HST",
      tax_amount: 0.03,
      total_with_tax: 0.25,
      base_rate_cad: 0.22,
      priority_multiplier: 1,
      sovereignty_premium: 0,
      multi_gpu_discount: 0,
    });
    apiMocks.launchInstance.mockResolvedValue({
      instance: { job_id: "job-spot-1", status: "queued", docker_image: "nvcr.io/nvidia/pytorch:24.12-py3" },
    });
    apiMocks.classifyLaunchError.mockImplementation((err: Error) => ({ message: err.message }));
  });

  it(
    "shows spot rate without bid input and submits pricing_mode=spot",
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

      expect(screen.queryByLabelText(/bid/i)).not.toBeInTheDocument();
      expect(screen.queryByPlaceholderText(/bid/i)).not.toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: /pytorch/i }));

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
          }),
        );
      });

      const payload = apiMocks.launchInstance.mock.calls[0][0];
      expect(payload).not.toHaveProperty("max_bid");
      expect(payload).not.toHaveProperty("maxBid");
    },
    15000,
  );
});