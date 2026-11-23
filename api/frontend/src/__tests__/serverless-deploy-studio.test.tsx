import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { mockEndpoint } from "./serverless-test-helpers";

const apiMocks = vi.hoisted(() => ({
  createServerlessEndpoint: vi.fn(),
}));

const routerMocks = vi.hoisted(() => ({
  push: vi.fn(),
}));

const toastMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    createServerlessEndpoint: apiMocks.createServerlessEndpoint,
  };
});
vi.mock("next/navigation", () => ({ useRouter: () => routerMocks }));
vi.mock("sonner", () => ({ toast: toastMocks }));
vi.mock("posthog-js", () => ({
  default: { capture: vi.fn(), captureException: vi.fn() },
}));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
}));
vi.mock("framer-motion", async () => {
  const ReactModule = await import("react");
  const div = ReactModule.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => ReactModule.createElement("div", { ...props, ref }, children),
  );
  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    motion: { div },
  };
});

import { DeployStudio } from "@/features/serverless/deploy-studio";

function clickContinue() {
  fireEvent.click(screen.getByRole("button", { name: /dash\.serverless\.continue/i }));
}

const LIVE = {
  gpus: [
    { gpu_model: "RTX 3060", vram_gb: 12, region: "ca-on", province: "ON", count_available: 1, price_per_hour_cad: 0.12 },
  ],
  inventory: [
    { gpu_model: "RTX 2060", vram_gb: 6, region: "ca-on", province: "ON", count_total: 1, count_available: 0, count_allocated: 1, count_held: 0, count_unavailable: 0 },
    { gpu_model: "RTX 3060", vram_gb: 12, region: "ca-on", province: "ON", count_total: 1, count_available: 1, count_allocated: 0, count_held: 0, count_unavailable: 0 },
  ],
};

describe("DeployStudio", () => {
  vi.setConfig({ testTimeout: 20_000 });

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    apiMocks.createServerlessEndpoint.mockResolvedValue({ endpoint: mockEndpoint });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("advances through steps and deploys a preset endpoint", async () => {
    render(<DeployStudio gpuData={LIVE} canWrite />);

    expect(screen.getByText("dash.serverless.method_title")).toBeInTheDocument();
    clickContinue();

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.source_title")).toBeInTheDocument();
    });
    clickContinue();

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.hardware_title")).toBeInTheDocument();
    });

    const [gpuSelect] = screen.getAllByRole("combobox");
    fireEvent.change(gpuSelect, { target: { value: "RTX 3060" } });
    clickContinue();

    clickContinue();
    clickContinue();

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.review_title")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /dash\.serverless\.deploy/i }));

    await waitFor(() => {
      expect(apiMocks.createServerlessEndpoint).toHaveBeenCalled();
      expect(routerMocks.push).toHaveBeenCalledWith(
        `/dashboard/inference?endpoint=${encodeURIComponent(mockEndpoint.endpoint_id)}`,
      );
    });
  });

  it("uses normalized live inventory regions for deployment", async () => {
    render(
      <DeployStudio
        canWrite
        gpuData={{
          gpus: [
            {
              gpu_model: "RTX 4090",
              vram_gb: 24,
              region: "",
              province: "ON",
              count_available: 1,
              price_per_hour_cad: 1.5,
            },
          ],
        }}
      />,
    );

    clickContinue();
    clickContinue();

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.hardware_title")).toBeInTheDocument();
    });

    const [, countrySelect, regionSelect] = screen.getAllByRole("combobox");
    await waitFor(() => {
      expect(countrySelect).toHaveValue("CA");
      expect(regionSelect).toHaveValue("ca-on");
    });

    clickContinue();
    clickContinue();
    clickContinue();
    fireEvent.click(await screen.findByRole("button", { name: /dash\.serverless\.deploy/i }));

    await waitFor(() => {
      expect(apiMocks.createServerlessEndpoint).toHaveBeenCalledWith(
        expect.objectContaining({
          gpu_tier: "RTX 4090",
          region: "ca-on",
        }),
      );
    });
  });

  it("lists only the live fleet, busy models labelled, and never a catalog", async () => {
    render(<DeployStudio gpuData={LIVE} canWrite />);
    clickContinue();
    clickContinue();
    await waitFor(() => {
      expect(screen.getByText("dash.serverless.hardware_title")).toBeInTheDocument();
    });
    const [gpuSelect, countrySelect, regionSelect] = screen.getAllByRole("combobox");
    const labels = Array.from((gpuSelect as HTMLSelectElement).options).map((o) => o.value).filter(Boolean);
    expect(labels).toEqual(["RTX 3060", "RTX 2060"]);
    // The free model is preselected; the busy one stays choosable, its workers queue.
    expect(gpuSelect).toHaveValue("RTX 3060");
    expect(countrySelect).toHaveValue("CA");
    expect(regionSelect).toHaveValue("ca-on");
    fireEvent.change(gpuSelect, { target: { value: "RTX 2060" } });
    expect(await screen.findByText("dash.serverless.gpu_all_busy")).toBeInTheDocument();
  });

  it("says so when the fleet lists no GPUs, instead of inventing some", async () => {
    render(<DeployStudio gpuData={{ gpus: [], inventory: [] }} canWrite />);
    clickContinue();
    clickContinue();
    expect(await screen.findByText("dash.serverless.no_gpus_title")).toBeInTheDocument();
    const [gpuSelect, countrySelect, regionSelect] = screen.getAllByRole("combobox");
    expect(gpuSelect).toBeDisabled();
    expect(countrySelect).toBeDisabled();
    expect(regionSelect).toBeDisabled();
    expect((gpuSelect as HTMLSelectElement).options).toHaveLength(1);
    expect(screen.queryByText(/A100|H100|B200/)).toBeNull();
  });

  it("reports a failed GPU read as a failure, not as an empty fleet", async () => {
    render(<DeployStudio gpuData={{ gpus: [], inventory: [], failed: true }} canWrite />);
    clickContinue();
    clickContinue();
    expect(await screen.findByText("dash.serverless.gpus_load_failed")).toBeInTheDocument();
  });

  it("disables deploy for viewers", async () => {
    render(<DeployStudio gpuData={LIVE} canWrite={false} />);

    clickContinue();
    clickContinue();
    fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "RTX 3060" } });
    clickContinue();
    clickContinue();
    clickContinue();

    const deployBtn = await screen.findByRole("button", { name: /dash\.serverless\.deploy/i });
    expect(deployBtn).toBeDisabled();
    expect(apiMocks.createServerlessEndpoint).not.toHaveBeenCalled();
  });
});
