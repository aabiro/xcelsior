import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { mockEndpoint } from "./serverless-test-helpers";

// The phone Quick Deploy sheet (prod, 2026-11-22): its GPU list was a fixed
// catalog (B200, H200, H100…) whenever the fleet had nothing free, its Region
// list was then empty ("No Options"), and a successful deploy landed on a 404.

const apiMocks = vi.hoisted(() => ({
  createServerlessEndpoint: vi.fn(),
  fetchAvailableGPUs: vi.fn(),
}));
const routerMocks = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, ...apiMocks };
});
vi.mock("next/navigation", () => ({ useRouter: () => routerMocks }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }));
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

import { DeployServerlessModal } from "@/components/mobile/deploy-serverless-modal";

const BUSY_FLEET = {
  ok: true,
  gpus: [],
  inventory: [
    { gpu_model: "RTX 2060", vram_gb: 6, region: "ca-on", province: "ON", count_total: 1, count_available: 0, count_allocated: 1, count_held: 0, count_unavailable: 0 },
    { gpu_model: "RTX 3060", vram_gb: 12, region: "ca-on", province: "ON", count_total: 1, count_available: 0, count_allocated: 1, count_held: 0, count_unavailable: 0 },
  ],
};

function gpuSelect() {
  return screen.getByLabelText("dash.serverless.gpu_type") as HTMLSelectElement;
}
function countrySelect() {
  return screen.getByLabelText("dash.serverless.country") as HTMLSelectElement;
}
function regionSelect() {
  return screen.getByLabelText("dash.serverless.province") as HTMLSelectElement;
}

describe("DeployServerlessModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.createServerlessEndpoint.mockResolvedValue({ endpoint: mockEndpoint });
  });

  it("lists the real fleet with its regions when every GPU is busy", async () => {
    apiMocks.fetchAvailableGPUs.mockResolvedValue(BUSY_FLEET);
    render(<DeployServerlessModal open onClose={vi.fn()} canWrite />);
    await waitFor(() => expect(gpuSelect()).not.toBeDisabled());
    const models = Array.from(gpuSelect().options).map((o) => o.value).filter(Boolean);
    expect(models).toEqual(["RTX 2060", "RTX 3060"]);
    expect(screen.queryByText(/B200|H200|A100/)).toBeNull();

    fireEvent.change(gpuSelect(), { target: { value: "RTX 3060" } });
    // Country, then province by name; the region code stays underneath.
    expect(countrySelect()).toHaveValue("CA");
    expect(countrySelect().selectedOptions[0].textContent).toBe("Canada");
    expect(regionSelect()).toHaveValue("ca-on");
    expect(Array.from(regionSelect().options).map((o) => o.value)).toEqual(["ca-on"]);
    expect(screen.getByText("dash.serverless.gpu_all_busy")).toBeInTheDocument();
  });

  it("says there are no GPUs, and offers no deploy, when the fleet is empty", async () => {
    apiMocks.fetchAvailableGPUs.mockResolvedValue({ ok: true, gpus: [], inventory: [] });
    render(<DeployServerlessModal open onClose={vi.fn()} canWrite />);
    expect(await screen.findByText("dash.serverless.no_gpus_title")).toBeInTheDocument();
    expect(gpuSelect()).toBeDisabled();
    expect(gpuSelect().options).toHaveLength(1);
    expect(countrySelect()).toBeDisabled();
    expect(regionSelect()).toBeDisabled();
    expect(screen.getByRole("button", { name: /dash\.serverless\.deploy/ })).toBeDisabled();
  });

  it("says the list failed to load, and retries on request", async () => {
    apiMocks.fetchAvailableGPUs.mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(BUSY_FLEET);
    render(<DeployServerlessModal open onClose={vi.fn()} canWrite />);
    expect(await screen.findByText("dash.serverless.gpus_load_failed")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "dash.serverless.gpus_retry" }));
    await waitFor(() => expect(gpuSelect()).not.toBeDisabled());
  });

  it("opens the new endpoint at its real page after a deploy", async () => {
    apiMocks.fetchAvailableGPUs.mockResolvedValue(BUSY_FLEET);
    const onClose = vi.fn();
    render(<DeployServerlessModal open onClose={onClose} canWrite />);
    await waitFor(() => expect(gpuSelect()).not.toBeDisabled());
    fireEvent.change(gpuSelect(), { target: { value: "RTX 3060" } });
    fireEvent.click(screen.getByRole("button", { name: /dash\.serverless\.deploy/ }));
    await waitFor(() =>
      expect(apiMocks.createServerlessEndpoint).toHaveBeenCalledWith(
        expect.objectContaining({ gpu_tier: "RTX 3060", region: "ca-on" }),
      ),
    );
    expect(routerMocks.push).toHaveBeenCalledWith(
      `/dashboard/inference?endpoint=${encodeURIComponent(mockEndpoint.endpoint_id)}`,
    );
    expect(onClose).toHaveBeenCalled();
  });
});
