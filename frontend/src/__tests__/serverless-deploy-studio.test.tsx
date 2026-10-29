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
    render(<DeployStudio gpus={[]} canWrite />);

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
    fireEvent.change(gpuSelect, { target: { value: "A100" } });
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
        gpus={[
          {
            gpu_model: "RTX 4090",
            vram_gb: 24,
            region: "",
            province: "ON",
            count_available: 1,
            price_per_hour_cad: 1.5,
          },
        ]}
      />,
    );

    clickContinue();
    clickContinue();

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.hardware_title")).toBeInTheDocument();
    });

    const [, regionSelect] = screen.getAllByRole("combobox");
    await waitFor(() => {
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

  it("disables deploy for viewers", async () => {
    render(<DeployStudio gpus={[]} canWrite={false} />);

    clickContinue();
    clickContinue();
    fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "A100" } });
    clickContinue();
    clickContinue();
    clickContinue();

    const deployBtn = await screen.findByRole("button", { name: /dash\.serverless\.deploy/i });
    expect(deployBtn).toBeDisabled();
    expect(apiMocks.createServerlessEndpoint).not.toHaveBeenCalled();
  });
});