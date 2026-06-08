import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
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
}));

vi.mock("@/lib/api", () => apiMocks);
vi.mock("next/navigation", () => ({ useRouter: () => routerMocks }));
vi.mock("sonner", () => ({ toast: toastMocks }));
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

describe("DeployStudio", () => {
  vi.setConfig({ testTimeout: 20_000 });

  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.createServerlessEndpoint.mockResolvedValue({ endpoint: mockEndpoint });
  });

  it("advances through steps and deploys a preset endpoint", async () => {
    render(<DeployStudio gpus={[]} canWrite />);

    expect(screen.getByText("dash.serverless.method_title")).toBeInTheDocument();
    fireEvent.click(screen.getByText("dash.serverless.continue"));

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.source_title")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("dash.serverless.continue"));

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.hardware_title")).toBeInTheDocument();
    });

    const [gpuSelect] = screen.getAllByRole("combobox");
    fireEvent.change(gpuSelect, { target: { value: "A100" } });
    fireEvent.click(screen.getByText("dash.serverless.continue"));

    fireEvent.click(screen.getByText("dash.serverless.continue"));
    fireEvent.click(screen.getByText("dash.serverless.continue"));

    await waitFor(() => {
      expect(screen.getByText("dash.serverless.review_title")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("dash.serverless.deploy"));

    await waitFor(() => {
      expect(apiMocks.createServerlessEndpoint).toHaveBeenCalled();
      expect(routerMocks.push).toHaveBeenCalledWith(`/dashboard/inference/${mockEndpoint.endpoint_id}`);
    });
  });

  it("disables deploy for viewers", async () => {
    render(<DeployStudio gpus={[]} canWrite={false} />);

    fireEvent.click(screen.getByText("dash.serverless.continue"));
    fireEvent.click(screen.getByText("dash.serverless.continue"));
    fireEvent.change(screen.getAllByRole("combobox")[0], { target: { value: "A100" } });
    fireEvent.click(screen.getByText("dash.serverless.continue"));
    fireEvent.click(screen.getByText("dash.serverless.continue"));
    fireEvent.click(screen.getByText("dash.serverless.continue"));

    const deployBtn = await screen.findByText("dash.serverless.deploy");
    expect(deployBtn.closest("button")).toBeDisabled();
    expect(apiMocks.createServerlessEndpoint).not.toHaveBeenCalled();
  });
});