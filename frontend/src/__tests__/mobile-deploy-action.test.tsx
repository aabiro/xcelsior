import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MobileDeployAction } from "@/components/mobile/mobile-deploy-action";

vi.mock("@/lib/desktop/runtime", () => ({
  useDesktopRuntime: () => ({
    state: { isStandalonePwa: true, isNativeDesktop: false },
  }),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/dashboard",
}));

vi.mock("@/lib/api", () => ({
  fetchAvailableGPUs: vi.fn().mockResolvedValue({
    gpus: [{ gpu_model: "RTX 4090", region: "ca-east", price_per_hour_cad: 1.5 }],
  }),
  createServerlessEndpoint: vi.fn(),
}));

vi.mock("sonner", () => ({
  toast: {
    message: vi.fn(),
    error: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock("@/components/instances/launch-instance-modal", () => ({
  LaunchInstanceModal: ({ open }: { open: boolean }) =>
    open ? <div data-testid="launch-instance-modal" /> : null,
}));

describe("MobileDeployAction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders track selector and hold CTA when serverless enabled", () => {
    render(<MobileDeployAction serverlessEnabled canWrite />);
    expect(screen.getByText("dash.mobile.track_instance")).toBeInTheDocument();
    expect(screen.getByText("dash.mobile.track_serverless")).toBeInTheDocument();
    expect(screen.getByText("dash.mobile.action_idle")).toBeInTheDocument();
  });

  it("renders instance-only CTA when serverless disabled", () => {
    render(<MobileDeployAction serverlessEnabled={false} canWrite />);
    expect(screen.getByText("dash.mobile.action_idle_instances")).toBeInTheDocument();
  });

  it("hides when user cannot write", () => {
    const { container } = render(<MobileDeployAction serverlessEnabled canWrite={false} />);
    expect(container).toBeEmptyDOMElement();
  });


});