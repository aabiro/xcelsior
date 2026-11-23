import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
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

const navigation = vi.hoisted(() => ({ pathname: "/dashboard" }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => navigation.pathname,
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

vi.mock("@/components/mobile/deploy-serverless-modal", () => ({
  DeployServerlessModal: ({ open }: { open: boolean }) => open ? <div data-testid="deploy-serverless-modal" /> : null,
}));

describe("MobileDeployAction", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    navigation.pathname = "/dashboard";
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

  it.each([
    "/dashboard/hosts/host-id",
    "/dashboard/settings",
    "/dashboard/hosting",
    "/dashboard/earnings",
    "/dashboard/inference/new",
    // An instance's logs and terminal: the dock covered the Terminal card.
    "/dashboard/instances/925934a5",
  ])("keeps editable forms clear on %s", (pathname) => {
    navigation.pathname = pathname;
    const { container } = render(<MobileDeployAction serverlessEnabled canWrite />);
    expect(container).toBeEmptyDOMElement();
  });

  it.each(["/dashboard/instances", "/dashboard/instances/new", "/dashboard"])(
    "stays on %s, where launching is the point",
    (pathname) => {
      navigation.pathname = pathname;
      render(<MobileDeployAction serverlessEnabled canWrite />);
      expect(screen.getByText("dash.mobile.action_idle")).toBeInTheDocument();
    },
  );

  it("sits just above the browser bar, not lifted mid-screen", () => {
    const { container } = render(<MobileDeployAction serverlessEnabled canWrite />);
    const dock = container.querySelector(".fixed") as HTMLElement;
    expect(dock.className).toContain("bottom-0");
    expect(dock.className).not.toMatch(/5\.5rem|4rem/);
  });

  it("hides when user cannot write", () => {
    const { container } = render(<MobileDeployAction serverlessEnabled canWrite={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("opens the instance form after two touch-style taps", () => {
    render(<MobileDeployAction serverlessEnabled canWrite />);
    const action = screen.getByRole("button", { name: "dash.mobile.action_idle_label" });
    fireEvent.click(action, { detail: 1 });
    expect(screen.queryByTestId("launch-instance-modal")).toBeNull();
    fireEvent.click(action, { detail: 1 });
    expect(screen.getByTestId("launch-instance-modal")).toBeInTheDocument();
  });

  it("opens the selected serverless form on keyboard activation", () => {
    render(<MobileDeployAction serverlessEnabled canWrite />);
    fireEvent.click(screen.getByText("dash.mobile.track_serverless"));
    fireEvent.click(screen.getByRole("button", { name: "dash.mobile.action_idle_label" }), { detail: 0 });
    expect(screen.getByTestId("deploy-serverless-modal")).toBeInTheDocument();
    expect(screen.queryByTestId("launch-instance-modal")).toBeNull();
  });

  // The dock (z-48) drew over the phone navigation drawer's last links
  // (owner review, 2026-11-23). While the drawer is open it steps aside.
  it("steps aside while the navigation drawer is open", () => {
    const { container } = render(<MobileDeployAction serverlessEnabled canWrite suppressed />);
    const dock = container.querySelector("[aria-hidden='true']") as HTMLElement;
    expect(dock).not.toBeNull();
    expect(dock.className).toContain("pointer-events-none");
    expect(dock.className).toContain("opacity-0");
    expect(dock.hasAttribute("inert")).toBe(true);
  });
});
