import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { mockEndpoint, viewerUser, writerUser } from "./serverless-test-helpers";

const apiMocks = vi.hoisted(() => ({
  listServerlessEndpoints: vi.fn(),
  deleteServerlessEndpoint: vi.fn(),
  fetchAvailableGPUs: vi.fn(),
}));

const authMocks = vi.hoisted(() => ({
  useAuth: vi.fn(),
}));

const toastMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listServerlessEndpoints: apiMocks.listServerlessEndpoints,
    deleteServerlessEndpoint: apiMocks.deleteServerlessEndpoint,
    fetchAvailableGPUs: apiMocks.fetchAvailableGPUs,
  };
});
vi.mock("@/lib/auth", () => ({ useAuth: authMocks.useAuth }));
vi.mock("@/hooks/useEventStream", () => ({ useEventStream: () => ({ status: "disconnected" }) }));
vi.mock("@/components/team/team-context-banner", () => ({ TeamContextBanner: () => null }));
vi.mock("sonner", () => ({ toast: toastMocks }));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));
vi.mock("next/link", () => ({
  default: ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => (
    <a href={href} className={className}>{children}</a>
  ),
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

import InferencePage from "@/app/(dashboard)/dashboard/inference/page";

describe("InferencePage (serverless list)", () => {
  vi.setConfig({ testTimeout: 15_000 });
  beforeEach(() => {
    vi.clearAllMocks();
    authMocks.useAuth.mockReturnValue({ user: writerUser });
    apiMocks.listServerlessEndpoints.mockResolvedValue({ endpoints: [mockEndpoint] });
    apiMocks.fetchAvailableGPUs.mockResolvedValue({ gpus: [] });
    apiMocks.deleteServerlessEndpoint.mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));
    vi.stubGlobal("confirm", vi.fn(() => true));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders endpoint list and create-endpoint control for writers", async () => {
    render(<InferencePage />);
    await waitFor(() => {
      expect(screen.getAllByText("test-llama").length).toBeGreaterThan(0);
    });
    expect(screen.getByRole("button", { name: /^Create$/i })).toBeInTheDocument();
  });

  it("shows view-only deploy control for team viewers", async () => {
    authMocks.useAuth.mockReturnValue({ user: viewerUser });
    render(<InferencePage />);
    await waitFor(() => {
      expect(screen.getByText("dash.serverless.view_only")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^Create$/i })).not.toBeInTheDocument();
  });

  it("hides delete control for team viewers", async () => {
    authMocks.useAuth.mockReturnValue({ user: viewerUser });
    render(<InferencePage />);
    await waitFor(() => expect(screen.getAllByText("test-llama").length).toBeGreaterThan(0));
    expect(document.querySelector(".lucide-trash2")).toBeNull();
  });
});