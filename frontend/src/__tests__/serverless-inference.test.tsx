import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { mockEndpoint, viewerUser, writerUser } from "./serverless-test-helpers";

const apiMocks = vi.hoisted(() => ({
  listServerlessEndpoints: vi.fn(),
  deleteServerlessEndpoint: vi.fn(),
}));

const authMocks = vi.hoisted(() => ({
  useAuth: vi.fn(),
}));

const toastMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);
vi.mock("@/lib/auth", () => ({ useAuth: authMocks.useAuth }));
vi.mock("@/hooks/useEventStream", () => ({ useEventStream: () => ({ status: "disconnected" }) }));
vi.mock("@/components/team/team-context-banner", () => ({ TeamContextBanner: () => null }));
vi.mock("sonner", () => ({ toast: toastMocks }));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
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
    apiMocks.deleteServerlessEndpoint.mockResolvedValue({ ok: true });
    vi.stubGlobal("confirm", vi.fn(() => true));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders endpoint list and Deploy Studio link for writers", async () => {
    render(<InferencePage />);
    await waitFor(() => {
      expect(screen.getByText("test-llama")).toBeInTheDocument();
    });
    const studioLink = screen.getByRole("link", { name: /dash.serverless.open_studio/i });
    expect(studioLink).toHaveAttribute("href", "/dashboard/inference/new");
  });

  it("shows view-only deploy control for team viewers", async () => {
    authMocks.useAuth.mockReturnValue({ user: viewerUser });
    render(<InferencePage />);
    await waitFor(() => {
      expect(screen.getByText("dash.serverless.view_only")).toBeInTheDocument();
    });
    expect(screen.queryByRole("link", { name: /dash.serverless.open_studio/i })).not.toBeInTheDocument();
  });

  it("hides delete control for team viewers", async () => {
    authMocks.useAuth.mockReturnValue({ user: viewerUser });
    render(<InferencePage />);
    await waitFor(() => expect(screen.getByText("test-llama")).toBeInTheDocument());
    expect(document.querySelector(".lucide-trash2")).toBeNull();
  });
});