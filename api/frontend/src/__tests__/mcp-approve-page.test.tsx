import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// The page where a person answers an agent's request for more access. What
// matters: it says what would be granted in the server's own plain words,
// marks the one that spends money, shows the agent's text as a quotation, and
// nothing is sent until Approve or Decline is pressed.

const api = vi.hoisted(() => ({ getMcpPermissionRequest: vi.fn(), decideMcpPermissionRequest: vi.fn() }));
const params = vi.hoisted(() => ({ code: "ABCDE-FGHJK" as string | null }));
vi.mock("@/lib/api", () => api);
vi.mock("next/navigation", () => ({ useSearchParams: () => ({ get: () => params.code }) }));
vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => <a href={href}>{children}</a>,
}));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import McpApprovePage from "@/app/(dashboard)/dashboard/mcp/approve/page";

const pending = {
  code: "ABCDE-FGHJK",
  status: "pending",
  surface: "mcp",
  key_prefix: "xcel_ai_…7890",
  reason: "Top up $20 to finish the run",
  requested: [
    { scope: "instances:write", description: "Launch, cancel, and terminate GPU instances", spends_money: false },
    { scope: "billing:topup", description: "Charge your saved card to top up your wallet", spends_money: true },
  ],
  created_at: "2026-11-12T20:00:00Z",
  expires_at: "2026-11-12T20:15:00Z",
};

describe("MCP approval page", () => {
  beforeEach(() => {
    api.getMcpPermissionRequest.mockReset();
    api.decideMcpPermissionRequest.mockReset();
    params.code = "ABCDE-FGHJK";
  });

  it("shows what would be granted, warns about money, and quotes the agent", async () => {
    api.getMcpPermissionRequest.mockResolvedValue({ ok: true, request: pending });
    render(<McpApprovePage />);
    expect(await screen.findByText("Launch, cancel, and terminate GPU instances")).toBeInTheDocument();
    expect(screen.getByText("Charge your saved card to top up your wallet")).toBeInTheDocument();
    expect(screen.getByText("dash.mcp.approve.money_warning")).toBeInTheDocument();
    expect(screen.getByText("“Top up $20 to finish the run”")).toBeInTheDocument();
    expect(screen.getByText("xcel_ai_…7890")).toBeInTheDocument();
    expect(api.getMcpPermissionRequest).toHaveBeenCalledWith("ABCDE-FGHJK");
    // Reading the page answers nothing.
    expect(api.decideMcpPermissionRequest).not.toHaveBeenCalled();
  });

  it("has no money warning when nothing requested spends money", async () => {
    api.getMcpPermissionRequest.mockResolvedValue({
      ok: true,
      request: { ...pending, requested: [pending.requested[0]] },
    });
    render(<McpApprovePage />);
    await screen.findByText("Launch, cancel, and terminate GPU instances");
    expect(screen.queryByText("dash.mcp.approve.money_warning")).not.toBeInTheDocument();
  });

  it("approves and tells the user to go back to the agent", async () => {
    api.getMcpPermissionRequest.mockResolvedValue({ ok: true, request: pending });
    api.decideMcpPermissionRequest.mockResolvedValue({ ok: true, request: { ...pending, status: "approved" } });
    render(<McpApprovePage />);
    fireEvent.click(await screen.findByRole("button", { name: "dash.mcp.approve.approve" }));
    await waitFor(() => expect(api.decideMcpPermissionRequest).toHaveBeenCalledWith("ABCDE-FGHJK", true));
    expect(await screen.findByText("dash.mcp.approve.approved_title")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "dash.mcp.approve.approve" })).not.toBeInTheDocument();
  });

  it("declines", async () => {
    api.getMcpPermissionRequest.mockResolvedValue({ ok: true, request: pending });
    api.decideMcpPermissionRequest.mockResolvedValue({ ok: true, request: { ...pending, status: "declined" } });
    render(<McpApprovePage />);
    fireEvent.click(await screen.findByRole("button", { name: "dash.mcp.approve.decline" }));
    await waitFor(() => expect(api.decideMcpPermissionRequest).toHaveBeenCalledWith("ABCDE-FGHJK", false));
    expect(await screen.findByText("dash.mcp.approve.declined_title")).toBeInTheDocument();
  });

  it("offers nothing to approve for an expired link", async () => {
    api.getMcpPermissionRequest.mockResolvedValue({ ok: true, request: { ...pending, status: "expired" } });
    render(<McpApprovePage />);
    expect(await screen.findByText("dash.mcp.approve.expired_title")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "dash.mcp.approve.approve" })).not.toBeInTheDocument();
  });

  it("treats someone else's link, and no code at all, as nothing to approve", async () => {
    api.getMcpPermissionRequest.mockRejectedValue(new Error("404"));
    const { unmount } = render(<McpApprovePage />);
    expect(await screen.findByText("dash.mcp.approve.missing_title")).toBeInTheDocument();
    unmount();

    params.code = null;
    api.getMcpPermissionRequest.mockClear();
    render(<McpApprovePage />);
    expect(screen.getByText("dash.mcp.approve.missing_title")).toBeInTheDocument();
    expect(api.getMcpPermissionRequest).not.toHaveBeenCalled();
  });
});
