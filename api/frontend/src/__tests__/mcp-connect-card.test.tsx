import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// The card loads each tab's key once, shows it in full (the server now keeps
// Quick Connect keys encrypted so it can return them on every visit), and
// recovers from a failed load with Retry.

const api = vi.hoisted(() => ({ getMcpQuickConnect: vi.fn() }));
vi.mock("@/lib/api", () => api);
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: { user_id: "u1" } }) }));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => (key.endsWith("_prompt") ? "Connect with key {token} at {mcp_url}" : key),
  }),
}));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { McpConnectCard } from "@/components/dashboard/mcp-connect-card";

const KEY = "xcel_ai_full_key_value_1234567890";
const issued = {
  ok: true,
  client_id: "c1",
  access_token: KEY,
  key_id: "k1",
  key_prefix: "xcel_ai_…7890",
  in_use: true,
  last_used_at: "2026-10-12T00:00:00Z",
  expires_in: null,
  scopes: [],
  mcp_url: "https://mcp.xcelsior.ca/mcp",
  api_url: "https://xcelsior.ca",
};

describe("MCP connect card", () => {
  beforeEach(() => api.getMcpQuickConnect.mockReset());

  it("loads the tab's key once and shows it in full", async () => {
    api.getMcpQuickConnect.mockResolvedValue(issued);
    render(<McpConnectCard />);
    await waitFor(() => expect(screen.getByRole("button", { name: "dash.mcp.copy_token" })).toHaveTextContent(KEY));
    expect(api.getMcpQuickConnect).toHaveBeenCalledTimes(1);
    expect(api.getMcpQuickConnect).toHaveBeenCalledWith(false, "mcp");
    expect(screen.queryByText("xcel_ai_…7890")).not.toBeInTheDocument();
  });

  it("recovers from a failed load with Retry", async () => {
    api.getMcpQuickConnect.mockRejectedValueOnce(new Error("offline")).mockResolvedValue(issued);
    render(<McpConnectCard />);
    const retry = await screen.findByRole("button", { name: "dash.mcp.retry" });
    expect(api.getMcpQuickConnect).toHaveBeenCalledTimes(1);
    fireEvent.click(retry);
    await waitFor(() => expect(screen.getByRole("button", { name: "dash.mcp.copy_token" })).toHaveTextContent(KEY));
    expect(api.getMcpQuickConnect).toHaveBeenCalledTimes(2);
  });
});
