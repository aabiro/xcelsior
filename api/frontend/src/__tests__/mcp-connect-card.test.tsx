import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// The card loads each tab's key once, shows it in full (the server now keeps
// Quick Connect keys encrypted so it can return them on every visit), and
// recovers from a failed load with Retry.

const api = vi.hoisted(() => ({ getMcpQuickConnect: vi.fn(), setMcpQuickConnectPermissions: vi.fn() }));
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
  beforeEach(() => {
    api.getMcpQuickConnect.mockReset();
    api.setMcpQuickConnectPermissions.mockReset();
  });

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

  // ── Permissions: Read / Write / Custom ──────────────────────────────────

  const WRITE = ["instances:read", "instances:write", "instances:connect", "volumes:read", "volumes:write", "billing:read"];
  const READ = ["instances:read", "volumes:read", "billing:read"];
  const withPermissions = {
    ...issued,
    scopes: WRITE,
    permission_preset: "write",
    permission_presets: { read: READ, write: WRITE },
  };
  // The mocked `t` returns the key, which the picker treats as "no
  // translation" and replaces with the bare resource and verb.
  const box = (resource: string, verb: string) =>
    screen.getByRole("checkbox", { name: `${resource}: ${verb}` });
  const tab = (id: string) => screen.getByRole("tab", { name: `dash.mcp.perm.tab_${id}` });

  it("opens on the saved preset with its boxes already ticked", async () => {
    api.getMcpQuickConnect.mockResolvedValue(withPermissions);
    render(<McpConnectCard />);
    await waitFor(() => expect(tab("write")).toHaveAttribute("aria-selected", "true"));
    expect(screen.getAllByRole("checkbox")).toHaveLength(WRITE.length);
    for (const checkbox of screen.getAllByRole("checkbox")) expect(checkbox).toBeChecked();
    // Nothing to save until something changes.
    expect(screen.queryByRole("button", { name: "dash.mcp.perm.save" })).not.toBeInTheDocument();
  });

  it("Read ticks only the read boxes and saves without replacing the key", async () => {
    api.getMcpQuickConnect.mockResolvedValue(withPermissions);
    api.setMcpQuickConnectPermissions.mockResolvedValue({
      ok: true,
      client_id: "c1",
      permission_preset: "read",
      scopes: READ,
      permission_presets: { read: READ, write: WRITE },
      keys_updated: 1,
    });
    render(<McpConnectCard />);
    fireEvent.click(await screen.findByRole("tab", { name: "dash.mcp.perm.tab_read" }));
    expect(box("instances", "read")).toBeChecked();
    expect(box("billing", "read")).toBeChecked();
    expect(box("instances", "write")).not.toBeChecked();
    expect(box("instances", "connect")).not.toBeChecked();
    expect(box("volumes", "write")).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: "dash.mcp.perm.save" }));
    await waitFor(() => expect(api.setMcpQuickConnectPermissions).toHaveBeenCalledWith("mcp", "read", READ));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "dash.mcp.perm.save" })).not.toBeInTheDocument(),
    );
    expect(tab("read")).toHaveAttribute("aria-selected", "true");
    // Same key on screen, and it was never asked for again.
    expect(screen.getByRole("button", { name: "dash.mcp.copy_token" })).toHaveTextContent(KEY);
    expect(api.getMcpQuickConnect).toHaveBeenCalledTimes(1);
  });

  it("Custom starts with everything ticked", async () => {
    api.getMcpQuickConnect.mockResolvedValue({ ...withPermissions, scopes: READ, permission_preset: "read" });
    render(<McpConnectCard />);
    fireEvent.click(await screen.findByRole("tab", { name: "dash.mcp.perm.tab_custom" }));
    for (const checkbox of screen.getAllByRole("checkbox")) expect(checkbox).toBeChecked();
  });

  it("ticking write ticks read, and editing a preset moves to Custom", async () => {
    api.getMcpQuickConnect.mockResolvedValue({ ...withPermissions, scopes: ["billing:read"], permission_preset: "custom" });
    render(<McpConnectCard />);
    await waitFor(() => expect(tab("custom")).toHaveAttribute("aria-selected", "true"));
    expect(box("volumes", "read")).not.toBeChecked();
    fireEvent.click(box("volumes", "write"));
    expect(box("volumes", "write")).toBeChecked();
    expect(box("volumes", "read")).toBeChecked();
    // Unticking read takes the write that depends on it with it.
    fireEvent.click(box("volumes", "read"));
    expect(box("volumes", "write")).not.toBeChecked();

    fireEvent.click(tab("read"));
    fireEvent.click(box("instances", "connect"));
    expect(tab("custom")).toHaveAttribute("aria-selected", "true");
    expect(box("instances", "read")).toBeChecked();
    expect(box("instances", "write")).not.toBeChecked();
  });

  it("Undo puts the picker back to what the key holds", async () => {
    api.getMcpQuickConnect.mockResolvedValue(withPermissions);
    render(<McpConnectCard />);
    fireEvent.click(await screen.findByRole("tab", { name: "dash.mcp.perm.tab_read" }));
    fireEvent.click(screen.getByRole("button", { name: "dash.mcp.perm.reset" }));
    expect(tab("write")).toHaveAttribute("aria-selected", "true");
    expect(box("instances", "write")).toBeChecked();
    expect(api.setMcpQuickConnectPermissions).not.toHaveBeenCalled();
  });

  it("shows an approved extra as a box that can be removed and survives a Custom save", async () => {
    const held = [...WRITE, "billing:topup"];
    api.getMcpQuickConnect.mockResolvedValue({
      ...withPermissions,
      scopes: held,
      permission_preset: "custom",
      approved_extras: ["billing:topup"],
    });
    api.setMcpQuickConnectPermissions.mockResolvedValue({
      ok: true,
      client_id: "c1",
      permission_preset: "custom",
      scopes: held.filter((s) => s !== "volumes:write"),
      permission_presets: { read: READ, write: WRITE },
      approved_extras: ["billing:topup"],
      keys_updated: 1,
    });
    render(<McpConnectCard />);
    await waitFor(() => expect(box("billing", "topup")).toBeChecked());
    expect(screen.getAllByRole("checkbox")).toHaveLength(WRITE.length + 1);

    // Editing something else keeps it in what is saved.
    fireEvent.click(box("volumes", "write"));
    fireEvent.click(screen.getByRole("button", { name: "dash.mcp.perm.save" }));
    await waitFor(() =>
      expect(api.setMcpQuickConnectPermissions).toHaveBeenCalledWith(
        "mcp",
        "custom",
        held.filter((s) => s !== "volumes:write"),
      ),
    );

    // Choosing a preset drops it: no tab grants it.
    fireEvent.click(tab("write"));
    expect(box("billing", "topup")).not.toBeChecked();
  });
});
