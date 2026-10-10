import { act, cleanup, render, screen } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WebTerminal } from "@/components/terminal/WebTerminal";

vi.mock("@/lib/api", () => ({
  createTerminalTicket: vi.fn(async () => ({ ok: true, ticket: "test-ticket", expires_in: 30 })),
}));
vi.mock("@xterm/xterm", () => ({ Terminal: class {
  cols = 80; rows = 24; unicode = { activeVersion: "" }; options = {};
  loadAddon() {} open() {} focus() {} dispose() {} write() {} onData() {}
  attachCustomWheelEventHandler() {} attachCustomKeyEventHandler() {}
} }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: class { fit() {} } }));
vi.mock("@xterm/addon-search", () => ({ SearchAddon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/addon-unicode11", () => ({ Unicode11Addon: class {} }));
vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: class { dispose() {} } }));

class Socket {
  static OPEN = 1;
  static instances: Socket[] = [];
  readyState = 1;
  binaryType = "";
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor() { Socket.instances.push(this); }
}

beforeEach(async () => {
  // Resolve the mocked module before Strict Mode starts concurrent imports.
  await import("@xterm/xterm");
  vi.useFakeTimers();
  Socket.instances = [];
  vi.stubGlobal("WebSocket", Socket);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function openTerminal() {
  render(<WebTerminal instanceId="startup-job" />);
  await act(async () => { await vi.dynamicImportSettled(); });
  expect(Socket.instances).toHaveLength(1);
  const socket = Socket.instances[0];
  act(() => socket.onopen?.());
  return socket;
}

describe("terminal startup", () => {
  it("opens only one connection after the Strict Mode setup and cleanup cycle", async () => {
    render(<StrictMode><WebTerminal instanceId="startup-job" /></StrictMode>);
    await act(async () => { await vi.dynamicImportSettled(); });
    expect(Socket.instances).toHaveLength(1);
  });

  it("keeps a long image pull connected while receiving progress without pong frames", async () => {
    const socket = await openTerminal();
    expect(screen.getByText("Connecting")).toBeInTheDocument();
    for (let i = 0; i < 5; i += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(20_000);
        socket.onmessage?.({ data: JSON.stringify({ type: "status", message: "Pulling image…", retry: true }) });
      });
    }
    expect(socket.close).not.toHaveBeenCalled();
    expect(Socket.instances).toHaveLength(1);
    act(() => socket.onmessage?.({ data: JSON.stringify({ type: "status", message: "Connected to startup-job", retry: false }) }));
    expect(screen.getByText("Connected")).toBeInTheDocument();
  });

  it("still closes a genuinely silent connection", async () => {
    const socket = await openTerminal();
    await act(async () => { await vi.advanceTimersByTimeAsync(50_000); });
    expect(socket.close).toHaveBeenCalledWith(4000, "stale");
  });

  it("preserves the actual startup failure when the server closes the socket", async () => {
    const socket = await openTerminal();
    const message = "Instance failed during startup: required volume mount failed";
    act(() => {
      socket.onmessage?.({ data: JSON.stringify({ type: "error", message, code: 4003 }) });
      socket.onclose?.({ code: 4003, reason: "" });
    });
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reconnect" })).toBeInTheDocument();
  });
});
