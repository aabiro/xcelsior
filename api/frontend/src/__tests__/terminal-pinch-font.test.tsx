import { act, cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  attachPinchToResize,
  pinchFontSize,
  storeFontSize,
  storedFontSize,
  TERMINAL_FONT_DEFAULT,
  TERMINAL_FONT_MAX,
  TERMINAL_FONT_MIN,
} from "@/components/terminal/pinch-font";
import { WebTerminal } from "@/components/terminal/WebTerminal";

const KEY = "xcelsior.terminal.fontSize";

type Point = [number, number];

function touch(el: Element, type: string, points: Point[]) {
  const ev = new Event(type, { bubbles: true, cancelable: true });
  const touches = points.map(([clientX, clientY]) => ({ clientX, clientY }));
  Object.defineProperty(ev, "touches", { value: touches });
  el.dispatchEvent(ev);
  return ev;
}

describe("pinch arithmetic", () => {
  it("follows the fingers and stays inside the limits", () => {
    expect(pinchFontSize(13, 100, 100)).toBe(13);
    expect(pinchFontSize(13, 100, 50)).toBe(7);
    expect(pinchFontSize(13, 100, 10)).toBe(TERMINAL_FONT_MIN);
    expect(pinchFontSize(13, 100, 400)).toBe(TERMINAL_FONT_MAX);
    expect(pinchFontSize(13, 0, 80)).toBe(13);
  });
});

describe("the remembered size", () => {
  beforeEach(() => window.localStorage.clear());

  it("is the default until a pinch changes it", () => {
    expect(storedFontSize()).toBe(TERMINAL_FONT_DEFAULT);
    storeFontSize(8);
    expect(window.localStorage.getItem(KEY)).toBe("8");
    expect(storedFontSize()).toBe(8);
    storeFontSize(TERMINAL_FONT_DEFAULT);
    expect(window.localStorage.getItem(KEY)).toBeNull();
  });

  it("ignores a value it did not write", () => {
    window.localStorage.setItem(KEY, "huge");
    expect(storedFontSize()).toBe(TERMINAL_FONT_DEFAULT);
    window.localStorage.setItem(KEY, "200");
    expect(storedFontSize()).toBe(TERMINAL_FONT_MAX);
  });

  it("falls back to the default when storage is refused", () => {
    const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    const set = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("SecurityError");
    });
    expect(storedFontSize()).toBe(TERMINAL_FONT_DEFAULT);
    expect(() => storeFontSize(9)).not.toThrow();
    get.mockRestore();
    set.mockRestore();
  });
});

describe("pinching the terminal", () => {
  function setup() {
    const outer = document.createElement("div");
    const inner = document.createElement("div"); // stands in for xterm's own elements
    outer.appendChild(inner);
    document.body.appendChild(outer);
    let size = 13;
    const font = { get: () => size, set: vi.fn((n: number) => { size = n; }), done: vi.fn() };
    const xterm = vi.fn();
    inner.addEventListener("touchmove", xterm);
    const detach = attachPinchToResize(outer, font);
    return { outer, inner, font, xterm, detach, size: () => size };
  }

  it("two fingers spreading make the text bigger, and the page does not zoom", () => {
    const t = setup();
    touch(t.inner, "touchstart", [[100, 100], [200, 100]]);
    const move = touch(t.inner, "touchmove", [[50, 100], [250, 100]]);
    expect(t.size()).toBe(TERMINAL_FONT_MAX); // 13 x 2, capped
    expect(move.defaultPrevented).toBe(true);
    expect(t.xterm).not.toHaveBeenCalled();
    touch(t.inner, "touchend", [[50, 100]]);
    expect(t.font.done).toHaveBeenCalledTimes(1);
    expect(t.font.done).toHaveBeenCalledWith(TERMINAL_FONT_MAX);
  });

  it("two fingers closing make it smaller, so wide output fits", () => {
    const t = setup();
    touch(t.inner, "touchstart", [[100, 100], [300, 100]]);
    touch(t.inner, "touchmove", [[150, 100], [250, 100]]);
    expect(t.size()).toBe(7);
  });

  it("leaves one finger to xterm, so scrolling still works", () => {
    const t = setup();
    touch(t.inner, "touchstart", [[100, 100]]);
    const move = touch(t.inner, "touchmove", [[100, 40]]);
    expect(move.defaultPrevented).toBe(false);
    expect(t.xterm).toHaveBeenCalledTimes(1);
    expect(t.font.set).not.toHaveBeenCalled();
    expect(t.font.done).not.toHaveBeenCalled();
  });

  it("stops Safari's own pinch from zooming the page", () => {
    const t = setup();
    const ev = new Event("gesturestart", { bubbles: true, cancelable: true });
    t.inner.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

  it("stops listening when detached", () => {
    const t = setup();
    t.detach();
    touch(t.inner, "touchstart", [[100, 100], [200, 100]]);
    touch(t.inner, "touchmove", [[50, 100], [250, 100]]);
    expect(t.font.set).not.toHaveBeenCalled();
  });
});

// ── In the terminal itself ────────────────────────────────────────────────────

const created: { options: Record<string, unknown>; el: HTMLElement | null }[] = [];

vi.mock("@/lib/api", () => ({
  createTerminalTicket: vi.fn(async () => ({ ok: true, ticket: "t", expires_in: 30 })),
}));
vi.mock("@xterm/xterm", () => ({
  Terminal: class {
    cols = 36; rows = 20; unicode = { activeVersion: "" };
    options: Record<string, unknown>;
    el: HTMLElement | null = null;
    constructor(options: Record<string, unknown>) {
      this.options = { ...options };
      created.push(this);
    }
    loadAddon() {} focus() {} dispose() {} write() {} onData() {}
    open(el: HTMLElement) { this.el = el; }
    attachCustomWheelEventHandler() {} attachCustomKeyEventHandler() {}
  },
}));
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
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor(public url = "") { Socket.instances.push(this); }
}

describe("the web terminal", () => {
  beforeEach(async () => {
    await import("@xterm/xterm");
    created.length = 0;
    Socket.instances = [];
    window.localStorage.clear();
    vi.stubGlobal("WebSocket", Socket);
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  async function open() {
    render(<WebTerminal instanceId="pinch-job" />);
    await act(async () => { await vi.dynamicImportSettled(); });
    const socket = Socket.instances[0];
    act(() => socket.onopen?.());
    const term = created[created.length - 1];
    return { socket, term, screen: term.el as HTMLElement };
  }

  it("opens at the size this browser last pinched to", async () => {
    window.localStorage.setItem(KEY, "9");
    const { term } = await open();
    expect(term.options.fontSize).toBe(9);
  });

  it("tells the shell its new size when pinched, and remembers it", async () => {
    const { socket, term, screen } = await open();
    expect(term.options.fontSize).toBe(TERMINAL_FONT_DEFAULT);
    socket.send.mockClear();
    act(() => {
      touch(screen, "touchstart", [[100, 100], [300, 100]]);
      touch(screen, "touchmove", [[150, 100], [250, 100]]);
      touch(screen, "touchend", []);
    });
    expect(term.options.fontSize).toBe(7);
    const frames = socket.send.mock.calls.map(([raw]) => JSON.parse(raw as string));
    expect(frames).toContainEqual({ type: "resize", cols: 36, rows: 20 });
    expect(window.localStorage.getItem(KEY)).toBe("7");
  });
});
