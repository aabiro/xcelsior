"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import "@xterm/xterm/css/xterm.css";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, Maximize2, Minimize2, X } from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────────

type ConnState =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error";

interface WebTerminalProps {
  instanceId: string;
  onClose?: () => void;
}

// ── Reconnect constants ────────────────────────────────────────────────────────

const MAX_RECONNECT_ATTEMPTS = 8;
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_CAP_MS = 30_000;

// ── Cookie helper ─────────────────────────────────────────────────────────────

function _getSessionToken(): string {
  // Prefer xcelsior_session; fall back to xcelsior_token (legacy)
  for (const name of ["xcelsior_session", "xcelsior_token"]) {
    const entry = document.cookie.split("; ").find((c) => c.startsWith(`${name}=`));
    if (entry) return decodeURIComponent(entry.split("=")[1] ?? "");
  }
  return "";
}

// ── Badge helper ──────────────────────────────────────────────────────────────

function _badgeVariant(state: ConnState): "active" | "warning" | "failed" | "default" {
  if (state === "connected") return "active";
  if (state === "error") return "failed";
  if (state === "disconnected") return "default";
  return "warning";
}

// ── Component ─────────────────────────────────────────────────────────────────

/**
 * Production web terminal powered by xterm.js v6 with:
 * - Binary WebSocket protocol (raw PTY bytes → xterm; control frames are JSON text)
 * - WebGL renderer (graceful Canvas fallback)
 * - SearchAddon (Ctrl+F), Unicode11Addon, WebLinksAddon
 * - Ctrl+R → sends \x12 reverse-search signal to shell (not reimplemented client-side)
 * - Exponential backoff reconnect (up to 8 attempts)
 * - WebGL texture atlas cleared on tab visibility restore (fixes glyph artifacts)
 * - customGlyphs: true, convertEol: false (PTY handles CRLF natively)
 * - macOptionIsMeta: true, scrollback: 10000
 *
 * Dynamically imported to avoid SSR issues.
 */
export function WebTerminal({ instanceId, onClose }: WebTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const termRef = useRef<import("@xterm/xterm").Terminal | null>(null);
  const fitAddonRef = useRef<import("@xterm/addon-fit").FitAddon | null>(null);
  const webglAddonRef = useRef<import("@xterm/addon-webgl").WebglAddon | null>(null);
  const searchAddonRef = useRef<import("@xterm/addon-search").SearchAddon | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const visChangeRef = useRef<(() => void) | null>(null);

  const [connState, setConnState] = useState<ConnState>("connecting");
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [maximized, setMaximized] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const searchOpenRef = useRef(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // ── Terminal fit helper ─────────────────────────────────────────────────────
  const _fit = useCallback(() => {
    const term = termRef.current;
    const fitAddon = fitAddonRef.current;
    const ws = wsRef.current;
    if (!fitAddon || !term) return;
    try {
      fitAddon.fit();
    } catch {
      // ignore ResizeObserver errors before terminal is attached
    }
    webglAddonRef.current?.clearTextureAtlas?.();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
    }
  }, []);

  // ── Search helpers ──────────────────────────────────────────────────────────
  const _searchNext = useCallback(() => {
    searchAddonRef.current?.findNext(searchQuery, {
      incremental: false,
      regex: false,
      caseSensitive: false,
    });
  }, [searchQuery]);

  const _searchPrev = useCallback(() => {
    searchAddonRef.current?.findPrevious(searchQuery, {
      incremental: false,
      regex: false,
      caseSensitive: false,
    });
  }, [searchQuery]);

  // ── Connect ─────────────────────────────────────────────────────────────────
  const connect = useCallback(async () => {
    if (!containerRef.current || !mountedRef.current) return;

    // Dynamic imports — avoids SSR; tree-shaken from server bundle
    const { Terminal } = await import("@xterm/xterm");
    const { FitAddon } = await import("@xterm/addon-fit");
    const { WebLinksAddon } = await import("@xterm/addon-web-links");
    const { SearchAddon } = await import("@xterm/addon-search");
    const { Unicode11Addon } = await import("@xterm/addon-unicode11");

    if (!mountedRef.current) return;

    // Dispose old terminal instance before creating a new one
    if (termRef.current) {
      termRef.current.dispose();
      termRef.current = null;
      webglAddonRef.current = null;
      searchAddonRef.current = null;
    }

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily:
        "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace",
      theme: {
        background: "#0b0f1a",
        foreground: "#e2e8f0",
        cursor: "#38bdf8",
        cursorAccent: "#0b0f1a",
        selectionBackground: "#38bdf833",
        black: "#1e293b",
        red: "#ef4444",
        green: "#22c55e",
        yellow: "#eab308",
        blue: "#3b82f6",
        magenta: "#a855f7",
        cyan: "#06b6d4",
        white: "#f1f5f9",
        brightBlack: "#334155",
        brightWhite: "#f8fafc",
      },
      scrollback: 10_000,
      allowProposedApi: true,
      customGlyphs: true,
      convertEol: false,       // PTY handles CRLF; setting true breaks vim/htop
      macOptionIsMeta: true,
    });
    termRef.current = term;

    const fitAddon = new FitAddon();
    fitAddonRef.current = fitAddon;
    const searchAddon = new SearchAddon();
    searchAddonRef.current = searchAddon;
    const unicode11Addon = new Unicode11Addon();

    term.loadAddon(fitAddon);
    term.loadAddon(searchAddon);
    term.loadAddon(unicode11Addon);
    term.loadAddon(new WebLinksAddon());

    // Activate Unicode 11 (wider emoji / CJK)
    term.unicode.activeVersion = "11";

    term.open(containerRef.current);
    fitAddon.fit();
    term.focus();

    // WebGL renderer — graceful Canvas fallback
    try {
      const { WebglAddon } = await import("@xterm/addon-webgl");
      if (mountedRef.current) {
        const wgl = new WebglAddon();
        // Dispose WebGL addon on context loss to prevent blank screen
        wgl.onContextLoss(() => {
          wgl.dispose();
          webglAddonRef.current = null;
        });
        term.loadAddon(wgl);
        webglAddonRef.current = wgl;
      }
    } catch {
      // Canvas fallback is automatic — no action needed
    }

    // Clear texture atlas on tab restore to fix glyph artifacts after sleep
    if (visChangeRef.current) {
      document.removeEventListener("visibilitychange", visChangeRef.current);
    }
    const _onVisChange = () => {
      if (!document.hidden) {
        webglAddonRef.current?.clearTextureAtlas?.();
        _fit();
      }
    };
    visChangeRef.current = _onVisChange;
    document.addEventListener("visibilitychange", _onVisChange);

    // ── Keyboard interceptors ───────────────────────────────────────────────
    term.attachCustomKeyEventHandler((ev: KeyboardEvent) => {
      const ws = wsRef.current;
      const open = ws && ws.readyState === WebSocket.OPEN;

      // Ctrl+F → open search panel (suppress default browser find)
      if (ev.ctrlKey && ev.key === "f" && ev.type === "keydown") {
        setSearchOpen(true);
        setTimeout(() => searchInputRef.current?.focus(), 0);
        return false;
      }

      // Ctrl+R → send reverse-search signal \x12 to shell
      if (ev.ctrlKey && ev.key === "r" && ev.type === "keydown") {
        if (open) ws.send(JSON.stringify({ type: "input", data: "\x12" }));
        return false;
      }

      // Ctrl+C — copy selection if any, otherwise pass through as SIGINT
      if (ev.ctrlKey && !ev.shiftKey && ev.key === "c" && ev.type === "keydown") {
        const sel = term.getSelection();
        if (sel) {
          navigator.clipboard.writeText(sel).catch(() => {});
          return false;
        }
        // No selection → fall through: xterm sends \x03 (SIGINT) to shell
      }

      // Ctrl+Shift+V → paste from clipboard
      if (ev.ctrlKey && ev.shiftKey && ev.key === "V" && ev.type === "keydown") {
        navigator.clipboard.readText().then((text) => {
          if (open && text) ws.send(JSON.stringify({ type: "input", data: text }));
        }).catch(() => {});
        return false;
      }

      // Escape → close search panel if open
      if (ev.key === "Escape" && ev.type === "keydown") {
        if (searchOpenRef.current) {
          setSearchOpen(false);
          setTimeout(() => termRef.current?.focus(), 0);
          return false;
        }
      }

      return true;
    });

    // ── WebSocket ───────────────────────────────────────────────────────────
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? window.location.origin;
    const wsBase = apiBase.replace(/^https?:/, protocol);
    const token = _getSessionToken();
    const wsUrl = `${wsBase}/ws/terminal/${instanceId}${token ? `?token=${encodeURIComponent(token)}` : ""}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    setConnState("connecting");

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      reconnectAttemptsRef.current = 0;
      setConnState("connected");
      setStatusMsg("");
      ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
      term.focus();
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!termRef.current) return;

      if (event.data instanceof ArrayBuffer) {
        // Binary frame: raw PTY bytes — write directly (fastest path)
        termRef.current.write(new Uint8Array(event.data));
        return;
      }

      // Text frame: JSON control message
      try {
        const msg = JSON.parse(event.data as string) as {
          type: string;
          message?: string;
          code?: number;
          retry?: boolean;
          ts?: number;
        };

        switch (msg.type) {
          case "status":
            setStatusMsg(msg.message ?? "");
            if (msg.message && termRef.current) {
              if (msg.retry) {
                // Progress updates overwrite same line (container polling)
                termRef.current.write(
                  `\r\x1b[2m${msg.message}\x1b[0m\x1b[K`
                );
              } else {
                // Final status (e.g., "Connected") on its own line
                termRef.current.write(
                  `\r\x1b[2m${msg.message}\x1b[0m\r\n`
                );
              }
            }
            if (msg.retry) setConnState("connecting");
            else setConnState("connected");
            break;

          case "error":
            termRef.current.write(
              `\r\n\x1b[31m\u26a0 ${msg.message ?? "Error"}\x1b[0m\r\n`
            );
            setConnState("error");
            setStatusMsg(msg.message ?? "Error");
            break;

          case "exit":
            termRef.current.write(
              `\r\n\x1b[33m[Process exited with code ${msg.code ?? 0}]\x1b[0m\r\n`
            );
            setConnState("disconnected");
            break;

          case "pong":
            // Keepalive acknowledged — no UI action needed
            break;
        }
      } catch {
        // Malformed JSON — ignore silently
      }
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      if (termRef.current) {
        termRef.current.write("\r\n\x1b[33m[Connection closed]\x1b[0m\r\n");
      }

      const att = reconnectAttemptsRef.current;
      if (att < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttemptsRef.current = att + 1;
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** att, RECONNECT_CAP_MS);
        setConnState("reconnecting");
        setStatusMsg(`Reconnecting in ${Math.round(delay / 1000)}s\u2026`);
        reconnectTimerRef.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, delay);
      } else {
        if (termRef.current) {
          termRef.current.write(
            "\r\n\x1b[31m[Could not reconnect. Click Reconnect to try again.]\x1b[0m\r\n"
          );
        }
        setConnState("disconnected");
        setStatusMsg("Connection lost");
      }
    };

    ws.onerror = () => {
      // onclose fires immediately after onerror — let it handle reconnect
      setConnState("error");
    };

    // ── Input ───────────────────────────────────────────────────────────────
    term.onData((data: string) => {
      const w = wsRef.current;
      if (w && w.readyState === WebSocket.OPEN) {
        w.send(JSON.stringify({ type: "input", data }));
      }
      // Keep auth idle timer alive — xterm captures key events before document
      window.dispatchEvent(new Event("keydown"));
    });

    // ── Resize observer ─────────────────────────────────────────────────────
    if (observerRef.current) {
      observerRef.current.disconnect();
    }
    const obs = new ResizeObserver(() => _fit());
    obs.observe(containerRef.current!);
    observerRef.current = obs;

    return () => {
      if (visChangeRef.current) {
        document.removeEventListener("visibilitychange", visChangeRef.current);
        visChangeRef.current = null;
      }
      obs.disconnect();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instanceId]);

  // ── Mount / unmount ─────────────────────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;
    let cleanupInner: (() => void) | undefined;
    connect().then((fn) => { cleanupInner = fn; });
    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      observerRef.current?.disconnect();
      wsRef.current?.close();
      webglAddonRef.current?.dispose();
      termRef.current?.dispose();
      cleanupInner?.();
    };
  }, [connect]);

  // ── Refit after maximize toggle ─────────────────────────────────────────────
  useEffect(() => {
    const id = setTimeout(_fit, 50);
    return () => clearTimeout(id);
  }, [maximized, _fit]);

  // ── Search panel auto-focus / cleanup ──────────────────────────────────────
  useEffect(() => {
    searchOpenRef.current = searchOpen;
    if (searchOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 0);
    } else {
      termRef.current?.focus();
      setSearchQuery("");
      searchAddonRef.current?.clearDecorations?.();
    }
  }, [searchOpen]);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <Card
      className={`flex flex-col overflow-hidden ${
        maximized ? "fixed inset-4 z-50 rounded-xl shadow-2xl" : "h-full"
      }`}
    >
      {/* ── Toolbar ── */}
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 bg-[#0b0f1a]">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted">Terminal</span>

          {/* Connection state badge */}
          {(connState === "connecting" || connState === "reconnecting") ? (
            <Badge variant="warning" className="text-[10px] px-1.5 py-0 flex items-center gap-1">
              <Loader2 className="h-2.5 w-2.5 animate-spin text-sky-400" />
              {connState === "reconnecting" ? "Reconnecting" : "Connecting"}
            </Badge>
          ) : (
            <Badge
              variant={_badgeVariant(connState)}
              className="text-[10px] px-1.5 py-0"
            >
              {connState === "connected"
                ? (statusMsg || "Connected")
                : connState === "disconnected"
                ? "Disconnected"
                : "Error"}
            </Badge>
          )}
        </div>

        <div className="flex items-center gap-1">
          {(connState === "disconnected" || connState === "error") && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs"
              onClick={() => {
                reconnectAttemptsRef.current = 0;
                connect();
              }}
            >
              Reconnect
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => setMaximized((m) => !m)}
          >
            {maximized ? (
              <Minimize2 className="h-3 w-3" />
            ) : (
              <Maximize2 className="h-3 w-3" />
            )}
          </Button>
          {onClose && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {/* ── Terminal canvas ── */}
      <div
        ref={containerRef}
        className="relative flex-1 bg-[#0b0f1a] p-1"
        style={{ minHeight: 0 }}
        onClick={() => termRef.current?.focus()}
      >
        {/* Ctrl+F search panel — overlay inside terminal area */}
        {searchOpen && (
          <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1 rounded border border-border bg-[#0f172a] px-2 py-1 shadow-lg">
            <input
              ref={searchInputRef}
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                if (e.target.value) {
                  searchAddonRef.current?.findNext(e.target.value, {
                    incremental: true,
                    regex: false,
                    caseSensitive: false,
                  });
                }
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.shiftKey ? _searchPrev() : _searchNext();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setSearchOpen(false);
                }
              }}
              placeholder="Find\u2026"
              className="h-6 w-40 bg-transparent text-xs text-slate-200 outline-none placeholder:text-slate-500"
              aria-label="Terminal search"
              spellCheck={false}
            />
            <button
              onClick={_searchPrev}
              className="px-1 text-slate-400 hover:text-slate-200"
              aria-label="Previous match"
            >
              &#8593;
            </button>
            <button
              onClick={_searchNext}
              className="px-1 text-slate-400 hover:text-slate-200"
              aria-label="Next match"
            >
              &#8595;
            </button>
            <button
              onClick={() => setSearchOpen(false)}
              className="px-1 text-slate-400 hover:text-slate-200"
              aria-label="Close search"
            >
              &#215;
            </button>
          </div>
        )}
      </div>
    </Card>
  );
}
