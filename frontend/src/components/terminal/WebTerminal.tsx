"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Maximize2, Minimize2, X, Wifi, WifiOff } from "lucide-react";

interface WebTerminalProps {
  instanceId: string;
  onClose?: () => void;
}

/**
 * xterm.js-powered web terminal that connects via WebSocket to
 * /ws/terminal/{instanceId} for an interactive shell session.
 *
 * Dynamically imports xterm to avoid SSR issues.
 */
export function WebTerminal({ instanceId, onClose }: WebTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const termRef = useRef<unknown>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [maximized, setMaximized] = useState(false);

  const connect = useCallback(async () => {
    if (!containerRef.current) return;

    // Dynamic import of xterm (avoids SSR)
    const { Terminal } = await import("@xterm/xterm");
    const { FitAddon } = await import("@xterm/addon-fit");
    const { WebLinksAddon } = await import("@xterm/addon-web-links");

    // Cleanup old terminal
    if (termRef.current) {
      (termRef.current as InstanceType<typeof Terminal>).dispose();
    }

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      theme: {
        background: "#0b0f1a",
        foreground: "#e2e8f0",
        cursor: "#38bdf8",
        selectionBackground: "#38bdf833",
        black: "#1e293b",
        red: "#ef4444",
        green: "#22c55e",
        yellow: "#eab308",
        blue: "#3b82f6",
        magenta: "#a855f7",
        cyan: "#06b6d4",
        white: "#f1f5f9",
      },
      scrollback: 5000,
      allowProposedApi: true,
    });
    termRef.current = term;

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.loadAddon(new WebLinksAddon());

    term.open(containerRef.current);
    fitAddon.fit();

    // WebSocket connection
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const apiBase = process.env.NEXT_PUBLIC_API_URL || window.location.origin;
    const wsUrl = apiBase.replace(/^https?:/, protocol);
    const token = document.cookie
      .split("; ")
      .find((c) => c.startsWith("xcelsior_token="))
      ?.split("=")[1] || "";

    const ws = new WebSocket(`${wsUrl}/ws/terminal/${instanceId}?token=${encodeURIComponent(token)}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);

      // Send initial terminal size
      ws.send(JSON.stringify({
        type: "resize",
        cols: term.cols,
        rows: term.rows,
      }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "output") {
          term.write(msg.data);
        } else if (msg.type === "error") {
          term.write(`\r\n\x1b[31m${msg.message}\x1b[0m\r\n`);
          setError(msg.message);
        } else if (msg.type === "exit") {
          term.write(`\r\n\x1b[33m[Process exited with code ${msg.code}]\x1b[0m\r\n`);
          setConnected(false);
        }
      } catch {
        // Non-JSON message — write raw
        term.write(event.data);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      term.write("\r\n\x1b[33m[Connection closed]\x1b[0m\r\n");
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      setConnected(false);
    };

    // Send input to backend
    term.onData((data: string) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "input", data }));
      }
    });

    // Handle terminal resize
    const handleResize = () => {
      fitAddon.fit();
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          type: "resize",
          cols: term.cols,
          rows: term.rows,
        }));
      }
    };

    const observer = new ResizeObserver(handleResize);
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      ws.close();
      term.dispose();
    };
  }, [instanceId]);

  useEffect(() => {
    const cleanup = connect();
    return () => {
      cleanup?.then((fn) => fn?.());
      wsRef.current?.close();
    };
  }, [connect]);

  return (
    <Card
      className={`flex flex-col overflow-hidden ${
        maximized
          ? "fixed inset-4 z-50 rounded-xl shadow-2xl"
          : "h-[400px]"
      }`}
    >
      {/* Toolbar */}
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 bg-[#0b0f1a]">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-muted">Terminal</span>
          <Badge variant={connected ? "active" : error ? "failed" : "warning"} className="text-[10px] px-1.5 py-0">
            {connected ? "Connected" : error ? "Error" : "Disconnected"}
          </Badge>
          {connected ? (
            <Wifi className="h-3 w-3 text-emerald" />
          ) : (
            <WifiOff className="h-3 w-3 text-text-muted" />
          )}
        </div>
        <div className="flex items-center gap-1">
          {!connected && (
            <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={() => connect()}>
              Reconnect
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => setMaximized(!maximized)}
          >
            {maximized ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
          </Button>
          {onClose && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {/* Terminal container */}
      <div
        ref={containerRef}
        className="flex-1 bg-[#0b0f1a] p-1"
        style={{ minHeight: 0 }}
      />
    </Card>
  );
}
