"use client";

import { useEffect, useState } from "react";
import { CheckCircle, Loader2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  evaluateHostSetup,
  observeBrowser,
  type HostSetupObservation,
  type HostSetupRow,
} from "@/lib/host-setup-requirements";

const ROW_DELAY_MS = 280;

export async function observeThisBrowser(): Promise<HostSetupObservation> {
  const userAgent = typeof navigator === "undefined" ? "" : navigator.userAgent;
  let webgpu = false;
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    webgpu = true;
  }
  return observeBrowser({ userAgent, webgpu });
}

export function HostMachineSetupCard({
  observe = observeThisBrowser,
}: {
  observe?: () => Promise<HostSetupObservation>;
}) {
  const [phase, setPhase] = useState<"idle" | "running" | "done">("idle");
  const [rows, setRows] = useState<HostSetupRow[]>([]);
  const [visible, setVisible] = useState(0);

  useEffect(() => {
    if (phase !== "running" || rows.length === 0 || visible >= rows.length) return;
    const timer = setTimeout(() => setVisible((count) => count + 1), ROW_DELAY_MS);
    return () => clearTimeout(timer);
  }, [phase, rows, visible]);

  useEffect(() => {
    if (phase === "running" && rows.length > 0 && visible >= rows.length) {
      setPhase("done");
    }
  }, [phase, rows.length, visible]);

  async function runScan() {
    setPhase("running");
    setRows([]);
    setVisible(0);
    const observation = await observe();
    setRows(evaluateHostSetup(observation));
  }

  const shown = rows.slice(0, visible);

  return (
    <div className="mt-3 rounded-lg border border-border/50 bg-surface/60 p-3.5" data-testid="host-setup-card">
      <Button
        type="button"
        variant="outline"
        className="h-10 border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/10"
        onClick={() => void runScan()}
        disabled={phase === "running"}
      >
        Test your machine setup
      </Button>
      {phase === "running" && (rows.length === 0 || visible < rows.length) ? (
        <div className="mt-3 flex items-center gap-2 text-xs text-text-muted" role="status" data-testid="host-setup-spinner">
          <Loader2 className="h-4 w-4 animate-spin text-accent-cyan" aria-hidden="true" />
          Checking this machine
        </div>
      ) : null}
      {shown.length > 0 ? (
        <ul className="mt-3 space-y-2" aria-live="polite">
          {shown.map((row) => (
            <li
              key={row.id}
              data-testid="host-setup-row"
              data-requirement={row.id}
              data-mark={row.mark}
              className={cn(
                "flex items-start gap-2 text-xs transition-colors",
                row.mark === "pass" && "text-emerald",
                row.mark === "fail" && "text-red-400",
                row.mark === "unresolved" && "text-text-muted",
              )}
            >
              {row.mark === "pass" ? (
                <CheckCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              ) : row.mark === "fail" ? (
                <XCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              ) : (
                <span className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
              )}
              <span>
                <span className="font-medium text-text-primary">{row.label}</span>
                <span className="mt-0.5 block">{row.detail}</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
