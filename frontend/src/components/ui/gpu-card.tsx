import { cn } from "@/lib/utils";
import { Cpu } from "lucide-react";

interface GpuCardProps {
  model: string;
  vram?: number;
  price?: number;
  status?: string;
  className?: string;
  children?: React.ReactNode;
}

export function GpuCard({ model, vram, price, status, className, children }: GpuCardProps) {
  const statusColor =
    status === "active" || status === "available"
      ? "text-emerald"
      : status === "offline"
        ? "text-accent-red"
        : "text-text-muted";

  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-surface p-5 card-hover",
        className,
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-ice-blue/10">
            <Cpu className="h-4 w-4 text-ice-blue" />
          </div>
          <div>
            <p className="font-semibold text-sm">{model}</p>
            {vram != null && (
              <p className="text-xs text-text-muted">{vram}GB VRAM</p>
            )}
          </div>
        </div>
        {status && (
          <span className={cn("text-xs font-medium capitalize", statusColor)}>
            {status}
          </span>
        )}
      </div>
      {price != null && (
        <p className="text-lg font-bold font-mono mb-1">
          ${price.toFixed(2)} <span className="text-xs text-text-muted font-normal">CAD/hr</span>
        </p>
      )}
      {children}
    </div>
  );
}
