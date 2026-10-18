"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

/** Terminal-style code block: window chrome (filename + traffic lights) + inline copy. */
export function CodeBlock({
  filename,
  code,
  onCopy,
  className,
}: {
  filename?: string;
  code: string;
  /** Optional side-effect after copy (e.g. a toast). Clipboard write is handled here. */
  onCopy?: () => void;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard.writeText(code);
    setCopied(true);
    onCopy?.();
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-border bg-surface shadow-sm",
        className,
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border bg-surface-hover px-3 py-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
          {filename && <span className="ml-2 truncate font-mono text-[11px] text-text-muted">{filename}</span>}
        </div>
        <button
          type="button"
          onClick={copy}
          className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-surface hover:text-text-primary"
        >
          {copied ? <Check className="h-3 w-3 text-emerald" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-xs leading-relaxed text-text-primary">{code}</pre>
    </div>
  );
}