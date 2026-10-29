"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function CopyableText({
  text,
  className,
  mono = true,
}: {
  text: string;
  className?: string;
  mono?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button
      type="button"
      onClick={copy}
      className={cn(
        "inline-flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors cursor-pointer max-w-full",
        mono && "font-mono",
        className,
      )}
      title="Copy"
    >
      <span className="truncate">{text}</span>
      {copied ? <Check className="h-3 w-3 shrink-0 text-emerald" /> : <Copy className="h-3 w-3 shrink-0" />}
    </button>
  );
}