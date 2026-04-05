"use client";

import { useEffect, type ReactNode } from "react";
import { X } from "lucide-react";

interface DialogProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: ReactNode;
  maxWidth?: string;
}

export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  maxWidth = "max-w-lg",
}: DialogProps) {
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className={`relative z-10 w-full ${maxWidth} rounded-xl border border-border bg-surface shadow-2xl mx-4 max-h-[85vh] flex flex-col`}>
        <div className="flex items-center justify-between px-6 pt-5 pb-3">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {description && <p className="mt-0.5 text-sm text-text-secondary">{description}</p>}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-surface-hover text-text-muted hover:text-text-primary transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="px-6 pb-6 overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}
