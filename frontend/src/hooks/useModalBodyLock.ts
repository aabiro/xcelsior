"use client";

import { useEffect } from "react";

/** Prevent background scroll while a modal is open (iOS-safe). */
export function useModalBodyLock(open: boolean) {
  useEffect(() => {
    if (!open || typeof document === "undefined") return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);
}