"use client";

/** Suppress text hydration warnings for client-i18n legal/long-form pages. */
export function HydrationGuard({ children }: { children: React.ReactNode }) {
  return <div suppressHydrationWarning>{children}</div>;
}