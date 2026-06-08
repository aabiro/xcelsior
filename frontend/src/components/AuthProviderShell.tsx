"use client";

import { AuthProvider } from "@/lib/auth";

/** Wrap routes that need session state (dashboard + auth entry). Omit on pure marketing pages. */
export function AuthProviderShell({ children }: { children: React.ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}