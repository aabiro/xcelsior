"use client";

import { WalletConnectProvider } from "@/lib/wallet-connect";

/** Dashboard-only providers (WalletConnect/AppKit is heavy — keep off marketing routes). */
export function DashboardProviders({ children }: { children: React.ReactNode }) {
  return <WalletConnectProvider>{children}</WalletConnectProvider>;
}