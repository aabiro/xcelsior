"use client";

import { DesktopRuntimeProvider } from "@/lib/desktop/runtime";
import { WalletConnectProvider } from "@/lib/wallet-connect";

/** Dashboard-only providers (WalletConnect/AppKit + native desktop runtime are heavy). */
export function DashboardProviders({ children }: { children: React.ReactNode }) {
  return (
    <DesktopRuntimeProvider>
      <WalletConnectProvider>{children}</WalletConnectProvider>
    </DesktopRuntimeProvider>
  );
}