"use client";

import dynamic from "next/dynamic";
import { DesktopRuntimeProvider } from "@/lib/desktop/runtime";
import { WalletConnectProvider } from "@/lib/wallet-connect";

const DesktopAppRuntime = dynamic(
  () => import("@/components/DesktopAppRuntime").then((m) => ({ default: m.DesktopAppRuntime })),
  { ssr: false },
);

/** Dashboard-only providers (WalletConnect/AppKit + native desktop runtime are heavy). */
export function DashboardProviders({ children }: { children: React.ReactNode }) {
  return (
    <DesktopRuntimeProvider>
      <DesktopAppRuntime />
      <WalletConnectProvider>{children}</WalletConnectProvider>
    </DesktopRuntimeProvider>
  );
}