"use client";

import { AuthProvider } from "@/lib/auth";
import { DesktopRuntimeProvider } from "@/lib/desktop/runtime";
import { ThemeProvider } from "@/lib/theme";
import { LocaleProvider } from "@/lib/locale";
import { WalletConnectProvider } from "@/lib/wallet-connect";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <DesktopRuntimeProvider>
      <ThemeProvider>
        <LocaleProvider>
          <WalletConnectProvider>
            <AuthProvider>{children}</AuthProvider>
          </WalletConnectProvider>
        </LocaleProvider>
      </ThemeProvider>
    </DesktopRuntimeProvider>
  );
}
