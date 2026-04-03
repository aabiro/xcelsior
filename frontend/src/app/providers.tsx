"use client";

import { AuthProvider } from "@/lib/auth";
import { ThemeProvider } from "@/lib/theme";
import { LocaleProvider } from "@/lib/locale";
import { WalletConnectProvider } from "@/lib/wallet-connect";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <LocaleProvider>
        <WalletConnectProvider>
          <AuthProvider>{children}</AuthProvider>
        </WalletConnectProvider>
      </LocaleProvider>
    </ThemeProvider>
  );
}
