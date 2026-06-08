"use client";

import { ThemeProvider } from "@/lib/theme";
import { LocaleProvider } from "@/lib/locale";

/** Root providers — AuthProvider is mounted only on dashboard + auth routes (see AuthProviderShell). */
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <LocaleProvider>{children}</LocaleProvider>
    </ThemeProvider>
  );
}