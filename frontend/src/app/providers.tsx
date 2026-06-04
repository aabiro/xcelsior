"use client";

import { AuthProvider } from "@/lib/auth";
import { DesktopRuntimeProvider } from "@/lib/desktop/runtime";
import { ThemeProvider } from "@/lib/theme";
import { LocaleProvider } from "@/lib/locale";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <DesktopRuntimeProvider>
      <ThemeProvider>
        <LocaleProvider>
          <AuthProvider>{children}</AuthProvider>
        </LocaleProvider>
      </ThemeProvider>
    </DesktopRuntimeProvider>
  );
}
