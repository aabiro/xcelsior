import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import { DashboardShell } from "./dashboard-shell";
import { ApiStatusBanner } from "@/components/ApiStatusBanner";
import { AuthProviderShell } from "@/components/AuthProviderShell";
import { DashboardProviders } from "./dashboard-providers";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
  preload: false,
});

export const metadata: Metadata = {
  title: "Dashboard",
  robots: { index: false, follow: false },
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className={geistMono.variable}>
      <AuthProviderShell>
        <DashboardProviders>
          <ApiStatusBanner />
          <DashboardShell>{children}</DashboardShell>
        </DashboardProviders>
      </AuthProviderShell>
    </div>
  );
}
