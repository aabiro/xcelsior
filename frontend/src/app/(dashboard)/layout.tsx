import type { Metadata } from "next";
import { DashboardShell } from "./dashboard-shell";
import { ApiStatusBanner } from "@/components/ApiStatusBanner";
import { AuthProviderShell } from "@/components/AuthProviderShell";
import { DashboardProviders } from "./dashboard-providers";

export const metadata: Metadata = {
  title: "Dashboard",
  robots: { index: false, follow: false },
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProviderShell>
      <DashboardProviders>
        <ApiStatusBanner />
        <DashboardShell>{children}</DashboardShell>
      </DashboardProviders>
    </AuthProviderShell>
  );
}
