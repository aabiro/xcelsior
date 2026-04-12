import type { Metadata } from "next";
import { DashboardShell } from "./dashboard-shell";
import { ApiStatusBanner } from "@/components/ApiStatusBanner";

export const metadata: Metadata = {
  title: "Dashboard",
  robots: { index: false, follow: false },
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <ApiStatusBanner />
      <DashboardShell>{children}</DashboardShell>
    </>
  );
}
