"use client";

import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";

const DesktopAppRuntime = dynamic(
  () => import("@/components/DesktopAppRuntime").then((m) => ({ default: m.DesktopAppRuntime })),
  { ssr: false },
);
const InstallBanner = dynamic(
  () => import("@/components/InstallBanner").then((m) => ({ default: m.InstallBanner })),
  { ssr: false },
);
const ServiceWorkerRegistrar = dynamic(
  () =>
    import("@/components/ServiceWorkerRegistrar").then((m) => ({
      default: m.ServiceWorkerRegistrar,
    })),
  { ssr: false },
);

export function RootClientWidgets() {
  const pathname = usePathname();
  const onDashboard = Boolean(pathname?.startsWith("/dashboard"));

  return (
    <>
      {onDashboard ? <DesktopAppRuntime /> : null}
      <InstallBanner />
      <ServiceWorkerRegistrar />
    </>
  );
}