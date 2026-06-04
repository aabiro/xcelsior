"use client";

import dynamic from "next/dynamic";

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
  return (
    <>
      <DesktopAppRuntime />
      <InstallBanner />
      <ServiceWorkerRegistrar />
    </>
  );
}