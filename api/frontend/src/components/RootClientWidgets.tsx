"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

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

function useIdleReady(timeoutMs = 5000) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (typeof window.requestIdleCallback === "function") {
      const id = window.requestIdleCallback(() => setReady(true), { timeout: timeoutMs });
      return () => window.cancelIdleCallback(id);
    }
    const id = window.setTimeout(() => setReady(true), Math.min(timeoutMs, 2500));
    return () => window.clearTimeout(id);
  }, [timeoutMs]);

  return ready;
}

export function RootClientWidgets() {
  const pathname = usePathname();
  const onDashboard = Boolean(pathname?.startsWith("/dashboard"));
  const pwaReady = useIdleReady(onDashboard ? 1500 : 5000);

  return (
    <>
      {pwaReady ? (
        <>
          <InstallBanner />
          <ServiceWorkerRegistrar />
        </>
      ) : null}
    </>
  );
}