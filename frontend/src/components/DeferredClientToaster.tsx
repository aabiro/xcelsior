"use client";

import dynamic from "next/dynamic";

const ClientToaster = dynamic(
  () => import("@/components/ClientToaster").then((m) => ({ default: m.ClientToaster })),
  { ssr: false },
);

export function DeferredClientToaster() {
  return <ClientToaster />;
}